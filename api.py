import os
import sys
import io
import time
import math
import base64
import wave
import asyncio
from typing import Optional
from contextlib import asynccontextmanager

import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, Response
import warnings
try:
    from speechbrain.inference import SepformerSeparation, EncoderClassifier
except ImportError:
    from speechbrain.pretrained import SepformerSeparation, EncoderClassifier

try:
    import onnxruntime as ort
except ImportError:
    ort = None

try:
    from torchcodec.decoders import AudioDecoder
except Exception:
    AudioDecoder = None


# ===========  配置区域  ==============
SEP_MODELS = {}
CLS = None
PRELOAD_TIMES = {
    "sepformer_2": 0.0,
    "sepformer_3": 0.0,
    "classifier": 0.0,
}
SEP_SR = 16000
CLS_SR = 16000
MAIN_DEVICE = "cpu"
MATCH_DEVICE = "cpu"
ENABLE_ONNX = False
FORCE_ONNX_CPU = False
ONNX_DIR = os.path.join(os.path.dirname(__file__), "onnx")

# 物理核心数，用于 ONNX 推理加速，具体数值得自己试试，太大也会变慢，要适中
NUM_THREADS = 16
print(f"Using {NUM_THREADS} threads for ONNX inference")

# ============================================================

def _is_torch_tensor(x):
    try:
        return isinstance(x, torch.Tensor)
    except Exception:
        return False


def _load_audio_mono_bytes(b):
    try:
        y, sr = sf.read(io.BytesIO(b))
        if y.ndim == 2:
            y = y.mean(axis=1)
        y = y.astype(np.float32)
        return y, int(sr)
    except Exception:
        pass
    try:
        if AudioDecoder is not None:
            dec = AudioDecoder(src=io.BytesIO(b))
            y_dec, sr = dec.decode()
            if _is_torch_tensor(y_dec):
                y_t = y_dec
                if y_t.dim() == 2 and y_t.shape[0] > 1:
                    y_t = y_t.mean(dim=0)
                y = y_t.detach().cpu().numpy().astype("float32")
            else:
                y_np = np.array(y_dec)
                if y_np.ndim == 2:
                    if y_np.shape[0] > 1:
                        y_np = y_np.mean(axis=0)
                    else:
                        y_np = y_np.mean(axis=1)
                y = y_np.astype(np.float32)
            return y, int(sr)
    except Exception:
        pass
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
            y_t, sr = torchaudio.load(io.BytesIO(b))
        if y_t.shape[0] > 1:
            y_t = y_t.mean(dim=0)
        y = y_t.detach().cpu().numpy().astype("float32")
        return y, int(sr)
    except Exception:
        pass
    raise RuntimeError("无法读取音频")


def _resample_np(y, sr_from, sr_to):
    if sr_from == sr_to:
        return y
    ratio = float(sr_to) / float(sr_from)
    new_len = int(math.floor(len(y) * ratio))
    x_old = np.linspace(0.0, 1.0, num=len(y), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=new_len, endpoint=False)
    y_new = np.interp(x_new, x_old, y).astype(np.float32)
    return y_new


def _resample_torch(y_t, sr_from, sr_to):
    if sr_from == sr_to:
        return y_t
    try:
        return torchaudio.functional.resample(y_t, sr_from, sr_to)
    except Exception:
        y_np = y_t.detach().cpu().numpy()
        y_np = _resample_np(y_np, sr_from, sr_to)
        return torch.tensor(y_np).unsqueeze(0)


def _wav_bytes(y, sr):
    try:
        y_np = y.detach().cpu().numpy() if _is_torch_tensor(y) else y
        bio = io.BytesIO()
        sf.write(bio, y_np.astype(np.float32), sr, format="WAV")
        return bio.getvalue()
    except Exception:
        pass
    y_np = y.detach().cpu().numpy() if _is_torch_tensor(y) else y
    y_np = np.clip(y_np, -1.0, 1.0)
    y_i16 = (y_np * 32767.0).astype(np.int16)
    bio = io.BytesIO()
    wf = wave.open(bio, "wb")
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(sr)
    wf.writeframes(y_i16.tobytes())
    wf.close()
    return bio.getvalue()



class RedirectStream(object):
    def __init__(self):
        pass

    def __enter__(self):
        self.devnull_fd = os.open(os.devnull, os.O_WRONLY)
        self.saved_stdout_fd = os.dup(1)
        self.saved_stderr_fd = os.dup(2)
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(self.devnull_fd, 1)
        os.dup2(self.devnull_fd, 2)
        return self

    def __exit__(self, *args):
        os.dup2(self.saved_stdout_fd, 1)
        os.dup2(self.saved_stderr_fd, 2)
        os.close(self.saved_stdout_fd)
        os.close(self.saved_stderr_fd)
        os.close(self.devnull_fd)


class OnnxSepformer:
    def __init__(self, path, sample_rate=8000, device="cpu"):
        if ort is None:
            raise ImportError("onnxruntime is required for ONNX support")
        
        self.path = path
        self.hparams = type("HParams", (), {"sample_rate": sample_rate})()
        
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 4  # Fatal only
        
        # Set graph optimization level
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        if device == "cuda":
            # Priority: CUDA -> CPU
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            try:
                with RedirectStream():
                    self.sess = ort.InferenceSession(path, providers=providers, sess_options=sess_options)
            except Exception as e:
                print(f"Failed to initialize ONNX session with CUDA: {e}")
                self.sess = None
            
            # Check if we actually got a GPU provider
            current_providers = self.sess.get_providers() if self.sess else []
            if "CUDAExecutionProvider" not in current_providers:
                 print("Falling back to CPU execution provider for ONNX")
                 try:
                    providers = ["CPUExecutionProvider"]
                    self.sess = ort.InferenceSession(path, providers=providers, sess_options=sess_options)
                 except Exception as e:
                     print(f"Failed to initialize ONNX session with CPU: {e}")
                     self.sess = None
        else:
            if FORCE_ONNX_CPU:
                # 使用量化后的模型（如果存在）
                # 假设量化模型与原模型在同一目录，优先级：_static_int8.onnx > _int8.onnx
                path_static_int8 = path.replace(".onnx", "_static_int8.onnx")
                path_int8 = path.replace(".onnx", "_int8.onnx")
                
                if os.path.exists(path_static_int8):
                    path = path_static_int8
                elif os.path.exists(path_int8):
                    path = path_int8
                    
                # CPU 优化配置
                sess_options.intra_op_num_threads = NUM_THREADS
                sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                
                self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"], sess_options=sess_options)
            else:
                path_static_int8 = path.replace(".onnx", "_static_int8.onnx")
                path_int8 = path.replace(".onnx", "_int8.onnx")
                
                if os.path.exists(path_static_int8):
                    path = path_static_int8
                elif os.path.exists(path_int8):
                    path = path_int8
                self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"], sess_options=sess_options)
            
    def warmup(self):
        if self.sess:
            dummy_input = np.random.randn(1, 8000).astype(np.float32)
            input_name = self.sess.get_inputs()[0].name
            try:
                self.sess.run(None, {input_name: dummy_input})
                print(f"Warmup successful for {input_name}")
            except Exception as e:
                print(f"Warmup failed for {input_name}: {e}")

    def separate_batch(self, mix):
        if self.sess is None:
            raise RuntimeError("ONNX model session is not initialized")
        input_name = self.sess.get_inputs()[0].name
        
        is_fp16 = self.path.endswith("_fp16.onnx")
        
        if "CUDAExecutionProvider" in self.sess.get_providers() and mix.device.type == "cuda":
            if is_fp16:
                mix = mix.contiguous().half() 
                elem_type = np.float16
            else:
                mix = mix.contiguous().float() 
                elem_type = np.float32
            
            data_ptr = mix.data_ptr()
            shape = tuple(mix.shape)
            
            io_binding = self.sess.io_binding()
            
            io_binding.bind_input(
                name=input_name,
                device_type='cuda',
                device_id=mix.device.index if mix.device.index is not None else 0,
                element_type=elem_type,
                shape=shape,
                buffer_ptr=data_ptr
            )
            
            output_name = self.sess.get_outputs()[0].name
            io_binding.bind_output(output_name, 'cuda')
            
            self.sess.run_with_iobinding(io_binding)
            
            ort_output = io_binding.get_outputs()[0]
            
            try:
                from torch.utils.dlpack import from_dlpack
                out_tensor = from_dlpack(ort_output.to_dlpack())
                if out_tensor.dtype == torch.float16:
                    out_tensor = out_tensor.float()
                return out_tensor
            except Exception:
                out_np = ort_output.numpy()
                return torch.from_numpy(out_np).to(mix.device).float()
        else:
            mix_np = mix.detach().cpu().numpy()
            if is_fp16:
                mix_np = mix_np.astype(np.float16)
            
            out = self.sess.run(None, {input_name: mix_np})[0]
            out_tensor = torch.from_numpy(out).to(mix.device)
            if out_tensor.dtype == torch.float16:
                out_tensor = out_tensor.float()
            return out_tensor


class OnnxClassifier:
    def __init__(self, path, feature_extractor, sample_rate=16000, device="cpu"):
        if ort is None:
            raise ImportError("onnxruntime is required for ONNX support")
            
        self.path = path
        self.compute_features = feature_extractor
        self.hparams = type("HParams", (), {"sample_rate": sample_rate})()
        
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 4  # Fatal only
        
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        if device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            try:
                with RedirectStream():
                    self.sess = ort.InferenceSession(path, providers=providers, sess_options=sess_options)
            except Exception as e:
                print(f"Failed to initialize ONNX session with CUDA: {e}")
                self.sess = None
            
            current_providers = self.sess.get_providers() if self.sess else []
            if "CUDAExecutionProvider" not in current_providers:
                 print("Falling back to CPU execution provider for ONNX")
                 try:
                    providers = ["CPUExecutionProvider"]
                    self.sess = ort.InferenceSession(path, providers=providers, sess_options=sess_options)
                 except Exception as e:
                     print(f"Failed to initialize ONNX session with CPU: {e}")
                     self.sess = None
        else:
            if FORCE_ONNX_CPU:
                path_static_int8 = path.replace(".onnx", "_static_int8.onnx")
                path_int8 = path.replace(".onnx", "_int8.onnx")
                
                if os.path.exists(path_static_int8):
                    path = path_static_int8
                elif os.path.exists(path_int8):
                    path = path_int8
                    
                sess_options.intra_op_num_threads = NUM_THREADS
                sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                
                self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"], sess_options=sess_options)
            else:
                path_static_int8 = path.replace(".onnx", "_static_int8.onnx")
                path_int8 = path.replace(".onnx", "_int8.onnx")
                
                if os.path.exists(path_static_int8):
                    path = path_static_int8
                elif os.path.exists(path_int8):
                    path = path_int8
                self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"], sess_options=sess_options)

    def warmup(self):
        if self.sess:
            dummy_input = np.random.randn(1, 100, 80).astype(np.float32)
            input_name = self.sess.get_inputs()[0].name
            try:
                self.sess.run(None, {input_name: dummy_input})
                print(f"Warmup successful for {input_name}")
            except Exception as e:
                print(f"Warmup failed for {input_name}: {e}")

    def encode_batch(self, wavs):
        if self.sess is None:
            raise RuntimeError("ONNX model session is not initialized")
        feats = self.compute_features(wavs)
        input_name = self.sess.get_inputs()[0].name
        
        is_fp16 = self.path.endswith("_fp16.onnx")
        
        if "CUDAExecutionProvider" in self.sess.get_providers() and wavs.device.type == "cuda":
            if is_fp16:
                feats = feats.contiguous().half()
                elem_type = np.float16
            else:
                feats = feats.contiguous().float()
                elem_type = np.float32
                
            data_ptr = feats.data_ptr()
            shape = tuple(feats.shape)
            
            io_binding = self.sess.io_binding()
            
            io_binding.bind_input(
                name=input_name,
                device_type='cuda',
                device_id=wavs.device.index if wavs.device.index is not None else 0,
                element_type=elem_type,
                shape=shape,
                buffer_ptr=data_ptr
            )
            
            output_name = self.sess.get_outputs()[0].name
            io_binding.bind_output(output_name, 'cuda')
            
            self.sess.run_with_iobinding(io_binding)
            
            ort_output = io_binding.get_outputs()[0]
            
            try:
                from torch.utils.dlpack import from_dlpack
                out_tensor = from_dlpack(ort_output.to_dlpack())
                if out_tensor.dtype == torch.float16:
                    out_tensor = out_tensor.float()
                return out_tensor
            except Exception:
                out_np = ort_output.numpy()
                return torch.from_numpy(out_np).to(wavs.device).float()
        else:
            feats_np = feats.detach().cpu().numpy()
            if is_fp16:
                feats_np = feats_np.astype(np.float16)
            out = self.sess.run(None, {input_name: feats_np})[0]
            out_tensor = torch.from_numpy(out).to(wavs.device)
            if out_tensor.dtype == torch.float16:
                out_tensor = out_tensor.float()
            return out_tensor


class SepformerWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder = model.mods.encoder
        self.masknet = model.mods.masknet
        self.decoder = model.mods.decoder
        self.num_spks = model.hparams.num_spks
        
    def forward(self, mix):
        mix_w = self.encoder(mix)
        est_mask = self.masknet(mix_w)
        mix_w = torch.stack([mix_w] * self.num_spks)
        sep_h = mix_w * est_mask

        est_source = torch.cat(
            [
                self.decoder(sep_h[i]).unsqueeze(-1)
                for i in range(self.num_spks)
            ],
            dim=-1,
        )

        T_origin = mix.size(1)
        T_est = est_source.size(1)
        if T_origin > T_est:
            est_source = F.pad(est_source, (0, 0, 0, T_origin - T_est))   
        else:
            est_source = est_source[:, :T_origin, :]
        
        return est_source


@asynccontextmanager
async def lifespan(app: FastAPI):
    global SEP_MODELS, CLS, PRELOAD_TIMES, SEP_SR, CLS_SR, MAIN_DEVICE, MATCH_DEVICE
    if FORCE_ONNX_CPU:
        MAIN_DEVICE = "cpu"
        MATCH_DEVICE = "cpu"
    else:
        MAIN_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        MATCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    t0 = time.time()

    if ENABLE_ONNX:
        # ... (Existing ONNX Loading Logic) ...
        sep2_path = os.path.join(ONNX_DIR, "sepformer_wsj02mix.onnx")
        sep2_path_fp16 = sep2_path.replace(".onnx", "_fp16.onnx")
        sep2_path_static_int8 = sep2_path.replace(".onnx", "_static_int8.onnx")
        sep2_path_int8 = sep2_path.replace(".onnx", "_int8.onnx")
        
        final_path = None
        if os.path.exists(sep2_path_fp16):
            final_path = sep2_path_fp16
        elif os.path.exists(sep2_path_static_int8):
            final_path = sep2_path_static_int8
        elif os.path.exists(sep2_path_int8):
            final_path = sep2_path_int8
        elif os.path.exists(sep2_path):
            final_path = sep2_path
            
        if final_path:
            print(f"Loading ONNX model for sepformer_wsj02mix: {final_path}")
            try:
                SEP_MODELS["2"] = OnnxSepformer(final_path, sample_rate=8000, device=MAIN_DEVICE)
            except Exception as e:
                print(f"Failed to load ONNX model {final_path}: {e}")
                final_path = None 
        
        if not final_path:
            print("No valid ONNX model found for sepformer_wsj02mix, falling back to PyTorch.")
            SEP_MODELS["2"] = SepformerSeparation.from_hparams(
                source="speechbrain/sepformer-wsj02mix",
                savedir=os.path.join("pretrained_models", "sepformer-wsj02mix"),
                run_opts={"device": MAIN_DEVICE},
            )
    else:
        SEP_MODELS["2"] = SepformerSeparation.from_hparams(
            source="speechbrain/sepformer-wsj02mix",
            savedir=os.path.join("pretrained_models", "sepformer-wsj02mix"),
            run_opts={"device": MAIN_DEVICE},
        )

    PRELOAD_TIMES["sepformer_2"] = time.time() - t0
    SEP_SR = int(getattr(SEP_MODELS["2"].hparams, "sample_rate", 16000))
    t1 = time.time()

    if ENABLE_ONNX:
        # ... (Existing ONNX Loading Logic) ...
        sep3_path = os.path.join(ONNX_DIR, "sepformer_wsj03mix.onnx")
        sep3_path_fp16 = sep3_path.replace(".onnx", "_fp16.onnx")
        sep3_path_static_int8 = sep3_path.replace(".onnx", "_static_int8.onnx")
        sep3_path_int8 = sep3_path.replace(".onnx", "_int8.onnx")
        
        final_path = None
        if os.path.exists(sep3_path_fp16):
            final_path = sep3_path_fp16
        elif os.path.exists(sep3_path_static_int8):
            final_path = sep3_path_static_int8
        elif os.path.exists(sep3_path_int8):
            final_path = sep3_path_int8
        elif os.path.exists(sep3_path):
            final_path = sep3_path
            
        if final_path:
            print(f"Loading ONNX model for sepformer_wsj03mix: {final_path}")
            try:
                SEP_MODELS["3"] = OnnxSepformer(final_path, sample_rate=8000, device=MAIN_DEVICE)
            except Exception as e:
                print(f"Failed to load ONNX model {final_path}: {e}")
                final_path = None

        if not final_path:
            print("No valid ONNX model found for sepformer_wsj03mix, falling back to PyTorch.")
            SEP_MODELS["3"] = SepformerSeparation.from_hparams(
                source="speechbrain/sepformer-wsj03mix",
                savedir=os.path.join("pretrained_models", "sepformer-wsj03mix"),
                run_opts={"device": MAIN_DEVICE},
            )
    else:
        SEP_MODELS["3"] = SepformerSeparation.from_hparams(
            source="speechbrain/sepformer-wsj03mix",
            savedir=os.path.join("pretrained_models", "sepformer-wsj03mix"),
            run_opts={"device": MAIN_DEVICE},
        )

    PRELOAD_TIMES["sepformer_3"] = time.time() - t1
    t2 = time.time()

    sb_cls = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=os.path.join("pretrained_models", "spkrec-ecapa-voxceleb"),
        run_opts={"device": "cpu" if ENABLE_ONNX else MATCH_DEVICE},
    )

    cls_loaded_onnx = False
    if ENABLE_ONNX:
        cls_path = os.path.join(ONNX_DIR, "ecapa_voxceleb.onnx")
        cls_path_fp16 = cls_path.replace(".onnx", "_fp16.onnx")
        cls_path_static_int8 = cls_path.replace(".onnx", "_static_int8.onnx")
        cls_path_int8 = cls_path.replace(".onnx", "_int8.onnx")
        
        final_path = None
        if os.path.exists(cls_path_fp16):
            final_path = cls_path_fp16
        elif os.path.exists(cls_path_static_int8):
            final_path = cls_path_static_int8
        elif os.path.exists(cls_path_int8):
            final_path = cls_path_int8
        elif os.path.exists(cls_path):
            final_path = cls_path
            
        if final_path:
            print(f"Loading ONNX model for ecapa_voxceleb: {final_path}")
            try:
                feature_extractor = sb_cls.mods.compute_features
                CLS = OnnxClassifier(final_path, feature_extractor, sample_rate=16000, device=MATCH_DEVICE)
                cls_loaded_onnx = True
            except Exception as e:
                print(f"Failed to load ONNX model {final_path}: {e}")
                cls_loaded_onnx = False

    if not cls_loaded_onnx:
        print("Loading PyTorch model for ecapa_voxceleb...")
        if ENABLE_ONNX:
             if MATCH_DEVICE != "cpu":
                 sb_cls = sb_cls.to(MATCH_DEVICE)
        CLS = sb_cls
    else:
        pass 
    PRELOAD_TIMES["classifier"] = time.time() - t2
    CLS_SR = int(getattr(CLS.hparams, "sample_rate", 16000))

    # ----------------------------------------------------------------
    # 优化点 1: 始终执行 Warmup (即使不是 ONNX)
    # ----------------------------------------------------------------
    print("Warming up models...")
    
    # Warmup separation models
    dummy_input = torch.randn(1, 16000).to(MAIN_DEVICE)
    
    for key, model in SEP_MODELS.items():
        try:
            if ENABLE_ONNX:
                is_fp16 = hasattr(model, 'path') and model.path.endswith("_fp16.onnx")
                current_input = dummy_input.clone()
                if is_fp16:
                    current_input = current_input.half()
                model.separate_batch(current_input)
            else:
                # PyTorch path
                model.separate_batch(dummy_input)
        except Exception as e:
            print(f"Warmup failed for sepformer_{key}: {e}")

    # Warmup classifier
    if CLS is not None:
        dummy_wavs = torch.randn(1, 16000).to(MATCH_DEVICE)
        try:
            is_fp16 = ENABLE_ONNX and hasattr(CLS, 'path') and CLS.path.endswith("_fp16.onnx")
            if is_fp16:
                dummy_wavs = dummy_wavs.half()
            CLS.encode_batch(dummy_wavs)
        except Exception as e:
            print(f"Warmup failed for classifier: {e}")
            
    print("Models warmed up!")
    
    yield
    
    SEP_MODELS.clear()
    if CLS is not None:
        CLS = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Models cleared.")


async def run_warmup():
    print("Starting model warmup...")
    loop = asyncio.get_running_loop()
    
    if "2" in SEP_MODELS:
        await loop.run_in_executor(None, SEP_MODELS["2"].warmup)
    if "3" in SEP_MODELS:
        await loop.run_in_executor(None, SEP_MODELS["3"].warmup)
    if CLS:
        await loop.run_in_executor(None, CLS.warmup)
        
    print("Model warmup completed!")


app = FastAPI(lifespan=lifespan)


def _separate(yb, num_speakers, normalize=True):
    model = SEP_MODELS.get(str(num_speakers))
    if model is None:
        raise RuntimeError("模型未加载")
    t_sep_start = time.time()
    try:
        with torch.inference_mode():
            est_sources = model.separate_batch(yb)
            if MAIN_DEVICE == "cuda":
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
    except RuntimeError as e:
        msg = str(e)
        if ("CUDA" in msg) or ("device-side assert" in msg):
            model = SEP_MODELS.get(str(num_speakers))
            with torch.inference_mode():
                est_sources = model.separate_batch(yb.cpu())
        else:
            raise
    t_sep_end = time.time()
    
    # ----------------------------------------------------------------
    # 优化点 2: 保持数据在 Device 上，利用 GPU 进行归一化和排序
    # ----------------------------------------------------------------
    # 优先保持在原设备，如果 est_sources 不是 Tensor (例如 ONNX 输出 numpy)，转为 Tensor
    sources = est_sources if _is_torch_tensor(est_sources) else torch.tensor(est_sources, device=MAIN_DEVICE)
    sources = sources.detach().float() # 不立即转 CPU
    
    # 处理维度
    if sources.ndim == 3:
        if sources.shape[0] == 1:
            sources = sources.squeeze(0)
        elif sources.shape[1] == 1:
            sources = sources.squeeze(1)
        else:
            sources = sources.squeeze()
    if sources.ndim == 1:
        sources = sources.unsqueeze(0)
        
    # 确保 shape 为 (Sources, Time)
    if sources.ndim == 2:
        s0, s1 = sources.shape
        if s0 > s1 and s1 <= 8:
            sources = sources.transpose(0, 1)
            
    if normalize:
        m = sources.abs().max() # GPU 操作
        if m > 0:
            sources = sources / m * 0.95
            
    energies = sources.pow(2).mean(dim=1) # GPU 操作
    order = torch.argsort(energies, descending=True).tolist() # 转回 CPU list
    # sources 保持在 GPU
    
    return sources, order, t_sep_end - t_sep_start


def _match_best(sources, sources_sr, tgt_y, tgt_sr, threshold):
    t_match_compute_start = time.time()
    
    # ----------------------------------------------------------------
    # 优化点 3: 批量处理 (Batch Processing)
    # ----------------------------------------------------------------
    
    # 1. 准备目标音频 (Target)
    # 直接将 numpy 转为 Tensor 并移至设备
    tgt_y_t = torch.from_numpy(tgt_y).unsqueeze(0).to(MATCH_DEVICE).float()
    
    # 重采样 Target (直接从 tgt_sr -> CLS_SR)
    if tgt_sr != CLS_SR:
        try:
            tgt_y_t = torchaudio.functional.resample(tgt_y_t, tgt_sr, CLS_SR)
        except Exception:
             # 回退：如果 GPU 重采样失败，使用原逻辑
             tgt_y_np = _resample_np(tgt_y, tgt_sr, CLS_SR)
             tgt_y_t = torch.from_numpy(tgt_y_np).unsqueeze(0).to(MATCH_DEVICE).float()
             
    with torch.inference_mode():
        # 提取目标声纹
        # EncoderClassifier 通常输出 (Batch, 1, EmbDim)
        tgt_emb = CLS.encode_batch(tgt_y_t).view(-1) # Flatten 为 (EmbDim,)

    # 2. 准备源音频 (Sources) - 此时 sources 应该已经在 Device 上 (来自 _separate)
    sources = sources.to(MATCH_DEVICE) # 如果设备不同则迁移，通常相同
    
    # 批量重采样 (Batch Resample)
    if sources_sr != CLS_SR:
        try:
            # torchaudio 支持 (..., Time) 形状，会自动处理 Batch 维度
            sources_rs = torchaudio.functional.resample(sources, sources_sr, CLS_SR)
        except Exception:
            # 回退：循环处理
            sources_rs_list = []
            for i in range(sources.shape[0]):
                 # 单个处理
                 s_item = sources[i].unsqueeze(0)
                 try:
                     s_rs = torchaudio.functional.resample(s_item, sources_sr, CLS_SR)
                 except:
                     # 极端回退
                     s_np = s_item.cpu().numpy()
                     s_np = _resample_np(s_np.squeeze(), sources_sr, CLS_SR)
                     s_rs = torch.from_numpy(s_np).unsqueeze(0).to(MATCH_DEVICE)
                 sources_rs_list.append(s_rs)
            sources_rs = torch.cat(sources_rs_list, dim=0)
    else:
        sources_rs = sources

    # 3. 批量提取声纹 (Batch Encode)
    with torch.inference_mode():
        # 输入形状: (N_Sources, Time)
        # 输出形状: (N_Sources, 1, EmbDim)
        embs = CLS.encode_batch(sources_rs)
        # 调整为 (N_Sources, EmbDim)
        embs = embs.view(sources_rs.shape[0], -1)

    # 4. 向量化计算相似度 (Vectorized Cosine Similarity)
    # tgt_emb: (EmbDim,)
    # embs: (N, EmbDim)
    # 对 dim=1 进行余弦相似度计算
    sims_t = F.cosine_similarity(embs, tgt_emb.unsqueeze(0), dim=1)
    sims = sims_t.cpu().tolist()

    best_idx = int(np.argmax(sims)) if len(sims) > 0 else 0
    if len(sims) > 0 and sims[best_idx] <= threshold:
        best_idx = None
        
    t_match_compute_end = time.time()
    return best_idx, sims, t_match_compute_end - t_match_compute_start


@app.post("/separate-match")
async def separate_match(
    mixed: UploadFile = File(...),
    target: UploadFile = File(...),
    num_speakers: int = Form(2),
    normalize: bool = Form(True),
    match_threshold: float = Form(0.25),
):
    t_total_start = time.time()
    mixed_bytes = await mixed.read()
    target_bytes = await target.read()
    mix_y, mix_sr = _load_audio_mono_bytes(mixed_bytes)
    tgt_y, tgt_sr = _load_audio_mono_bytes(target_bytes)
    
    # ----------------------------------------------------------------
    # 优化: 使用 torchaudio 重采样 Mix，避免 numpy 插值
    # ----------------------------------------------------------------
    mix_t = torch.from_numpy(mix_y).float()
    mix_rs_t = _resample_torch(mix_t, mix_sr, SEP_SR)
    x = mix_rs_t.unsqueeze(0).to(MAIN_DEVICE)
    
    sources, order, t_sep = _separate(x, num_speakers, normalize=normalize)
    
    # 优化: 直接传递 tgt_sr，在内部一次性重采样到 CLS_SR
    best_idx, sims, t_match = _match_best(sources, SEP_SR, tgt_y, tgt_sr, match_threshold)
    
    if best_idx is None:
        return JSONResponse(content={"code": -1, "message": "没有目标人声音"})

    best_audio = sources[best_idx]
    wav_b = _wav_bytes(best_audio, SEP_SR)
    matched_idx = best_idx + 1
    similarity = sims[best_idx]

    t_total_end = time.time()
    return JSONResponse(
        content={
            "matched_speaker_index": matched_idx,
            "similarity": similarity,
            "audio_wav_base64": base64.b64encode(wav_b).decode("ascii"),
            "timings": {
                "preload": PRELOAD_TIMES,
                "separation_time_sec": round(t_sep, 6),
                "match_compute_time_sec": round(t_match, 6),
                "total_time_sec": round(t_total_end - t_total_start, 6),
            },
            "device": {
                "separation": MAIN_DEVICE,
                "match": MATCH_DEVICE,
            },
        }
    )


@app.post("/separate-match-wav")
async def separate_match_wav(
    mixed: UploadFile = File(...),
    target: UploadFile = File(...),
    num_speakers: int = Form(2),
    normalize: bool = Form(True),
    match_threshold: float = Form(0.25),
):
    t_total_start = time.time()
    mixed_bytes = await mixed.read()
    target_bytes = await target.read()
    mix_y, mix_sr = _load_audio_mono_bytes(mixed_bytes)
    tgt_y, tgt_sr = _load_audio_mono_bytes(target_bytes)
    
    mix_t = torch.from_numpy(mix_y).float()
    mix_rs_t = _resample_torch(mix_t, mix_sr, SEP_SR)
    x = mix_rs_t.unsqueeze(0).to(MAIN_DEVICE)
    
    sources, order, t_sep = _separate(x, num_speakers, normalize=normalize)
    
    best_idx, sims, t_match = _match_best(sources, SEP_SR, tgt_y, tgt_sr, match_threshold)
    
    if best_idx is None:
        return JSONResponse(content={"code": -1, "message": "没有目标人声音"})

    best_audio = sources[best_idx]
    wav_b = _wav_bytes(best_audio, SEP_SR)
    matched_idx = best_idx + 1
    similarity = sims[best_idx]

    t_total_end = time.time()
    headers = {
        "X-Matched-Speaker-Index": str(matched_idx),
        "X-Similarity": str(similarity),
        "X-Separation-Time-Sec": str(round(t_sep, 6)),
        "X-Match-Compute-Time-Sec": str(round(t_match, 6)),
        "X-Total-Time-Sec": str(round(t_total_end - t_total_start, 6)),
        "X-Device-Separation": MAIN_DEVICE,
        "X-Device-Match": MATCH_DEVICE,
        "Content-Disposition": "attachment; filename=matched_best.wav",
    }
    return Response(content=wav_b, media_type="application/octet-stream", headers=headers)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
