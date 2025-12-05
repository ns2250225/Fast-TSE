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
import torchaudio
import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, Response
try:
    from speechbrain.inference import SepformerSeparation, EncoderClassifier
except ImportError:
    from speechbrain.pretrained import SepformerSeparation, EncoderClassifier

try:
    import onnxruntime as ort
except ImportError:
    ort = None


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
        y, sr = torchaudio.load(io.BytesIO(b))
        if y.shape[0] > 1:
            y = y.mean(dim=0)
        y = y.detach().cpu().numpy().astype("float32")
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
    try:
        bio = io.BytesIO()
        y_t = y.detach().cpu().unsqueeze(0) if _is_torch_tensor(y) else torch.tensor(y).unsqueeze(0)
        torchaudio.save(bio, y_t, sr, format="wav")
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
                # 假设量化模型与原模型在同一目录，后缀为 _int8.onnx
                path_int8 = path.replace(".onnx", "_int8.onnx")
                if os.path.exists(path_int8):
                    path = path_int8
                    
                # CPU 优化配置
                sess_options.intra_op_num_threads = 4  # 物理核心数
                sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                
                self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"], sess_options=sess_options)
            else:
                # 允许使用 GPU (如果未强制 CPU) 但 device 参数为 "cpu" 的情况 (例如作为后备)
                # 但通常 device 参数由调用者控制。如果调用者传入 "cpu"，我们尊重它。
                # 不过，如果存在 int8 模型，我们也可以尝试加载它，因为 onnxruntime-gpu 也能运行 int8 模型
                path_int8 = path.replace(".onnx", "_int8.onnx")
                if os.path.exists(path_int8):
                    path = path_int8
                self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"], sess_options=sess_options)
            
    def warmup(self):
        if self.sess:
            # Create dummy input for warmup
            # Input: mix, shape: ['batch_size', 'time']
            # Use a reasonable length, e.g., 1 second at 8000Hz
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
        # mix: torch tensor (batch, time)
        input_name = self.sess.get_inputs()[0].name
        
        # Check if model expects FP16 input (based on file name or inference)
        is_fp16 = self.path.endswith("_fp16.onnx")
        
        # Use IO Binding for GPU inference if available
        if "CUDAExecutionProvider" in self.sess.get_providers() and mix.device.type == "cuda":
            # Input is already on GPU (torch tensor)
            # Create OrtValue from the existing GPU tensor without copying
            # Note: We need to ensure the tensor is contiguous and correct type
            
            if is_fp16:
                mix = mix.contiguous().half() # Convert to float16
                elem_type = np.float16
            else:
                mix = mix.contiguous().float() # Convert to float32
                elem_type = np.float32
            
            # Get data pointer and shape
            data_ptr = mix.data_ptr()
            shape = tuple(mix.shape)
            
            io_binding = self.sess.io_binding()
            
            # Bind input (already on GPU)
            io_binding.bind_input(
                name=input_name,
                device_type='cuda',
                device_id=mix.device.index if mix.device.index is not None else 0,
                element_type=elem_type,
                shape=shape,
                buffer_ptr=data_ptr
            )
            
            # Bind output to GPU
            # We don't know the exact output shape beforehand for dynamic axes, 
            # but bind_output without shape will allocate it on the device.
            output_name = self.sess.get_outputs()[0].name
            io_binding.bind_output(output_name, 'cuda')
            
            # Run
            self.sess.run_with_iobinding(io_binding)
            
            # Get output as OrtValue (still on GPU)
            ort_output = io_binding.get_outputs()[0]
            
            # Convert to Torch Tensor (zero-copy if possible, but dlpack is safest)
            # ORT -> DLPack -> Torch
            try:
                from torch.utils.dlpack import from_dlpack
                out_tensor = from_dlpack(ort_output.to_dlpack())
                # If model output FP16, convert back to FP32 if needed by downstream
                if out_tensor.dtype == torch.float16:
                    out_tensor = out_tensor.float()
                return out_tensor
            except Exception:
                # Fallback: copy to CPU then to GPU (slower)
                out_np = ort_output.numpy()
                return torch.from_numpy(out_np).to(mix.device).float()
        else:
            # CPU fallback or CPU device
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
                # 假设量化模型与原模型在同一目录，后缀为 _int8.onnx
                path_int8 = path.replace(".onnx", "_int8.onnx")
                if os.path.exists(path_int8):
                    path = path_int8
                    
                # CPU 优化配置
                sess_options.intra_op_num_threads = 4  # 物理核心数
                sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                
                self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"], sess_options=sess_options)
            else:
                # 允许使用 GPU (如果未强制 CPU) 但 device 参数为 "cpu" 的情况 (例如作为后备)
                # 但通常 device 参数由调用者控制。如果调用者传入 "cpu"，我们尊重它。
                # 不过，如果存在 int8 模型，我们也可以尝试加载它，因为 onnxruntime-gpu 也能运行 int8 模型
                path_int8 = path.replace(".onnx", "_int8.onnx")
                if os.path.exists(path_int8):
                    path = path_int8
                self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"], sess_options=sess_options)

    def warmup(self):
        if self.sess:
            # Create dummy input for warmup
            # Input: mel_spectrogram, shape: ['batch_size', 'time', 80]
            # Use a reasonable length, e.g., 1 second -> ~100 frames
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
        # wavs: torch tensor (batch, time)
        # Feature extraction
        feats = self.compute_features(wavs)
        input_name = self.sess.get_inputs()[0].name
        
        # Check if model expects FP16 input (based on file name or inference)
        is_fp16 = self.path.endswith("_fp16.onnx")
        
        # Use IO Binding for GPU inference if available
        if "CUDAExecutionProvider" in self.sess.get_providers() and wavs.device.type == "cuda":
            # Input features are on GPU
            
            if is_fp16:
                feats = feats.contiguous().half()
                elem_type = np.float16
            else:
                feats = feats.contiguous().float()
                elem_type = np.float32
                
            # Get data pointer and shape
            data_ptr = feats.data_ptr()
            shape = tuple(feats.shape)
            
            io_binding = self.sess.io_binding()
            
            # Bind input (already on GPU)
            io_binding.bind_input(
                name=input_name,
                device_type='cuda',
                device_id=wavs.device.index if wavs.device.index is not None else 0,
                element_type=elem_type,
                shape=shape,
                buffer_ptr=data_ptr
            )
            
            # Bind output to GPU
            output_name = self.sess.get_outputs()[0].name
            io_binding.bind_output(output_name, 'cuda')
            
            # Run
            self.sess.run_with_iobinding(io_binding)
            
            # Get output as OrtValue (still on GPU)
            ort_output = io_binding.get_outputs()[0]
            
            # Convert to Torch Tensor (zero-copy if possible, but dlpack is safest)
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
        sep2_path = os.path.join(ONNX_DIR, "sepformer_wsj02mix.onnx")
        # Ensure model exists, otherwise export it
        if not os.path.exists(sep2_path) and not os.path.exists(sep2_path.replace(".onnx", "_int8.onnx")) and not os.path.exists(sep2_path.replace(".onnx", "_fp16.onnx")):
            print(f"Model {sep2_path} not found. Exporting from SpeechBrain...")
            temp_model = SepformerSeparation.from_hparams(
                source="speechbrain/sepformer-wsj02mix",
                savedir=os.path.join("pretrained_models", "sepformer-wsj02mix"),
                run_opts={"device": "cpu"},
            )
            os.makedirs(ONNX_DIR, exist_ok=True)
            # Dummy input for export: [batch, time]
            dummy_input = torch.randn(1, 16000)
            torch.onnx.export(
                temp_model.mods.encoder.model,
                (dummy_input,),
                sep2_path,
                input_names=["mix"],
                output_names=["est_sources"],
                dynamic_axes={"mix": {0: "batch", 1: "time"}, "est_sources": {0: "batch", 2: "time"}},
                opset_version=17
            )
            print(f"Exported {sep2_path}")
            del temp_model
            
        # Automatically use quantized/optimized model if available
        # Priority: FP16 -> INT8 -> FP32
        sep2_path_fp16 = sep2_path.replace(".onnx", "_fp16.onnx")
        sep2_path_int8 = sep2_path.replace(".onnx", "_int8.onnx")
        
        if os.path.exists(sep2_path_fp16):
            sep2_path = sep2_path_fp16
        elif os.path.exists(sep2_path_int8):
            sep2_path = sep2_path_int8
            
        SEP_MODELS["2"] = OnnxSepformer(sep2_path, sample_rate=8000, device=MAIN_DEVICE)
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
        sep3_path = os.path.join(ONNX_DIR, "sepformer_wsj03mix.onnx")
        # Ensure model exists, otherwise export it
        if not os.path.exists(sep3_path) and not os.path.exists(sep3_path.replace(".onnx", "_int8.onnx")) and not os.path.exists(sep3_path.replace(".onnx", "_fp16.onnx")):
            print(f"Model {sep3_path} not found. Exporting from SpeechBrain...")
            temp_model = SepformerSeparation.from_hparams(
                source="speechbrain/sepformer-wsj03mix",
                savedir=os.path.join("pretrained_models", "sepformer-wsj03mix"),
                run_opts={"device": "cpu"},
            )
            os.makedirs(ONNX_DIR, exist_ok=True)
            # Dummy input for export: [batch, time]
            dummy_input = torch.randn(1, 16000)
            torch.onnx.export(
                temp_model.mods.encoder.model,
                (dummy_input,),
                sep3_path,
                input_names=["mix"],
                output_names=["est_sources"],
                dynamic_axes={"mix": {0: "batch", 1: "time"}, "est_sources": {0: "batch", 2: "time"}},
                opset_version=17
            )
            print(f"Exported {sep3_path}")
            del temp_model
            
        # Automatically use quantized/optimized model if available
        # Priority: FP16 -> INT8 -> FP32
        sep3_path_fp16 = sep3_path.replace(".onnx", "_fp16.onnx")
        sep3_path_int8 = sep3_path.replace(".onnx", "_int8.onnx")
        
        if os.path.exists(sep3_path_fp16):
            sep3_path = sep3_path_fp16
        elif os.path.exists(sep3_path_int8):
            sep3_path = sep3_path_int8
            
        SEP_MODELS["3"] = OnnxSepformer(sep3_path, sample_rate=8000, device=MAIN_DEVICE)
    else:
        SEP_MODELS["3"] = SepformerSeparation.from_hparams(
            source="speechbrain/sepformer-wsj03mix",
            savedir=os.path.join("pretrained_models", "sepformer-wsj03mix"),
            run_opts={"device": MAIN_DEVICE},
        )

    PRELOAD_TIMES["sepformer_3"] = time.time() - t1
    t2 = time.time()

    if ENABLE_ONNX:
        # Load a lightweight classifier just for feature extraction
        sb_cls = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=os.path.join("pretrained_models", "spkrec-ecapa-voxceleb"),
            run_opts={"device": "cpu"},
        )
        feature_extractor = sb_cls.mods.compute_features
        cls_path = os.path.join(ONNX_DIR, "ecapa_voxceleb.onnx")
        # Ensure model exists, otherwise export it
        if not os.path.exists(cls_path) and not os.path.exists(cls_path.replace(".onnx", "_int8.onnx")) and not os.path.exists(cls_path.replace(".onnx", "_fp16.onnx")):
            print(f"Model {cls_path} not found. Exporting from SpeechBrain...")
            os.makedirs(ONNX_DIR, exist_ok=True)
            # Dummy input for export: [batch, time, 80]
            dummy_input = torch.randn(1, 100, 80)
            # Use the feature_extractor logic to get embeddings from features
            # Note: sb_cls.mods.embedding_model is the core ECAPA model
            torch.onnx.export(
                sb_cls.mods.embedding_model,
                (dummy_input,),
                cls_path,
                input_names=["feats"],
                output_names=["emb"],
                dynamic_axes={"feats": {0: "batch", 1: "time"}, "emb": {0: "batch"}},
                opset_version=17
            )
            print(f"Exported {cls_path}")

        # Automatically use quantized/optimized model if available
        # Priority: FP16 -> INT8 -> FP32
        cls_path_fp16 = cls_path.replace(".onnx", "_fp16.onnx")
        cls_path_int8 = cls_path.replace(".onnx", "_int8.onnx")
        
        if os.path.exists(cls_path_fp16):
            cls_path = cls_path_fp16
        elif os.path.exists(cls_path_int8):
            cls_path = cls_path_int8
            
        CLS = OnnxClassifier(cls_path, feature_extractor, sample_rate=16000, device=MATCH_DEVICE)
    else:
        CLS = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=os.path.join("pretrained_models", "spkrec-ecapa-voxceleb"),
            run_opts={"device": MATCH_DEVICE},
        )
    PRELOAD_TIMES["classifier"] = time.time() - t2
    CLS_SR = int(getattr(CLS.hparams, "sample_rate", 16000))

    # Run dummy inference for warmup
    print("Warming up models...")
    
    # Warmup separation models
    dummy_input = torch.randn(1, 16000).to(MAIN_DEVICE)
    
    # Check if model is FP16 and cast dummy input accordingly
    # For ONNX models, we need to check the underlying session or path
    if ENABLE_ONNX:
        for key, model in SEP_MODELS.items():
            try:
                # Check if this specific model instance uses an FP16 ONNX file
                is_fp16 = hasattr(model, 'path') and model.path.endswith("_fp16.onnx")
                
                current_input = dummy_input.clone()
                if is_fp16:
                    current_input = current_input.half()
                
                model.separate_batch(current_input)
            except Exception as e:
                print(f"Warmup failed for sepformer_{key}: {e}")
    else:
        # PyTorch models usually handle autocast or explicit types
        for key, model in SEP_MODELS.items():
            try:
                model.separate_batch(dummy_input)
            except Exception as e:
                print(f"Warmup failed for sepformer_{key}: {e}")

    # Warmup classifier
    if CLS is not None:
        dummy_wavs = torch.randn(1, 16000).to(MATCH_DEVICE)
        try:
            # Check if classifier is FP16 ONNX
            is_fp16 = ENABLE_ONNX and hasattr(CLS, 'path') and CLS.path.endswith("_fp16.onnx")
            
            if is_fp16:
                dummy_wavs = dummy_wavs.half()
                
            CLS.encode_batch(dummy_wavs)
        except Exception as e:
            print(f"Warmup failed for classifier: {e}")
            
    print("Models warmed up!")
    
    yield
    
    # Clean up
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
    sources = est_sources if _is_torch_tensor(est_sources) else torch.tensor(est_sources)
    sources = sources.detach().cpu().float()
    if sources.ndim == 3:
        if sources.shape[0] == 1:
            sources = sources.squeeze(0)
        elif sources.shape[1] == 1:
            sources = sources.squeeze(1)
        else:
            sources = sources.squeeze()
    if sources.ndim == 1:
        sources = sources.unsqueeze(0)
    if sources.ndim == 2:
        s0, s1 = sources.shape
        if s0 > s1 and s1 <= 8:
            sources = sources.transpose(0, 1)
    if normalize:
        m = sources.abs().max().item()
        if m > 0:
            sources = sources / m * 0.95
    energies = sources.pow(2).mean(dim=1)
    order = torch.argsort(energies, descending=True).tolist()
    order = order[: int(sources.shape[0])]
    return sources, order, t_sep_end - t_sep_start


def _match_best(sources, sr, tgt_y, threshold):
    t_match_compute_start = time.time()
    tgt_y_t = torch.tensor(_resample_np(tgt_y, sr, CLS_SR)).unsqueeze(0).to(MATCH_DEVICE)
    with torch.inference_mode():
        tgt_emb = CLS.encode_batch(tgt_y_t).detach().cpu().float().view(-1)
    sims = []
    for i in range(int(sources.shape[0])):
        y = sources[i]
        y_t = y.unsqueeze(0)
        y_rs = _resample_torch(y_t, sr, CLS_SR).to(MATCH_DEVICE)
        with torch.inference_mode():
            emb = CLS.encode_batch(y_rs).detach().cpu().float().view(-1)
        d = torch.clamp(emb.norm() * tgt_emb.norm(), min=1e-8)
        s = torch.dot(emb, tgt_emb) / d
        sims.append(float(s.item()))
    best_idx = int(max(range(len(sims)), key=lambda k: sims[k])) if len(sims) > 0 else 0
    # Check threshold
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
    mix_rs = _resample_np(mix_y, mix_sr, SEP_SR)
    x = torch.tensor(mix_rs).unsqueeze(0).to(MAIN_DEVICE)
    sources, order, t_sep = _separate(x, num_speakers, normalize=normalize)
    best_idx, sims, t_match = _match_best(sources, SEP_SR, _resample_np(tgt_y, tgt_sr, SEP_SR), match_threshold)
    
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
    mix_rs = _resample_np(mix_y, mix_sr, SEP_SR)
    x = torch.tensor(mix_rs).unsqueeze(0).to(MAIN_DEVICE)
    sources, order, t_sep = _separate(x, num_speakers, normalize=normalize)
    best_idx, sims, t_match = _match_best(sources, SEP_SR, _resample_np(tgt_y, tgt_sr, SEP_SR), match_threshold)
    
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
