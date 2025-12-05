import os
import sys
import io
import time
import math
import base64
import wave
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
ENABLE_QUANTIZATION = False
ENABLE_ONNX = False
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
        
        self.hparams = type("HParams", (), {"sample_rate": sample_rate})()
        
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 4  # Fatal only

        if device == "cuda":
            # Priority: TensorRT -> CUDA -> CPU
            providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
            
            # Suppress TensorRT failure logs by redirecting stderr/stdout temporarily
            try:
                with RedirectStream():
                    self.sess = ort.InferenceSession(path, providers=providers, sess_options=sess_options)
            except Exception:
                self.sess = None

            # Check if we actually got a GPU provider
            current_providers = self.sess.get_providers() if self.sess else []
            if "TensorrtExecutionProvider" not in current_providers and "CUDAExecutionProvider" not in current_providers:
                # Fallback to CUDA only
                try:
                    with RedirectStream():
                        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                        self.sess = ort.InferenceSession(path, providers=providers, sess_options=sess_options)
                except Exception:
                    pass
        else:
            self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"], sess_options=sess_options)

    def separate_batch(self, mix):
        # mix: torch tensor (batch, time)
        mix_np = mix.detach().cpu().numpy()
        input_name = self.sess.get_inputs()[0].name
        out = self.sess.run(None, {input_name: mix_np})[0]
        return torch.from_numpy(out).to(mix.device)


class OnnxClassifier:
    def __init__(self, path, feature_extractor, sample_rate=16000, device="cpu"):
        if ort is None:
            raise ImportError("onnxruntime is required for ONNX support")
            
        self.compute_features = feature_extractor
        self.hparams = type("HParams", (), {"sample_rate": sample_rate})()
        
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 4  # Fatal only
        
        if device == "cuda":
            # Priority: TensorRT -> CUDA -> CPU
            providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
            try:
                with RedirectStream():
                    self.sess = ort.InferenceSession(path, providers=providers, sess_options=sess_options)
            except Exception:
                self.sess = None
                
            # Check if we actually got a GPU provider
            current_providers = self.sess.get_providers() if self.sess else []
            if "TensorrtExecutionProvider" not in current_providers and "CUDAExecutionProvider" not in current_providers:
                # Fallback to CUDA only
                try:
                    with RedirectStream():
                        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                        self.sess = ort.InferenceSession(path, providers=providers, sess_options=sess_options)
                except Exception:
                    pass
        else:
             self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"], sess_options=sess_options)

    def encode_batch(self, wavs):
        # wavs: torch tensor (batch, time)
        # Feature extraction
        feats = self.compute_features(wavs)
        feats_np = feats.detach().cpu().numpy()
        input_name = self.sess.get_inputs()[0].name
        out = self.sess.run(None, {input_name: feats_np})[0]
        return torch.from_numpy(out).to(wavs.device)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global SEP_MODELS, CLS, PRELOAD_TIMES, SEP_SR, CLS_SR, MAIN_DEVICE, MATCH_DEVICE
    if ENABLE_QUANTIZATION:
        MAIN_DEVICE = "cpu"
        MATCH_DEVICE = "cpu"
    else:
        MAIN_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        MATCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Configure TensorRT paths if enabled and using CUDA
    if ENABLE_ONNX and MAIN_DEVICE == "cuda":
        try:
            import tensorrt
            trt_path = os.path.dirname(tensorrt.__file__)
            # Add TensorRT libs to LD_LIBRARY_PATH (Linux) and PATH (Windows)
            # This helps onnxruntime find libnvinfer.so / nvinfer.dll
            
            # Linux
            current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
            if trt_path not in current_ld_path:
                os.environ["LD_LIBRARY_PATH"] = f"{trt_path}{os.pathsep}{current_ld_path}"
                # Also try to add the lib folder inside tensorrt package if it exists
                trt_lib_path = os.path.join(trt_path, "lib")
                if os.path.exists(trt_lib_path):
                     os.environ["LD_LIBRARY_PATH"] = f"{trt_lib_path}{os.pathsep}{os.environ['LD_LIBRARY_PATH']}"

            # Windows
            current_path = os.environ.get("PATH", "")
            if trt_path not in current_path:
                os.environ["PATH"] = f"{trt_path}{os.pathsep}{current_path}"
                trt_lib_path = os.path.join(trt_path, "lib")
                if os.path.exists(trt_lib_path):
                     os.environ["PATH"] = f"{trt_lib_path}{os.pathsep}{os.environ['PATH']}"
            
            # Re-import onnxruntime might be needed if it was already imported, 
            # but usually setting env var before session creation is enough if ORT loads libs dynamically.
            # However, ORT might have already cached load failures.
        except ImportError:
            pass

    t0 = time.time()

    if ENABLE_ONNX:
        sep2_path = os.path.join(ONNX_DIR, "sepformer_wsj02mix.onnx")
        SEP_MODELS["2"] = OnnxSepformer(sep2_path, sample_rate=8000, device=MAIN_DEVICE)
    else:
        SEP_MODELS["2"] = SepformerSeparation.from_hparams(
            source="speechbrain/sepformer-wsj02mix",
            savedir=os.path.join("pretrained_models", "sepformer-wsj02mix"),
            run_opts={"device": MAIN_DEVICE},
        )
        if ENABLE_QUANTIZATION:
            try:
                # Try 'masknet' which is common for SepFormer
                inner_model = SEP_MODELS["2"].mods["masknet"]
                quantized_inner_model = torch.quantization.quantize_dynamic(
                    inner_model,
                    {torch.nn.Linear},
                    dtype=torch.qint8
                )
                SEP_MODELS["2"].mods["masknet"] = quantized_inner_model
            except (AttributeError, KeyError):
                try:
                    # Fallback to 'model'
                    inner_model = SEP_MODELS["2"].mods["model"]
                    quantized_inner_model = torch.quantization.quantize_dynamic(
                        inner_model,
                        {torch.nn.Linear},
                        dtype=torch.qint8
                    )
                    SEP_MODELS["2"].mods["model"] = quantized_inner_model
                except Exception:
                     print(f"Warning: Failed to quantize sepformer_2. Available mods: {list(SEP_MODELS['2'].mods.keys()) if hasattr(SEP_MODELS['2'], 'mods') else 'No mods'}")

    PRELOAD_TIMES["sepformer_2"] = time.time() - t0
    SEP_SR = int(getattr(SEP_MODELS["2"].hparams, "sample_rate", 16000))
    t1 = time.time()

    if ENABLE_ONNX:
        sep3_path = os.path.join(ONNX_DIR, "sepformer_wsj03mix.onnx")
        SEP_MODELS["3"] = OnnxSepformer(sep3_path, sample_rate=8000, device=MAIN_DEVICE)
    else:
        SEP_MODELS["3"] = SepformerSeparation.from_hparams(
            source="speechbrain/sepformer-wsj03mix",
            savedir=os.path.join("pretrained_models", "sepformer-wsj03mix"),
            run_opts={"device": MAIN_DEVICE},
        )
        if ENABLE_QUANTIZATION:
            try:
                inner_model = SEP_MODELS["3"].mods["masknet"]
                quantized_inner_model = torch.quantization.quantize_dynamic(
                    inner_model,
                    {torch.nn.Linear},
                    dtype=torch.qint8
                )
                SEP_MODELS["3"].mods["masknet"] = quantized_inner_model
            except (AttributeError, KeyError):
                  try:
                     inner_model = SEP_MODELS["3"].mods["model"]
                     quantized_inner_model = torch.quantization.quantize_dynamic(
                         inner_model,
                         {torch.nn.Linear},
                         dtype=torch.qint8
                     )
                     SEP_MODELS["3"].mods["model"] = quantized_inner_model
                  except Exception:
                     print(f"Warning: Failed to quantize sepformer_3. Available mods: {list(SEP_MODELS['3'].mods.keys()) if hasattr(SEP_MODELS['3'], 'mods') else 'No mods'}")

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
        CLS = OnnxClassifier(cls_path, feature_extractor, sample_rate=16000, device=MATCH_DEVICE)
    else:
        CLS = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=os.path.join("pretrained_models", "spkrec-ecapa-voxceleb"),
            run_opts={"device": MATCH_DEVICE},
        )
        if ENABLE_QUANTIZATION:
            try:
                inner_model = CLS.mods["embedding_model"]
                quantized_inner_model = torch.quantization.quantize_dynamic(
                    inner_model,
                    {torch.nn.Linear},
                    dtype=torch.qint8
                )
                CLS.mods["embedding_model"] = quantized_inner_model
            except Exception:
                try:
                    inner_model = CLS.modules.embedding_model
                    quantized_inner_model = torch.quantization.quantize_dynamic(
                        inner_model,
                        {torch.nn.Linear},
                        dtype=torch.qint8
                    )
                    CLS.modules.embedding_model = quantized_inner_model
                except Exception:
                    print("Warning: Failed to quantize classifier")
    PRELOAD_TIMES["classifier"] = time.time() - t2
    CLS_SR = int(getattr(CLS.hparams, "sample_rate", 16000))
    yield
    SEP_MODELS.clear()
    CLS = None
    PRELOAD_TIMES = {
        "sepformer_2": 0.0,
        "sepformer_3": 0.0,
        "classifier": 0.0,
    }


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
