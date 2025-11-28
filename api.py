import os
import io
import time
import math
import base64
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, Response
from contextlib import asynccontextmanager


def _is_torch_tensor(x):
    try:
        import torch
        return isinstance(x, torch.Tensor)
    except Exception:
        return False


def _load_audio_mono_bytes(b):
    try:
        import soundfile as sf
        import numpy as np
        y, sr = sf.read(io.BytesIO(b))
        if y.ndim == 2:
            y = y.mean(axis=1)
        y = y.astype(np.float32)
        return y, int(sr)
    except Exception:
        pass
    try:
        import torchaudio
        import torch
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
    import numpy as np
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
        import torchaudio
        return torchaudio.functional.resample(y_t, sr_from, sr_to)
    except Exception:
        import torch
        y_np = y_t.detach().cpu().numpy()
        y_np = _resample_np(y_np, sr_from, sr_to)
        return torch.tensor(y_np).unsqueeze(0)


def _wav_bytes(y, sr):
    try:
        import soundfile as sf
        import numpy as np
        y_np = y.detach().cpu().numpy() if _is_torch_tensor(y) else y
        bio = io.BytesIO()
        sf.write(bio, y_np.astype(np.float32), sr, format="WAV")
        return bio.getvalue()
    except Exception:
        pass
    try:
        import torch
        import torchaudio
        bio = io.BytesIO()
        y_t = y.detach().cpu().unsqueeze(0) if _is_torch_tensor(y) else torch.tensor(y).unsqueeze(0)
        torchaudio.save(bio, y_t, sr, format="wav")
        return bio.getvalue()
    except Exception:
        pass
    import numpy as np
    import wave
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    global SEP_MODELS, CLS, PRELOAD_TIMES, SEP_SR, CLS_SR, MAIN_DEVICE, MATCH_DEVICE
    import torch
    from speechbrain.pretrained import SepformerSeparation, EncoderClassifier
    MAIN_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MATCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    SEP_MODELS["2"] = SepformerSeparation.from_hparams(
        source="speechbrain/sepformer-wsj02mix",
        savedir=os.path.join("pretrained_models", "sepformer-wsj02mix"),
        run_opts={"device": MAIN_DEVICE},
    )
    PRELOAD_TIMES["sepformer_2"] = time.time() - t0
    SEP_SR = int(getattr(SEP_MODELS["2"].hparams, "sample_rate", 16000))
    t1 = time.time()
    SEP_MODELS["3"] = SepformerSeparation.from_hparams(
        source="speechbrain/sepformer-wsj03mix",
        savedir=os.path.join("pretrained_models", "sepformer-wsj03mix"),
        run_opts={"device": MAIN_DEVICE},
    )
    PRELOAD_TIMES["sepformer_3"] = time.time() - t1
    t2 = time.time()
    CLS = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=os.path.join("pretrained_models", "spkrec-ecapa-voxceleb"),
        run_opts={"device": MATCH_DEVICE},
    )
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
    import torch
    import numpy as np
    model = SEP_MODELS.get(str(num_speakers))
    if model is None:
        raise RuntimeError("模型未加载")
    t_sep_start = time.time()
    try:
        with torch.no_grad():
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
            with torch.no_grad():
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


def _match_best(sources, sr, tgt_y):
    import torch
    t_match_compute_start = time.time()
    tgt_y_t = torch.tensor(_resample_np(tgt_y, sr, CLS_SR)).unsqueeze(0).to(MATCH_DEVICE)
    with torch.no_grad():
        tgt_emb = CLS.encode_batch(tgt_y_t).detach().cpu().float().view(-1)
    sims = []
    for i in range(int(sources.shape[0])):
        y = sources[i]
        y_t = y.unsqueeze(0)
        y_rs = _resample_torch(y_t, sr, CLS_SR).to(MATCH_DEVICE)
        with torch.no_grad():
            emb = CLS.encode_batch(y_rs).detach().cpu().float().view(-1)
        d = torch.clamp(emb.norm() * tgt_emb.norm(), min=1e-8)
        s = torch.dot(emb, tgt_emb) / d
        sims.append(float(s.item()))
    best_idx = int(max(range(len(sims)), key=lambda k: sims[k])) if len(sims) > 0 else 0
    t_match_compute_end = time.time()
    return best_idx, sims, t_match_compute_end - t_match_compute_start


@app.post("/separate-match")
async def separate_match(
    mixed: UploadFile = File(...),
    target: UploadFile = File(...),
    num_speakers: int = Form(2),
    normalize: bool = Form(True),
):
    t_total_start = time.time()
    mixed_bytes = await mixed.read()
    target_bytes = await target.read()
    mix_y, mix_sr = _load_audio_mono_bytes(mixed_bytes)
    tgt_y, tgt_sr = _load_audio_mono_bytes(target_bytes)
    import torch
    mix_rs = _resample_np(mix_y, mix_sr, SEP_SR)
    x = torch.tensor(mix_rs).unsqueeze(0).to(MAIN_DEVICE)
    sources, order, t_sep = _separate(x, num_speakers, normalize=normalize)
    best_idx, sims, t_match = _match_best(sources, SEP_SR, _resample_np(tgt_y, tgt_sr, SEP_SR))
    best_audio = sources[best_idx]
    wav_b = _wav_bytes(best_audio, SEP_SR)
    t_total_end = time.time()
    return JSONResponse(
        content={
            "matched_speaker_index": best_idx + 1,
            "similarity": sims[best_idx] if len(sims) > 0 else 0.0,
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
):
    t_total_start = time.time()
    mixed_bytes = await mixed.read()
    target_bytes = await target.read()
    mix_y, mix_sr = _load_audio_mono_bytes(mixed_bytes)
    tgt_y, tgt_sr = _load_audio_mono_bytes(target_bytes)
    import torch
    mix_rs = _resample_np(mix_y, mix_sr, SEP_SR)
    x = torch.tensor(mix_rs).unsqueeze(0).to(MAIN_DEVICE)
    sources, order, t_sep = _separate(x, num_speakers, normalize=normalize)
    best_idx, sims, t_match = _match_best(sources, SEP_SR, _resample_np(tgt_y, tgt_sr, SEP_SR))
    best_audio = sources[best_idx]
    wav_b = _wav_bytes(best_audio, SEP_SR)
    t_total_end = time.time()
    headers = {
        "X-Matched-Speaker-Index": str(best_idx + 1),
        "X-Similarity": str(sims[best_idx] if len(sims) > 0 else 0.0),
        "X-Separation-Time-Sec": str(round(t_sep, 6)),
        "X-Match-Compute-Time-Sec": str(round(t_match, 6)),
        "X-Total-Time-Sec": str(round(t_total_end - t_total_start, 6)),
        "X-Device-Separation": MAIN_DEVICE,
        "X-Device-Match": MATCH_DEVICE,
        "Content-Disposition": "attachment; filename=matched_best.wav",
    }
    return Response(content=wav_b, media_type="application/octet-stream", headers=headers)
