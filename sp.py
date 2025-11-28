import argparse
import os
import sys
import time
import math


def _is_torch_tensor(x):
    try:
        import torch
        return isinstance(x, torch.Tensor)
    except Exception:
        return False


def _save_wav_wave(x, sr, path):
    import numpy as np
    import wave
    y = x.detach().cpu().numpy() if _is_torch_tensor(x) else x
    y = np.clip(y, -1.0, 1.0)
    y_int16 = (y * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(y_int16.tobytes())


def _save_wav(y, sr, path):
    try:
        import soundfile as sf
        y_np = y.detach().cpu().numpy() if _is_torch_tensor(y) else y
        sf.write(path, y_np, sr)
        return
    except Exception:
        pass
    try:
        import torch
        import torchaudio
        y_t = y.detach().cpu().unsqueeze(0) if _is_torch_tensor(y) else torch.tensor(y).unsqueeze(0)
        torchaudio.save(path, y_t, sr)
        return
    except Exception:
        pass
    _save_wav_wave(y, sr, path)


def _load_audio_mono(path):
    try:
        import soundfile as sf
        y, sr = sf.read(path)
        import numpy as np
        if y.ndim == 2:
            y = y.mean(axis=1)
        y = y.astype(np.float32)
        return y, int(sr)
    except Exception:
        pass
    try:
        import torchaudio
        y, sr = torchaudio.load(path)
        import torch
        if y.shape[0] > 1:
            y = y.mean(dim=0)
        y = y.detach().cpu().numpy().astype("float32")
        return y, int(sr)
    except Exception:
        pass
    raise RuntimeError("无法读取音频: {}".format(path))


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


def main():
    parser = argparse.ArgumentParser(description="使用SpeechBrain将多人混合语音分离为各独立说话人音频")
    parser.add_argument("input", help="输入混合音频文件路径")
    parser.add_argument("--num_speakers", "-n", type=int, default=2, choices=[2, 3], help="说话人数，选择2或3")
    parser.add_argument("--model", "-m", type=str, default=None, help="自定义SpeechBrain模型仓库名")
    parser.add_argument("--savedir", type=str, default=None, help="下载模型的本地目录")
    parser.add_argument("--output_prefix", type=str, default=None, help="输出文件名前缀")
    parser.add_argument("--normalize", action="store_true", help="对分离结果进行幅度归一化")
    parser.add_argument("--target", "-t", type=str, default=None, help="目标说话人音频文件路径")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"],
        help="推理设备(cpu/cuda)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print("输入文件不存在: {}".format(args.input), file=sys.stderr)
        sys.exit(1)

    try:
        import torch
        from speechbrain.pretrained import SepformerSeparation
    except Exception:
        print(
            "未安装speechbrain或其依赖。请运行: pip install speechbrain torch torchaudio soundfile",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.model is None:
        if args.num_speakers == 2:
            model = "speechbrain/sepformer-wsj02mix"
            savedir = args.savedir or os.path.join("pretrained_models", "sepformer-wsj02mix")
        else:
            model = "speechbrain/sepformer-wsj03mix"
            savedir = args.savedir or os.path.join("pretrained_models", "sepformer-wsj03mix")
    else:
        model = args.model
        savedir = args.savedir or os.path.join("pretrained_models", args.model.replace("/", "_"))

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    t_total_start = time.time()
    t_model_start = time.time()
    separation_model = SepformerSeparation.from_hparams(
        source=model, savedir=savedir, run_opts={"device": args.device}
    )
    t_model_end = time.time()
    used_device = args.device

    sr = int(getattr(separation_model.hparams, "sample_rate", 16000))

    t_sep_start = time.time()
    try:
        with torch.no_grad():
            est_sources = separation_model.separate_file(args.input)
            if args.device == "cuda":
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
        t_sep_end = time.time()
    except RuntimeError as e:
        msg = str(e)
        if ("CUDA" in msg) or ("device-side assert" in msg):
            print("CUDA出错，切换到CPU重新分离", file=sys.stderr)
            separation_model = SepformerSeparation.from_hparams(
                source=model, savedir=savedir, run_opts={"device": "cpu"}
            )
            with torch.no_grad():
                est_sources = separation_model.separate_file(args.input)
            used_device = "cpu"
            t_sep_end = time.time()
        else:
            raise

    import torch
    if isinstance(est_sources, (list, tuple)):
        try:
            est_sources = torch.stack([t.detach().cpu() if _is_torch_tensor(t) else torch.tensor(t) for t in est_sources], dim=0)
        except Exception:
            est_sources = [t.detach().cpu().numpy() if _is_torch_tensor(t) else t for t in est_sources]
            min_len = min(len(x) for x in est_sources)
            est_sources = torch.stack([torch.tensor(x[:min_len]) for x in est_sources], dim=0)
    sources = est_sources if _is_torch_tensor(est_sources) else torch.tensor(est_sources)
    sources = sources.detach().cpu().float()
    sources = torch.nan_to_num(sources, nan=0.0, posinf=1.0, neginf=-1.0)
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

    num_est = int(sources.shape[0])
    base = args.output_prefix or os.path.splitext(os.path.basename(args.input))[0]
    out_dir = os.getcwd()

    if args.normalize:
        max_abs = sources.abs().max().item()
        if max_abs > 0:
            sources = sources / max_abs * 0.95

    energies = sources.pow(2).mean(dim=1)
    order = torch.argsort(energies, descending=True).tolist()
    order = order[:num_est]
    for idx in range(num_est):
        spk_idx = order[idx] if idx < len(order) else idx
        if spk_idx >= num_est:
            break
        y = sources[spk_idx]
        out_path = os.path.join(out_dir, f"{base}_spk{idx + 1}.wav")
        _save_wav(y, sr, out_path)
        print(out_path)

    t_total_end = time.time()
    print("模型加载耗时: {:.3f}s".format(t_model_end - t_model_start))
    print("分离耗时({}): {:.3f}s".format(used_device, t_sep_end - t_sep_start))
    print("总耗时: {:.3f}s".format(t_total_end - t_total_start))

    if args.target is not None:
        t_match_start = time.time()
        t_match_model_start = None
        t_match_model_end = None
        t_match_compute_start = None
        t_match_compute_end = None
        try:
            tgt_y, tgt_sr = _load_audio_mono(args.target)
            from speechbrain.pretrained import EncoderClassifier
            import torch
            match_device = "cuda" if (torch.cuda.is_available() and args.device == "cuda") else "cpu"
            t_match_model_start = time.time()
            cls = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=os.path.join("pretrained_models", "spkrec-ecapa-voxceleb"),
                run_opts={"device": match_device},
            )
            t_match_model_end = time.time()
            cls_sr = int(getattr(cls.hparams, "sample_rate", 16000))
            t_match_compute_start = time.time()
            try:
                tgt_y_t = torch.tensor(_resample_np(tgt_y, tgt_sr, cls_sr)).unsqueeze(0).to(match_device)
                with torch.no_grad():
                    tgt_emb = cls.encode_batch(tgt_y_t).detach().cpu().float().view(-1)
                sims = []
                for i in range(num_est):
                    y = sources[i]
                    y_t = y.unsqueeze(0)
                    y_rs = _resample_torch(y_t, sr, cls_sr).to(match_device)
                    with torch.no_grad():
                        emb = cls.encode_batch(y_rs).detach().cpu().float().view(-1)
                    denom = torch.clamp(emb.norm() * tgt_emb.norm(), min=1e-8)
                    sim = torch.dot(emb, tgt_emb) / denom
                    sim = float(sim.item())
                    sims.append(sim)
            except RuntimeError as e:
                msg = str(e)
                if ("CUDA" in msg) or ("device-side assert" in msg):
                    match_device = "cpu"
                    cls = EncoderClassifier.from_hparams(
                        source="speechbrain/spkrec-ecapa-voxceleb",
                        savedir=os.path.join("pretrained_models", "spkrec-ecapa-voxceleb"),
                        run_opts={"device": match_device},
                    )
                    tgt_y_t = torch.tensor(_resample_np(tgt_y, tgt_sr, cls_sr)).unsqueeze(0)
                    with torch.no_grad():
                        tgt_emb = cls.encode_batch(tgt_y_t).detach().cpu().float().view(-1)
                    sims = []
                    for i in range(num_est):
                        y = sources[i]
                        y_t = y.unsqueeze(0)
                        y_rs = _resample_torch(y_t, sr, cls_sr)
                        with torch.no_grad():
                            emb = cls.encode_batch(y_rs).detach().cpu().float().view(-1)
                        denom = torch.clamp(emb.norm() * tgt_emb.norm(), min=1e-8)
                        sim = torch.dot(emb, tgt_emb) / denom
                        sim = float(sim.item())
                        sims.append(sim)
                else:
                    raise
            best_idx = int(max(range(len(sims)), key=lambda k: sims[k])) if len(sims) > 0 else 0
            best_path = os.path.join(out_dir, f"{base}_target_best.wav")
            _save_wav(sources[best_idx], sr, best_path)
            print(best_path)
            print("匹配说话人: spk{} 相似度: {:.4f}".format(best_idx + 1, sims[best_idx] if len(sims) > 0 else 0.0))
            t_match_compute_end = time.time()
            t_match_end = time.time()
            print("目标匹配-模型加载耗时({}): {:.3f}s".format(match_device, t_match_model_end - t_match_model_start))
            print("目标匹配-计算耗时({}): {:.3f}s".format(match_device, t_match_compute_end - t_match_compute_start))
            print("目标匹配耗时: {:.3f}s".format(t_match_end - t_match_start))
        except Exception as e:
            print("目标匹配失败: {}".format(e), file=sys.stderr)
            t_match_end = time.time()
            d_model = (t_match_model_end - t_match_model_start) if (t_match_model_start is not None and t_match_model_end is not None) else 0.0
            d_compute = (t_match_compute_end - t_match_compute_start) if (t_match_compute_start is not None and t_match_compute_end is not None) else 0.0
            print("目标匹配-模型加载耗时: {:.3f}s".format(d_model))
            print("目标匹配-计算耗时: {:.3f}s".format(d_compute))
            print("目标匹配耗时: {:.3f}s".format(t_match_end - t_match_start))


if __name__ == "__main__":
    main()
