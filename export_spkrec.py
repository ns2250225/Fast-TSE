import torch
import speechbrain
from speechbrain.inference.speaker import EncoderClassifier
import os
from huggingface_hub import snapshot_download
import pathlib
import shutil
import urllib.request
import huggingface_hub
import speechbrain.utils.fetching as sb_fetch

def _fetch_copy(
    filename,
    source,
    savedir="./pretrained_model_checkpoints",
    overwrite=False,
    save_filename=None,
    use_auth_token=False,
    revision=None,
    huggingface_cache_dir=None,
):
    if save_filename is None:
        save_filename = filename
    savedir = pathlib.Path(savedir)
    savedir.mkdir(parents=True, exist_ok=True)
    fetch_from = None
    if isinstance(source, sb_fetch.FetchSource):
        fetch_from, source = source
    sourcefile = f"{source}/{filename}"
    destination = savedir / save_filename
    if destination.exists() and not overwrite:
        return destination
    if pathlib.Path(source).is_dir() and fetch_from not in [
        sb_fetch.FetchFrom.HUGGING_FACE,
        sb_fetch.FetchFrom.URI,
    ]:
        sourcepath = pathlib.Path(sourcefile).absolute()
        sb_fetch._missing_ok_unlink(destination)
        shutil.copyfile(sourcepath, destination)
        return destination
    if (
        str(source).startswith("http:") or str(source).startswith("https:")
    ) or fetch_from is sb_fetch.FetchFrom.URI:
        urllib.request.urlretrieve(sourcefile, destination)
    else:
        try:
            fetched_file = huggingface_hub.hf_hub_download(
                repo_id=source,
                filename=filename,
                use_auth_token=use_auth_token,
                revision=revision,
                cache_dir=huggingface_cache_dir,
            )
        except Exception as e:
            raise ValueError("File not found on HF hub") from e
        sourcepath = pathlib.Path(fetched_file).absolute()
        sb_fetch._missing_ok_unlink(destination)
        shutil.copyfile(sourcepath, destination)
    return destination

sb_fetch.fetch = _fetch_copy
import speechbrain.inference.interfaces as sb_int
import speechbrain.utils.parameter_transfer as sb_pt
sb_int.fetch = _fetch_copy
sb_pt.fetch = _fetch_copy

def export_ecapa_to_onnx(output_path="ecapa_voxceleb.onnx"):
    print(">>> 1. Loading Pretrained Model from SpeechBrain...")
    # 加载预训练模型
    # source 会自动从 HuggingFace 下载
    savedir = "pretrained_models/spkrec-ecapa-voxceleb"
    try:
        classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=savedir,
        )
    except OSError as e:
        if getattr(e, "winerror", None) == 1314:
            os.makedirs(savedir, exist_ok=True)
            snapshot_download(
                repo_id="speechbrain/spkrec-ecapa-voxceleb",
                local_dir=savedir,
                local_dir_use_symlinks=False,
            )
            classifier = EncoderClassifier.from_hparams(
                source=savedir,
                savedir=savedir,
            )
        else:
            raise
    
    # 获取核心模型组件
    # [...](asc_slot://start-slot-4)这里的 embedding_model 是 ECAPA-TDNN 网络
    # mean_var_norm 是输入特征归一化层
    embedding_model = classifier.mods.embedding_model
    normalizer = classifier.mods.mean_var_norm
    
    # 设置为评估模式
    embedding_model.eval()
    normalizer.eval()

    print(">>> 2. Preparing Wrapper Model...")
    
    # 定义一个 Wrapper 类，将归一化和主模型打包在一起
    # 这样 ONNX 模型就可以直接接受未归一化的 Mel 频谱特征
    class EcapaWrapper(torch.nn.Module):
        def __init__(self, norm, model):
            super().__init__()
            self.norm = norm
            self.model = model

        def forward(self, x):
            # SpeechBrain 的 Normalizer 通常需要 lengths，但推理时 batch=1 或定长
            # 这里简化处理，假设输入为 (Batch, Time, Feats)
            # 归一化
            x = self.norm(x, torch.ones(x.shape[0], device=x.device))
            # [...](asc_slot://start-slot-6)提取 Embedding
            # 注意：ECAPA-TDNN 的 forward 通常也接受 lengths，传 None 默认全长
            out = self.model(x) 
            return out

    # 实例化 Wrapper
    full_model = EcapaWrapper(normalizer, embedding_model)
    
    print(">>> 3. Creating Dummy Input...")
    # ECAPA-TDNN 默认接受的特征维度通常是 80 (Fbank)
    # 我们可以通过 classifier 的 compute_features 验证一下
    # 生成 1 秒的随机波形
    dummy_audio = torch.randn(1, 16000) 
    # 计算特征以获取正确的维度 (Batch, Time, Channels)
    dummy_features = classifier.mods.compute_features(dummy_audio)
    print(f"    Detected input feature shape: {dummy_features.shape}")
    
    # 创建用于导出的 dummy 输入
    # 使用动态轴，这样 ONNX 模型可以接受任意长度的音频特征
    dummy_input = torch.randn(1, 200, 80) # (Batch, Time, Channels)

    print(f">>> 4. Exporting to ONNX: {output_path} ...")
    torch.onnx.export(
        full_model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17, # 建议使用 11 或 12 以获得最佳兼容性
        do_constant_folding=True,
        input_names=['mel_spectrogram'],
        output_names=['embedding'],
        dynamic_axes={
            'mel_spectrogram': {0: 'batch_size', 1: 'time'}, # 允许变长输入
            'embedding': {0: 'batch_size'}
        }
    )
    print(">>> Export successful!")

    # 验证 ONNX 模型
    import onnx
    try:
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print(">>> ONNX model check passed.")
    except Exception as e:
        print(f">>> ONNX model check failed: {e}")

if __name__ == "__main__":
    export_ecapa_to_onnx()
