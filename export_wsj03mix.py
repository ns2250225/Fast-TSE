import torch
import torch.nn as nn
from speechbrain.inference.separation import SepformerSeparation
from huggingface_hub import snapshot_download
import pathlib
import shutil
import urllib.request
import huggingface_hub
import speechbrain.utils.fetching as sb_fetch
import speechbrain.inference.interfaces as sb_int
import speechbrain.utils.parameter_transfer as sb_pt

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
sb_int.fetch = _fetch_copy
sb_pt.fetch = _fetch_copy

# --- 1. 修正后的 Wrapper 类 ---
class SepFormerWrapper(nn.Module):
    def __init__(self, sb_model):
        super(SepFormerWrapper, self).__init__()
        self.encoder = sb_model.mods.encoder
        self.masknet = sb_model.mods.masknet
        self.decoder = sb_model.mods.decoder
        self.num_spks = sb_model.hparams.num_spks
        
    def forward(self, mix):
        """
        Args:
            mix: (Batch, Time)
        """
        # 1. Encoder: (Batch, Time) -> (Batch, Feats, Time)
        mix_w = self.encoder(mix)
        
        # 2. MaskNet: 输出通常是 (Batch, Spks, Feats, Time)
        est_mask = self.masknet(mix_w)
        
        # 3. Apply Mask (修正维度匹配问题)
        # mix_w 是 (Batch, Feats, Time)
        # est_mask 是 (Batch, Spks, Feats, Time)
        # 我们将 mix_w 在 dim=1 增加一个维度变成 (Batch, 1, Feats, Time)
        # 这样它就可以自动广播以匹配 Spks 维度
        mix_w_expanded = mix_w.unsqueeze(1)
        
        # 逐元素相乘: (Batch, 1, F, T) * (Batch, Spks, F, T) -> (Batch, Spks, F, T)
        sep_h = mix_w_expanded * est_mask
        
        # 4. Decoder
        # Decoder 期望输入 (Batch * Spks, Feats, Time)
        batch, spks, feats, time = sep_h.shape
        
        # 将 Batch 和 Spks 合并
        sep_h_flat = sep_h.view(batch * spks, feats, time)
        
        # 解码: (Batch * Spks, Feats, Time) -> (Batch * Spks, Time)
        est_source_flat = self.decoder(sep_h_flat)
        
        # 恢复维度: (Batch * Spks, Time) -> (Batch, Spks, Time)
        est_source = est_source_flat.view(batch, spks, -1)
        
        # 最终输出调整为 (Batch, Time, Spks) 以符合常见习惯，或者保持 (Batch, Spks, Time)
        # 这里我们输出 (Batch, Time, Spks)
        est_source = est_source.transpose(1, 2)
        
        return est_source

# --- 2. 加载模型 ---
print("正在加载 SpeechBrain 模型...")
# 确保 savedir 存在，避免重复下载
savedir = "pretrained_models/sepformer-wsj03mix"
try:
    sb_model = SepformerSeparation.from_hparams(
        source="speechbrain/sepformer-wsj03mix",
        savedir=savedir
    )
except OSError as e:
    if getattr(e, "winerror", None) == 1314:
        pathlib.Path(savedir).mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id="speechbrain/sepformer-wsj03mix",
            local_dir=savedir,
            local_dir_use_symlinks=False,
        )
        sb_model = SepformerSeparation.from_hparams(
            source=savedir,
            savedir=savedir
        )
    else:
        raise

# --- 3. 实例化 Wrapper ---
model_wrapper = SepFormerWrapper(sb_model)
model_wrapper.eval()

# --- 4. 创建虚拟输入 (尽量使用对齐的大小以减少 TracerWarning) ---
# 为了减少 "if gap > 0" 的警告，我们可以尝试使用一个能被 chunk size 整除的长度。
# 但动态轴设置好后，onnx 应该能处理不同长度。
# 这里使用 16000 (2秒 8k音频)
dummy_input = torch.randn(1, 16000)

# --- 5. 导出 ---
onnx_path = "sepformer_wsj03mix.onnx"
print(f"正在导出到 {onnx_path} ...")

try:
    torch.onnx.export(
        model_wrapper,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=17,          # 保持 17 以支持 STFT
        do_constant_folding=True,
        input_names=['mix'],
        output_names=['est_sources'],
        dynamic_axes={
            'mix': {0: 'batch_size', 1: 'time'},
            'est_sources': {0: 'batch_size', 1: 'time'}
        }
    )
    print("导出成功！忽略上面的 TracerWarning 是正常的。")
    
except Exception as e:
    print(f"导出仍然失败: {e}")
