import torch
import torch.nn as nn
from speechbrain.inference.separation import SepformerSeparation
import os

# ==========================================
# 1. 更加健壮的模型包装器
# [...](asc_slot://start-slot-1)==========================================
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
        Returns:
            est_source: (Batch, Time, Spks)
        """
        # --- 关键修改 1: 动态获取 Batch Size ---
        # 使用 .size(0) 而不是 .shape[0]，这样 ONNX 会将其视为动态变量而不是常量
        batch_size = mix.size(0)
        
        # 1. Encoder: (Batch, Time) -> (Batch, Feats, Time)
        mix_w = self.encoder(mix)
        
        # 2. MaskNet
        est_mask = self.masknet(mix_w)
        
        # 3. [...](asc_slot://start-slot-3)Apply Mask
        # (Batch, Feats, Time) -> (Batch, 1, Feats, Time)
        mix_w_expanded = mix_w.unsqueeze(1)
        
        # (Batch, 1, F, T) * (Batch, Spks, F, T) -> (Batch, Spks, F, T)
        sep_h = mix_w_expanded * est_mask
        
        # 准备进入 Decoder
        # 获取特征维度 (F) 和时间维度 (T)
        # 注意：这里 F 和 T 可以用 shape 获取，因为特征维通常是固定的，时间维是动态轴会自动处理
        feats = sep_h.shape[2] 
        # time = sep_h.shape[3] 
        
        # 4. Reshape for Decoder
        # 将 (Batch, Spks, ...) 合并为 (Batch * Spks, ...)
        # 使用 reshape 而不是 view，更加稳健
        sep_h_flat = sep_h.reshape(batch_size * self.num_spks, feats, -1)
        
        # 5. Decoder
        est_source_flat = self.decoder(sep_h_flat)
        
        # --- 关键修改 2: 处理 Decoder 可能的 3D 输出 (B, 1, T) ---
        if est_source_flat.dim() == 3 and est_source_flat.size(1) == 1:
            est_source_flat = est_source_flat.squeeze(1)
        elif est_source_flat.dim() == 3 and est_source_flat.size(2) == 1:
             est_source_flat = est_source_flat.squeeze(2)
            
        # 此时 est_source_flat 应该是 (Batch * Spks, Time)
        
        # 6. 恢复形状
        # (Batch * Spks, Time) -> (Batch, Spks, Time)
        # 使用动态的 batch_size 变量
        est_source = est_source_flat.view(batch_size, self.num_spks, -1)
        
        # 7. 转置为 (Batch, Time, Spks)
        est_source = est_source.transpose(1, 2)
        
        return est_source

def export_onnx():
    model_source = "speechbrain/sepformer-wsj02mix"
    onnx_path = "sepformer_wsj02mix.onnx"
    
    print(f"--- 正在加载模型: {model_source} ---")
    sb_model = SepformerSeparation.from_hparams(
        source=model_source,
        savedir="pretrained_models/sepformer-wsj02mix"
    )
    
    model_wrapper = SepFormerWrapper(sb_model)
    model_wrapper.eval()
    
    # 虚拟输入
    dummy_input = torch.randn(1, 16000)
    
    print(f"--- 正在导出到 {onnx_path} ---")
    try:
        torch.onnx.export(
            model_wrapper,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=17, 
            do_constant_folding=True,
            input_names=['mix'],
            output_names=['est_sources'],
            dynamic_axes={
                'mix': {0: 'batch_size', 1: 'time'},
                'est_sources': {0: 'batch_size', 1: 'time'}
            }
        )
        print(f"✅ 导出成功！文件: {onnx_path}")

    except Exception as e:
        print(f"❌ 导出失败: {e}")
        return

    # [...](asc_slot://start-slot-9)--- 验证步骤 ---
    try:
        import onnxruntime as ort
        import numpy as np
        
        print("\n--- 最终验证 ---")
        ort_session = ort.InferenceSession(onnx_path)
        
        # 测试 Batch=1
        test_input = np.random.randn(1, 24000).astype(np.float32)
        ort_outs = ort_session.run(None, {'mix': test_input})
        out_shape = ort_outs[0].shape
        print(f"输入 (1, 24000) -> 输出 {out_shape}")
        
        if len(out_shape) == 3 and out_shape[0] == 1 and out_shape[2] == 2:
            print("✅ 验证完美通过！格式为 (Batch, Time, Spks)")
        else:
            print("❌ 格式依然不对，请检查上面的输出形状。")
            
    except Exception as e:
        print(f"验证出错: {e}")

if __name__ == "__main__":
    export_onnx()
