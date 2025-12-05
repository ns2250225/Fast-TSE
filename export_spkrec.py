import torch
import speechbrain
from speechbrain.inference.speaker import EncoderClassifier
import os

def export_ecapa_to_onnx(output_path="ecapa_voxceleb.onnx"):
    print(">>> 1. Loading Pretrained Model from SpeechBrain...")
    # 加载预训练模型
    # source 会自动从 HuggingFace 下载
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", 
        savedir="pretrained_models/spkrec-ecapa-voxceleb"
    )
    
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
