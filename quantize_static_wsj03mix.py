import os
import random
import numpy as np
import librosa
import onnx
from onnxruntime.quantization import CalibrationDataReader, quantize_static, QuantFormat, QuantType

class Synthetic3MixDataReader(CalibrationDataReader):
    """
    自动将单人语音混合成 3人混合语音 用于校准
    """
    def __init__(self, audio_folder, input_name, num_samples=50, chunk_seconds=4.0):
        self.audio_paths = [os.path.join(audio_folder, f) for f in os.listdir(audio_folder) if f.endswith('.wav')]
        if len(self.audio_paths) < 3:
            raise ValueError("文件夹里的 wav 文件太少，至少需要 3 个不同的文件来合成 3mix 数据！")
            
        self.input_name = input_name
        self.num_samples = num_samples # 生成多少条校准数据
        self.sr = 8000  # SepFormer 必须是 8k
        self.max_len = int(chunk_seconds * self.sr)
        self.iter_count = 0

    def _load_and_fix(self, path):
        # 加载并重采样到 8k
        audio, _ = librosa.load(path, sr=self.sr, mono=True)
        # 长度处理：截断或补零
        if len(audio) > self.max_len:
            start = random.randint(0, len(audio) - self.max_len)
            audio = audio[start : start + self.max_len]
        else:
            padding = np.zeros(self.max_len - len(audio))
            audio = np.concatenate((audio, padding))
        return audio

    def get_next(self):
        if self.iter_count >= self.num_samples:
            return None
        
        # 1. 随机抽取 3 个不同的音频文件
        sources = random.sample(self.audio_paths, 3)
        
        # 2. 读取并处理
        s1 = self._load_and_fix(sources[0])
        s2 = self._load_and_fix(sources[1])
        s3 = self._load_and_fix(sources[2])
        
        # 3. 【核心】合成 3mix 信号
        # 直接相加
        mix = s1 + s2 + s3
        
        # 4. 防止爆音 (Clipping)，稍微归一化一下，保持在 [-1, 1] 之间
        # 模拟真实环境，混合声音通常会有幅度叠加
        max_amp = np.max(np.abs(mix))
        if max_amp > 0:
            mix = mix / max_amp * 0.9 # 缩放到 0.9 以防万一
            
        # 5. 调整形状 [Batch, Time] -> [1, 32000]
        input_tensor = mix[np.newaxis, :].astype(np.float32)
        
        self.iter_count += 1
        return {self.input_name: input_tensor}

# ================= 配置区 =================
MODEL_PATH = 'sepformer_wsj03mix.onnx'     # 你的 3mix 模型路径
OUTPUT_PATH = 'sepformer_wsj03mix_static_int8.onnx' # 输出路径
AUDIO_FOLDER = './test_audio_dir'         # 【重要】这里放任意的单人语音wav文件即可
# =========================================

# 1. 自动探测输入节点名称
print(f"正在加载模型: {MODEL_PATH} ...")
model = onnx.load(MODEL_PATH)
input_node_name = model.graph.input[0].name
print(f"检测到模型输入节点名称为: '{input_node_name}'")

# 2. 准备数据读取器
print("初始化数据读取器 (自动合成 3-Mix)...")
try:
    dr = Synthetic3MixDataReader(AUDIO_FOLDER, input_node_name)
except ValueError as e:
    print(f"错误: {e}")
    exit(1)

# 3. 开始量化
print("开始静态量化 (这可能需要几分钟)...")

quantize_static(
    model_input=MODEL_PATH,
    model_output=OUTPUT_PATH,
    calibration_data_reader=dr,
    
    # CPU 加速关键配置
    quant_format=QuantFormat.QOperator,  # 使用 QOperator 格式以获得 CPU 最佳性能
    weight_type=QuantType.QInt8,         # 权重用带符号 INT8
    activation_type=QuantType.QUInt8,    # 激活值用无符号 UINT8
    
    per_channel=True,  # CNN/Audio 模型建议开启
    reduce_range=False # 如果量化后杂音很重，可以尝试改为 True
)

print(f"✅ 量化完成！模型已保存至: {OUTPUT_PATH}")
print("提示: 推理时请确保输入音频也是 8000Hz 采样率。")
