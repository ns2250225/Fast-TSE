import os
import random
import numpy as np
import librosa
import onnx
from onnx import shape_inference
from onnxruntime.quantization import (
    CalibrationDataReader, 
    quantize_static, 
    QuantFormat, 
    QuantType
)

# ================= 数据读取器 (保持不变) =================
class VoxCelebFeatureDataReader(CalibrationDataReader):
    def __init__(self, audio_folder, input_name, num_samples=50, chunk_seconds=3.0):
        self.audio_paths = [os.path.join(audio_folder, f) for f in os.listdir(audio_folder) if f.endswith('.wav')]
        if not self.audio_paths:
            raise ValueError(f"错误：文件夹 {audio_folder} 中没有找到 .wav 文件！")
        
        self.input_name = input_name
        self.num_samples = num_samples
        self.sr = 16000  
        self.max_len = int(chunk_seconds * self.sr) 
        self.iter_count = 0

    def compute_fbank(self, waveform):
        melspec = librosa.feature.melspectrogram(
            y=waveform, sr=self.sr, n_fft=400, hop_length=160, n_mels=80,
            fmin=0, fmax=8000, window='hamming'
        )
        log_melspec = np.log(melspec + 1e-6)
        log_melspec = log_melspec.T  # [Time, 80]
        input_tensor = log_melspec[np.newaxis, :].astype(np.float32) # [1, Time, 80]
        return input_tensor

    def _load_and_process(self, path):
        audio, _ = librosa.load(path, sr=self.sr, mono=True)
        if len(audio) > self.max_len:
            start = random.randint(0, len(audio) - self.max_len)
            audio = audio[start : start + self.max_len]
        else:
            tile_count = (self.max_len // len(audio)) + 1
            audio = np.tile(audio, tile_count)[:self.max_len]
        return self.compute_fbank(audio)

    def get_next(self):
        if self.iter_count >= self.num_samples:
            return None
        wav_path = random.choice(self.audio_paths)
        data = self._load_and_process(wav_path)
        self.iter_count += 1
        return {self.input_name: data}

# ================= 配置区 =================
MODEL_PATH = 'ecapa_voxceleb.onnx'   
PREPROCESSED_MODEL = 'spkrec-ecapa-inferred.onnx'
OUTPUT_PATH = 'ecapa_voxceleb_static_int8.onnx'
AUDIO_FOLDER = './test_audio_dir'          
# =========================================

# --- Step 0: 手动形状推导 ---
print("正在加载模型并修复形状信息 (Shape Inference)...")
model_proto = onnx.load(MODEL_PATH)
inferred_model = shape_inference.infer_shapes(model_proto, check_type=True, strict_mode=False, data_prop=True)
onnx.save(inferred_model, PREPROCESSED_MODEL)
print(f"预处理完成，已保存: {PREPROCESSED_MODEL}")

# --- Step 1: 探测输入节点 ---
input_node_name = inferred_model.graph.input[0].name
print(f"输入节点名称: '{input_node_name}'")

# --- Step 2: 初始化读取器 ---
dr = VoxCelebFeatureDataReader(AUDIO_FOLDER, input_node_name)

# --- Step 3: 执行静态量化 ---
print("开始静态量化...")

# 【如果你依然遇到 "Invalid model" 错误】
# 请尝试将下面这行改为 target_format = QuantFormat.QDQ
target_format = QuantFormat.QDQ 

try:
    quantize_static(
        model_input=PREPROCESSED_MODEL, 
        model_output=OUTPUT_PATH,
        calibration_data_reader=dr,
        quant_format=target_format, 
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
        per_channel=True,
        reduce_range=False
        # optimize_model=False  <-- 已移除此参数
    )
    print(f"✅ 量化成功！模型已保存至: {OUTPUT_PATH}")

except Exception as e:
    print(f"\n❌ 量化失败: {e}")
    print("\n⚠️ 调试建议: 如果报错包含 'Invalid model' 或 'unknown initializers'，")
    print("请将代码中的 target_format 改为 QuantFormat.QDQ 再试一次。")
