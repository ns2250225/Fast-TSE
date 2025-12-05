import os
import numpy as np
import librosa
from onnxruntime.quantization import CalibrationDataReader, quantize_static, QuantFormat, QuantType

class SepFormerDataReader(CalibrationDataReader):
    def __init__(self, audio_folder, input_name, chunk_seconds=4.0):
        """
        audio_folder: 存放混合音频 .wav 的文件夹
        input_name: ONNX 模型的输入节点名称 (通常是 'input' 或 'speech_mix')
        chunk_seconds: 校准时截取的音频长度，默认4秒
        """
        self.audio_paths = [os.path.join(audio_folder, f) for f in os.listdir(audio_folder) if f.endswith('.wav')][:50] # 取50个足矣
        self.input_name = input_name
        self.sr = 8000  # 【关键】WSJ0-2mix 模型必须用 8k 采样率
        self.max_len = int(chunk_seconds * self.sr)
        self.enum_data = None

    def preprocess(self, wav_path):
        # 1. 加载音频并重采样到 8000Hz
        # librosa 加载默认就是 float32, 归一化到 [-1, 1]
        audio, _ = librosa.load(wav_path, sr=self.sr, mono=True)
        
        # 2. 长度处理：为了 Batch 处理，统一裁剪或 Padding 到固定长度
        if len(audio) > self.max_len:
            audio = audio[:self.max_len]
        else:
            # 如果短了就补零
            padding = np.zeros(self.max_len - len(audio))
            audio = np.concatenate((audio, padding))
            
        # 3. 调整形状
        # SepFormer ONNX 输入通常是 [Batch, Time] -> [1, 32000]
        # 如果你的模型输入是 [Batch, 1, Time]，请改为 audio[np.newaxis, np.newaxis, :]
        input_tensor = audio[np.newaxis, :].astype(np.float32) 
        
        return input_tensor

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter(self.audio_paths)
        
        wav_path = next(self.enum_data, None)
        if wav_path:
            return {self.input_name: self.preprocess(wav_path)}
        return None

# --- 执行量化 ---

# 1. 确定输入节点名
# 你可以用 Netron 打开 onnx 看，或者用 python onnx 库查
# 假设 SepFormer 的输入叫做 'mix'
input_node_name = 'mix' 

# 2. 准备数据
# 请确保 'test_audio_dir' 里面放的是 8k 或其他采样率的 wav 文件
# 如果没有现成的混合音频，你可以随便找点人声放进去，代码会自动读取
dr = SepFormerDataReader("test_audio_dir", input_node_name)

quantize_static(
    model_input='sepformer_wsj02mix.onnx',
    model_output='sepformer_wsj02mix_static_int8.onnx',
    calibration_data_reader=dr,
    quant_format=QuantFormat.QOperator, # CPU 推荐
    weight_type=QuantType.QInt8,
    per_channel=True # 建议开启
)
