import torch
import pyaudio
import numpy as np
from speechbrain.pretrained import SpeakerRecognition
from speechbrain.pretrained import SepformerSeparation as separator
from scipy.io.wavfile import write

# 设置设备：检查是否有GPU支持
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 初始化说话人识别模型并加载到GPU
print("加载说话人识别模型...")
speaker_rec = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", savedir="tmpdir_spk").to(DEVICE)

# 加载目标说话人声纹
target_audio_path = "target_speaker.wav"
print(f"加载目标说话人音频：{target_audio_path}...")
target_speaker_embedding = speaker_rec.encode_file(target_audio_path).to(DEVICE)

# 初始化分离模型并加载到GPU
print("加载说话人分离模型...")
separator_model = separator.from_hparams(source="speechbrain/sepformer-libri-960h", savedir="tmpdir_separation").to(DEVICE)

# 设置麦克风参数
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
RECORD_SECONDS = 5

# 设置麦克风
p = pyaudio.PyAudio()

# 用来存储音频帧
def record_audio():
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    frames = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    stream.stop_stream()
    stream.close()
    audio_data = b''.join(frames)
    return np.frombuffer(audio_data, dtype=np.int16)

# 使用说话人分离模型将多个说话人的声音分离
def separate_speakers(audio_data):
    # 使用SpeechBrain的Sepformer模型进行多说话人分离
    est_sources, _ = separator_model.separate_batch(torch.tensor(audio_data).unsqueeze(0).to(DEVICE))
    return est_sources

# 比较音频数据与目标说话人的声纹
def match_speaker(audio_data):
    # 提取当前音频数据的说话人声纹
    embedding = speaker_rec.encode_audio(torch.tensor(audio_data).float().to(DEVICE))
    score, _ = speaker_rec.verify_batch(embedding, target_speaker_embedding)
    return score

print("开始监听并提取指定说话人的声音...")

# 实时音频流处理
while True:
    # 从麦克风获取音频数据
    audio_data = record_audio()

    # 使用说话人分离模型分离音频数据中的多个说话人的声音
    separated_sources = separate_speakers(audio_data)

    # 检查每个分离的声音，并进行声纹匹配
    for i, source in enumerate(separated_sources):
        score = match_speaker(source.cpu().numpy())  # 将结果从GPU转移到CPU进行处理

        if score > 0.8:  # 设置一个阈值，如果匹配度高，认为是目标说话人
            print(f"识别到目标说话人（Source {i+1}），保存音频...")
            write(f"recognized_audio_source_{i+1}.wav", RATE, source.cpu().numpy())
        else:
            print(f"Source {i+1} 没有识别到目标说话人的声音。")
