# 快速的目标说话人语音提取服务
基于speechbrain的目标说话人提取服务（TSE），平均响应400ms

本项目提供两种使用方式：
- 命令行脚本：将混合多人语音分离为独立说话人音频，并可匹配目标说话人
- HTTP 服务：FastAPI 接口，支持文件上传，返回最匹配说话人的音频与耗时统计

## 环境与安装
- Python 3.9+（推荐）
- 安装依赖：

```bash
pip install -r requirements.txt
```

requirements.txt 中包含：`torch`、`torchaudio`、`speechbrain`、`soundfile`、`numpy<2.0`、`fastapi`、`uvicorn`、`python-multipart`、`onnx`、`onnxruntime` 等。

说明：
- **GPU 支持**：默认优先使用 CUDA（如可用）。
- **ONNX Runtime 加速**：支持使用 ONNX Runtime 进行推理加速（CPU/GPU）。
- **模型预热**：首次运行会自动下载 SpeechBrain 预训练模型到 `pretrained_models/`。

## 命令行脚本
文件：`sp.py`

功能：
- 对混合语音进行源分离（支持 2/3 人）
- 可按能量排序输出各路音频到当前目录
- 支持传入目标人音频，使用 ECAPA 说话人嵌入进行相似度匹配，输出最匹配音频
- 打印模型加载、分离、匹配的耗时统计

基本用法：

```bash
# 两人分离
python sp.py mixed.wav -n 2

# 三人分离
python sp.py mixed.wav -n 3

# 分离并匹配目标人
python sp.py mixed.wav -n 2 -t target.wav

# 强制使用CPU
python sp.py mixed.wav -n 2 -t target.wav --device cpu
```

常用参数：
- `-n, --num_speakers`：说话人数（2 或 3）
- `-t, --target`：目标说话人音频路径
- `--device`：推理设备，`cpu` 或 `cuda`
- `--normalize`：对分离结果做幅度归一化
- `-m, --model`：自定义 SpeechBrain 模型仓库名（默认使用 SepFormer 2/3 人模型）
- `--output_prefix`：自定义输出文件前缀

## HTTP 服务
文件：`api.py`

### 配置项 (api.py)
在 `api.py` 开头部分可以配置以下选项：
- `ENABLE_ONNX = True`: 启用 ONNX Runtime 加速（推荐）。
- `FORCE_ONNX_CPU = True`: 强制使用 CPU 进行 ONNX 推理，并启用 INT8 量化模型（需先运行量化脚本）。

### 模型量化 (推荐 CPU 环境使用, 运行export的脚本可以导出32的onnx，之后放到onnx目录再执行量化脚本生成int8的onnx)
如果在 CPU 环境下运行，建议使用量化模型以获得最佳性能（速度提升约 2-3 倍）。

1. 确保 `onnx/` 目录下有原始 `.onnx` 模型文件。
2. 运行量化脚本：
   ```bash
   python quantize.py
   ```
   脚本会自动生成 `_int8.onnx` 后缀的量化模型。
3. 在 `api.py` 中设置 `FORCE_ONNX_CPU = True`。

### 启动服务

```bash
# 方式一：直接运行脚本
python api.py

# 方式二：使用 uvicorn 命令
uvicorn api:app --host 0.0.0.0 --port 8000
```

启动行为：
- 应用生命周期的 `lifespan` 在接收请求前预加载模型。
- 自动进行模型预热（Warmup），避免首次请求延迟。
- 如果启用了 ONNX，会自动加载 TensorRT（GPU）或 Quantized INT8（CPU）模型。

### 接口文档

#### JSON 端点
`POST /separate-match`
- `multipart/form-data`
- 字段：
  - `mixed`：混合音频文件
  - `target`：目标人音频文件
  - `num_speakers`：说话人数（默认 2）
  - `normalize`：是否归一化（默认 true）
  - `match_threshold`：匹配相似度阈值（默认 0.25）

示例：

```bash
curl -X POST \
  -F "mixed=@mixed.wav" \
  -F "target=@target.wav" \
  -F "num_speakers=2" \
  -F "match_threshold=0.25" \
  http://localhost:8000/separate-match
```

返回 JSON：
- 成功匹配：
  - `matched_speaker_index`：最匹配说话人序号（从 1 开始）
  - `similarity`：余弦相似度分数
  - `audio_wav_base64`：最匹配音频的 WAV Base64
  - `timings`：预加载、分离、匹配计算、总耗时
  - `device`：分离与匹配所用设备
- 未找到匹配（相似度 <= match_threshold）：
  - `code`: -1
  - `message`: "没有目标人声音"

#### 字节流端点
`POST /separate-match-wav`
- `multipart/form-data`
- 字段与 JSON 端点一致
- 响应：
  - 成功匹配：`application/octet-stream`，直接返回 WAV 字节。响应头包含元数据：
    - `X-Matched-Speaker-Index`
    - `X-Similarity`
    - `X-Separation-Time-Sec`
    - `X-Match-Compute-Time-Sec`
    - `X-Total-Time-Sec`
    - `X-Device-Separation`
    - `X-Device-Match`
  - 未找到匹配（相似度 <= match_threshold）：返回 JSON `{"code": -1, "message": "没有目标人声音"}`，Content-Type 为 `application/json`。

示例下载：

```bash
curl -X POST \
  -F "mixed=@mixed.wav" \
  -F "target=@target.wav" \
  -F "num_speakers=2" \
  -F "match_threshold=0.25" \
  http://localhost:8000/separate-match-wav -o best.wav
```

## 说明与建议
- 音频采样率自动重采样到模型支持的采样率（SepFormer/ECAPA 默认 16k）
- 如果混合人数不确定，建议先用 `num_speakers=3` 尝试
- 目标匹配的相似度基于说话人嵌入余弦相似度，建议目标音频尽量干净且不少于 2~3 秒
- Windows 下如需 GPU，请安装与 CUDA 版本匹配的 PyTorch

## 项目结构
- `sp.py`：命令行分离与目标匹配脚本
- `api.py`：FastAPI 服务，提供 JSON 与字节流端点
- `quantize.py`: ONNX 模型量化脚本
- `requirements.txt`：依赖列表
- `onnx/`: 存放 ONNX 模型文件

## 许可
此项目用于演示/实验用途。
