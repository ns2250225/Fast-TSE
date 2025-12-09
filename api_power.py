import io
import os
import time
import shutil
import tempfile
import asyncio
import torch
import torchaudio
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from speechbrain.inference.separation import SepformerSeparation

# --- 全局变量 ---
model = None
device = None
gpu_lock = asyncio.Lock()  # 必须加锁，防止并发请求导致 GPU 显存冲突

# --- 1. 生命周期管理 (启动加载 & 预热) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, device
    print("⏳ [Startup] 正在初始化环境...")
    
    # 硬件配置
    if torch.cuda.is_available():
        device = "cuda"
        # 关闭 benchmark 避免首次动态搜索算法耗时
        torch.backends.cudnn.benchmark = False 
        # Ampere 架构开启 TF32
        if torch.cuda.get_device_capability()[0] >= 8:
            torch.set_float32_matmul_precision('high')
        print(f"🚀 使用设备: GPU ({torch.cuda.get_device_name(0)})")
    else:
        device = "cpu"
        print("⚠️ 使用设备: CPU")

    # 加载模型
    print("⏳ [Startup] 正在加载模型 (常驻内存)...")
    run_opts = {"device": device}
    model = SepformerSeparation.from_hparams(
        source="speechbrain/sepformer-wsj03mix",
        savedir="pretrained_models/sepformer-wsj03mix",
        run_opts=run_opts
    )
    model.eval()

    # 预热 GPU
    if device == "cuda":
        print("🔥 [Startup] 正在预热 GPU...")
        dummy_input = torch.randn(1, 8000).to(device)
        with torch.inference_mode():
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                _ = model.separate_batch(dummy_input)
        torch.cuda.synchronize()
        print("✅ [Startup] 预热完成，服务已就绪！")
    
    yield
    
    # 关闭时清理
    print("🛑 [Shutdown] 服务关闭，清理资源...")
    if device == "cuda":
        torch.cuda.empty_cache()

# --- 2. 初始化 FastAPI ---
app = FastAPI(title="Audio Separation API", lifespan=lifespan)

# --- 3. 核心接口逻辑 ---
@app.post("/separate")
async def separate_audio_endpoint(file: UploadFile = File(...)):
    """
    上传混合音频，返回分离后能量最大的音频流 (WAV格式)
    """
    global model, device

    # 步骤 A: 保存上传的文件到临时目录
    # SpeechBrain 需要文件路径作为输入
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_input:
        shutil.copyfileobj(file.file, temp_input)
        temp_input_path = temp_input.name

    try:
        # 步骤 B: 获取 GPU 锁并执行推理
        # 使用 async with gpu_lock 确保同一时间只有一个请求在使用 GPU
        async with gpu_lock:
            start_time = time.time()
            
            # --- 极速推理核心 ---
            with torch.inference_mode():
                if device == "cuda":
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        est_sources = model.separate_file(path=temp_input_path)
                else:
                    est_sources = model.separate_file(path=temp_input_path)
                
                # --- 能量筛选 (GPU内完成) ---
                # est_sources: [batch=1, time, sources]
                # 计算平方和能量，找出最大值的索引
                energies = est_sources.pow(2).sum(dim=1).squeeze()
                best_idx = torch.argmax(energies).item()
                best_source = est_sources[:, :, best_idx]
            
            if device == "cuda":
                torch.cuda.synchronize()
            
            infer_time = time.time() - start_time
            print(f"✅ [Request] 推理完成，耗时: {infer_time:.4f}s | 选中源索引: {best_idx}")

        # 步骤 C: 将结果写入内存 Buffer (不写磁盘，速度更快)
        # 必须转回 float32 否则 wav 编码会报错
        source_cpu = best_source.detach().cpu().float()
        
        buffer = io.BytesIO()
        torchaudio.save(buffer, source_cpu, 8000, format="wav")
        buffer.seek(0) # 指针回到开头

        # 步骤 D: 返回流式响应
        return StreamingResponse(
            buffer, 
            media_type="audio/wav",
            headers={"Content-Disposition": f"attachment; filename=best_source_{best_idx}.wav"}
        )

    except Exception as e:
        return {"error": str(e)}
        
    finally:
        # 步骤 E: 清理临时输入文件
        if os.path.exists(temp_input_path):
            os.remove(temp_input_path)

if __name__ == "__main__":
    import uvicorn
    # 启动服务
    uvicorn.run(app, host="0.0.0.0", port=8000)
