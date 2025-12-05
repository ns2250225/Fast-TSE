from onnxruntime.quantization import quantize_dynamic, QuantType
import os
import glob
import onnx
import shutil

# ONNX 目录路径
onnx_dir = os.path.join(os.path.dirname(__file__), 'onnx')

# 获取目录下所有 .onnx 文件
onnx_files = glob.glob(os.path.join(onnx_dir, '*.onnx'))

print(f"找到 {len(onnx_files)} 个 ONNX 文件在 {onnx_dir}")

# 设置一个明确的、无中文、短路径的临时目录
TEMP_WORK_DIR = "D:\\FastTSE_Quant_Temp"

# 关键：强制修改系统临时目录环境变量
# 这将迫使 onnxruntime 内部使用的 tempfile 模块使用我们指定的目录
# 从而彻底绕过 C:\Users\马清雄... 路径
os.environ['TEMP'] = TEMP_WORK_DIR
os.environ['TMP'] = TEMP_WORK_DIR

if not os.path.exists(TEMP_WORK_DIR):
    try:
        os.makedirs(TEMP_WORK_DIR, exist_ok=True)
    except Exception as e:
        print(f"❌ 无法创建临时目录 {TEMP_WORK_DIR}: {e}")
        print("请检查 D 盘是否存在或是否有写入权限。")
        exit(1)

print(f"已强制设置系统临时目录到: {TEMP_WORK_DIR}")

for model_fp32 in onnx_files:
    # 跳过已经量化的文件（假设量化文件以 _int8.onnx 或 _fp16.onnx 结尾）
    if model_fp32.endswith('_int8.onnx') or model_fp32.endswith('_fp16.onnx'):
        continue
    
    # 跳过可能残留的中间文件
    if model_fp32.endswith('_inferred_temp.onnx'):
        continue
        
    filename = os.path.basename(model_fp32)
    name_without_ext = os.path.splitext(filename)[0]
    model_quant = os.path.join(onnx_dir, f"{name_without_ext}_fp16.onnx")
    
    print(f"\n正在量化模型 (FP16): {filename} ...")
    
    # 定义临时文件路径
    temp_input = os.path.join(TEMP_WORK_DIR, "input.onnx")
    temp_output = os.path.join(TEMP_WORK_DIR, "output.onnx")
    
    try:
        # 1. 将原始模型拷贝到临时目录
        print(f"  正在拷贝模型到临时目录...")
        shutil.copy(model_fp32, temp_input)
        
        # 2. 执行量化
        # 此时我们对 input.onnx 进行操作。
        # 由于我们已经修改了 TEMP/TMP 环境变量，onnxruntime 内部生成的所有临时文件
        # 都应该会去 D:\FastTSE_Quant_Temp，从而避免中文路径问题。
        print(f"  正在执行 FP16 转换...")
        
        # 使用 convert_float_to_float16 进行 FP16 转换
        # 注意：FP16 转换通常在 float16 模块中
        # 我们检查 onnxruntime 是否有 float16 转换工具
        
        try:
            from onnxruntime.quantization.quant_utils import QuantType
            from onnxruntime.quantization import quantize_dynamic
            # 这里的 quantize_dynamic 主要用于 int8，对于 fp16，我们需要其他方法
            # 更好的方法是使用 onnx 库本身或 onnxmltools，但 requirements.txt 里只有 onnx 和 onnxruntime
            
            # 让我们尝试使用 onnxruntime 的 convert_float_to_float16
            from onnxruntime.transformers.float16 import convert_float_to_float16
            
            # 先加载模型
            import onnx
            model = onnx.load(temp_input)
            
            # 转换
            model_fp16 = convert_float_to_float16(model)
            
            # 保存
            onnx.save(model_fp16, temp_output)
            
        except ImportError:
            print("  onnxruntime.transformers.float16 未找到，尝试使用 quantize_dynamic (但这通常是 int8)...")
            # 如果找不到 fp16 工具，回退到 int8 (虽然用户要求 fp16，但如果没有工具也没办法)
            # 或者我们可以尝试手动转换节点... 但太复杂了
            # 这里我们假设用户安装了完整的 onnxruntime，其中包含 transformers 工具
            raise ImportError("需要 onnxruntime-gpu 或安装了 transformers 工具的 onnxruntime 版本来支持 FP16 转换")

        
        # 3. 将结果拷回
        print(f"  正在保存量化结果...")
        if os.path.exists(temp_output):
            shutil.copy(temp_output, model_quant)
            
            print(f"✅ 量化成功！生成文件: {os.path.basename(model_quant)}")
            
            size_fp32 = os.path.getsize(model_fp32) / 1024 / 1024
            size_quant = os.path.getsize(model_quant) / 1024 / 1024
            print(f"原模型大小: {size_fp32:.2f} MB")
            print(f"量化后大小: {size_quant:.2f} MB")
            print(f"压缩率: {(1 - size_quant/size_fp32)*100:.2f}%")
        else:
            print(f"❌ 量化失败：未生成输出文件")
        
    except Exception as e:
        print(f"❌ 量化失败 {filename}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理临时目录中的文件，为下一个模型做准备
        try:
            # 清理所有文件，保持干净
            for f in glob.glob(os.path.join(TEMP_WORK_DIR, "*")):
                try:
                    if os.path.isfile(f):
                        os.remove(f)
                    elif os.path.isdir(f):
                        shutil.rmtree(f)
                except:
                    pass
        except:
            pass

# 最终尝试清理临时目录
try:
    if os.path.exists(TEMP_WORK_DIR):
        shutil.rmtree(TEMP_WORK_DIR)
except:
    pass
