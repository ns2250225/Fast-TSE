import os
import glob
import shutil
import numpy as np
import onnx
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat
from onnxruntime.quantization.shape_inference import quant_pre_process


class RandomDataReader(CalibrationDataReader):
    def __init__(self, model_path, count=10):
        self.model_path = model_path
        self.count = count
        self.enum_data = None
        self.input_specs = self._get_input_specs()
    
    def _get_input_specs(self):
        model = onnx.load(self.model_path)
        specs = []
        for input in model.graph.input:
            shape = []
            for d in input.type.tensor_type.shape.dim:
                if d.dim_value > 0:
                    shape.append(d.dim_value)
                else:
                    # Dynamic dimension strategy
                    # Usually audio models have [Batch, Time, Channels] or [Batch, Time]
                    # Let's assume dynamic dim is Time, and set a reasonable length e.g. 200
                    # If it's batch, we usually set to 1
                    shape.append(200 if len(shape) > 0 else 1) 
            
            # Special case adjustments based on inspection or common knowledge
            # If shape ended up being [1, 200, 80] (MelSpec) -> Reasonable
            # If shape ended up being [1, 200] (Raw Audio) -> Reasonable
            
            specs.append({'name': input.name, 'shape': shape, 'type': input.type.tensor_type.elem_type})
        return specs

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter(range(self.count))
        
        try:
            next(self.enum_data)
        except StopIteration:
            return None

        input_feed = {}
        for spec in self.input_specs:
            # Generate random float data
            data = np.random.rand(*spec['shape']).astype(np.float32)
            input_feed[spec['name']] = data
        
        return input_feed

# ONNX 目录路径
onnx_dir = os.path.join(os.path.dirname(__file__), 'onnx')

# 获取目录下所有 .onnx 文件
onnx_files = glob.glob(os.path.join(onnx_dir, '*.onnx'))

print(f"找到 {len(onnx_files)} 个 ONNX 文件在 {onnx_dir}")

# 设置一个明确的、无中文、短路径的临时目录
TEMP_WORK_DIR = "D:\\FastTSE_Quant_Temp"

# 关键：强制修改系统临时目录环境变量
os.environ['TEMP'] = TEMP_WORK_DIR
os.environ['TMP'] = TEMP_WORK_DIR

if not os.path.exists(TEMP_WORK_DIR):
    try:
        os.makedirs(TEMP_WORK_DIR, exist_ok=True)
    except Exception as e:
        print(f"❌ 无法创建临时目录 {TEMP_WORK_DIR}: {e}")
        exit(1)

print(f"已强制设置系统临时目录到: {TEMP_WORK_DIR}")

for model_fp32 in onnx_files:
    # 跳过已经量化的文件
    if '_int8' in model_fp32 or '_fp16' in model_fp32 or '_static' in model_fp32:
        continue
    
    # 跳过中间文件
    if model_fp32.endswith('_inferred_temp.onnx'):
        continue
        
    filename = os.path.basename(model_fp32)
    name_without_ext = os.path.splitext(filename)[0]
    # 命名为 _static_int8 以示区别
    model_quant = os.path.join(onnx_dir, f"{name_without_ext}_static_int8.onnx")
    
    print(f"\n正在静态量化模型 (Static INT8): {filename} ...")
    
    # 定义临时文件路径
    temp_input = os.path.join(TEMP_WORK_DIR, "input.onnx")
    temp_processed = os.path.join(TEMP_WORK_DIR, "processed.onnx")
    temp_output = os.path.join(TEMP_WORK_DIR, "output.onnx")
    
    try:
        # 1. 将原始模型拷贝到临时目录
        print(f"  正在拷贝模型到临时目录...")
        shutil.copy(model_fp32, temp_input)
        
        # 1.1 特殊处理：修复 ecapa_voxceleb 的输入形状
        # 解决 Shape Inference 失败的问题 (Time dim too small or unknown)
        use_preprocess = False
        model_to_quantize = temp_input
        
        if 'ecapa_voxceleb' in filename:
             print(f"  [Fix] 针对 ecapa_voxceleb 强制设置时间维度为 200...")
             model = onnx.load(temp_input)
             inp = model.graph.input[0]
             # 假设 shape 是 [Batch, Time, Feats] = [1, 1, 80]
             # 修改 Time 维度 (索引 1)
             if len(inp.type.tensor_type.shape.dim) == 3:
                 inp.type.tensor_type.shape.dim[1].dim_value = 200
             onnx.save(model, temp_input)
             use_preprocess = True

        # 1.5 预处理模型 (关键：修复 Shape Inference 问题)
        # 仅对 ecapa_voxceleb 启用，因为 sepformer 在预处理时会失败
        if use_preprocess:
            print(f"  正在执行预处理 (quant_pre_process)...")
            quant_pre_process(temp_input, temp_processed)
            model_to_quantize = temp_processed
        
        # 2. 准备校准数据读取器 (使用预处理后的模型或原模型)
        print(f"  正在准备校准数据 (Random)...")
        dr = RandomDataReader(model_to_quantize, count=5) # 少量校准数据用于演示
        
        # 2.5 准备排除列表 (针对 ecapa_voxceleb 的 ASP 模块)
        # ASP 模块包含全局统计信息计算，容易导致 shape inference 失败或量化误差
        nodes_to_exclude = []
        if 'ecapa_voxceleb' in filename:
            print(f"  正在分析需要排除的节点 (ASP)...")
            model_for_exclude = onnx.load(model_to_quantize)
            for node in model_for_exclude.graph.node:
                # 排除名字中包含 asp 的节点，或者输出包含 asp 的节点
                if 'asp' in node.name or (len(node.output) > 0 and 'asp' in node.output[0]):
                     nodes_to_exclude.append(node.name)
            print(f"  已将 {len(nodes_to_exclude)} 个 ASP 相关节点加入排除列表")

        # 3. 执行静态量化
        print(f"  正在执行 Static INT8 转换 (QOperator)...")
        
        # 强制检查并降低 ai.onnx.ml opset 版本，避免 onnxruntime 加载失败
        # onnxruntime 1.14+ 可能不支持 ai.onnx.ml v5 (开发中)，官方支持到 v3
        # 我们在量化前加载模型，检查 opset imports
        model_check = onnx.load(model_to_quantize)
        for opset in model_check.opset_import:
            if opset.domain == 'ai.onnx.ml' and opset.version > 3:
                print(f"  [Warning] 发现 ai.onnx.ml opset version {opset.version}，强制降级为 3...")
                opset.version = 3
                onnx.save(model_check, model_to_quantize)
                break
        
        quantize_static(
            model_input=model_to_quantize,
            model_output=temp_output,
            calibration_data_reader=dr,
            quant_format=QuantFormat.QOperator, # 强制使用 QOperator
            weight_type=QuantType.QInt8, # 权重类型 usually QInt8 for static
            activation_type=QuantType.QInt8, # 激活类型 usually QInt8 for static
            nodes_to_exclude=nodes_to_exclude
        )
        
        # 4. 将结果拷回
        print(f"  正在保存量化结果...")
        if os.path.exists(temp_output):
            
            # 4.1 恢复动态形状 (针对 ecapa_voxceleb)
            if 'ecapa_voxceleb' in filename:
                print(f"  [Restore] 恢复 ecapa_voxceleb 的动态时间维度...")
                model_out = onnx.load(temp_output)
                inp = model_out.graph.input[0]
                # 恢复 dim 1 为动态 ("time")
                if len(inp.type.tensor_type.shape.dim) == 3:
                    # 清除 dim_value, 设置 dim_param
                    if inp.type.tensor_type.shape.dim[1].HasField("dim_value"):
                        inp.type.tensor_type.shape.dim[1].ClearField("dim_value")
                    inp.type.tensor_type.shape.dim[1].dim_param = "time"
                
                # 同时也需要修复输出形状 (如果也是静态的)
                for out in model_out.graph.output:
                     # emb output is [Batch, EmbDim], usually batch is dynamic, embdim is fixed.
                     # feats was [Batch, Time, Feats].
                     # ecapa output usually pools over time, so output shape shouldn't depend on time dim value.
                     pass

                onnx.save(model_out, temp_output)

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
        # 清理临时目录
        try:
            for f in glob.glob(os.path.join(TEMP_WORK_DIR, "*")):
                try:
                    if os.path.isfile(f): os.remove(f)
                    elif os.path.isdir(f): shutil.rmtree(f)
                except: pass
        except: pass

# 最终清理
try:
    if os.path.exists(TEMP_WORK_DIR):
        shutil.rmtree(TEMP_WORK_DIR)
except:
    pass
