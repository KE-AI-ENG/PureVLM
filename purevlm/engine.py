
import os
from collections import OrderedDict

from safetensors import safe_open
from safetensors.torch import load_file

import torch
from transformers import AutoTokenizer

from purevlm.model.config import Qwen3VLConfig, Qwen3VLTextConfig, Qwen3VLVisionConfig
from purevlm.model.qwen3_vl import Qwen3VLForCausalLM
from purevlm.layer.quantization.quant_config import QuantizationConfig

from purevlm.utils import weight_loading

def load_safetensors_fast_(model, safetensors_files, strict=False):
    print("开始批量加载 safetensors 文件...")
    complete_state_dict = {}

    for i, safetensor_file in enumerate(safetensors_files):
        print(f"加载文件 ({i+1}/{len(safetensors_files)}): {os.path.basename(safetensor_file)}")
        try:
            state_dict = load_file(safetensor_file, device="cpu")
            complete_state_dict.update(state_dict)
        except Exception as e:
            print(f"  错误: {e}")
            continue

    if 'lm_head.weight' not in complete_state_dict:
        complete_state_dict['lm_head.weight'] = complete_state_dict['model.language_model.embed_tokens.weight']

    missing_keys, unexpected_keys = model.load_state_dict(complete_state_dict, strict=strict)

    if missing_keys:
        print(f"  缺失键数量: {len(missing_keys)}")
    if unexpected_keys:
        print(f"  意外键数量: {len(unexpected_keys)}")

    print("模型权重加载完成...")
    return model

def load_safetensors_(model, safetensors_files, strict=False, chunk_size=64, device="cpu"):
    """
    分块读取keys，避免一次性加载过多
    """
    print("开始分块加载safetensors文件...")
    
    complete_state_dict = OrderedDict()
    
    for i, safetensor_file in enumerate(safetensors_files):
        print(f"\n处理文件 ({i+1}/{len(safetensors_files)}): {os.path.basename(safetensor_file)}")
        
        try:
            with safe_open(safetensor_file, framework="pt", device="cpu") as f:
                print(f"  开始分块读取参数...")
                
                # 分块处理keys
                keys_buffer = []
                total_processed = 0
                
                for key in f.keys():  # 直接遍历迭代器
                    keys_buffer.append(key)
                    
                    # 当缓冲区达到chunk_size时处理一批
                    if len(keys_buffer) >= chunk_size:
                        print(f"    处理参数块: {total_processed} - {total_processed + len(keys_buffer)}")
                        
                        for k in keys_buffer:
                            complete_state_dict[k] = f.get_tensor(k)
                        
                        total_processed += len(keys_buffer)
                        keys_buffer.clear()
                
                # 处理剩余的keys
                if keys_buffer:
                    print(f"    处理最后一块: {total_processed} - {total_processed + len(keys_buffer)}")
                    for k in keys_buffer:
                        complete_state_dict[k] = f.get_tensor(k)
                    total_processed += len(keys_buffer)
                
                print(f"  文件处理完成，共 {total_processed} 个参数")
                
        except Exception as e:
            print(f"  错误: {e}")
            continue
    
    print(f"\n总共加载 {len(complete_state_dict)} 个参数")

    #lm_head.weight not in complete_state_dict
    if 'lm_head.weight' not in complete_state_dict:
        complete_state_dict['lm_head.weight'] = complete_state_dict['model.language_model.embed_tokens.weight']
    
    # 加载到模型
    loaded, failed = weight_loading(model, complete_state_dict, device=device)
    if failed:
        raise ValueError(f"加载权重失败: {len(failed)} 键未能加载，具体信息: {failed}")

    print("模型权重load完成...")
    
    return

def create_model(
    model_dir, 
    device="cuda",
    torch_dtype=torch.bfloat16,
    batch_size = 1,
    max_seq_len = 4096,
    using_online_quant = False
):
    """
    创建模型, 并从包含safetensors文件的checkpoints目录加载权重
    
    Args:
        model_dir (str): 模型目录路径
        device: 目标设备
        torch_dtype: 数据类型
        batch_size: 批处理大小
        max_seq_len: 输入+输出最大序列长度, 用于初始化kvcache
        using_online_quant: 在线量化原始模型, 默认为False
    
    Returns:
        model: 加载完成的模型
    """
    # 查找所有safetensors文件
    safetensors_files = []
    for file in os.listdir(model_dir):
        if file.endswith('.safetensors'):
            safetensors_files.append(os.path.join(model_dir, file))
    
    if not safetensors_files:
        raise ValueError(f"在目录 {model_dir} 中未找到safetensors文件")
    
    # 按文件名排序确保正确的加载顺序
    safetensors_files.sort()
    print(f"找到 {len(safetensors_files)} 个safetensors文件")
    
    # 尝试加载配置文件
    config = None
    config_path = os.path.join(model_dir, "config.json")
    if os.path.exists(config_path):
        try:
            import json
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            
            # 根据配置字典创建配置对象
            vision_config = Qwen3VLVisionConfig(**config_dict.get('vision_config', {}))
            text_config = Qwen3VLTextConfig(**config_dict.get('text_config', {}))
            if not using_online_quant:
                quantization_config = QuantizationConfig(**config_dict.get('quantization_config', {}))
            else:
                project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                onl_q_config_path = os.path.join(project_path, "examples/online_quantization_marlin.json")
                with open(onl_q_config_path, 'r', encoding='utf-8') as f:
                    onl_q_config_dict = json.load(f)
                quantization_config = QuantizationConfig(**onl_q_config_dict.get('quantization_config', {}))
            config = Qwen3VLConfig(
                vision_config=vision_config,
                text_config=text_config,
                quantization_config=quantization_config,
                **{k: v for k, v in config_dict.items() 
                   if k not in ['vision_config', 'text_config', 'quantization_config']}
            )
            print(f"从 {config_path} 加载配置成功")
        except Exception as e:
            print(f"加载配置文件失败: {e}，使用默认配置")

    # 创建默认配置（如果没有提供）
    if config is None:
        vision_config = Qwen3VLVisionConfig()
        text_config = Qwen3VLTextConfig()
        config = Qwen3VLConfig(
            vision_config=vision_config,
            text_config=text_config
        )
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # 创建模型
    print("创建Qwen3VL模型...")
    model = Qwen3VLForCausalLM(config, tokenizer=tokenizer, batch_size=batch_size, max_length=max_seq_len, device=device, dtype=torch_dtype)
    
    # 加载权重
    load_safetensors_(model, safetensors_files, device=device)

    return model