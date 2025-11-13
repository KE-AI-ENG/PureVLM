import os
import json
from collections import OrderedDict

import torch
from transformers import AutoTokenizer
from safetensors import safe_open

from purevlm.layer.qlinear import QLinear
from purevlm.utils.sample import sample_next_token
from purevlm.model.text_model import KVCache

def find_attr_path(root_obj, target_obj, root_name="model"):
    """
    在 root_obj 中递归查找 target_obj 的路径名
    支持对象、列表、元组、字典
    """
    visited = set()

    def _search(obj, path):
        if id(obj) in visited:
            return None
        visited.add(id(obj))

        if obj is target_obj:
            return path

        # 对象属性
        if hasattr(obj, "__dict__"):
            for attr_name, attr_value in obj.__dict__.items():
                sub_path = f"{path}.{attr_name}"
                result = _search(attr_value, sub_path)
                if result:
                    return result

        # 列表 / 元组
        if isinstance(obj, (list, tuple)):
            for idx, item in enumerate(obj):
                sub_path = f"{path}[{idx}]"
                result = _search(item, sub_path)
                if result:
                    return result

        # 字典
        if isinstance(obj, dict):
            for key, value in obj.items():
                sub_path = f"{path}[{repr(key)}]"
                result = _search(value, sub_path)
                if result:
                    return result

        return None

    return _search(root_obj, root_name)

def get_obj_by_path(root_obj, path_str):
    current_obj = root_obj
    tokens = []
    i = 0
    while i < len(path_str):
        if path_str[i] == '.':
            i += 1
            start = i
            while i < len(path_str) and path_str[i] not in '.[':
                i += 1
            tokens.append(('attr', path_str[start:i]))
        elif path_str[i] == '[':
            i += 1
            start = i
            while i < len(path_str) and path_str[i] != ']':
                i += 1
            key_str = path_str[start:i]
            i += 1
            try:
                key = eval(key_str)
            except Exception:
                key = key_str
            tokens.append(('item', key))
        else:
            start = i
            while i < len(path_str) and path_str[i] not in '.[':
                i += 1
            tokens.append(('attr', path_str[start:i]))

    # 遍历 tokens 获取对象
    for typ, val in tokens:
        if typ == 'attr':
            # 如果是 list/tuple 并且 val 是数字字符串，按索引访问
            if isinstance(current_obj, (list, tuple)) and val.isdigit():
                current_obj = current_obj[int(val)]
            else:
                current_obj = getattr(current_obj, val)
        elif typ == 'item':
            current_obj = current_obj[val]

    return current_obj

def weight_loading(model, checkpoint, device='cuda'):
    loaded_keys = []
    failed_keys = []

    for key, tensor in checkpoint.items():
        try:
            device_tensor = tensor.to(device)

            if key.endswith(".weight") or key.endswith(".weight_packed") or key.endswith(".weight_shape") or key.endswith(".weight_scale"):
                layer_path = key.rsplit(".", 1)[0]
                layer_obj = get_obj_by_path(model, layer_path)

                if isinstance(layer_obj, QLinear):
                    layer_obj.set_weight(key, device_tensor)
                else:
                    layer_obj.weight = device_tensor
            else:
                parent_path, attr_name = key.rsplit(".", 1)
                parent_obj = get_obj_by_path(model, parent_path)
                setattr(parent_obj, attr_name, device_tensor)

            loaded_keys.append(key)

        except Exception as e:
            failed_keys.append((key, str(e)))

    return loaded_keys, failed_keys

class InferEngine:
    def __init__(self, ckpt_path = "", device="cuda", torch_dtype=torch.bfloat16, batch_size=1, max_seq_len=4096, path_online_quant=""):
        self.device = device
        self.torch_dtype = torch_dtype
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.path_online_quant = path_online_quant
        self.model = None
        self.tokenizer = None
        self.config = None

        # create model and build kv_cache
        self.create_model(ckpt_path)
        self.kv_cache = KVCache(self.config.text_config)
        self.kv_cache.allocate(batch_size=batch_size, max_len=max_seq_len, device=device, dtype=torch_dtype)

    def _load_safetensors_(self, safetensors_files, strict=False, chunk_size=64, device="cpu"):
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

        if 'lm_head.weight' not in complete_state_dict:
            complete_state_dict['lm_head.weight'] = complete_state_dict['model.language_model.embed_tokens.weight']

        loaded, failed = weight_loading(self.model, complete_state_dict, device=device)
        if failed:
            raise ValueError(f"加载权重失败: {len(failed)} 键未能加载，具体信息: {failed}")

        print("模型权重load完成...")

    def _create_config_and_model(self, architectures, config_dict):
        """
        根据 architectures 创建对应的 Config 和 Model
        后续支持新模型时，只需在这里加 elif 分支
        """
        if not architectures:
            raise ValueError("architectures 列表为空，无法创建模型")

        arch_str = architectures[0].lower()

        # === Qwen3VL 系列 ===
        if "qwen3vl" in arch_str:
            from purevlm.utils.configs.qwen3_vl_config import Qwen3VLConfig, Qwen3VLTextConfig, Qwen3VLVisionConfig
            from purevlm.layer.quantization.quant_config import QuantizationConfig
            from purevlm.model.qwen3_vl import Qwen3VLForCausalLM
            vision_config = Qwen3VLVisionConfig(**config_dict.get('vision_config', {}))
            text_config = Qwen3VLTextConfig(**config_dict.get('text_config', {}))

            if self.path_online_quant == "":
                quantization_config = QuantizationConfig(**config_dict.get('quantization_config', {}))
            else:
                with open(self.path_online_quant, 'r', encoding='utf-8') as f:
                    onl_q_config_dict = json.load(f)
                quantization_config = QuantizationConfig(**onl_q_config_dict.get('quantization_config', {}))

            config = Qwen3VLConfig(
                vision_config=vision_config,
                text_config=text_config,
                quantization_config=quantization_config,
                **{k: v for k, v in config_dict.items()
                   if k not in ['vision_config', 'text_config', 'quantization_config']}
            )

            model = Qwen3VLForCausalLM(
                config,
                tokenizer=self.tokenizer,
                device=self.device,
            )
            print(f"创建Qwen3VL模型成功")
            return config, model

        # === 未来支持其他模型 ===
        elif "llama" in arch_str:
            # TODO: 这里可以添加 LLaMAConfig + LLaMAForCausalLM 的创建逻辑
            raise NotImplementedError("LLaMA 系列暂未实现")
        else:
            raise ValueError(f"暂不支持的 architectures: {architectures}")

    def create_model(self, model_dir):
        """根据 config.json 中 architectures 创建模型并加载权重"""
        # 加载配置文件
        config_path = os.path.join(model_dir, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"{config_path} 不存在")

        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)

        architectures = config_dict.get("architectures", [])

        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        # 创建 Config 和 Model
        self.config, self.model = self._create_config_and_model(architectures, config_dict)

        # 加载权重
        # 查找 safetensors 文件
        safetensors_files = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith('.safetensors')]
        if not safetensors_files:
            raise ValueError(f"在目录 {model_dir} 中未找到safetensors文件")
        safetensors_files.sort()
        print(f"找到 {len(safetensors_files)} 个safetensors文件")
        self._load_safetensors_(safetensors_files, device=self.device)

        return
    
    def generate(self, prompts=None, images=None, generated_len=128, temperature=1.0, do_sample=True, top_p=1.0, top_k=0, repetition_penalty=1.0, presence_penalty=0.0):
        """text生成函数"""

        input_ids, image_values, image_grid_thw, attention_mask, cache_position = self.model.processor.tokenize_inputs(images, prompts, self.config, self.device)

        output_ids = torch.zeros((1,0), dtype=torch.int64, device=self.device)

        prefill_lengths = input_ids.shape[1]

        for _ in range(generated_len):
            logits = self.model.forward(
                input_ids = input_ids,
                pixel_values = image_values,
                past_key_values = self.kv_cache,
                image_grid_thw = image_grid_thw,
                cache_position = cache_position
            )

            # Sample next token using sample module
            next_token = sample_next_token(
                logits,
                temperature=temperature,
                do_sample=do_sample,
                top_k=top_k,
                top_p=top_p,
                generated_ids=output_ids,
                repetition_penalty=repetition_penalty,
                presence_penalty=presence_penalty
            )

            #make next-step inputs
            input_ids = next_token
            image_values = None
            attention_mask = torch.cat([attention_mask, torch.ones((1,1), dtype=torch.int64, device=self.device)], dim=-1)
            # cache_position = torch.tensor([cache_position[-1]+1], dtype=torch.int64, device=self.device)
            cache_position = torch.tensor([prefill_lengths+output_ids.shape[1]], dtype=torch.int, device=self.device)

            output_ids = torch.cat([output_ids, next_token], dim=-1)

            # 检查是否生成了结束token
            if next_token.item() == self.config.text_config.eos_token_id:
                break

        #model reset
        self.kv_cache.clear()
        self.model.model.rope_deltas = None

        decode_lenths = output_ids.shape[1]

        return output_ids, prefill_lengths, decode_lenths