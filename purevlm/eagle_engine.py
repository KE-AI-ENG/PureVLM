import os
import json
from collections import OrderedDict

import torch
from transformers import AutoTokenizer
from safetensors import safe_open

from purevlm.layer.qlinear import QLinear
from purevlm.utils.sample import sample_next_token
from purevlm.model.text_model import KVCache
from purevlm.engine import InferEngine, weight_loading

class EagleInferEngine:
    def __init__(self, ckpt_path = "", eagle_model_path = "", device="cuda", torch_dtype=torch.bfloat16, batch_size=1, max_seq_len=4096, path_online_quant="", use_cuda_graph=False):
        
        self.target_engine = InferEngine(
            ckpt_path=ckpt_path,
            device=device,
            torch_dtype=torch_dtype,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            path_online_quant=path_online_quant,
            use_cuda_graph=use_cuda_graph
        )
        # set capure layers for aux_hidden_states
        self.target_engine.model.model.language_model.layers_to_capture = [
            2, 
            self.target_engine.config.text_config.num_hidden_layers // 2,
            self.target_engine.config.text_config.num_hidden_layers -3]

        self.device = device
        self.torch_dtype = torch_dtype
        self.draft_model = None
        self.draft_config = None
        self.path_online_quant = path_online_quant
        self.use_cuda_graph = use_cuda_graph

        # create draft model
        self.create_draft_model(eagle_model_path)
        self.draft_kv_cache = KVCache(self.draft_config)
        self.draft_kv_cache.allocate(batch_size=batch_size, max_len=max_seq_len, device=device, dtype=torch_dtype)


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

        loaded, failed = weight_loading(self.draft_model, complete_state_dict, device=device)
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

        elif "eagle3" in arch_str:
            from purevlm.utils.configs.qwen3_vl_config import Qwen3EagleConfig
            from purevlm.model.qwen3_eagle import Qwen3EagleModel
            config = Qwen3EagleConfig(**config_dict)

            model = Qwen3EagleModel(
                config,
                quant_config=None
            )

            print(f"创建Qwen3Eagle模型成功")
            return config, model
        # === 未来支持其他模型 ===
        elif "llama" in arch_str:
            # TODO: 这里可以添加 LLaMAConfig + LLaMAForCausalLM 的创建逻辑
            raise NotImplementedError("LLaMA 系列暂未实现")
        else:
            raise ValueError(f"暂不支持的 architectures: {architectures}")

    def create_draft_model(self, model_dir):
        """根据 config.json 中 architectures 创建模型并加载权重"""
        # 加载配置文件
        config_path = os.path.join(model_dir, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"{config_path} 不存在")

        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)

        architectures = config_dict.get("architectures", [])

        # 创建 Config 和 Model
        self.draft_config, self.draft_model = self._create_config_and_model(architectures, config_dict)

        # 加载权重
        # 查找 safetensors 文件
        safetensors_files = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith('.safetensors')]
        if not safetensors_files:
            raise ValueError(f"在目录 {model_dir} 中未找到safetensors文件")
        safetensors_files.sort()
        print(f"找到 {len(safetensors_files)} 个safetensors文件")
        self._load_safetensors_(safetensors_files, device=self.device)

        self.draft_model.embed_tokens.weight = self.target_engine.model.model.language_model.embed_tokens.weight

        return
    
    def init_cuda_graph_decode(self, batch_size=1):
        """
        初始化并捕获 decode 阶段的 CUDA Graph
        """
        # 1. 创建静态输入缓冲区
        self.decode_static_input_ids = torch.zeros((batch_size, 1), dtype=torch.int64, device=self.device)
        self.decode_static_cache_position = torch.tensor([0], dtype=torch.int, device=self.device)
        self.static_output = torch.zeros((batch_size, self.config.text_config.vocab_size), dtype=self.torch_dtype, device=self.device)

        # 2. Warmup 一次，确保所有 kernel 已经加载
        _ = self.model.forward_decode(
            input_ids=self.decode_static_input_ids,
            past_key_values=self.kv_cache,
            cache_position=self.decode_static_cache_position
        )

        torch.cuda.synchronize()

        # 3. 捕获 CUDA Graph
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self.static_output.copy_(self.model.forward_decode(
            input_ids=self.decode_static_input_ids,
            past_key_values=self.kv_cache,
            cache_position=self.decode_static_cache_position
        ))

        self.decode_graph = g
        print("[CUDA Graph] Decode 阶段捕获完成")

    def generate(self, prompts=None, images=None, generated_len=128, temperature=1.0, do_sample=True, top_p=1.0, top_k=0, repetition_penalty=1.0, presence_penalty=0.0):
        """text生成函数"""

        input_ids, image_values, image_grid_thw = self.target_engine.model.processor.tokenize_inputs(images, prompts, self.target_engine.config, self.device)

        if self.target_engine.config.text_config.use_flash_attn:
            cache_position = torch.tensor([0], dtype=torch.int, device=self.device)
        else:
            cache_position = torch.tensor([i for i in range(input_ids.shape[-1])], dtype=torch.int, device=self.device)

        output_ids = torch.zeros((1,0), dtype=torch.int64, device=self.device)

        # ===== Target model prefill 阶段 =====
        logits, aux_hidden_states,prefill_position_ids = self.target_engine.model.forward_prefill(
            input_ids = input_ids,
            pixel_values = image_values,
            past_key_values = self.target_engine.kv_cache,
            image_grid_thw = image_grid_thw,
            cache_position = cache_position
        )
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
        prefill_lengths = input_ids.shape[1]
        output_ids = torch.cat([output_ids, next_token], dim=-1)

        # ===== Draft model prefill 阶段 =====
        # draft_input_ids = torch.cat([input_ids, next_token], dim=-1)
        hidden_states = self.draft_model.forward(
            input_ids = input_ids,
            position_ids=prefill_position_ids,
            past_key_values = self.draft_kv_cache,
            cache_position = cache_position,
            aux_hidden_states = aux_hidden_states,
            rope_deltas = self.target_engine.model.model.rope_deltas
        )
        last_hidden_states = hidden_states[:, -1, :] #get last hidden_states
        logits = self.draft_model.lm_head(last_hidden_states)
        draft_first_token = sample_next_token(
            logits,
            temperature=temperature,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            generated_ids=output_ids,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty
        )

        # ===== Decode 阶段 =====
        cur_seq_len = prefill_lengths
        eos_token_id = self.target_engine.tokenizer.eos_token_id if hasattr(self.target_engine.tokenizer, 'eos_token_id') else None

        for l in range(generated_len-1):
            # === draft model decode num steps ===
            draft_decode_input_id = draft_first_token
            draft_tokens = draft_first_token
            for draft_step in range(3):
                hidden_states = self.draft_model.forward(
                    input_ids = draft_decode_input_id,
                    past_key_values = self.draft_kv_cache,
                    cache_position = torch.tensor([cur_seq_len + draft_step], dtype=torch.int, device=self.device),
                    aux_hidden_states = last_hidden_states.unsqueeze(1),
                    rope_deltas = self.target_engine.model.model.rope_deltas
                )
                hidden_states = hidden_states

                last_hidden_states = hidden_states[:, -1, :] #get last hidden_states
                logits = self.draft_model.lm_head(last_hidden_states)
                draft_token = sample_next_token(
                    logits,
                    temperature=temperature,
                    do_sample=do_sample,
                    top_k=top_k,
                    top_p=top_p,
                    generated_ids=output_ids,
                    repetition_penalty=repetition_penalty,
                    presence_penalty=presence_penalty
                )
                draft_decode_input_id = draft_token
                draft_tokens = torch.cat([draft_tokens, draft_token], dim=-1)

            # ==== target model verify draft tokens =====
            verify_input_ids = torch.cat([next_token, draft_tokens], dim=-1)
            cache_position = torch.tensor([cur_seq_len + i for i in range(draft_tokens.shape[-1]+1)], dtype=torch.int, device=self.device)
            logits, aux_hidden_states, _ = self.target_engine.model.forward_prefill(
                input_ids = verify_input_ids,
                pixel_values = None,
                past_key_values = self.target_engine.kv_cache,
                image_grid_thw = image_grid_thw,
                cache_position = cache_position,
                verify = True
            )
            verified_tokens = sample_next_token(
                logits,
                temperature=temperature,
                do_sample=do_sample,
                top_k=top_k,
                top_p=top_p,
                generated_ids=output_ids,
                repetition_penalty=repetition_penalty,
                presence_penalty=presence_penalty
            )
            # 验证和接受tokens
            verified_tokens = verified_tokens.squeeze()  # [seq_len]
            draft_tokens_to_verify = verify_input_ids.squeeze()[1:]  # [draft_len]

            # 找到连续正确的tokens (从头开始)
            correct_mask = draft_tokens_to_verify == verified_tokens[:-1]

            # 找到第一个不匹配的位置
            num_accepted = 0
            for i in range(len(correct_mask)):
                if correct_mask[i]:
                    num_accepted += 1
                else:
                    break  # 遇到第一个错误就停止
            
            accepted_tokens = verified_tokens[:num_accepted + 1].unsqueeze(0)
            # # 接受的tokens: 包括连续正确的draft tokens + target生成的下一个token
            # if num_accepted > 0:
            #     # 接受了部分draft tokens
            #     accepted_tokens = verified_tokens[:num_accepted + 1].unsqueeze(0)  # +1 for next token
            # else:
            #     # 一个都没接受,只用target生成的第一个token
            #     accepted_tokens = verified_tokens[:1].unsqueeze(0)

            next_token = accepted_tokens[:, -1:] # 保存最后一个token作为下一轮的输入
            output_ids = torch.cat([output_ids, accepted_tokens], dim=-1)

            # 检查是否生成了EOS token
            if eos_token_id is not None and (accepted_tokens == eos_token_id).any():
                print(f"Early stopping: EOS token generated at step {l}")
                break

            # 计算需要回滚的draft tokens数量，生成最后一个token
            target_num_rejected = len(draft_tokens_to_verify) - accepted_tokens.shape[-1]
            draft_num_rejected = len()

            # 删除target kvcache中被拒绝的部分
            # if num_rejected > 0:
            #     self.target_engine.kv_cache.delete(0, num_rejected)

            # 删除draft kvcache中被拒绝的部分
            # if num_rejected > 0:
            #     self.draft_kv_cache.delete(0, num_rejected)


            # 更新序列长度
            cur_seq_len += num_accepted + 1

            # 使用accepted tokens更新draft model的hidden states
            # 选择对应的aux_hidden_states
            # accepted_aux_hidden_states = torch.cat([aux[:, :num_accepted + 1, :] for aux in aux_hidden_states], dim=-1)
            accepted_aux_hidden_states = [aux[:, :num_accepted + 1, :] for aux in aux_hidden_states]
            # draft model前向传播接受的tokens以更新其状态
            cache_position_accepted = torch.tensor([cur_seq_len +1 + i for i in range(accepted_tokens.shape[-1])], dtype=torch.int, device=self.device)
            hidden_states = self.draft_model.forward(
                input_ids = accepted_tokens,
                past_key_values = self.draft_kv_cache,
                cache_position = cache_position_accepted,
                aux_hidden_states = accepted_aux_hidden_states,
                rope_deltas = self.target_engine.model.model.rope_deltas
            )


            # 更新last_hidden_states为下一轮draft生成准备
            last_hidden_states = hidden_states[:, -1, :]

            # 为下一轮draft生成第一个token
            logits = self.draft_model.lm_head(last_hidden_states)
            draft_first_token = sample_next_token(
                logits,
                temperature=temperature,
                do_sample=do_sample,
                top_k=top_k,
                top_p=top_p,
                generated_ids=output_ids,
                repetition_penalty=repetition_penalty,
                presence_penalty=presence_penalty
            )






        #model reset
        self.target_engine.kv_cache.clear()
        self.target_engine.model.model.rope_deltas = None
        self.draft_kv_cache.clear()

        decode_lenths = output_ids.shape[1]

        return output_ids, prefill_lengths, decode_lenths