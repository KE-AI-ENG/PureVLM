import argparse
import time
from PIL import Image
import torch

from purevlm.eagle_engine import EagleInferEngine

def inference_example(model_path=None, eagle_model_path=None, prompts=None, image_path=None, temp=0.7, max_generated_len=128, sys_prompts=None, topk=0, topp=1.0, repetition_penalty=1.0, presence_penalty=0.0, path_online_quant="", use_cuda_graph=False):
    """推理示例，带 warmup 和 token 数统计"""

    # ===== 模型创建 =====
    model_start_time = time.time()
    inf_engine_ = EagleInferEngine(ckpt_path=model_path,eagle_model_path=eagle_model_path, path_online_quant=path_online_quant, use_cuda_graph=use_cuda_graph)
    torch.cuda.synchronize()
    model_end_time = time.time()
    model_load_time = model_end_time - model_start_time
    print(f"模型初始化耗时: {model_load_time:.4f} 秒")

    # ===== 图片加载 =====
    if image_path is None:
        images = None
    else:
        images = Image.open(image_path).convert('RGB')

    # ===== 拼接 Prompt =====
    if images is not None:
        input_prompts = (
            f'<|im_start|>system\n{sys_prompts}<|im_end|>\n'
            f'<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{prompts}<|im_end|>\n'
            f'<|im_start|>assistant\n'
        )
    else:
        input_prompts = (
            f'<|im_start|>system\n{sys_prompts}<|im_end|>\n'
            f'<|im_start|>user\n{prompts}<|im_end|>\n'
            f'<|im_start|>assistant\n'
        )

    # ===== Warmup =====
    with torch.no_grad():
        _,_,_ = inf_engine_.generate(
            prompts=input_prompts,
            images=images,
            generated_len=8,
            temperature=temp,
            do_sample=temp > 0
        )

    torch.cuda.synchronize()
    print("Warmup completed.")

    # ===== prefill time =====
    prefill_start_time = time.time()
    with torch.no_grad():
        _,_,_ = inf_engine_.generate(
            prompts=input_prompts,
            images=images,
            generated_len=1,
            temperature=temp,
            do_sample=temp > 0
        )
    torch.cuda.synchronize()
    prefill_time = time.time()-prefill_start_time

    # ===== inference =====
    start_time = time.time()

    with torch.no_grad():
        generated_ids, prefill_token_len, generated_token_len = inf_engine_.generate(
            prompts = input_prompts,
            images = images,
            generated_len = max_generated_len,
            temperature = temp,
            do_sample = temp > 0,
            top_p = topp,
            top_k = topk,
            repetition_penalty = repetition_penalty,
            presence_penalty = presence_penalty
        )

    torch.cuda.synchronize()  # 确保 GPU 完成推理
    end_time = time.time()

    # ===== Token 数统计 =====
    # total_tokens = sum(len(ids) for ids in generated_ids)

    elapsed_time = end_time - start_time
    throughput = (generated_token_len-1) / (elapsed_time-prefill_time)

    output_text = inf_engine_.tokenizer.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    print(f"推理耗时: {elapsed_time:.4f} 秒")
    print(f"生成 Token 数: {generated_token_len}")
    print(f"Prefill latency: {prefill_time:.4f} 秒")
    print(f"Decode Throughput: {throughput:.2f} tokens/sec")

    return output_text

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog='vlm-inf',
        description='VLM inference config.',
        epilog='Copyright(r), 2025'
    )
    parser.add_argument('-m', '--model-path', type=str, default="./", required=True, help='Model Path.')
    parser.add_argument('-em', '--eagle-model-path', type=str, default="./", help='Eagle Model Path.')
    parser.add_argument('-im', '--image-path', type=str, default=None, help="Input image path")
    parser.add_argument('-p', '--prompts', required=True, type=str, default="Describe this image.", help="Input Prompt")
    parser.add_argument("--sys-prompt", type=str, default='你是一个智能家居助手，根据用户提供的图片以及用户正在寻找物品,请仔细分析图片中的所有物品和位置，如果发现用户要找的物品，请详细描述其位置和周围环境。如果没有发现，回答没找到，要回答没周到原因。回答格式：是否找到，以及周围描述，原因。', help='system prompt')
    parser.add_argument('-t', '--temperature', type=float, default=0.7, help="Sample Temperature")
    parser.add_argument('--max-gen-len', type=int, default=128, help="Max generated token length")
    parser.add_argument('--topk', type=int, default = 0, help="topk sampling")
    parser.add_argument('--topp', type=float, default = 1.0, help="topp sampling")
    parser.add_argument('--repetition-penalty', type=float, default=1.0, help="Repetition penalty")
    parser.add_argument('--presence-penalty', type=float, default=0.0, help="Presence penalty")
    parser.add_argument('--use-cuda-graph', action='store_true', help="Use cuda graph for inference, default False")
    
    parser.add_argument("-q", '--using-online-quant', type=str, default="", help="Path for online quantization json. Not '' indicates using online quantization for base model, eg. Qwen3-VL-8B-Instruct")
    
    args = parser.parse_args()

    generated_text = inference_example(
        model_path=args.model_path,
        eagle_model_path=args.eagle_model_path,
        prompts=args.prompts,
        image_path=args.image_path,
        temp=args.temperature,
        max_generated_len=args.max_gen_len,
        sys_prompts=args.sys_prompt,
        path_online_quant=args.using_online_quant,
        use_cuda_graph=args.use_cuda_graph,
    )

    print(generated_text)