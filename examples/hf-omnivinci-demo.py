
"""
Example script for image understanding using the model.
This script demonstrates how to:
1. Load the model and processor
2. Process image input
3. Generate description output
Usage:
    python example_mini_image.py --model_path <path_to_model> --image_path <path_to_image>
"""

from transformers import AutoProcessor, AutoModel, AutoConfig, AutoModelForCausalLM
import torch
import os
import argparse

# Configuration
parser = argparse.ArgumentParser(description="Image understanding example")
parser.add_argument("--model_path", type=str, default="./", help="Path to the model")
parser.add_argument("--image_path", type=str, required=True, help="Path to the image file")
parser.add_argument("--max_new_tokens", type=int, default=1024, help="Maximum number of tokens to generate")
parser.add_argument("--prompt", type=str, default="Describe the image in detail.", help="Text prompt for the model")

args = parser.parse_args()

model_path = args.model_path
image_path = args.image_path
generation_kwargs = {"max_new_tokens": args.max_new_tokens, "max_length": 99999999}

config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

model = AutoModel.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto"
)

processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
generation_config = model.default_generation_config
generation_config.update(**generation_kwargs)

conversation = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "你是一个智能家居助手，根据用户提供的图片以及用户正在寻找物品,请仔细分析图片中的所有物品和位置，如果发现用户要找的物品，请详细描述其位置和周围环境。如果没有发现，回答没找到，要回答没周到原因。回答格式：是否找到，以及周围描述，原因。"
            }
        ]
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": "请帮我找下音响"}
        ]
    }
]
text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)

inputs = processor([text])

output_ids = model.generate(
    input_ids=inputs.input_ids,
    media=getattr(inputs, 'media', None),
    media_config=getattr(inputs, 'media_config', None),
    generation_config=generation_config,
)
print(processor.tokenizer.batch_decode(output_ids, skip_special_tokens=True))