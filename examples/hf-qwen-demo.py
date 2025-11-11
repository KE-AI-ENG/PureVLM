from transformers import Qwen3VLMoeForConditionalGeneration, Qwen3VLForConditionalGeneration, AutoProcessor

# moe models
# model_path = "/root/hf-models/Qwen3-VL-235B-A22B-Instruct"
model_path = "/root/pretrained-models/Qwen/Qwen3-VL-30B-A3B-Instruct"

# dense models
# model_path = "/root/hf-models/Qwen3-VL/Qwen3-VL-32B-Instruct"
# model_path = "/root/hf-models/Qwen3-VL/Qwen3-VL-8B-Instruct"
# model_path = "/root/hf-models/Qwen3-VL/Qwen3-VL-4B-Instruct"
# model_path = "/root/hf-models/Qwen3-VL/Qwen3-VL-2B-Instruct"

# default: Load the model on the available device(s)
if "2B" in model_path or "4B" in model_path or "8B" in model_path or "32B" in model_path: #dense model
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, dtype="auto", device_map="auto"
    )
else: #moe model
    model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
        model_path, dtype="auto", device_map="auto"
    )

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen3-VL-30B-A3B-Instruct",
#     dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

processor = AutoProcessor.from_pretrained(model_path)

messages = [
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
            {
                "type": "image",
                "image": "./jiaju-demo.png",
            },
            {"type": "text", "text": "请帮我找下音响"},
        ],
        # "content": [
        #     {"type": "text", "text": "Who are you?"},
        # ],
    }
]

# Preparation for inference
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
