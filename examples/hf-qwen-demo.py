from transformers import Qwen3VLMoeForConditionalGeneration, Qwen3VLForConditionalGeneration, AutoProcessor

model_path = "/root/pretrained-models/Qwen/Qwen3-VL-30B-A3B-Instruct"
model_path = "/root/hf-models/Qwen3-VL/Qwen3-VL-8B-Instruct"

# default: Load the model on the available device(s)
if "4B" or "8B" in model_path: #dense model
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
        "role": "user",
        # "content": [
        #     {
        #         "type": "image",
        #         "image": "./demo-qwen.jpeg",
        #     },
        #     {"type": "text", "text": "Describe this image."},
        # ],
        "content": [
            {"type": "text", "text": "Who are you?"},
        ],
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
