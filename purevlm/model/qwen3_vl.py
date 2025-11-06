"""
Qwen3-VL Main Model
Integrates vision and text models for multimodal tasks
"""
from typing import Optional

import torch
from torch import Tensor

import purevlm.layer as L

# Import from separated modules
from purevlm.model.vision_model import VisionModel
from purevlm.model.text_model import TextModel, KVCache
from purevlm.model.processor import Qwen3VLProcessor
from purevlm.model.sample import sample_next_token
from purevlm.model.mm_utils import get_rope_index, get_placeholder_mask


class Qwen3VLModel:
    """Qwen3-VL主模型"""
    def __init__(self, config, device='cuda'):
        super().__init__()
        self.visual = VisionModel(config.vision_config, device=device, quant_config=config.quantization_config)
        self.language_model = TextModel(config.text_config, quant_config=config.quantization_config)
        self.rope_deltas = None  # cache rope_deltas here
        self.config = config

    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None):
        """Encode images into continuous embeddings.

        Args:
            pixel_values: Image pixel values
            image_grid_thw: Image grid dimensions (temporal, height, width)

        Returns:
            Tuple of (image_embeds, deepstack_image_embeds)
        """
        pixel_values = pixel_values.type(self.visual.dtype)
        image_embeds, deepstack_image_embeds = self.visual.forward(pixel_values, grid_thw=image_grid_thw)
        split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        image_embeds = torch.split(image_embeds, split_sizes)
        return image_embeds, deepstack_image_embeds

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        past_key_values: Optional[KVCache] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        """
        inputs_embeds = self.language_model.embed_tokens(input_ids)

        image_mask = None

        if pixel_values is not None:
            image_embeds, deepstack_image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask = get_placeholder_mask(
                self.config, input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        visual_pos_masks = None
        deepstack_visual_embeds = None
        if image_mask is not None:
            image_mask = image_mask[..., 0]
            visual_pos_masks = image_mask
            deepstack_visual_embeds = deepstack_image_embeds

        # Calculate RoPE index once per generation in the pre-fill stage only.
        # When compiling, we can't check tensor values thus we check only input length
        # It is safe to assume that `length!=1` means we're in pre-fill because compiled
        # models currently cannot do asssisted decoding
        if self.rope_deltas is None:
            position_ids, rope_deltas = get_rope_index(
                self.config,
                input_ids,
                image_grid_thw,
                attention_mask=None
            )
            self.rope_deltas = rope_deltas
        # then use the prev pre-calculated rope-deltas to get the correct position ids
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            position_ids = ((cache_position[0] + self.rope_deltas)
                            .repeat_interleave(batch_size // self.rope_deltas.shape[0], dim=0)  # repeat for batch
                            .unsqueeze(0).expand(3, -1, -1)) # expand for 3 dims

        outputs = self.language_model.forward(
            input_ids=None,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )

        return outputs


class Qwen3VLForCausalLM:
    """用于因果语言建模的Qwen3-VL模型"""
    def __init__(self, config, tokenizer=None, batch_size = 1, max_length = 4096, device='cuda', dtype=torch.bfloat16):
        super().__init__()
        self.model = Qwen3VLModel(config, device=device)
        self.lm_head = L.QLinear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False, quant_config=config.quantization_config)
        self.kv_cache = KVCache(config.text_config)

        self.kv_cache.allocate(batch_size=batch_size, max_len=max_length)

        self.config = config

        self.tokenizer = tokenizer

        # Create processor
        self.processor = Qwen3VLProcessor(
            tokenizer=tokenizer,
            patch_size=config.vision_config.patch_size,
            spatial_merge_size=config.vision_config.spatial_merge_size,
            temporal_patch_size=config.vision_config.temporal_patch_size,
        )

        self.device = device
        self.dtype = dtype

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            past_key_values: Optional[KVCache] = None,
            pixel_values: Optional[torch.Tensor] = None,
            image_grid_thw: Optional[torch.LongTensor] = None,
            cache_position: Optional[torch.LongTensor] = None
    ) -> Tensor:

        outputs = self.model.forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            past_key_values=past_key_values,
            cache_position=cache_position,
        )

        last_hidden_states = outputs[:, -1, :] #get last hidden_states
        logits = self.lm_head(last_hidden_states)

        return logits

    def generate(self, prompts=None, images=None, generated_len=128, temperature=1.0, do_sample=True, top_p=1.0, top_k=0, repetition_penalty=1.0, presence_penalty=0.0):
        """text生成函数"""

        input_ids, image_values, image_grid_thw, attention_mask, cache_position = self.processor.tokenize_inputs(images, prompts, self.config, self.device)

        output_ids = torch.zeros((1,0), dtype=torch.int64, device=self.device)

        prefill_lengths = input_ids.shape[1]

        for _ in range(generated_len):
            logits = self.forward(
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
            cache_position = torch.tensor([cache_position[-1]+1], dtype=torch.int64, device=self.device)

            output_ids = torch.cat([output_ids, next_token], dim=-1)

            # 检查是否生成了结束token
            if next_token.item() == 151643:  # 151643是结束token
                break

        #model reset
        self.kv_cache.clear()
        self.model.rope_deltas = None

        decode_lenths = output_ids.shape[1]

        return output_ids, prefill_lengths, decode_lenths
