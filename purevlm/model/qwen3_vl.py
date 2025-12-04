"""
Qwen3-VL Main Model
Integrates vision and text models for multimodal tasks
"""
from typing import Optional

import torch
from torch import Tensor

from purevlm.utils.preprocess.qwen3vl import Qwen3VLProcessor
import purevlm.layer as L
from purevlm.model.vision_model import VisionModel
from purevlm.model.text_model import TextModel, KVCache

def get_rope_index(
    config,
    input_ids: Optional[torch.LongTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculate RoPE position indices for multimodal inputs.

    Different from the original implementation, Qwen3VL use timestamps rather than absolute time position ids.

    Args:
        config: Model configuration
        input_ids: Input token IDs
        image_grid_thw: Image grid dimensions (temporal, height, width)
        attention_mask: Attention mask

    Returns:
        Tuple of (position_ids, mrope_position_deltas)
    """
    spatial_merge_size = config.vision_config.spatial_merge_size
    image_token_id = config.image_token_id
    vision_start_token_id = config.vision_start_token_id
    mrope_position_deltas = []

    if input_ids is not None and image_grid_thw is not None:
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        image_index = 0
        attention_mask = attention_mask.to(total_input_ids.device)

        for i, input_ids in enumerate(total_input_ids):
            input_ids = input_ids[attention_mask[i] == 1]
            image_nums = 0
            vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
            vision_tokens = input_ids[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()
            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images = image_nums

            for _ in range(image_nums):
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1

                t, h, w = (
                    image_grid_thw[image_index][0],
                    image_grid_thw[image_index][1],
                    image_grid_thw[image_index][2],
                )
                image_index += 1
                remain_images -= 1
                ed = ed_image
                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(),
                    h.item() // spatial_merge_size,
                    w.item() // spatial_merge_size,
                )
                text_len = ed - st

                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                # llm_grid_t is always 1 for images
                t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))

        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, mrope_position_deltas
    else:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, 1, -1)
                .expand(3, input_ids.shape[0], -1)
            )
            mrope_position_deltas = torch.zeros(
                [input_ids.shape[0], 1],
                device=input_ids.device,
                dtype=input_ids.dtype,
            )

        return position_ids, mrope_position_deltas


def get_placeholder_mask(
    config,
    input_ids: torch.LongTensor,
    inputs_embeds: torch.FloatTensor,
    image_features: Optional[torch.FloatTensor] = None,
) -> torch.Tensor:
    """Get mask for image placeholder positions.

    Obtains image placeholder mask from input_ids or inputs_embeds, and checks that
    the placeholder token count is equal to the length of image features.

    Args:
        config: Model configuration
        input_ids: Input token IDs
        inputs_embeds: Input embeddings
        image_features: Image features to be inserted

    Returns:
        Boolean mask indicating image placeholder positions

    Raises:
        ValueError: If image features and placeholder tokens don't match
    """
    if input_ids is None:
        # This case is unusual, typically we have input_ids
        special_image_mask = inputs_embeds == inputs_embeds  # Placeholder logic
        special_image_mask = special_image_mask.all(-1)
    else:
        special_image_mask = input_ids == config.image_token_id

    n_image_tokens = special_image_mask.sum()
    special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)

    if image_features is not None and inputs_embeds[special_image_mask].numel() != image_features.numel():
        raise ValueError(
            f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {image_features.shape[0]}"
        )

    return special_image_mask


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

    def forward_prefill(
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
            position_ids = ((cache_position + self.rope_deltas)
                            .repeat_interleave(batch_size // self.rope_deltas.shape[0], dim=0)  # repeat for batch
                            .unsqueeze(0).expand(3, -1, -1)) # expand for 3 dims

        outputs = self.language_model.forward_prefill(
            input_ids=None,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )

        return outputs, position_ids
    
    def forward_decode(
        self,
        input_ids: torch.LongTensor = None,
        past_key_values: Optional[KVCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        inputs_embeds = self.language_model.embed_tokens(input_ids)

        if self.config.text_config.use_flash_attn:
            position_ids = None
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            position_ids = ((cache_position[0] + self.rope_deltas)
                            .repeat_interleave(batch_size // self.rope_deltas.shape[0], dim=0)  # repeat for batch
                            .unsqueeze(0).expand(3, -1, -1)) # expand for 3 dims

        outputs = self.language_model.forward_decode(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
        )

        return outputs

class Qwen3VLForCausalLM:
    """用于因果语言建模的Qwen3-VL模型"""
    def __init__(self, config, tokenizer=None, device='cuda'):
        super().__init__()
        self.model = Qwen3VLModel(config, device=device)
        self.lm_head = L.QLinear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False, quant_config=config.quantization_config)

        # Create processor
        self.processor = Qwen3VLProcessor(
            tokenizer=tokenizer,
            patch_size=config.vision_config.patch_size,
            spatial_merge_size=config.vision_config.spatial_merge_size,
            temporal_patch_size=config.vision_config.temporal_patch_size,
        )

    def forward_prefill(
            self,
            input_ids: torch.LongTensor = None,
            past_key_values: Optional[KVCache] = None,
            pixel_values: Optional[torch.Tensor] = None,
            image_grid_thw: Optional[torch.LongTensor] = None,
            cache_position: Optional[torch.LongTensor] = None,  
            verify: bool = False,
    ) -> Tensor:

        outputs, prefill_position_ids = self.model.forward_prefill(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            past_key_values=past_key_values,
            cache_position=cache_position,
        )
        hidden_states, aux_hidden_states = outputs

        if not verify:
            last_hidden_states = hidden_states[:, -1, :] #get last hidden_states
        else:
            last_hidden_states = hidden_states
        logits = self.lm_head(last_hidden_states)

        return logits, aux_hidden_states, prefill_position_ids
    
    def forward_decode(
            self,
            input_ids: torch.LongTensor = None,
            past_key_values: Optional[KVCache] = None,
            cache_position: Optional[torch.LongTensor] = None
    ) -> Tensor:

        outputs = self.model.forward_decode(
            input_ids=input_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
        )

        logits = self.lm_head(outputs.squeeze(1))  # outputs is of shape (batch_size, 1, hidden_size)

        return logits
