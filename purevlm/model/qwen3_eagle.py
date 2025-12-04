from typing import Optional
import torch

import purevlm.layer as L

from purevlm.model.text_model import TextDecoderLayer, TextMLP, TextRotaryEmbedding, KVCache
from purevlm.utils.configs.qwen3_vl_config import Qwen3EagleConfig

class Qwen3EagleDecoderLayer(TextDecoderLayer):
    def __init__(self, config: Qwen3EagleConfig, layer_idx: int, quant_config=None):
        super().__init__(config, layer_idx, quant_config)
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.self_attn.q_proj = L.QLinear(2*config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias, quant_config=quant_config)
        self.self_attn.k_proj = L.QLinear(2*config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias, quant_config=quant_config)
        self.self_attn.v_proj = L.QLinear(2*config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias, quant_config=quant_config)

        self.mlp = TextMLP(config, quant_config=quant_config)
        self.hidden_norm = L.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


    def forward(
        self,
        inputs_embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[KVCache] = None,
        cache_position: Optional[torch.IntTensor] = None,
    ):
        residual = inputs_embeds
        inputs_embeds = self.input_layernorm(inputs_embeds)
        hidden_states = self.hidden_norm(hidden_states)
        hidden_states = torch.cat([inputs_embeds, hidden_states], dim=-1)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states
    
class Qwen3EagleModel:
    def __init__(self, config: Qwen3EagleConfig, quant_config=None):
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.config = config
        self.embed_tokens = L.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        if hasattr(config, "target_hidden_size"):
            self.hidden_size_in = config.target_hidden_size
        else:
            self.hidden_size_in = config.hidden_size

        self.fc = L.QLinear(
            self.hidden_size_in * 3,
            config.hidden_size,
            bias=getattr(config, "bias", False),
            quant_config=quant_config,
        )

        self.midlayer = Qwen3EagleDecoderLayer(config, 0, quant_config=quant_config)
        self.norm = L.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = TextRotaryEmbedding(config=config)
        self.lm_head = L.QLinear(
            config.hidden_size,
            config.draft_vocab_size,
            bias=False,
            quant_config=quant_config,
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[KVCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        # args for deepstack
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[list[torch.Tensor]] = None,
        # args for eagle3
        aux_hidden_states: Optional[torch.FloatTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
    ):
        r"""
        visual_pos_masks (`torch.Tensor` of shape `(batch_size, seqlen)`, *optional*):
            The mask of the visual positions.
        deepstack_visual_embeds (`list[torch.Tensor]`, *optional*):
            The deepstack visual embeddings. The shape is (num_layers, visual_seqlen, embed_dim).
            The feature is extracted from the different visual encoder layers, and fed to the decoder
            hidden states. It's from the paper DeepStack(https://arxiv.org/abs/2406.04334).
        """

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if isinstance(aux_hidden_states, list):
            aux_hidden_states = torch.cat(aux_hidden_states, dim=-1)
        if aux_hidden_states.shape[-1] != inputs_embeds.shape[-1]:
            aux_hidden_states = self.fc(aux_hidden_states)

        if position_ids is None:
            batch_size, seq_length, _ = inputs_embeds.shape
            position_ids = ((cache_position[0] + rope_deltas)
                            .repeat_interleave(batch_size // rope_deltas.shape[0], dim=0)  # repeat for batch
                            .unsqueeze(0).expand(3, -1, -1)) # expand for 3 dims

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        # decoder layers
        layer_outputs = self.midlayer.forward(
            inputs_embeds = inputs_embeds,
            hidden_states = aux_hidden_states,
            position_embeddings = position_embeddings,
            attention_mask = None,
            past_key_values = past_key_values,
            cache_position = cache_position,
        )

        hidden_states = layer_outputs

        hidden_states = self.norm(hidden_states)
        return hidden_states