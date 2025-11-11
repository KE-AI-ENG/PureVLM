"""
Qwen3-VL Text Model Components
"""
from typing import Optional

import torch
import torch.nn as nn

import purevlm.layer as L
from purevlm.utils.configs.qwen3_vl_config import Qwen3VLTextConfig

class KVCache:
    def __init__(self, config):
        self.max_seq_len = config.max_position_embeddings
        self.layer_num = config.num_hidden_layers
        self.head_num = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.cur_seq_len = 0

    def allocate(self, batch_size = 1, max_len=None, dtype=torch.bfloat16, device="cuda"):
        self.max_cache_len = self.max_seq_len if max_len is None else max_len
        self.key_states = [torch.zeros((batch_size, self.head_num, self.max_cache_len, self.head_dim), dtype=dtype, device=device) for i in range(self.layer_num)]
        self.value_states = [torch.zeros((batch_size, self.head_num, self.max_cache_len, self.head_dim), dtype=dtype, device=device) for i in range(self.layer_num)]

    def update(self, key_states, value_states, layer_idx, cache_position):
        self.key_states[layer_idx].index_copy_(2, cache_position, key_states)
        self.value_states[layer_idx].index_copy_(2, cache_position, value_states)

        # Only update cur_seq_len at layer 0
        if layer_idx == 0:
            self.cur_seq_len = cache_position[-1].item() + 1

        return (self.key_states[layer_idx][:, :, :self.cur_seq_len, :],
                self.value_states[layer_idx][:, :, :self.cur_seq_len, :])

    def get_seq_length(self):
        return self.cur_seq_len

    def clear(self):
        self.cur_seq_len = 0


def _compute_default_rope_parameters(
    config: Optional[Qwen3VLTextConfig] = None,
    device: Optional["torch.device"] = None,
    seq_len: Optional[int] = None,
) -> tuple["torch.Tensor", float]:
    """
    Computes the inverse frequencies according to the original RoPE implementation
    Args:
        config ([`~transformers.PreTrainedConfig`]):
            The model configuration. This function assumes that the config will provide at least the following
            properties:

            *   rope_theta (`float`): The base wavelength from which the inverse frequencies will be derived.
            *   hidden_size (`int`): The numerator when deriving a head_dim, if not provided directly.
            *   num_attention_heads (`int`): The denominator when deriving a head_dim, if not provided directly.

            Additionally, this function will make use of the following properties if they are found in the config:

            *   head_dim (`int`, *optional*): The size of the key-value heads in the model. If None, this value will be
                derived as hidden_size // num_attention_heads.
            *   partial_rotary_factor (`float`, *optional*): If less than 1.0, inverse frequencies will be returned for
                the first fraction of the head_dim. Defaults to 1.0.
        device (`torch.device`):
            The device to use for initialization of the inverse frequencies.
        seq_len (`int`, *optional*):
            The current sequence length. Unused for this type of RoPE.

    Returns:
        Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
    """
    base = config.rope_theta
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    dim = int(head_dim * partial_rotary_factor)

    attention_factor = 1.0  # Unused in this type of RoPE

    # Compute the inverse frequencies
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
    return inv_freq, attention_factor


class TextRotaryEmbedding:
    def __init__(self, config: Qwen3VLTextConfig, device=None):
        super().__init__()
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", "default")
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = _compute_default_rope_parameters

        self.inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        # self.register_buffer("inv_freq", inv_freq, persistent=False)
        # self.original_inv_freq = self.inv_freq

        self.mrope_section = config.rope_scaling.get("mrope_section", [24, 20, 20])

    def apply_interleaved_mrope(self, freqs, mrope_section):
        """Apply interleaved MRoPE to 3D rotary embeddings.
        Reorganizes frequency layout from chunked [TTT...HHH...WWW] to
        interleaved [THTHWHTHW...TT], preserving frequency continuity.
        args:
            x: (3, bs, seq_len, head_dim // 2)
            mrope_section: (3,)
        returns:
            x_t: (bs, seq_len, head_dim // 2)
        """
        freqs_t = freqs[0]  # just overwrite the first dimension T
        for dim, offset in enumerate((1, 2), start=1):  # H, W
            length = mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t

    def __call__(self, x, position_ids):
        # In contrast to other models, Qwen3VL has different position ids for the grids
        # So we expand the inv_freq to shape (3, ...)
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, :, None, :].float()  # shape (3, bs, 1, positions)

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        # return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
        rotary_emb_merged = torch.cat((cos.squeeze(0)[:,:(self.config.head_dim//2)], sin.squeeze(0)[:,:(self.config.head_dim//2)]), dim=-1).to(x.dtype)
        return rotary_emb_merged


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class TextAttention:
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen3VLTextConfig, layer_idx: int, quant_config=None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = L.QLinear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias, quant_config=quant_config
        )
        self.k_proj = L.QLinear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias, quant_config=quant_config
        )
        self.v_proj = L.QLinear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias, quant_config=quant_config
        )
        self.o_proj = L.QLinear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias, quant_config=quant_config
        )
        self.q_norm = L.RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # unlike olmo, only on the head dim!
        self.k_norm = L.RMSNorm(
            self.head_dim, eps=config.rms_norm_eps
        )  # thus post q_norm does not need reshape

    def __call__(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[KVCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape))
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape))
        value_states = self.v_proj(hidden_states)

        L.RotEmb(query_states, key_states, self.head_dim, position_embeddings, is_neox=True)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.view(*input_shape, -1, self.head_dim).transpose(1, 2)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            #cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_position)

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attention_mask,
            scale=self.scaling,
            is_causal = query_states.shape[2] > 1 and attention_mask is None,
            enable_gqa = True
        )

        attn_output = attn_output.transpose(1,2).reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output


class TextMLP:
    def __init__(self, config, quant_config=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = L.QLinear(self.hidden_size, self.intermediate_size, bias=False, quant_config=quant_config)
        self.up_proj = L.QLinear(self.hidden_size, self.intermediate_size, bias=False, quant_config=quant_config)
        self.down_proj = L.QLinear(self.intermediate_size, self.hidden_size, bias=False, quant_config=quant_config)

    def __call__(self, x):
        down_proj = self.down_proj(nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

class Qwen3VLMoeTextExperts:
    def __init__(self, config, quant_config=None):
        super().__init__()
        self.num_experts = config.num_experts
        self.intermediate_size = config.moe_intermediate_size
        self.hidden_size = config.hidden_size
        self.expert_dim = self.intermediate_size

        # TODO support moe fused kernel, support quantization
        # self.gate_up_proj = L.QLinear(self.hidden_size, 2 * self.expert_dim * self.num_experts, bias=False, quant_config=quant_config)
        # self.down_proj = L.QLinear(self.expert_dim, self.hidden_size * self.num_experts, bias=False, quant_config=quant_config)

        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_size, 2 * self.expert_dim))
        self.down_proj = nn.Parameter(torch.empty((self.num_experts, self.expert_dim, self.hidden_size)))

    def __call__(
        self, hidden_states: torch.Tensor, routing_weights: torch.Tensor, router_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        When training it is more efficient to just loop over the experts and compute the output for each expert
        as otherwise the memory would explode.

        For inference we can sacrifice some memory and compute the output for all experts at once. By repeating the inputs.

        Args:
            hidden_states (torch.Tensor): (batch_size * token_num, hidden_size)
            routing_weights (torch.Tensor): (batch_size * token_num, num_experts)
            router_indices (torch.Tensor): (batch_size * token_num, top_k)
        Returns:
            torch.Tensor
        """
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.hidden_size)  # (num_tokens, hidden_size)
        hidden_states = hidden_states.repeat(self.num_experts, 1)
        hidden_states = hidden_states.view(self.num_experts, -1, self.hidden_size)
        # gate_up = torch.bmm(hidden_states, self.gate_up_proj.weight.view(self.num_experts, self.hidden_size, 2 * self.expert_dim))
        gate_up = torch.bmm(hidden_states, self.gate_up_proj)
        gate, up = gate_up.chunk(2, dim=-1)  # not supported for DTensors
        # next_states = torch.bmm((up * torch.nn.functional.silu(gate)), self.down_proj.weight.view(self.num_experts, self.expert_dim, self.hidden_size))
        next_states = torch.bmm((up * torch.nn.functional.silu(gate)), self.down_proj)
        next_states = next_states.reshape(self.num_experts, batch_size, -1, self.hidden_size)
        next_states = (
            next_states * routing_weights.transpose(0, 1).view(self.num_experts, batch_size, -1)[..., None]
        )
        next_states = next_states.sum(dim=0)
        return next_states

class Qwen3VLMoeTextSparseMoeBlock:
    def __init__(self, config, quant_config=None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.gate = L.QLinear(config.hidden_size, config.num_experts, bias=False)
        self.experts = Qwen3VLMoeTextExperts(config, quant_config=quant_config)

    def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # TODO support fused moe kernel, the current implementation is not efficient.
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.hidden_size)
        router_logits = self.gate(hidden_states)
        routing_weights = torch.nn.functional.softmax(router_logits, dim=-1, dtype=torch.float)
        routing_weights, router_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(router_logits.dtype)
        router_weights = torch.zeros_like(router_logits).scatter_(1, router_indices, routing_weights)
        hidden_states = hidden_states.reshape(batch_size, -1, self.hidden_size)
        routed_out = self.experts(hidden_states, router_weights, router_indices)
        return routed_out

class TextDecoderLayer:
    '''
    Supported Dense and MoE Decoder Layer of Qwen3-VL.
    '''
    def __init__(self, config: Qwen3VLTextConfig, layer_idx: int, quant_config=None):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = TextAttention(config=config, layer_idx=layer_idx, quant_config=quant_config)

        if config.num_experts is None: #dense model
            self.mlp = TextMLP(config, quant_config=quant_config)
        else: #moe model
            if (layer_idx not in config.mlp_only_layers) and (
                config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
            ):
                self.mlp = Qwen3VLMoeTextSparseMoeBlock(config, quant_config=quant_config)
            else:
                self.mlp = TextMLP(config, quant_config=quant_config)

        self.input_layernorm = L.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = L.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[KVCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
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


class TextModel:

    def __init__(self, config: Qwen3VLTextConfig, quant_config=None):
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = L.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = [TextDecoderLayer(config, layer_idx, quant_config=quant_config) for layer_idx in range(config.num_hidden_layers)]
        self.norm = L.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = TextRotaryEmbedding(config=config)

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

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        for layer_idx, decoder_layer in enumerate(self.layers):
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=None,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_outputs

            # add visual features to the hidden states of first several layers
            if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
                hidden_states = self._deepstack_process(
                    hidden_states,
                    visual_pos_masks,
                    deepstack_visual_embeds[layer_idx],
                )

        hidden_states = self.norm(hidden_states)

        return hidden_states

    def _deepstack_process(
        self, hidden_states: torch.Tensor, visual_pos_masks: torch.Tensor, visual_embeds: torch.Tensor
    ):
        visual_pos_masks = visual_pos_masks.to(hidden_states.device)
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
        hidden_states = hidden_states.clone()
        local_this = hidden_states[visual_pos_masks, :] + visual_embeds
        hidden_states[visual_pos_masks, :] = local_this
        return hidden_states
