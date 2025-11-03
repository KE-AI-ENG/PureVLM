import math
import functools
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import purevlm.layer as L

# from torch.nn.attention import sdpa_kernel, SDPBackend
# sdpa_backend = SDPBackend.CUDNN_ATTENTION

class Qwen3VLVisionConfig:
    def __init__(
        self,
        deepstack_visual_indexes = [5, 11, 17],
        depth = 24,
        hidden_act = "gelu_pytorch_tanh",
        hidden_size = 1024,
        in_channels = 3,
        initializer_range = 0.02,
        intermediate_size = 4096,
        model_type = "qwen3_vl",
        num_heads = 16,
        num_position_embeddings = 2304,
        out_hidden_size = 2560,
        patch_size = 16,
        spatial_merge_size = 2,
        temporal_patch_size = 2
    ):
        self.depth = depth
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.out_hidden_size = out_hidden_size
        self.num_position_embeddings = num_position_embeddings
        self.initializer_range = initializer_range
        self.deepstack_visual_indexes = deepstack_visual_indexes

class Qwen3VLTextConfig:
    def __init__(
        self,
        attention_bias = False,
        attention_dropout = 0.0,
        bos_token_id = 151643,
        dtype = "bfloat16",
        eos_token_id = 151645,
        head_dim = 128,
        hidden_act = "silu",
        hidden_size = 2560,
        initializer_range = 0.02,
        intermediate_size = 9728,
        max_position_embeddings = 262144,
        model_type = "qwen3_vl_text",
        num_attention_heads = 32,
        num_hidden_layers = 36,
        num_key_value_heads = 8,
        rms_norm_eps = 1e-06,
        rope_scaling = None,
        rope_theta = 5000000,
        tie_word_embeddings = True,
        use_cache = True,
        vocab_size = 151936
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.pad_token_id = None

class Qwen3VLConfig:
    """Qwen3-VL模型配置类"""
    def __init__(
        self,
        text_config=None,
        vision_config=None,
        architectures = ["Qwen3VLForConditionalGeneration"],
        model_type = "qwen3_vl",
        image_token_id=151655,
        tie_word_embeddings = True,
        transformers_version = "4.57.0.dev0",
        video_token_id=151656,
        vision_start_token_id=151652,
        vision_end_token_id=151653,
    ):
        self.vision_config = vision_config

        self.text_config = text_config

        self.image_token_id = image_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id

class KVCache:
    def __init__(self, config:Qwen3VLTextConfig):
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

class Qwen3VLVisionMLP:
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.linear_fc1 = L.QLinear(self.hidden_size, self.intermediate_size, bias=True)
        self.linear_fc2 = L.QLinear(self.intermediate_size, self.hidden_size, bias=True)

    def __call__(self, hidden_state):
        return self.linear_fc2(nn.functional.gelu(self.linear_fc1(hidden_state), approximate="tanh"))


class Qwen3VLVisionPatchEmbed:
    def __init__(self, config) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size

        kernel_size = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.proj = L.Conv3d(self.in_channels, self.embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=True)

    def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states


class Qwen3VLVisionRotaryEmbedding:

    def __init__(self, dim: int, theta: float = 10000.0, device='cuda', dtype=torch.bfloat16) -> None:
        super().__init__()
        self.inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.inv_freq = self.inv_freq.to(device).to(dtype)

    def __call__(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


class Qwen3VLVisionPatchMerger:
    def __init__(self, config: Qwen3VLVisionConfig, use_postshuffle_norm=False) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size * (config.spatial_merge_size**2)
        self.use_postshuffle_norm = use_postshuffle_norm
        self.norm = L.LayerNorm(self.hidden_size if use_postshuffle_norm else config.hidden_size, eps=1e-6)
        self.linear_fc1 = L.QLinear(self.hidden_size, self.hidden_size)
        self.linear_fc2 = L.QLinear(self.hidden_size, config.out_hidden_size)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x.view(-1, self.hidden_size) if self.use_postshuffle_norm else x).view(-1, self.hidden_size)
        x = self.linear_fc2(torch.nn.functional.gelu(self.linear_fc1(x)))
        return x


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    q_embed = q_embed.to(orig_q_dtype)
    k_embed = k_embed.to(orig_k_dtype)
    return q_embed, k_embed


class Qwen3VLVisionAttention:
    def __init__(self, config: Qwen3VLVisionConfig) -> None:
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.num_key_value_groups = 1  # needed for eager attention
        self.qkv = L.QLinear(self.dim, self.dim * 3, bias=True)
        self.proj = L.QLinear(self.dim, self.dim)
        self.scaling = self.head_dim**-0.5
        self.config = config
        self.attention_dropout = 0.0
        self.is_causal = False

    def __call__(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        # with sdpa_kernel(sdpa_backend):
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            scale=self.scaling,
            is_causal=False,
        )

        attn_output = attn_output.transpose(1,2).reshape(seq_length, -1).contiguous()
        attn_output = self.proj(attn_output)
        return attn_output


class Qwen3VLVisionBlock:
    def __init__(self, config) -> None:
        super().__init__()
        self.norm1 = L.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = L.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = Qwen3VLVisionAttention(config=config)
        self.mlp = Qwen3VLVisionMLP(config=config)

    def __call__(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            position_embeddings=position_embeddings,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class Qwen3VLTextRotaryEmbedding:
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


class Qwen3VLTextAttention:
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen3VLTextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = L.QLinear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = L.QLinear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = L.QLinear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = L.QLinear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
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
            #cache_kwargs = {"sin = sin, "cos = cos, "cache_position = cache_position}
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


class Qwen3VLTextMLP:
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = L.QLinear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = L.QLinear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = L.QLinear(self.intermediate_size, self.hidden_size, bias=False)

    def __call__(self, x):
        down_proj = self.down_proj(nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class Qwen3VLTextDecoderLayer:
    def __init__(self, config: Qwen3VLTextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Qwen3VLTextAttention(config=config, layer_idx=layer_idx)

        self.mlp = Qwen3VLTextMLP(config)
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


class Qwen3VLVisionModel:

    def __init__(self, config, device='cuda') -> None:
        super().__init__()
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size
        self.dtype = torch.bfloat16
        self.config = config

        self.patch_embed = Qwen3VLVisionPatchEmbed(
            config=config,
        )

        self.pos_embed = L.Embedding(config.num_position_embeddings, config.hidden_size)
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(head_dim // 2, device=device)

        self.blocks = [Qwen3VLVisionBlock(config) for _ in range(config.depth)]
        self.merger = Qwen3VLVisionPatchMerger(
            config=config,
            use_postshuffle_norm=False,
        )

        self.deepstack_visual_indexes = config.deepstack_visual_indexes
        self.deepstack_merger_list = [
                Qwen3VLVisionPatchMerger(
                    config=config,
                    use_postshuffle_norm=True,
                )
                for _ in range(len(config.deepstack_visual_indexes))
            ]

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        merge_size = self.spatial_merge_size

        max_hw = int(grid_thw[:, 1:].max().item())
        freq_table = self.rotary_pos_emb(max_hw)  # (max_hw, dim // 2)
        device = freq_table.device

        total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

        offset = 0
        for num_frames, height, width in grid_thw:
            merged_h, merged_w = height // merge_size, width // merge_size

            block_rows = torch.arange(merged_h, device=device)  # block row indices
            block_cols = torch.arange(merged_w, device=device)  # block col indices
            intra_row = torch.arange(merge_size, device=device)  # intra-block row offsets
            intra_col = torch.arange(merge_size, device=device)  # intra-block col offsets

            # Compute full-resolution positions
            row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
            col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]

            row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)

            coords = torch.stack((row_idx, col_idx), dim=-1)

            if num_frames > 1:
                coords = coords.repeat(num_frames, 1)

            num_tokens = coords.shape[0]
            pos_ids[offset : offset + num_tokens] = coords
            offset += num_tokens

        embeddings = freq_table[pos_ids]  # lookup rotary embeddings
        embeddings = embeddings.flatten(1)
        return embeddings

    def fast_pos_embed_interpolate(self, grid_thw):
        grid_ts, grid_hs, grid_ws = grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]
        device = grid_thw.device

        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]

        for t, h, w in zip(grid_ts, grid_hs, grid_ws):
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)

            h_idxs_floor = h_idxs.int()
            w_idxs_floor = w_idxs.int()
            h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)

            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor

            base_h = h_idxs_floor * self.num_grid_per_side
            base_h_ceil = h_idxs_ceil * self.num_grid_per_side

            indices = [
                (base_h[None].T + w_idxs_floor[None]).flatten(),
                (base_h[None].T + w_idxs_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
            ]

            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]

            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
        weight_tensor = torch.tensor(weight_list, dtype=self.pos_embed.weight.dtype, device=device)
        pos_embeds = self.pos_embed(idx_tensor).to(device) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

        patch_pos_embeds = patch_pos_embeds.split([h * w for h, w in zip(grid_hs, grid_ws)])

        patch_pos_embeds_permute = []
        merge_size = self.config.spatial_merge_size
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            pos_embed = pos_embed.repeat(t, 1)
            pos_embed = (
                pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            patch_pos_embeds_permute.append(pos_embed)
        patch_pos_embeds = torch.cat(patch_pos_embeds_permute)
        return patch_pos_embeds

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
                The final hidden states of the model.
            grid_thw (`torch.Tensor` of shape `(num_images, 3)`):
                The temporal, height and width of feature shape of each image in LLM.

        Returns:
            `torch.Tensor`: hidden_states.
        """
        hidden_states = self.patch_embed(hidden_states)

        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds

        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        # cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
        #     dim=0,
        #     # Select dtype based on the following factors:
        #     #  - FA2 requires that cu_seqlens_q must have dtype int32
        #     #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
        #     # See https://github.com/huggingface/transformers/pull/34852 for more information
        #     dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        # )
        # cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        deepstack_feature_lists = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=None,
                position_embeddings=position_embeddings,
            )
            if layer_num in self.deepstack_visual_indexes:
                deepstack_feature = self.deepstack_merger_list[self.deepstack_visual_indexes.index(layer_num)](
                    hidden_states
                )
                deepstack_feature_lists.append(deepstack_feature)

        hidden_states = self.merger(hidden_states)

        return hidden_states, deepstack_feature_lists


class Qwen3VLTextModel:

    def __init__(self, config: Qwen3VLTextConfig):
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = L.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = [Qwen3VLTextDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        self.norm = L.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3VLTextRotaryEmbedding(config=config)

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

class Qwen3VLModel:
    """Qwen3-VL主模型"""
    def __init__(self, config, device='cuda'):
        super().__init__()
        self.visual = Qwen3VLVisionModel(config.vision_config, device=device)
        self.language_model = Qwen3VLTextModel(config.text_config)
        self.rope_deltas = None  # cache rope_deltas here
        self.config = config

    def get_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Different from the original implementation, Qwen3VL use timestamps rather than absolute time position ids."""

        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        vision_start_token_id = self.config.vision_start_token_id
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

    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None):
        """
        Encodes images into continuous embeddings that can be forwarded to the language model. The deepstack visual features are also returned.

        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input images.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
        """
        pixel_values = pixel_values.type(self.visual.dtype)
        image_embeds, deepstack_image_embeds = self.visual.forward(pixel_values, grid_thw=image_grid_thw)
        split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        image_embeds = torch.split(image_embeds, split_sizes)
        return image_embeds, deepstack_image_embeds

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: Optional[torch.FloatTensor] = None,
    ):
        """
        Obtains image placeholder mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token count is
        equal to the length of image features. If the lengths are different, an error is raised.
        """
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id

        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if image_features is not None and inputs_embeds[special_image_mask].numel() != image_features.numel():
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {image_features.shape[0]}"
            )

        return special_image_mask

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
            image_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
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
            position_ids, rope_deltas = self.get_rope_index(
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
        self.lm_head = L.QLinear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.kv_cache = KVCache(config.text_config)

        self.kv_cache.allocate(batch_size=batch_size, max_len=max_length)

        self.config = config

        self.image_token = '<|image_pad|>'
        self.tokenizer = tokenizer

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

    def smart_resize(
        self, height: int, width: int, factor: int = 28, min_pixels: int = 56 * 56, max_pixels: int = 14 * 14 * 4 * 1280
    ):
        """Rescales the image so that the following conditions are met:

        1. Both dimensions (height and width) are divisible by 'factor'.

        2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

        3. The aspect ratio of the image is maintained as closely as possible.

        """
        if max(height, width) / min(height, width) > 200:
            raise ValueError(
                f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
            )
        h_bar = round(height / factor) * factor
        w_bar = round(width / factor) * factor
        if h_bar * w_bar > max_pixels:
            beta = math.sqrt((height * width) / max_pixels)
            h_bar = max(factor, math.floor(height / beta / factor) * factor)
            w_bar = max(factor, math.floor(width / beta / factor) * factor)
        elif h_bar * w_bar < min_pixels:
            beta = math.sqrt(min_pixels / (height * width))
            h_bar = math.ceil(height * beta / factor) * factor
            w_bar = math.ceil(width * beta / factor) * factor
        return h_bar, w_bar

    def prepare_inputs(self, images, text):

        if images is not None:
            from torchvision.transforms.v2 import functional as F

            merge_size = self.config.vision_config.spatial_merge_size
            temporal_patch_size = self.config.vision_config.temporal_patch_size
            patch_size = self.config.vision_config.patch_size
            longest_edge = 16777216
            shortest_edge = 65536
            image_mean = torch.tensor([127.5, 127.5, 127.5], dtype=torch.float32)
            image_std = torch.tensor([127.5, 127.5, 127.5], dtype=torch.float32)

            #image to pixel value
            image = F.pil_to_tensor(images).contiguous()
            height, width = image.shape[-2:]
            resized_height, resized_width = self.smart_resize(
                height,
                width,
                factor=patch_size * merge_size,
                min_pixels=shortest_edge,
                max_pixels=longest_edge,
            )
            image = F.resize(image, (resized_height, resized_width), interpolation=F.InterpolationMode.BICUBIC, antialias=True)
            image = F.normalize(image.to(dtype=torch.float32), image_mean, image_std)
            patches = image.unsqueeze(0).unsqueeze(0)
            repeats = patches[:, -1:].repeat(1, temporal_patch_size - 1, 1, 1, 1)
            patches = torch.cat([patches, repeats], dim=1)

            batch_size, resized_t, channel = patches.shape[:3]
            grid_t = resized_t // temporal_patch_size 
            grid_h = resized_height // patch_size 
            grid_w = resized_width // patch_size

            pixel_values = (patches
                    .view(batch_size,
                        resized_t // temporal_patch_size, temporal_patch_size,
                        channel,
                        resized_height // patch_size // merge_size, merge_size, patch_size,
                        resized_width // patch_size // merge_size, merge_size, patch_size)
                    .permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9) #  -> (batch, grid_t, grid_h, grid_w, merge_h, merge_w, channel, temp_patch_size, patch_h, patch_w)
                    .reshape(
                        batch_size * grid_t * grid_h * grid_w,
                        channel * temporal_patch_size * patch_size * patch_size,
                    )
                    .to(self.device)
            )
            image_grid_thw = torch.tensor([[grid_t, grid_h, grid_w]] * batch_size, dtype=torch.int64, device=self.device)
        else:
            pixel_values = None
            image_grid_thw = None
        
        text = [text]
        text = text.copy()
        if images is not None:
            # Replace image tokens with placeholders
            merge_length = merge_size**2
            index = 0
            for i in range(len(text)):
                while self.image_token in text[i]:
                    num_image_tokens = image_grid_thw[index].prod() // merge_length
                    text[i] = text[i].replace(self.image_token, "<|placeholder|>" * num_image_tokens, 1)
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.image_token)

        text_inputs = self.tokenizer(text)

        input_ids = torch.tensor(text_inputs.input_ids, dtype=torch.int64, device=self.device)
        attention_mask = torch.tensor(text_inputs.attention_mask, dtype=torch.int64, device=self.device)
        cache_position = torch.tensor([i for i in range(input_ids.shape[-1])], dtype=torch.int64, device=self.device)

        return input_ids, pixel_values, image_grid_thw, attention_mask, cache_position

    def compile_model(self):
        """使用 torch.compile 编译模型"""
        self.model = torch.compile(self.model, mode="reduce-overhead", fullgraph=True)
        self.lm_head = torch.compile(self.lm_head, mode="reduce-overhead", fullgraph=True)

    def _apply_sampling_penalties(
        self,
        logits,
        generated_ids,
        repetition_penalty=1.0,
        presence_penalty=0.0
    ):
        if generated_ids is None or generated_ids.numel() == 0 or (1.0 == repetition_penalty and 0.0 == presence_penalty):
            return logits

        batch_size = logits.size(0)
        for b in range(batch_size):
            seen_tokens = torch.unique(generated_ids[b])
            # repetition_penalty
            if repetition_penalty != 1.0:
                token_logits = logits[b, seen_tokens]
                neg_mask = token_logits < 0
                token_logits[neg_mask] *= repetition_penalty
                token_logits[~neg_mask] /= repetition_penalty
                logits[b, seen_tokens] = token_logits
            # presence_penalty
            if presence_penalty != 0.0:
                logits[b, seen_tokens] -= presence_penalty

        return logits


    def _top_k_top_p_filtering(self, logits, top_k=0, top_p=1.0):
        """过滤 logits 以支持 top_k 和 top_p"""
        batch_size, vocab_size = logits.size()

        # top_k
        if top_k > 0:
            top_k = min(top_k, vocab_size)
            kth_values = torch.topk(logits, top_k, dim=-1)[0][..., -1, None]
            logits = logits.masked_fill(logits < kth_values, float('-inf'))

        # top_p
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            # 保证至少一个 token
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False

            # 还原到原始索引
            for b in range(batch_size):
                indices_to_remove = sorted_indices[b, sorted_indices_to_remove[b]]
                logits[b, indices_to_remove] = float('-inf')

        return logits

    def generate_fast(self, prompts=None, images=None, generated_len=128, temperature=1.0, do_sample=True):
        """使用 CUDA Graph + torch.compile 优化推理"""
        self.eval()

        # 准备输入
        input_ids, image_values, image_grid_thw, position_ids, attention_mask, inputs_embeds, cache_position = \
            self.prepare_inputs(images, prompts)

        # 提前分配输出张量（固定形状）
        output_ids = torch.zeros((1, generated_len), dtype=torch.int64, device=self.device)
        logits_buf = torch.empty((1, self.config.text_config.vocab_size), dtype=torch.float32, device=self.device)

        # 用于 CUDA Graph 的静态输入
        static_input_ids = torch.zeros_like(input_ids)
        static_pixel_values = torch.zeros_like(image_values)
        static_image_grid_thw = torch.zeros_like(image_grid_thw)
        static_position_ids = torch.zeros_like(position_ids) if position_ids is not None else None
        static_attention_mask = torch.zeros_like(attention_mask)
        static_cache_position = torch.zeros_like(cache_position)

        # 预热一次，确保编译完成
        _ = self.forward(
            input_ids=input_ids,
            pixel_values=image_values,
            past_key_values=self.kv_cache,
            image_grid_thw=image_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position
        )

        # 捕获 CUDA Graph
        g = torch.cuda.CUDAGraph()

        # 将初始值复制到静态输入
        static_input_ids.copy_(input_ids)
        static_pixel_values.copy_(image_values)
        static_image_grid_thw.copy_(image_grid_thw)
        if static_position_ids is not None:
            static_position_ids.copy_(position_ids)
        static_attention_mask.copy_(attention_mask)
        static_cache_position.copy_(cache_position)

        with torch.cuda.graph(g):
            logits = self.forward(
                input_ids=static_input_ids,
                pixel_values=static_pixel_values,
                past_key_values=self.kv_cache,
                image_grid_thw=static_image_grid_thw,
                position_ids=static_position_ids,
                attention_mask=static_attention_mask,
                inputs_embeds=inputs_embeds,
                cache_position=static_cache_position
            )
            logits_buf.copy_(logits)

        # 推理循环
        cur_len = 0
        for step in range(generated_len):
            # 执行 CUDA Graph
            g.replay()

            if do_sample:
                probs = F.softmax(logits_buf / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits_buf, dim=-1, keepdim=True)

            output_ids[:, step] = next_token.squeeze(-1)

            # 更新静态输入（下一步推理）
            static_input_ids.copy_(next_token)
            static_pixel_values.zero_()  # 后续步骤不再输入图片
            static_attention_mask[:, step] = 1
            static_cache_position[0] = static_cache_position[0] + 1

            cur_len += 1
            if next_token.item() == 151643:  # 结束 token
                break

        self.kv_cache.clear()
        return output_ids[:, :cur_len]

    def generate(self, prompts=None, images=None, generated_len=128, temperature=1.0, do_sample=True, top_p=1.0, top_k=0, repetition_penalty=1.0, presence_penalty=0.0):
        """text生成函数"""

        input_ids, image_values, image_grid_thw, attention_mask, cache_position = self.prepare_inputs(images, prompts)

        output_ids = torch.zeros((1,0), dtype=torch.int64, device=self.device)
 
        for _ in range(generated_len):
            outputs = self.forward(
                input_ids = input_ids,
                pixel_values = image_values,
                past_key_values = self.kv_cache,
                image_grid_thw = image_grid_thw,
                cache_position = cache_position
            )
            
            # 应用惩罚
            outputs = self._apply_sampling_penalties(outputs, output_ids, repetition_penalty, presence_penalty)

            if do_sample:
                # 应用 top_k / top_p
                outputs = self._top_k_top_p_filtering(outputs, top_k=top_k, top_p=top_p)
                probs = F.softmax(outputs / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(outputs, dim=-1, keepdim=True)

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
        
        return output_ids