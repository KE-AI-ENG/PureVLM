"""
Qwen2.5-VL Model Configuration Classes
"""

class Qwen2_5_VLVisionConfig:
    def __init__(
        self,
        depth = 32,
        hidden_act = "silu",
        hidden_size = 1280,
        intermediate_size = 3420,
        num_heads = 16,
        in_chans = 3,
        out_hidden_size = 3584,
        patch_size = 14,
        spatial_merge_size = 2,
        spatial_patch_size = 14,
        window_size = 112,
        fullatt_block_indexes = [7,15,23,31],
        tokens_per_second = 2,
        temporal_patch_size = 2,
        in_channels = 3,
        torch_dtype = "bfloat16",
        model_type = "qwen2_5_vl"
    ):
        self.depth = depth
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.in_channels = in_chans
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size
        self.out_hidden_size = out_hidden_size
        self.window_size = window_size
        self.fullatt_block_indexes = fullatt_block_indexes
        self.tokens_per_second = tokens_per_second
        self.torch_dtype = torch_dtype

class Qwen2_5_VLConfig:
    """Qwen3-VL模型配置类"""
    def __init__(
        self,
        vision_config=None,
        quantization_config=None,
        architectures = ["Qwen2_5_VLForConditionalGeneration"],
        attention_bias = True,
        attention_dropout = 0.0,
        bos_token_id = 151643,
        eos_token_id = 151645,
        vision_start_token_id = 151652,
        vision_end_token_id = 151653,
        vision_token_id = 151654,
        image_token_id = 151655,
        video_token_id = 151656,
        hidden_act = "silu",
        hidden_size = 3584,
        initializer_range = 0.02,
        intermediate_size = 18944,
        max_position_embeddings = 128000,
        max_window_layers = 28,
        model_type = "qwen2_5_vl",
        num_attention_heads = 28,
        num_hidden_layers = 28,
        num_key_value_heads = 4,
        rms_norm_eps = 1e-06,
        rope_theta = 1000000.0,
        sliding_window = 32768,
        tie_word_embeddings = False,
        torch_dtype = "bfloat16",
        transformers_version = "4.41.2",
        use_cache = True,
        use_sliding_window = False,
        rope_scaling = {
            "type": "mrope",
            "mrope_section": [16,24,24]
        },
        vocab_size = 152064,
        pad_token_id = None
    ):
        self.vision_config = vision_config
        self.quantization_config = quantization_config

        self.architectures = architectures
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        self.vision_token_id = vision_token_id
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.initializer_range = initializer_range
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.max_window_layers = max_window_layers
        self.model_type = model_type
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.sliding_window = sliding_window
        self.torch_dtype = torch_dtype
        self.transformers_version = transformers_version
        self.use_cache = use_cache
        self.use_sliding_window = use_sliding_window
        self.rope_scaling = rope_scaling
        self.tie_word_embeddings = tie_word_embeddings
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.head_dim = self.hidden_size // self.num_attention_heads

        #moe config
        self.num_experts = None
