"""
Qwen3-VL Model Configuration Classes
"""

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
        quantization_config=None,
        dtype = "bfloat16",
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
        self.quantization_config = quantization_config

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
