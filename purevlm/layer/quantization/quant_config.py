class QuantizationConfig:
    def __init__(
        self, 
        config_groups: dict = None,
        format: str = None,
        ignore: list = None,
        quant_method: str = None,
        quantization_status: str = None,
        kv_cache_scheme = None,
        sparsity_config = None,
        transform_config = None,
        version = None,
        global_compression_ratio = None,
    ):
        self.config_groups = config_groups
        self.format = format
        self.ignore = ignore
        self.quant_method = quant_method
        self.quantization_status = quantization_status
        self.kv_cache_scheme = kv_cache_scheme
        self.sparsity_config = sparsity_config
        self.transform_config = transform_config
        self.version = version
        self.global_compression_ratio = global_compression_ratio