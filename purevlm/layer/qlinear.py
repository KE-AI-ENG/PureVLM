from typing import Optional
import torch

#TODO remove vllm dependency later
try:
    from vllm import _custom_ops as ops
    from vllm.scalar_type import scalar_types
except ImportError:
    print("Warning: vllm is not installed. Compressed-tensors quantization will not work.")

from purevlm.layer.quantization.quant_config import QuantizationConfig
from purevlm.layer.quantization.utils import marlin_make_workspace_new, get_scale_perms, marlin_make_empty

class QLinear:
    """
    QLinear is a quantized linear layer that can be used in neural networks.
    """
    def __init__(self, in_features, out_features, bias=True, quant_config:QuantizationConfig=None):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = None
        self.bias = None
        self.quant_config = quant_config
        self.quant_method = quant_config.quant_method if quant_config is not None else None

        # quant_ignore will be determined once when we know the layer name
        self._quant_ignore = None

        # compressed-tensors specific attributes
        if self.quant_method == "compressed-tensors":
            self.weight_packed = None
            self.weight_shape = None
            self.weight_scale = None
            self.g_idx_sort_indices = None
            self.w_zp = None
            self.g_idx = None
            self.workspace = None

            # Cache group config for repeated access
            self._group_config = None
            if quant_config and quant_config.config_groups and 'group_0' in quant_config.config_groups:
                self._group_config = quant_config.config_groups['group_0']['weights']

    @property
    def quant_ignore(self) -> bool:
        """
        Property to check if this layer should ignore quantization.
        Uses caching to avoid repeated computation.
        """
        if self._quant_ignore is None:
            # If no quant_config or no ignore list, ignore quantization
            if self.quant_config is None or self.quant_config.ignore is None:
                self._quant_ignore = True
            else:
                self._quant_ignore = False  # Will be updated when layer name is known
        return self._quant_ignore

    def _update_quant_ignore(self, layer_name: str):
        """
        Update the quant_ignore status based on layer name.
        Should be called once when the layer name is first known.

        Args:
            layer_name (str): The full name/key of the layer
        """
        if self.quant_config is None or self.quant_config.ignore is None:
            self._quant_ignore = True
            return

        # Check if layer_name matches any ignore pattern
        for ignore_pattern in self.quant_config.ignore:
            if ignore_pattern in layer_name:
                self._quant_ignore = True
                return

        self._quant_ignore = False

    def set_weight(self, weight_key: str, weight: torch.Tensor):
        """
        Set the weight for the QLinear layer.

        Args:
            weight_key (str): The key identifying which weight to set.
            weight (torch.Tensor): The weight tensor to set.
        """
        # Update quant_ignore only once when layer name is first seen
        if self._quant_ignore is None:
            self._update_quant_ignore(weight_key)

        # Set weight based on key and apply post-processing if needed
        if weight_key.endswith(".weight"):
            if self.quant_method is None or self.quant_ignore:
                if weight.shape != (self.out_features, self.in_features):
                    raise ValueError(
                        f"Weight shape {weight.shape} does not match expected shape "
                        f"{(self.out_features, self.in_features)}"
                    )
                self.weight = weight
            else:
                pass #TODO support online quantization later

        elif weight_key.endswith(".weight_packed"):
            # For compressed-tensors, repack
            if self.quant_method == "compressed-tensors":
                self.weight_packed = self.compressor_repack_weight(weight)
            else:
                self.weight_packed = weight

        elif weight_key.endswith(".weight_shape"):
            self.weight_shape = weight

        elif weight_key.endswith(".weight_scale"):
            # For compressed-tensors, permute scales
            if self.quant_method == "compressed-tensors":
                if self._group_config is None:
                    raise ValueError("Group config not initialized for compressed-tensors")
                self.weight_scale = self.marlin_permute_scales(
                    weight.transpose(0, 1).contiguous(),
                    size_k=self.in_features,
                    size_n=self.out_features,
                    group_size=self._group_config['group_size']
                )
            else:
                self.weight_scale = weight
            
    def compressor_repack_weight(self, packed_weight):
        """
        Repack the weight from the packed format.
        
        Args:
            packed_weight (torch.Tensor): The packed weight tensor.

        Returns:
            torch.Tensor: The repacked weight tensor.
        """
        device = packed_weight.device
        self.g_idx_sort_indices = marlin_make_empty(device)
        self.w_zp = marlin_make_empty(device)
        self.g_idx = marlin_make_empty(device)
        self.workspace = marlin_make_workspace_new(device)
        
        x = ops.gptq_marlin_repack(packed_weight.t().contiguous(),
                                        perm=self.g_idx_sort_indices,
                                        size_k=self.in_features,
                                        size_n=self.out_features,
                                        num_bits=self.quant_config.config_groups['group_0']['weights']['num_bits'])
        return x
        
    def marlin_permute_scales(self, s: torch.Tensor, size_k: int, size_n: int,
                            group_size: int) -> torch.Tensor:

        scale_perm, scale_perm_single = get_scale_perms()
        if group_size < size_k and group_size != -1:
            s = s.reshape((-1, len(scale_perm)))[:, scale_perm]
        else:
            s = s.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
        s = s.reshape((-1, size_n)).contiguous()

        return s

    def __call__(self, input):
        if self.quant_method is None or self.quant_ignore:
            return torch.nn.functional.linear(input, self.weight, self.bias)
        elif self.quant_method == "compressed-tensors":
            return self.apply_gptq_marlin_linear(
                input,
                self.weight_packed,
                self.weight_scale,
                self.w_zp,
                self.g_idx,
                self.g_idx_sort_indices,
                workspace=self.workspace,
                wtype=scalar_types.uint4b8,
                output_size_per_partition=self.out_features,
                input_size_per_partition=self.in_features,
                is_k_full=True,
                bias=self.bias,
                use_fp32_reduce=True
            )
        
    def apply_gptq_marlin_linear(
            self,
            input: torch.Tensor,
            weight: torch.Tensor,
            weight_scale: torch.Tensor,
            weight_zp: torch.Tensor,
            g_idx: torch.Tensor,
            g_idx_sort_indices: torch.Tensor,
            workspace: torch.Tensor,
            wtype,
            output_size_per_partition: int,
            input_size_per_partition: int,
            is_k_full: bool,
            bias: Optional[torch.Tensor] = None,
            use_fp32_reduce: bool = True) -> torch.Tensor:
        reshaped_x = input.reshape(-1, input.shape[-1])
        out_shape = input.shape[:-1] + (output_size_per_partition, )

        use_atomic_add = False

        output = ops.gptq_marlin_gemm(reshaped_x,
                                    None,
                                    weight,
                                    bias,
                                    weight_scale,
                                    None,
                                    weight_zp,
                                    g_idx,
                                    g_idx_sort_indices,
                                    workspace,
                                    wtype,
                                    size_m=reshaped_x.shape[0],
                                    size_n=output_size_per_partition,
                                    size_k=input_size_per_partition,
                                    is_k_full=is_k_full,
                                    use_atomic_add=use_atomic_add,
                                    use_fp32_reduce=use_fp32_reduce,
                                    is_zp_float=False)

        return output.reshape(out_shape)
