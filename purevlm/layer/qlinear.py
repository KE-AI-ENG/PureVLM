from typing import Optional
import torch

from vllm import _custom_ops as ops

from purevlm.layer.quantization.quant_config import QuantizationConfig
# from purevlm.layer.quantization.utils import scalar_types
from vllm.scalar_type import scalar_types,ScalarType

class QLinear:
    """
    QLinear is a quantized linear layer that can be used in neural networks.
    """
    def __init__(self, in_features, out_features, bias=True, quant_config:QuantizationConfig=None):
        super(QLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = None
        self.bias = None
        self.quant_config = quant_config

        # compressor quant
        if quant_config is not None:
            self.quant_method = quant_config.quant_method
        else:
            self.quant_method = "unquant"

        if self.quant_method == "compressed-tensors":
            self.weight_packed = None
            self.weight_shape = None
            self.weight_scale = None

    def __call__(self, input):
        if self.quant_config is None:
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

    def set_weight(self, weight_key, weight):
        """
        Set the weight for the QLinear layer.
        
        Args:
            weight (torch.Tensor): The weight tensor to set.
        """
        if weight_key.endswith(".weight"):
            if weight.shape != (self.out_features, self.in_features):
                raise ValueError(f"Weight shape {weight.shape} does not match expected shape {(self.out_features, self.in_features)}")
            self.weight = weight
        elif weight_key.endswith(".weight_packed"):
            self.weight_packed = weight
        elif weight_key.endswith(".weight_shape"):
            self.weight_shape = weight
        elif weight_key.endswith(".weight_scale"):
            self.weight_scale = weight

        # post process weight
        if self.quant_method == "compressed-tensors":
            if self.weight_packed is not None and weight_key.endswith(".weight_packed"):
                self.weight_packed = self.compressor_repack_weight(self.weight_packed)
            elif self.weight_scale is not None and weight_key.endswith(".weight_scale"):
                self.weight_scale = self.marlin_permute_scales(
                    self.weight_scale.transpose(0, 1).contiguous(),
                    size_k=self.in_features,
                    size_n=self.out_features,
                    group_size=self.quant_config.config_groups['group_0']['weights']['group_size']
                )
            
    def compressor_repack_weight(self, packed_weight):
        """
        Repack the weight from the packed format.
        
        Args:
            packed_weight (torch.Tensor): The packed weight tensor.

        Returns:
            torch.Tensor: The repacked weight tensor.
        """
        device = packed_weight.device
        self.g_idx_sort_indices = torch.nn.Parameter(torch.empty(0, dtype=torch.int, device=device),
                              requires_grad=False)
        self.w_zp = torch.nn.Parameter(torch.empty(0, dtype=torch.int, device=device),
                              requires_grad=False)
        self.g_idx = torch.nn.Parameter(torch.empty(0, dtype=torch.int, device=device),
                              requires_grad=False)
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


def marlin_make_workspace_new(device: torch.device,
                              max_blocks_per_sm: int = 1) -> torch.Tensor:
    # In the new marlin kernel, we use the num of threadblocks as workspace
    # size. The num of threadblocks is sms_count * max_blocks_per_sm.
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    return torch.zeros(sms * max_blocks_per_sm,
                       dtype=torch.int,
                       device=device,
                       requires_grad=False)

def get_scale_perms():
    scale_perm: list[int] = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single: list[int] = []
    for i in range(4):
        scale_perm_single.extend(
            [2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return scale_perm, scale_perm_single