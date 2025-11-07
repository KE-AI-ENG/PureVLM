import torch
from .scalar_type import ScalarType
import purevlm.cuda_ops as cuda_ops

# marlin
def gptq_marlin_gemm(
    a: torch.Tensor,
    c: torch.Tensor | None,
    b_q_weight: torch.Tensor,
    b_bias: torch.Tensor | None,
    b_scales: torch.Tensor,
    global_scale: torch.Tensor | None,
    b_zeros: torch.Tensor | None,
    g_idx: torch.Tensor | None,
    perm: torch.Tensor | None,
    workspace: torch.Tensor,
    b_q_type: ScalarType,
    size_m: int,
    size_n: int,
    size_k: int,
    is_k_full: bool = True,
    use_atomic_add: bool = False,
    use_fp32_reduce: bool = False,
    is_zp_float: bool = False,
) -> torch.Tensor:
    return cuda_ops.gptq_marlin_gemm(
        a,
        c,
        b_q_weight,
        b_bias,
        b_scales,
        global_scale,
        b_zeros,
        g_idx,
        perm,
        workspace,
        b_q_type.id,
        size_m,
        size_n,
        size_k,
        is_k_full,
        use_atomic_add,
        use_fp32_reduce,
        is_zp_float,
    )

def gptq_marlin_repack(
    b_q_weight: torch.Tensor,
    perm: torch.Tensor,
    size_k: int,
    size_n: int,
    num_bits: int,
) -> torch.Tensor:
    return cuda_ops.gptq_marlin_repack(b_q_weight, perm, size_k, size_n, num_bits)