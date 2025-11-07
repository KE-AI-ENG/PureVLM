import torch
import numpy as np
from typing import Optional

from .scalar_type import ScalarType, scalar_types


GPTQ_MARLIN_TILE = 16
GPTQ_MARLIN_MIN_THREAD_N = 64
GPTQ_MARLIN_MAX_PARALLEL = 16


def get_pack_factor(num_bits):
    assert 32 % num_bits == 0, f"Unsupported num_bits = {num_bits}"
    return 32 // num_bits

def pack_rows(
    q_w: torch.Tensor,
    num_bits: int,
    size_k: int,
    size_n: int,
):
    assert q_w.shape == (size_k, size_n)

    pack_factor = get_pack_factor(num_bits)
    assert size_k % pack_factor == 0

    orig_device = q_w.device

    q_w = q_w.cpu().numpy().astype(np.uint32)

    q_res = np.zeros((size_k // pack_factor, size_n), dtype=np.uint32)

    for i in range(pack_factor):
        q_res |= q_w[i::pack_factor, :] << num_bits * i

    q_res = torch.from_numpy(q_res.astype(np.int32)).to(orig_device)
    return q_res

def quantize_weights(w: torch.Tensor,
                     quant_type: ScalarType,
                     group_size: Optional[int],
                     zero_points: bool = False,
                     ref_zero_points_after_scales: bool = False):
    assert quant_type.is_integer(), \
        "Floating point quantization may work but has not been tested"
    assert not zero_points or group_size is not None, \
        "to have group zero points, group_size must be provided "\
        "(-1 group_size is channelwise)"

    orig_device = w.device
    orig_type = w.dtype
    size_k, size_n = w.shape

    assert w.is_floating_point(), "w must be float"

    if group_size == -1:
        group_size = size_k

    # Reshape to [groupsize, -1]
    if group_size is not None and group_size < size_k:
        w = w.reshape((-1, group_size, size_n))
        w = w.permute(1, 0, 2)
        w = w.reshape((group_size, -1))

    # Compute scale for each group
    max_val = torch.max(w, 0, keepdim=True).values
    min_val = torch.min(w, 0, keepdim=True).values

    max_q_val = quant_type.max()
    min_q_val = quant_type.min()

    w_s = torch.Tensor([1.0]).to(w.device)  # unscaled case
    maybe_w_zp = None
    if group_size is not None:
        if zero_points:
            assert not quant_type.is_signed() and quant_type.max() > 0
            w_s = (max_val - min_val).clamp(min=1e-5) / quant_type.max()
            maybe_w_zp = torch.round(torch.abs(min_val / w_s)) \
                .clamp(min_q_val, max_q_val).int()
        else:
            # If the bias is such that there are no possible negative/positive
            #  values, set the max value to inf to avoid divide by 0
            w_s = torch.max(
                abs(max_val / (max_q_val if max_q_val != 0 else torch.inf)),
                abs(min_val / (min_q_val if min_q_val != 0 else torch.inf)))

    # Quantize
    w_q = torch.round(w / w_s).int() + (maybe_w_zp if zero_points else 0)
    w_q = torch.clamp(w_q, min_q_val, max_q_val)

    # Compute ref (dequantized)
    # For some kernels (namely Machete) the zero-points are applied after the
    # scales are applied, for this case computing the reference in similar way
    # allows us to use tighter error tolerances in our unit tests.
    if ref_zero_points_after_scales and maybe_w_zp is not None:
        w_ref = w_q.to(orig_type) * w_s - maybe_w_zp.to(orig_type) * w_s
    else:
        w_ref = (w_q - (maybe_w_zp if zero_points else 0)).to(orig_type) * w_s

    if quant_type.has_bias():
        w_q += quant_type.bias

    # Restore original shapes
    if group_size is not None and group_size < size_k:

        def reshape_w(w):
            w = w.reshape((group_size, -1, size_n))
            w = w.permute(1, 0, 2)
            w = w.reshape((size_k, size_n)).contiguous()
            return w

        w_q = reshape_w(w_q)
        w_ref = reshape_w(w_ref)
        w_s = w_s.reshape((-1, size_n)).contiguous()

    if maybe_w_zp is not None:
        maybe_w_zp = maybe_w_zp.reshape((-1, size_n)).contiguous()
        maybe_w_zp = maybe_w_zp.to(device=orig_device)

    return (
        w_ref.to(device=orig_device),
        w_q.to(device=orig_device),
        w_s if group_size is not None else None,
        maybe_w_zp,
    )

class MarlinWorkspace:

    def __init__(self, out_features, min_thread_n, max_parallel):
        assert (out_features % min_thread_n == 0), (
            "out_features = {} is undivisible by min_thread_n = {}".format(
                out_features, min_thread_n))

        max_workspace_size = ((out_features // min_thread_n) * max_parallel)

        self.scratch = torch.zeros(max_workspace_size,
                                   dtype=torch.int,
                                   device="cuda")


def get_weight_perm(num_bits: int):
    perm_list: list[int] = []
    for i in range(32):
        perm1: list[int] = []
        col = i // 4
        for block in [0, 1]:
            for row in [
                    2 * (i % 4),
                    2 * (i % 4) + 1,
                    2 * (i % 4 + 4),
                    2 * (i % 4 + 4) + 1,
            ]:
                perm1.append(16 * row + col + 8 * block)
        for j in range(4):
            perm_list.extend([p + 256 * j for p in perm1])

    perm = np.array(perm_list)

    if num_bits == 4:
        interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7])
    elif num_bits == 8:
        interleave = np.array([0, 2, 1, 3])
    else:
        raise Exception("num_bits must be 4 or 8, got {}".format(num_bits))

    perm = perm.reshape((-1, len(interleave)))[:, interleave].ravel()
    perm = torch.from_numpy(perm)
    return perm

def marlin_permute_weights(q_w, size_k, size_n, perm, tile=GPTQ_MARLIN_TILE):
    assert q_w.shape == (size_k, size_n)
    assert size_k % tile == 0, f"size_k = {size_k}, tile = {tile}"
    assert size_n % tile == 0, f"size_k = {size_n}, tile = {tile}"

    # Permute weights to 16x64 marlin tiles
    q_w = q_w.reshape((size_k // tile, tile, size_n // tile, tile))
    q_w = q_w.permute((0, 2, 1, 3))
    q_w = q_w.reshape((size_k // tile, size_n * tile))

    q_w = q_w.reshape((-1, perm.numel()))[:, perm].reshape(q_w.shape)

    return q_w

def marlin_weights(q_w, size_k, size_n, num_bits, perm):
    # Permute
    q_w = marlin_permute_weights(q_w, size_k, size_n, perm)

    # Pack
    pack_factor = get_pack_factor(num_bits)
    orig_device = q_w.device

    q_w = q_w.cpu().numpy().astype(np.uint32)

    q_packed = np.zeros((q_w.shape[0], q_w.shape[1] // pack_factor),
                        dtype=np.uint32)
    for i in range(pack_factor):
        q_packed |= q_w[:, i::pack_factor] << num_bits * i

    q_packed = torch.from_numpy(q_packed.astype(np.int32)).to(orig_device)

    return q_packed

def get_scale_perms():
    scale_perm: list[int] = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single: list[int] = []
    for i in range(4):
        scale_perm_single.extend(
            [2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return scale_perm, scale_perm_single

def marlin_permute_scales(s: torch.Tensor, size_k: int, size_n: int,
                          group_size: int) -> torch.Tensor:

    scale_perm, scale_perm_single = get_scale_perms()
    if group_size < size_k and group_size != -1:
        s = s.reshape((-1, len(scale_perm)))[:, scale_perm]
    else:
        s = s.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
    s = s.reshape((-1, size_n)).contiguous()

    return s

def pack_cols(
    q_w: torch.Tensor,
    num_bits: int,
    size_k: int,
    size_n: int,
):
    assert q_w.shape == (size_k, size_n)

    pack_factor = get_pack_factor(num_bits)
    assert size_n % pack_factor == 0

    orig_device = q_w.device

    q_w = q_w.cpu().numpy().astype(np.uint32)

    q_res = np.zeros((size_k, size_n // pack_factor), dtype=np.uint32)

    for i in range(pack_factor):
        q_res |= q_w[:, i::pack_factor] << num_bits * i

    q_res = torch.from_numpy(q_res.astype(np.int32)).to(orig_device)
    q_res = q_res.contiguous()

    return q_res

def marlin_zero_points(zp: torch.Tensor, size_k: int, size_n: int,
                       num_bits: int) -> torch.Tensor:
    # Permute zero-points in a similar way to scales, but do not use the
    # "single" permutation, since zero-points are applied on every MMA
    scale_perm, _ = get_scale_perms()
    zp = zp.reshape((-1, len(scale_perm)))[:, scale_perm]

    # Interleave column dim (for the dequantize code) and pack it to int32
    if num_bits == 4:
        interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7])
    elif num_bits == 8:
        interleave = np.array([0, 2, 1, 3])
    else:
        raise Exception("num_bits must be 4 or 8, got {}".format(num_bits))

    zp = zp.reshape((-1, len(interleave)))[:, interleave].ravel()
    zp = zp.reshape((-1, size_n)).contiguous()
    zp = pack_cols(zp, num_bits, size_k, size_n)

    return zp

def awq_marlin_quantize(w: torch.Tensor, quant_type: ScalarType,
                        group_size: int):
    size_k, size_n = w.shape

    # Normalize group_size
    if group_size == -1:
        group_size = size_k
    assert group_size <= size_k

    # Detect num groups
    assert size_k % group_size == 0
    num_groups = size_k // group_size

    # Quantize with zp
    w_ref, q_w, s, zp = quantize_weights(w,
                                         quant_type,
                                         group_size,
                                         zero_points=True)

    # Reformat to marlin
    weight_perm = get_weight_perm(quant_type.size_bits)
    marlin_q_w = marlin_weights(q_w, size_k, size_n, quant_type.size_bits,
                                weight_perm)
    marlin_s = marlin_permute_scales(s, size_k, size_n, group_size)
    marlin_zp = marlin_zero_points(zp, num_groups, size_n,
                                   quant_type.size_bits)

    # Create result
    res_list = [w_ref, marlin_q_w, marlin_s, marlin_zp]
    for i in range(len(res_list)):
        res_list[i] = res_list[i].to(w.device)

    return res_list


def preprocess(
        weight: torch.Tensor,   # shape[N,K]
        N: int,
        group_size: int,
    ):
    """
        quantize and reformat the weight of marlin, 
        quantize: per-channel-group
        shape: [n,k] -> [k/8/2, n*2]
        dtype: torch.bfloat16 -> torch.int32
    """
    device = weight.device
    weight_trans = weight.t()
    w_ref, marlin_q_w, marlin_s, marlin_zp = awq_marlin_quantize(
        weight_trans, scalar_types.uint4, group_size
    )
    g_idx = torch.empty(0, dtype=torch.int, device=device)
    sort_indices = torch.empty(0, dtype=torch.int, device=device)
    workspace = MarlinWorkspace(
            N, GPTQ_MARLIN_MIN_THREAD_N, GPTQ_MARLIN_MAX_PARALLEL
        )

    return marlin_q_w, marlin_s, marlin_zp, g_idx, sort_indices, workspace.scratch
