# adapted from https://github.com/Dao-AILab/flash-attention.git

from typing import Optional, Union
import torch
import purevlm.cuda_ops as cuda_ops


def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x

class FlashAttn:
    @staticmethod
    def flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),  # -1 means infinite context window
        softcap=0.0, # 0.0 means deactivated
        alibi_slopes=None,
        deterministic=False,
        return_attn_probs=False,
        block_table=None,
    ):
        """dropout_p should be set to 0.0 during evaluation
        Supports multi-query and grouped-query attention (MQA/GQA) by passing in K, V with fewer heads
        than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
        For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
        0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

        If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
        For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
            1 1 1 1 0
            1 1 1 1 1
        If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
            0 0
            0 0
            0 0
            1 0
            1 1
        If the row of the mask is all zero, the output will be zero.

        If window_size != (-1, -1), implements sliding window local attention. Query at position i
        will only attend to keys between
        [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

        Arguments:
            q: (total_q, nheads, headdim), where total_q = total number of query tokens in the batch.
            k: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
            v: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
            cu_seqlens_q: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
            of the sequences in the batch, used to index into q.
            cu_seqlens_k: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
            of the sequences in the batch, used to index into kv.
            max_seqlen_q: int. Maximum query sequence length in the batch.
            max_seqlen_k: int. Maximum key sequence length in the batch.
            dropout_p: float. Dropout probability.
            softmax_scale: float. The scaling of QK^T before applying softmax.
                Default to 1 / sqrt(headdim).
            causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
            window_size: (left, right). If not (-1, -1), implements sliding window local attention.
            softcap: float. Anything > 0 activates softcapping attention.
            alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
                (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
                is added to the attention score of query i and key j.
            deterministic: bool. Whether to use the deterministic implementation of the backward pass,
                which is slightly slower and uses more memory. The forward pass is always deterministic.
            return_attn_probs: bool. Whether to return the attention probabilities. This option is for
            testing only. The returned probabilities are not guaranteed to be correct
            (they might not have the right scaling).
        Return:
            out: (total, nheads, headdim).
            softmax_lse [optional, if return_attn_probs=True]: (nheads, total_q_seqlen). The
                logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
                normalization factor).
            S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen, seqlen).
                The output of softmax (possibly with different scaling). It also encodes the dropout
                pattern (negative means that location was dropped, nonnegative means it was kept).
        """
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        head_size_og = q.size(2)
        if head_size_og % 8 != 0:
            q = torch.nn.functional.pad(q, [0, 8 - head_size_og % 8])
            k = torch.nn.functional.pad(k, [0, 8 - head_size_og % 8])
            v = torch.nn.functional.pad(v, [0, 8 - head_size_og % 8])

        q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
        out_padded, softmax_lse, S_dmask, rng_state = cuda_ops.varlen_fwd(
            q,
            k,
            v,
            None,
            cu_seqlens_q,
            cu_seqlens_k,
            None,
            None,
            block_table,
            alibi_slopes,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p,
            softmax_scale,
            False,
            causal,
            window_size[0],
            window_size[1],
            softcap,
            return_attn_probs and dropout_p > 0,
            None,
        )

        out = out_padded[..., :head_size_og]
        return out if not return_attn_probs else (out, softmax_lse, S_dmask)


    @staticmethod
    def flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        k=None,
        v=None,
        rotary_cos=None,
        rotary_sin=None,
        cache_seqlens: Optional[Union[(int, torch.Tensor)]] = None,
        cache_batch_idx: Optional[torch.Tensor] = None,
        cache_leftpad: Optional[torch.Tensor] = None,
        block_table: Optional[torch.Tensor] = None,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),  # -1 means infinite context window
        softcap=0.0, # 0.0 means deactivated
        rotary_interleaved=True,
        alibi_slopes=None,
        num_splits=0,
        return_softmax_lse=False,
    ):
        """
        If k and v are not None, k_cache and v_cache will be updated *inplace* with the new values from
        k and v. This is useful for incremental decoding: you can pass in the cached keys/values from
        the previous step, and update them with the new keys/values from the current step, and do
        attention with the updated cache, all in 1 kernel.

        If you pass in k / v, you must make sure that the cache is large enough to hold the new values.
        For example, the KV cache could be pre-allocated with the max sequence length, and you can use
        cache_seqlens to keep track of the current sequence lengths of each sequence in the batch.

        Also apply rotary embedding if rotary_cos and rotary_sin are passed in. The key @k will be
        rotated by rotary_cos and rotary_sin at indices cache_seqlens, cache_seqlens + 1, etc.
        If causal or local (i.e., window_size != (-1, -1)), the query @q will be rotated by rotary_cos
        and rotary_sin at indices cache_seqlens, cache_seqlens + 1, etc.
        If not causal and not local, the query @q will be rotated by rotary_cos and rotary_sin at
        indices cache_seqlens only (i.e. we consider all tokens in @q to be at position cache_seqlens).

        See tests/test_flash_attn.py::test_flash_attn_kvcache for examples of how to use this function.

        Supports multi-query and grouped-query attention (MQA/GQA) by passing in KV with fewer heads
        than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
        For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
        0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

        If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
        For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
            1 1 1 1 0
            1 1 1 1 1
        If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
            0 0
            0 0
            0 0
            1 0
            1 1
        If the row of the mask is all zero, the output will be zero.

        If window_size != (-1, -1), implements sliding window local attention. Query at position i
        will only attend to keys between
        [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

        Note: Does not support backward pass.

        Arguments:
            q: (batch_size, seqlen, nheads, headdim)
            k_cache: (batch_size_cache, seqlen_cache, nheads_k, headdim) if there's no block_table,
                or (num_blocks, page_block_size, nheads_k, headdim) if there's a block_table (i.e. paged KV cache)
                page_block_size must be a multiple of 256.
            v_cache: (batch_size_cache, seqlen_cache, nheads_k, headdim) if there's no block_table,
                or (num_blocks, page_block_size, nheads_k, headdim) if there's a block_table (i.e. paged KV cache)
            k [optional]: (batch_size, seqlen_new, nheads_k, headdim). If not None, we concatenate
                k with k_cache, starting at the indices specified by cache_seqlens.
            v [optional]: (batch_size, seqlen_new, nheads_k, headdim). Similar to k.
            rotary_cos [optional]: (seqlen_ro, rotary_dim / 2). If not None, we apply rotary embedding
                to k and q. Only applicable if k and v are passed in. rotary_dim must be divisible by 16.
            rotary_sin [optional]: (seqlen_ro, rotary_dim / 2). Similar to rotary_cos.
            cache_seqlens: int, or (batch_size,), dtype torch.int32. The sequence lengths of the
                KV cache.
            cache_batch_idx: (batch_size,), dtype torch.int32. The indices used to index into the KV cache.
                If None, we assume that the batch indices are [0, 1, 2, ..., batch_size - 1].
                If the indices are not distinct, and k and v are provided, the values updated in the cache
                    might come from any of the duplicate indices.
            cache_leftpad: (batch_size,), dtype torch.int32. The index that the KV cache starts. If None, assume 0.
            block_table [optional]: (batch_size, max_num_blocks_per_seq), dtype torch.int32.
            softmax_scale: float. The scaling of QK^T before applying softmax.
                Default to 1 / sqrt(headdim).
            causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
            window_size: (left, right). If not (-1, -1), implements sliding window local attention.
            softcap: float. Anything > 0 activates softcapping attention.
            rotary_interleaved: bool. Only applicable if rotary_cos and rotary_sin are passed in.
                If True, rotary embedding will combine dimensions 0 & 1, 2 & 3, etc. If False,
                rotary embedding will combine dimensions 0 & rotary_dim / 2, 1 & rotary_dim / 2 + 1
                (i.e. GPT-NeoX style).
            alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
                (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
                is added to the attention score of query i and key j.
            num_splits: int. If > 1, split the key/value into this many chunks along the sequence.
            If num_splits == 1, we don't split the key/value. If num_splits == 0, we use a heuristic
            to automatically determine the number of splits.
            Don't change this unless you know what you are doing.
            return_softmax_lse: bool. Whether to return the logsumexp of the attention scores.

        Return:
            out: (batch_size, seqlen, nheads, headdim).
            softmax_lse [optional, if return_softmax_lse=True]: (batch_size, nheads, seqlen). The
                logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
                normalization factor).
        """
        assert k_cache.stride(-1) == 1, "k_cache must have contiguous last dimension"
        assert v_cache.stride(-1) == 1, "v_cache must have contiguous last dimension"
        q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        if cache_seqlens is not None and isinstance(cache_seqlens, int):
            cache_seqlens = torch.full(
                (q.shape[0],), cache_seqlens, dtype=torch.int32, device=k_cache.device
            )
            cache_seqlens = maybe_contiguous(cache_seqlens)
        cache_batch_idx = maybe_contiguous(cache_batch_idx)
        block_table = maybe_contiguous(block_table)
        out, softmax_lse = cuda_ops.fwd_kvcache(
            q,
            k_cache,
            v_cache,
            k,
            v,
            cache_seqlens,
            rotary_cos,
            rotary_sin,
            cache_batch_idx,
            cache_leftpad,
            block_table,
            alibi_slopes,
            None,
            softmax_scale,
            causal,
            window_size[0],
            window_size[1],
            softcap,
            rotary_interleaved,
            num_splits,
        )
        return (out, softmax_lse) if return_softmax_lse else out