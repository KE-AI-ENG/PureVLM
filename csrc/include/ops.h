#pragma once

#include <optional>
#include <torch/library.h>

#include <vector>
#include "scalar_type.hpp"

////////////////elementwise ops begin/////////////////
void rms_norm(torch::Tensor& out,
              torch::Tensor& input,
              torch::Tensor& weight,
              double epsilon);

void rotary_embedding(
    torch::Tensor& positions,  // [batch_size, seq_len] or [num_tokens]
    torch::Tensor& query,  // [batch_size, seq_len, num_heads * head_size] or
                           // [num_tokens, num_heads * head_size] or
                           // [batch_size, seq_len, num_heads, head_size] or
                           // [num_tokens, num_heads, head_size]
    std::optional<torch::Tensor> key,
    // null or
    // [batch_size, seq_len, num_kv_heads * head_size] or
    // [num_tokens, num_kv_heads * head_size] or
    // [batch_size, seq_len, num_heads, head_size] or
    // [num_tokens, num_heads, head_size]
    int64_t head_size,
    torch::Tensor& cos_sin_cache,  // [max_position, rot_dim]
    bool is_neox);
// /////////////////////////////elementwise ops end/////////////////////////////


// when __CUDA_ARCH__ >= 800, marlin can be used
torch::Tensor gptq_marlin_repack(torch::Tensor& b_q_weight, torch::Tensor& perm,
                                 int64_t size_k, int64_t size_n,
                                 int64_t num_bits);

torch::Tensor gptq_marlin_gemm(
    torch::Tensor& a, std::optional<torch::Tensor> c_or_none,
    torch::Tensor& b_q_weight,
    std::optional<torch::Tensor> const& b_bias_or_none, torch::Tensor& b_scales,
    std::optional<torch::Tensor> const& global_scale_or_none,
    std::optional<torch::Tensor> const& b_zeros_or_none,
    std::optional<torch::Tensor> const& g_idx_or_none,
    std::optional<torch::Tensor> const& perm_or_none, torch::Tensor& workspace,
    purevlm::ScalarTypeId const& b_q_type_id, int64_t size_m, int64_t size_n,
    int64_t size_k, bool is_k_full, bool use_atomic_add, bool use_fp32_reduce,
    bool is_zp_float);


// attention
std::vector<at::Tensor>
mha_varlen_fwd(at::Tensor &q,  // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
               const at::Tensor &k,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
               const at::Tensor &v,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
               std::optional<at::Tensor> &out_, // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
               const at::Tensor &cu_seqlens_q,  // b+1
               const at::Tensor &cu_seqlens_k,  // b+1
               std::optional<at::Tensor> &seqused_k, // b. If given, only this many elements of each batch element's keys are used.
               std::optional<const at::Tensor> &leftpad_k_, // batch_size
               std::optional<at::Tensor> &block_table_, // batch_size x max_num_blocks_per_seq
               std::optional<at::Tensor> &alibi_slopes_, // num_heads or b x num_heads
               int max_seqlen_q,
               const int max_seqlen_k,
               const float p_dropout,
               const float softmax_scale,
               const bool zero_tensors,
               bool is_causal,
               int window_size_left,
               int window_size_right,
               const float softcap,
               const bool return_softmax,
               std::optional<at::Generator> gen_);

std::vector<at::Tensor>
mha_fwd_kvcache(at::Tensor &q,                 // batch_size x seqlen_q x num_heads x head_size
                const at::Tensor &kcache,            // batch_size_c x seqlen_k x num_heads_k x head_size or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
                const at::Tensor &vcache,            // batch_size_c x seqlen_k x num_heads_k x head_size or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
                std::optional<const at::Tensor> &k_, // batch_size x seqlen_knew x num_heads_k x head_size
                std::optional<const at::Tensor> &v_, // batch_size x seqlen_knew x num_heads_k x head_size
                std::optional<const at::Tensor> &seqlens_k_, // batch_size
                std::optional<const at::Tensor> &rotary_cos_, // seqlen_ro x (rotary_dim / 2)
                std::optional<const at::Tensor> &rotary_sin_, // seqlen_ro x (rotary_dim / 2)
                std::optional<const at::Tensor> &cache_batch_idx_, // indices to index into the KV cache
                std::optional<const at::Tensor> &leftpad_k_, // batch_size
                std::optional<at::Tensor> &block_table_, // batch_size x max_num_blocks_per_seq
                std::optional<at::Tensor> &alibi_slopes_, // num_heads or batch_size x num_heads
                std::optional<at::Tensor> &out_,             // batch_size x seqlen_q x num_heads x head_size
                const float softmax_scale,
                bool is_causal,
                int window_size_left,
                int window_size_right,
                const float softcap,
                bool is_rotary_interleaved,   // if true, rotary combines indices 0 & 1, else indices 0 & rotary_dim / 2
                int num_splits
                );