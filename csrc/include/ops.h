#pragma once

#include <optional>
#include <torch/library.h>

#include <vector>
#include "core/scalar_type.hpp"

////////////////elementwise ops begin/////////////////
// void int8_quant(torch::Tensor& out,
//                             torch::Tensor const& input,
//                             torch::Tensor& scales, c10::optional<torch::Tensor> const& azp);
// void fp8_quant(
//     torch::Tensor& out, torch::Tensor const& input, torch::Tensor& scale,
//     c10::optional<torch::Tensor> const& scale_ub);
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



// torch::Tensor gptq_marlin_repack(torch::Tensor& b_q_weight, torch::Tensor& perm,
//                                  int64_t size_k, int64_t size_n,
//                                  int64_t num_bits);

// torch::Tensor gptq_marlin_repack_meta(torch::Tensor& b_q_weight,
//                                       torch::Tensor& perm, c10::SymInt size_k,
//                                       c10::SymInt size_n, int64_t num_bits);

// torch::Tensor gptq_marlin_gemm(
//     torch::Tensor& a, std::optional<torch::Tensor> c_or_none,
//     torch::Tensor& b_q_weight,
//     std::optional<torch::Tensor> const& b_bias_or_none, torch::Tensor& b_scales,
//     std::optional<torch::Tensor> const& global_scale_or_none,
//     std::optional<torch::Tensor> const& b_zeros_or_none,
//     std::optional<torch::Tensor> const& g_idx_or_none,
//     std::optional<torch::Tensor> const& perm_or_none, torch::Tensor& workspace,
//     vllm::ScalarTypeId const& b_q_type_id, int64_t size_m, int64_t size_n,
//     int64_t size_k, bool is_k_full, bool use_atomic_add, bool use_fp32_reduce,
//     bool is_zp_float);

// torch::Tensor awq_marlin_repack(torch::Tensor& b_q_weight, int64_t size_k,
//                                 int64_t size_n, int64_t num_bits);

// torch::Tensor awq_marlin_repack_meta(torch::Tensor& b_q_weight,
//                                      c10::SymInt size_k, c10::SymInt size_n,
//                                      int64_t num_bits);


// #if (defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 900 && __CUDA_ARCH__ < 1000) || \
// (defined HOST_CUDA_ARCH && HOST_CUDA_ARCH >= 900 && HOST_CUDA_ARCH < 1000)
// std::vector<torch::Tensor> 
// flash_mha(torch::Tensor &q,         // batch_size x seqlen_q x num_heads x head_size
//         const torch::Tensor &k,         // batch_size x seqlen_k x num_heads_k x head_size
//         const torch::Tensor &v,         // batch_size x seqlen_k x num_heads_k x head_size
//         const float softmax_scale,
//         bool is_causal
//         );
// void fp8_scaled_mm_sm90(torch::Tensor& out, torch::Tensor const& mat_a,
//                             torch::Tensor const& mat_b,
//                             torch::Tensor const& scales_a,
//                             torch::Tensor const& scales_b,
//                             const torch::Dtype& out_dtype,
//                             c10::optional<torch::Tensor> const& bias);

// void int8_scaled_mm_sm90(torch::Tensor& out, torch::Tensor const& a,
//                                      torch::Tensor const& b,
//                                      torch::Tensor const& a_scales,
//                                      torch::Tensor const& b_scales,
//                                      torch::Tensor const& azp_adj,
//                                      torch::Tensor const& azp,
//                                      std::optional<torch::Tensor> const& bias);
// #endif

// #if (defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 890 && __CUDA_ARCH__ < 900) || \
// (defined HOST_CUDA_ARCH && HOST_CUDA_ARCH >= 890 && HOST_CUDA_ARCH < 900)
// void fp8_scaled_mm_sm89(torch::Tensor& out, torch::Tensor const& mat_a,
//                             torch::Tensor const& mat_b,
//                             torch::Tensor const& scales_a,
//                             torch::Tensor const& scales_b,
//                             const torch::Dtype& out_dtype,
//                             c10::optional<torch::Tensor> const& bias);

// void int8_scaled_mm_sm89(torch::Tensor& out, torch::Tensor const& a,
//                                 torch::Tensor const& b,
//                                 torch::Tensor const& a_scales,
//                                 torch::Tensor const& b_scales,
//                                 torch::Tensor const& azp_adj,
//                                 torch::Tensor const& azp,
//                                 std::optional<torch::Tensor> const& bias);
// #endif

// #if (defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 800 && __CUDA_ARCH__ < 890) || \
// (defined HOST_CUDA_ARCH && HOST_CUDA_ARCH >= 800 && HOST_CUDA_ARCH < 890)
// void int8_scaled_mm_sm80(torch::Tensor& out, torch::Tensor const& a,
//                                 torch::Tensor const& b,
//                                 torch::Tensor const& a_scales,
//                                 torch::Tensor const& b_scales,
//                                 torch::Tensor const& azp_adj,
//                                 torch::Tensor const& azp,
//                                 std::optional<torch::Tensor> const& bias);
// #endif

// #if (defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 750 && __CUDA_ARCH__ < 800) || \
// (defined HOST_CUDA_ARCH && HOST_CUDA_ARCH >= 750 && HOST_CUDA_ARCH < 800)
// void int8_scaled_mm_sm75(torch::Tensor& out, torch::Tensor const& a,
//                                 torch::Tensor const& b,
//                                 torch::Tensor const& a_scales,
//                                 torch::Tensor const& b_scales,
//                                 torch::Tensor const& azp_adj,
//                                 torch::Tensor const& azp,
//                                 std::optional<torch::Tensor> const& bias);
// #endif
