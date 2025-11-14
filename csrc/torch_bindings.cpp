#include <cudaTypedefs.h>

#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>
#include <torch/extension.h>
#include <torch/python.h>
#include <torch/nn/functional.h>
#include <ATen/cuda/CUDAContext.h>

#include "ops.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // elementwise-type-ops
    m.def("rms_norm_", &rms_norm, "rms norm");
    m.def("rotary_emb_", &rotary_embedding, "rotary embedding");

    // gemm-type-ops
    m.def("gptq_marlin_gemm", &gptq_marlin_gemm, "GPTQ Marlin GEMM operation");
    m.def("gptq_marlin_repack", &gptq_marlin_repack, "gptq_marlin_repack");

}