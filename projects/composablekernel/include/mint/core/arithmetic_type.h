#pragma once
#if defined(MINT_BACKEND_CUDA)
#include <cuda_fp16.h>
#elif defined(MINT_BACKEND_ROCM)
#include <hip/hip_fp16.h>

#endif
#include <mint/config.h>
#include <mint/core/custom_bf16.h>
#include <mint/core/custom_fp8.h>

namespace mint {

// native scalar type
using fp32_t = float;

#if defined(MINT_BACKEND_CUDA)
using fp16_t = half;
#elif defined(MINT_BACKEND_ROCM)
using fp16_t = _Float16;
#endif

#ifdef MINT_BACKEND_ROCM
typedef float floatx16_t __attribute__((ext_vector_type(16)));
typedef float floatx4_t __attribute__((ext_vector_type(4)));

typedef fp16_t fp16x8_t __attribute__((ext_vector_type(8)));
typedef fp16_t fp16x4_t __attribute__((ext_vector_type(4)));
#endif

} // namespace mint
