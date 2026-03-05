#pragma once

#include <hip/hip_runtime.h>

// ROCm-specific configurations
#define MINT_WARP_SIZE 64

using stream_type = hipStream_t;

#define MINT_HOST_DEVICE __host__ __device__
#define MINT_HOST __host__
#define MINT_DEVICE __device__
#define MINT_HOST_INLINE __host__ inline

#if 1
#define MINT_GLOBAL_MEM
#else
#define MINT_GLOBAL_MEM __attribute__((address_space(1)))
#endif

#if 1
#define MINT_SHARED_MEM
#elif 1
#define MINT_SHARED_MEM __shared__
#elif 0
#define MINT_SHARED_MEM __attribute__((address_space(2)))
#endif

#define MINT_ALIAS_MAX_STRING_SIZE 16

// Hacks
#define MINT_HACK_HAS_CONSTEXPR_CLZ 0
