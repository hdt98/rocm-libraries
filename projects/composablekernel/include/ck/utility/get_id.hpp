// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"

namespace ck {

__host__ __device__ constexpr index_t get_warp_size()
{
#if defined(__GFX9__) || !defined(__HIP_DEVICE_COMPILE__)
    return 64;
#else
    return 32;
#endif
}

__device__ index_t get_thread_local_1d_id() { return threadIdx.x; }

__device__ index_t get_thread_global_1d_id() { return blockIdx.x * blockDim.x + threadIdx.x; }

__device__ index_t get_warp_local_1d_id() { return threadIdx.x / get_warp_size(); }

__device__ index_t get_block_1d_id() { return blockIdx.x; }

__device__ index_t get_grid_size() { return gridDim.x; }

__device__ index_t get_block_size() { return blockDim.x; }

__device__ index_t get_wavegroup_id()
{
#if defined(__gfx13__)
    return __builtin_amdgcn_wavegroup_id();
#else
    return 0;
#endif
}

__device__ index_t get_wave_id_in_wavegroup()
{
#if defined(__gfx13__)
    return __builtin_amdgcn_wave_id_in_wavegroup();
#else
    return 0;
#endif
}

__device__ index_t get_lane_id() { return __builtin_amdgcn_mbcnt_lo(-1, 0); }

} // namespace ck
