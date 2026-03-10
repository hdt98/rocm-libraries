// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace ck_tile {

struct LdsProbeKargs
{
    const uint32_t* p_should_read;
    const uint32_t* p_offset;
    int32_t instruction_mode; // 0=ds_read_b64, 1=ds_read_b96, 2=ds_read_b128, 3=ds_write_b64
};

struct LdsProbeKernel
{
    static constexpr int kBlockSize = 64; // 1 wavefront

    CK_TILE_HOST static constexpr dim3 GridSize() { return dim3(1); }
    CK_TILE_HOST static constexpr dim3 BlockSize() { return dim3(kBlockSize); }

    CK_TILE_HOST static LdsProbeKargs
    MakeKargs(const void* p_should_read, const void* p_offset, int32_t mode)
    {
        return LdsProbeKargs{static_cast<const uint32_t*>(p_should_read),
                             static_cast<const uint32_t*>(p_offset),
                             mode};
    }

    CK_TILE_DEVICE void operator()(LdsProbeKargs kargs) const
    {
        __shared__ char smem_buf[32768]; // 32KB static LDS (enough for bank probing)
        int* smem = reinterpret_cast<int*>(smem_buf);

        const int tid = threadIdx.x;

        uint32_t should_read          = kargs.p_should_read[tid];
        uint32_t shared_memory_offset = kargs.p_offset[tid];

        const uint32_t src_ptr = reinterpret_cast<uintptr_t>(&smem[0]);

        __builtin_amdgcn_s_waitcnt(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        if(should_read)
        {
            const uint32_t addr = src_ptr + shared_memory_offset;

            if(kargs.instruction_mode == 0)
            {
                // ds_read_b64: 64 bits = 8 bytes = 2 banks
                typedef __attribute__((__vector_size__(2 * sizeof(float)))) float floatx2_t;
                floatx2_t data;
                asm volatile("ds_read_b64 %0, %1 offset:%2"
                             : "=v"(data)
                             : "v"(addr), "i"(0)
                             : "memory");
            }
            else if(kargs.instruction_mode == 1)
            {
                // ds_read_b96: 96 bits = 12 bytes = 3 banks
                typedef __attribute__((__vector_size__(3 * sizeof(float)))) float floatx3_t;
                floatx3_t data;
                asm volatile("ds_read_b96 %0, %1 offset:%2"
                             : "=v"(data)
                             : "v"(addr), "i"(0)
                             : "memory");
            }
            else if(kargs.instruction_mode == 2)
            {
                // ds_read_b128: 128 bits = 16 bytes = 4 banks
                typedef __attribute__((__vector_size__(4 * sizeof(float)))) float floatx4_t;
                floatx4_t data;
                asm volatile("ds_read_b128 %0, %1 offset:%2"
                             : "=v"(data)
                             : "v"(addr), "i"(0)
                             : "memory");
            }
            else if(kargs.instruction_mode == 3)
            {
                // ds_write_b64: 64 bits = 8 bytes = 2 banks
                uint64_t garbage = 0x1234567890abcdef;
                asm volatile("ds_write_b64 %0, %1, offset:%2"
                             :
                             : "v"(addr), "v"(garbage), "i"(0));
            }
        }
    }
};

} // namespace ck_tile
