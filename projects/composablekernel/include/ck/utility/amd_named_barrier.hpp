// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#ifndef CK_AMD_NAMED_BARRIER_HPP
#define CK_AMD_NAMED_BARRIER_HPP

namespace ck {

#if defined(__gfx1300__) || defined(__gfx1301__) || defined(__gfx1302__)
#define __gfx13__
#endif

template <unsigned NameId>
class NamedBarrier
{
    public:
    __device__ void init(unsigned fakeId, unsigned count)
    {
#if defined(__gfx13__)
        __builtin_amdgcn_s_barrier_init(fakeId + NameId, count);
#endif
    }
    __device__ void join()
    {
#if defined(__gfx13__)
        __builtin_amdgcn_s_barrier_join(NameId);
#endif
    }
    __device__ void wait()
    {
#if defined(__gfx13__)
        __builtin_amdgcn_s_barrier_wait(NameId);
#endif
    }
    __device__ void signal()
    {
#if defined(__gfx13__)
        __builtin_amdgcn_s_barrier_signal_var(NameId);
#endif
    }

    template <bool async>
    __device__ void sync_lds()
    {
#if defined(__gfx13__)
        if constexpr(async)
        {
            asm volatile("s_wait_asynccnt 0x0 " ::);
        }
        else
        {
            asm volatile("s_wait_dscnt 0x0 " ::);
        }

        signal();
        wait();
#endif
    }
};

} // namespace ck

#endif // CK_AMD_NAMED_BARRIER_HPP
