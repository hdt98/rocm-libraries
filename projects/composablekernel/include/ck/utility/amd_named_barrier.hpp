// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#ifndef CK_AMD_NAMED_BARRIER_HPP
#define CK_AMD_NAMED_BARRIER_HPP

#define CK_USE_AMD_NAMED_BARRIER_ASM 1

namespace ck {

#if defined(__gfx1300__) || defined(__gfx1301__) || defined(__gfx1302__)
#define __gfx13__
#endif

#ifdef CK_USE_AMD_NAMED_BARRIER_ASM
template <unsigned NameId, unsigned Count>
class NamedBarrier
{
    public:
    __device__ void init()
    {
        constexpr uint32_t imm = NameId | (Count << 12);
        asm("s_movk_i32 m0, %0\n\ts_barrier_init m0" : : "n"(imm) : "memory");
    }
    __device__ void join()
    {
        constexpr uint32_t imm = NameId;
        asm("s_barrier_join %0" : : "n"(imm) : "memory");
    }
    __device__ void wait()
    {
        constexpr uint32_t imm = NameId;
        asm("s_barrier_wait %0" : : "n"(imm) : "memory");
    }
    __device__ void signal()
    {
        constexpr uint32_t imm = NameId;
        asm("s_movk_i32 m0, %0\n\ts_barrier_signal m0" : : "n"(imm) : "memory");
    }

    template <bool async>
    __device__ void sync_lds()
    {
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
    }
};
#else
template <unsigned Count>
class NamedBarrier
{
    public:
    __device__ NamedBarrier(__amdgpu_named_workgroup_barrier_t* bar) : bar_(bar) {}
    __device__ void init() { __builtin_amdgcn_s_barrier_init(bar_, Count); }
    __device__ void join() { __builtin_amdgcn_s_barrier_join(bar_); }
    __device__ void wait() { __builtin_amdgcn_s_barrier_wait(bar_id_); }
    __device__ void signal() { __builtin_amdgcn_s_barrier_signal_var(bar_); }
    template <bool async>
    __device__ void sync_lds()
    {
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
    }

    private:
    __amdgpu_named_workgroup_barrier_t* bar_;
    static constexpr uint32_t bar_id_ = 1;
};
#endif

} // namespace ck

#endif // CK_AMD_NAMED_BARRIER_HPP
