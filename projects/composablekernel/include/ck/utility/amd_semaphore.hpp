// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#ifndef CK_AMD_SEMAPHORE_HPP
#define CK_AMD_SEMAPHORE_HPP

namespace ck {

#if defined(__gfx1300__) || defined(__gfx1301__) || defined(__gfx1302__)
#define __gfx13__
#endif

template <unsigned waveIdInWavegroup, unsigned SemId>
class WavegroupSemaphore
{
    public:
    static_assert(SemId >= 1 && SemId <= 7, "GFX13 only support Sem 1-7");

    __device__ void init(unsigned count = 0, unsigned limit = 1, bool done = 0)
    {
#if defined(__gfx13__)
        if(__builtin_amdgcn_wave_id_in_wavegroup() == waveIdInWavegroup)
        {
            constexpr uint32_t SemaHwReg   = 0x00000024UL + SemId - 1;
            constexpr uint32_t HwRegOffset = 0;
            constexpr uint32_t HwRegBits   = 31;
            __builtin_amdgcn_s_setreg(SemaHwReg | (HwRegOffset << 6) | (HwRegBits << 11),
                                      count | (limit << 16) | (done << 28));
        }
#else
        ignore = count;
        ignore = limit;
        ignore = done;
#endif
    }

    __device__ void wait()
    {
#if defined(__gfx13__)
        asm("s_sema_wait %0" : : "n"(1 << (SemId - 1)) : "memory");
#endif
    }

    __device__ void signal()
    {
#if defined(__gfx13__)
        constexpr unsigned imm = SemId | (waveIdInWavegroup << 4) | (1 << 8);
        asm("s_sema_signal %0" : : "n"(imm) : "memory");
#endif
    }
};
} // namespace ck

#endif // CK_AMD_SEMAPHORE_HPP