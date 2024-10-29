// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#ifndef CK_AMD_SEMAPHORE_HPP
#define CK_AMD_SEMAPHORE_HPP

#include "amd_address_space.hpp"

#define CK_USE_AMD_SEMAPHORE_ASM 1

namespace ck {

#if defined(__gfx1300__) || defined(__gfx1301__) || defined(__gfx1302__)
#define __gfx13__
#endif

enum SemaphoreAddressSpaceMask : index_t
{
    SemaphoreAddressSpaceGlobal     = 1 << index_t(AddressSpaceEnum::Global),
    SemaphoreAddressSpaceShared     = 1 << index_t(AddressSpaceEnum::Lds),
    SemaphoreAddressSpaceLaneShared = 1 << 8,
    SemaphoreAddressSpaceAsync      = 1 << 9,
};

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
#if defined(CK_USE_AMD_SEMAPHORE_ASM)
            constexpr uint32_t SemaHwReg   = 0x00000024UL + SemId - 1;
            constexpr uint32_t HwRegOffset = 0;
            constexpr uint32_t HwRegBits   = 31;
            __builtin_amdgcn_s_setreg(SemaHwReg | (HwRegOffset << 6) | (HwRegBits << 11),
                                      count | (limit << 16) | (done << 28));
#else
            __builtin_amdgcn_s_sema_set_state(&_sema, count | (limit << 16) | (done << 28));
#endif
        }
#else
        ignore = count;
        ignore = limit;
        ignore = done;
#endif
    }

    template <index_t MemorySpaces>
    __device__ void wait()
    {
#if defined(__gfx13__)
#if defined(CK_USE_AMD_SEMAPHORE_ASM)
        asm("s_sema_wait %0" : : "n"(1 << (SemId - 1)) : "memory");
#else
        __builtin_amdgcn_s_sema_wait(&_sema);
#endif
#endif
        if constexpr(MemorySpaces & SemaphoreAddressSpaceGlobal)
            __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "global");
        if constexpr(MemorySpaces & SemaphoreAddressSpaceShared)
            __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "shared");
        if constexpr(MemorySpaces & SemaphoreAddressSpaceLaneShared)
            __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "laneshared");
    }

    template <index_t MemorySpaces>
    __device__ void signal(index_t count = 1)
    {
        if constexpr(MemorySpaces & SemaphoreAddressSpaceGlobal)
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "global");
        if constexpr(MemorySpaces & SemaphoreAddressSpaceShared)
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "shared");
        if constexpr(MemorySpaces & SemaphoreAddressSpaceLaneShared)
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "laneshared");

#if defined(__gfx13__)
#if defined(CK_USE_AMD_SEMAPHORE_ASM)
        constexpr unsigned imm = SemId | (waveIdInWavegroup << 4) | (1 << 8);
        asm("s_sema_signal %0" : : "n"(imm) : "memory");
#else
        __builtin_amdgcn_s_sema_signal(&_sema, count);
#endif
#endif
    }

#if !defined(CK_USE_AMD_SEMAPHORE_ASM)
    template <unsigned waveIdInWavegroup>
    struct SemaphoreType;

    template <>
    struct SemaphoreType<0>
    {
        using Type = __amdgpu_semaphore0_t;
    };

    template <>
    struct SemaphoreType<1>
    {
        using Type = __amdgpu_semaphore1_t;
    };

    template <>
    struct SemaphoreType<2>
    {
        using Type = __amdgpu_semaphore2_t;
    };

    template <>
    struct SemaphoreType<3>
    {
        using Type = __amdgpu_semaphore3_t;
    };

    template <>
    struct SemaphoreType<4>
    {
        using Type = __amdgpu_semaphore4_t;
    };

    template <>
    struct SemaphoreType<5>
    {
        using Type = __amdgpu_semaphore5_t;
    };

    template <>
    struct SemaphoreType<6>
    {
        using Type = __amdgpu_semaphore6_t;
    };

    template <>
    struct SemaphoreType<7>
    {
        using Type = __amdgpu_semaphore7_t;
    };

    SemaphoreType<waveIdInWavegroup>::Type _sema;
#endif
};
} // namespace ck

#endif // CK_AMD_SEMAPHORE_HPP