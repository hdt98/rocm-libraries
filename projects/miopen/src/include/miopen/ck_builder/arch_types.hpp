// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>

namespace miopen {
namespace conv {
namespace ck_builder {

// GPU Architecture Identifiers
// Corresponds to AMD GPU target architectures for CK Builder instance selection
enum class GpuArch : std::uint8_t
{
    ANY     = 0,  // No restriction (default) - compiles for all architectures
    GFX_908,      // CDNA1: AMD Instinct MI100
    GFX_90A,      // CDNA2: AMD Instinct MI200 series
    GFX_942,      // CDNA3: AMD Instinct MI300 series
    GFX_950,      // CDNA4: AMD Instinct MI350 series (future)
    GFX_1030,     // RDNA2: Radeon RX 6000 series
    GFX_1100,     // RDNA3: Radeon RX 7900 series
    GFX_1200,     // RDNA4: Radeon RX 9000 series (future)
    GFX_1201,     // RDNA4: Radeon RX 9000 series variant
};

// Architecture Family Grouping
// Used to categorize GPUs by instruction set and wave size
enum class ArchFamily : std::uint8_t
{
    ANY  = 0,  // No restriction
    CDNA,      // MFMA/XDL instructions, Wave64, gfx9xx series
    RDNA,      // WMMA instructions, Wave32, gfx1xxx series
};

// Get architecture family from specific GPU architecture
constexpr ArchFamily get_arch_family(GpuArch arch)
{
    switch(arch)
    {
    case GpuArch::GFX_908:
    case GpuArch::GFX_90A:
    case GpuArch::GFX_942:
    case GpuArch::GFX_950: return ArchFamily::CDNA;

    case GpuArch::GFX_1030:
    case GpuArch::GFX_1100:
    case GpuArch::GFX_1200:
    case GpuArch::GFX_1201: return ArchFamily::RDNA;

    default: return ArchFamily::ANY;
    }
}

// Check if architecture belongs to CDNA family (MFMA/XDL, Wave64)
constexpr bool is_cdna(GpuArch arch) { return get_arch_family(arch) == ArchFamily::CDNA; }

// Check if architecture belongs to RDNA family (WMMA, Wave32)
constexpr bool is_rdna(GpuArch arch) { return get_arch_family(arch) == ArchFamily::RDNA; }

// Convert architecture enum to string representation
constexpr const char* arch_to_string(GpuArch arch)
{
    switch(arch)
    {
    case GpuArch::GFX_908: return "gfx908";
    case GpuArch::GFX_90A: return "gfx90a";
    case GpuArch::GFX_942: return "gfx942";
    case GpuArch::GFX_950: return "gfx950";
    case GpuArch::GFX_1030: return "gfx1030";
    case GpuArch::GFX_1100: return "gfx1100";
    case GpuArch::GFX_1200: return "gfx1200";
    case GpuArch::GFX_1201: return "gfx1201";
    case GpuArch::ANY: return "any";
    default: return "unknown";
    }
}

} // namespace ck_builder
} // namespace conv
} // namespace miopen
