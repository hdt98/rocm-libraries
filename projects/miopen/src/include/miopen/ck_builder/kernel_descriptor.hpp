// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <miopen/ck_builder/arch_types.hpp>

namespace miopen {
namespace conv {
namespace ck_builder {

/**
 * @brief Kernel descriptor combining signature, algorithm, and target architecture.
 *
 * This template struct wraps a convolution kernel's signature (problem definition),
 * algorithm (tile sizes, pipeline config), and target GPU architecture.
 *
 * The architecture tagging enables compile-time filtering of kernel instances
 * based on the GPU_TARGETS CMake variable, reducing binary size and compile time.
 *
 * @tparam SIGNATURE Convolution signature (tensor layouts, data types, direction)
 * @tparam ALGORITHM Algorithm configuration (tile sizes, warp config, pipeline)
 * @tparam TARGET_ARCH Target GPU architecture (defaults to ANY - all architectures)
 *
 * Usage:
 *   // XDL config for CDNA (gfx942)
 *   constexpr auto my_descriptor = KernelDescriptor<sig, alg, GpuArch::GFX_942>{};
 *
 *   // Architecture-agnostic (compiles for all targets)
 *   constexpr auto generic_descriptor = KernelDescriptor<sig, alg>{};
 */
template <auto SIGNATURE, auto ALGORITHM, GpuArch TARGET_ARCH = GpuArch::ANY>
struct KernelDescriptor
{
    static constexpr auto signature   = SIGNATURE;
    static constexpr auto algorithm   = ALGORITHM;
    static constexpr GpuArch target_arch = TARGET_ARCH;
};

} // namespace ck_builder
} // namespace conv
} // namespace miopen
