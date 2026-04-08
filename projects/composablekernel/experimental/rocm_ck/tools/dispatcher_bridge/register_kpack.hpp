// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Load a self-describing kpack archive and register its GEMM kernels
// with a CK dispatcher Registry.

#pragma once

#include "kpack_kernel_instance.hpp"

#include <rocm_ck/gpu_arch.hpp>
#include <rocm_ck/kpack_spec_reader.hpp>
#include <rocm_ck/projection.hpp>

#include <ck_tile/dispatcher/registry.hpp>

#include <cstdio>
#include <memory>

namespace rocm_ck {

/// Load a kpack archive and register all compatible GEMM kernels.
///
/// Reads variant_specs from the kpack TOC, filters by current GPU and
/// supported features, projects each GemmSpec to a KernelKey, and
/// registers a KpackKernelInstance in the registry.
///
/// Returns the number of kernels successfully registered.
inline int registerKpackKernels(const char* kpack_path,
                                ck_tile::dispatcher::Registry& registry,
                                ck_tile::dispatcher::Registry::Priority priority =
                                    ck_tile::dispatcher::Registry::Priority::Normal)
{
    // Open the kpack archive (shared across all kernel instances)
    auto archive = std::make_shared<KpackArchive>();
    if(!archive->open(kpack_path))
        return 0;

    // Read variant specs from TOC
    auto variants = readVariantSpecs(kpack_path);
    if(variants.empty())
    {
        std::fprintf(stderr,
                     "No variant specs in kpack TOC — archive may predate "
                     "self-describing format\n");
        return 0;
    }

    // Detect GPU
    auto detected_target = detectGpuTarget();
    if(!detected_target)
    {
        std::fprintf(stderr, "Unsupported GPU for kpack registration\n");
        return 0;
    }
    std::string arch = getGpuArch();

    int registered = 0;
    int skipped    = 0;

    for(const auto& vi : variants)
    {
        // Only GEMM kernels for now
        if(vi.spec_type != "GemmSpec")
            continue;

        // Skip variants that don't target this GPU
        if(!vi.targets.contains(*detected_target))
        {
            ++skipped;
            continue;
        }

        // Skip preshuffle (requires host-side B matrix rearrangement)
        if(vi.gemm_spec.pipeline == Pipeline::Preshuffle)
        {
            std::fprintf(stderr, "  skip %s: preshuffle not supported\n", vi.name.c_str());
            ++skipped;
            continue;
        }

        // Skip quantized kernels (require special data preparation)
        if(vi.gemm_spec.group_size > 0)
        {
            std::fprintf(stderr, "  skip %s: quantized kernels not supported\n", vi.name.c_str());
            ++skipped;
            continue;
        }

        // Skip StreamK (persistent) kernels — grid calculation differs
        if(vi.gemm_spec.tile_partitioner == TilePartitioner::StreamK)
        {
            std::fprintf(stderr, "  skip %s: StreamK not supported\n", vi.name.c_str());
            ++skipped;
            continue;
        }

        auto key      = projectToDispatcher(vi.gemm_spec, arch);
        auto instance = std::make_shared<KpackKernelInstance>(key, vi.name, vi.gemm_spec, archive);

        if(registry.register_kernel(instance, priority))
            ++registered;
    }

    if(skipped > 0)
        std::fprintf(stderr, "  (%d variants skipped)\n", skipped);

    return registered;
}

} // namespace rocm_ck
