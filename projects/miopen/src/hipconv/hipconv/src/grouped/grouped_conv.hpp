#pragma once

#include "algo_entry.h"

#include <vector>

#include <hip/hip_runtime.h>

namespace grouped
{

std::vector<AlgoConfig> get_valid_configs(hipconv::Arch arch, const hipconv::Conv2dParams& par);

void launch(hipconv::Arch arch,
            AlgoConfig cfg,
            const hipconv::Conv2dParams& par,
            const void* in,
            const void* wei,
            void* out,
            void* workspace,
            hipStream_t stream = nullptr);

size_t get_workspace_size(hipconv::Arch arch, AlgoConfig cfg, const hipconv::Conv2dParams& par);

void get_tolerance(hipconv::Arch arch,
                   AlgoConfig cfg,
                   const hipconv::Conv2dParams& par,
                   float& atol,
                   float& rtol);

inline constexpr AlgorithmEntry algo_entry = {hipconv::Algorithm::Grouped,
                                              get_valid_configs,
                                              launch,
                                              get_workspace_size,
                                              get_tolerance};

} // namespace grouped
