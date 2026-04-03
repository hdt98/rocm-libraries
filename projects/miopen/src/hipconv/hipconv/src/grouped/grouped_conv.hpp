#pragma once

#include "algo_entry.h"

#include <vector>

#include <hip/hip_runtime.h>

namespace grouped
{

// Return vector of valid {kernel_variant, config_idx} pairs for the conv2d params.
//
// Returns an empty vector if the parameters are not supported.
std::vector<AlgoConfig> get_valid_configs(const hipconv::Conv2dParams& par);

// Launch the kernel with the given configuration and conv2d parameters.
void launch(AlgoConfig cfg,
            const hipconv::Conv2dParams& par,
            const void* in,
            const void* wei,
            void* out,
            void* workspace,
            hipStream_t stream = nullptr);

// Returns workspace size in bytes (0 if no workspace needed).
size_t get_workspace_size(AlgoConfig cfg, const hipconv::Conv2dParams& par);

// Get the expected numeric accuracy of the kernel.
void get_tolerance(AlgoConfig cfg, const hipconv::Conv2dParams& par, float& atol, float& rtol);

inline constexpr AlgorithmEntry algo_entry = {hipconv::Algorithm::Grouped,
                                              get_valid_configs,
                                              launch,
                                              get_workspace_size,
                                              get_tolerance};

} // namespace grouped
