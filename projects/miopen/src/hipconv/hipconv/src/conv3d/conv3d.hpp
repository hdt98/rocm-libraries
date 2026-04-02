#pragma once

#include "../algo_config.h"
#include "hipconv/conv3d_params.hpp"

#include <vector>
#include <hip/hip_runtime.h>

namespace conv3d
{

// Return valid {kernel_variant, config_idx} pairs for the given 3D conv params.
// Returns empty if the parameters are not supported.
std::vector<AlgoConfig> get_valid_configs(const hipconv::Conv3dParams& par);

// Launch the kernel with the given configuration.
void launch(AlgoConfig cfg,
            const hipconv::Conv3dParams& par,
            const void* in,
            const void* wei,
            void* out,
            hipStream_t stream);

// Numeric accuracy of the kernel.
void get_tolerance(AlgoConfig cfg, const hipconv::Conv3dParams& par, float& atol, float& rtol);

} // namespace conv3d
