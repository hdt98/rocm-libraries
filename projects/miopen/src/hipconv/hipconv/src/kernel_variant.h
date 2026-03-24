#pragma once

#include "hipconv/conv2d_params.hpp"
#include "launch_params.h"
#include <hip/hip_runtime.h>

// Define the matching functions for a conv2d kernel variant.
struct KernelVariant
{
    // Returns true if this variant supports the given parameters.
    bool (*is_applicable)(const hipconv::Conv2dParams& par);

    // Returns true if config_idx is compatible with the given parameters.
    bool (*config_is_compatible)(const hipconv::Conv2dParams& par, int config_idx);

    // Returns the launch parameters for a given config index.
    LaunchParams (*get_launch_params)(int config_idx, const hipconv::Conv2dParams& par);

    // Launch the kernel for a given config index.
    void (*launch)(int config_idx,
                   const LaunchParams& lp,
                   const hipconv::Conv2dParams& par,
                   const void* in,
                   const void* wei,
                   void* out,
                   hipStream_t stream);

    // Number of configurations used by this kernel.
    int num_configs;
};
