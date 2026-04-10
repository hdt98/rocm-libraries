#pragma once

#include "algo_config.h"
#include "hipconv/conv2d_params.hpp"

#include <hip/hip_runtime.h>

#include <vector>

// Per-architecture dispatch table. Each arch provides one of these for each
// algorithm family (direct, grouped).
struct ArchBackend
{
    std::vector<AlgoConfig> (*get_valid_configs)(const hipconv::Conv2dParams&);
    void (*launch)(AlgoConfig,
                   const hipconv::Conv2dParams&,
                   const void*,
                   const void*,
                   void*,
                   void*,
                   hipStream_t);
    size_t (*get_workspace_size)(AlgoConfig, const hipconv::Conv2dParams&);
    void (*get_tolerance)(AlgoConfig, const hipconv::Conv2dParams&, float&, float&);
};
