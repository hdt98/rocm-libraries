#pragma once

#include "algo_config.h"
#include "hipconv/hipconv.hpp"

#include <vector>

#include <hip/hip_runtime.h>

// Dispatch table entry for an algorithm.
// Each algorithm defines one of these in its header.
struct AlgorithmEntry
{
    hipconv::Algorithm algorithm;
    std::vector<AlgoConfig> (*get_valid_configs)(const hipconv::Conv2dParams& par);
    void (*launch)(AlgoConfig,
                   const hipconv::Conv2dParams&,
                   const void*,
                   const void*,
                   void*,
                   void* workspace,
                   hipStream_t);
    size_t (*get_workspace_size)(AlgoConfig, const hipconv::Conv2dParams&);
    void (*get_tolerance)(AlgoConfig, const hipconv::Conv2dParams&, float&, float&);
};
