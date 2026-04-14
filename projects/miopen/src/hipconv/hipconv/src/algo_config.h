#pragma once

// Selects a kernel variant and tuning configuration within an algorithm.
// Shared by all algorithms.
struct AlgoConfig
{
    int kernel_variant;
    int config_idx;
};
