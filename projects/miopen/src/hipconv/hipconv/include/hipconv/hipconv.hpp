#pragma once

// This file implements the public interface to hipconv

#include "conv2d_params.hpp"
#include "conv3d_params.hpp"
#include "export.hpp"

#include <hip/hip_runtime.h>

#include <optional>
#include <vector>

namespace hipconv
{

// All supported algorithms.
enum class Algorithm
{
    Grouped,
    Direct,
    Direct3d, // 3D forward convolution
};

// Kernel configuration selects an algorithm, a kernel variant within that
// algorithm, and a tuning configuration within that variant.
struct KernelConfig
{
    Algorithm algorithm;
    int kernel_variant; // selects the specialized kernel (e.g. 16c-fprop-fp16)
    int config_idx;     // selects the tuning config within that kernel's table
};

// All valid configs for the given params, best first. Empty if unsupported.
HIPCONV_API std::vector<KernelConfig> get_valid_configs(const Conv2dParams& par);

// All valid configs for the given params and algorithm.
HIPCONV_API std::vector<KernelConfig> get_valid_configs(const Conv2dParams& par, Algorithm algo);

// Best config, or nullopt if unsupported.
HIPCONV_API std::optional<KernelConfig> find_config(const Conv2dParams& par);

HIPCONV_API void launch(KernelConfig cfg,
                        const Conv2dParams& par,
                        const void* in,
                        const void* wei,
                        void* out,
                        hipStream_t stream = nullptr);

HIPCONV_API void get_tolerance(KernelConfig cfg, const Conv2dParams& par, float& atol, float& rtol);

// ---- 3D convolution API ----
// KernelConfig is reused (algorithm = Algorithm::Direct3d).

HIPCONV_API std::vector<KernelConfig> get_valid_configs(const Conv3dParams& par);

HIPCONV_API std::optional<KernelConfig> find_config(const Conv3dParams& par);

HIPCONV_API void launch(KernelConfig cfg,
                        const Conv3dParams& par,
                        const void* in,
                        const void* wei,
                        void* out,
                        hipStream_t stream = nullptr);

HIPCONV_API void get_tolerance(KernelConfig cfg, const Conv3dParams& par, float& atol, float& rtol);

} // namespace hipconv
