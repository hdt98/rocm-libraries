#pragma once

// This file implements the public interface to hipconv

#include "conv2d_params.hpp"
#include "export.hpp"

#include <hip/hip_runtime.h>

#include <optional>
#include <string_view>
#include <vector>

namespace hipconv
{

// Hardware architecture. Callers (e.g. MIOpen solvers) map their own arch
// string to one of these values and pass it into hipconv.
enum class Arch
{
    gfx942, // CDNA 3
    gfx950, // CDNA 4
    count_  // not a valid arch — used for array sizing
};

// Parse a GFX arch name (e.g. "gfx950", "gfx950:sramecc+:xnack-") into Arch.
inline std::optional<Arch> parse_arch(std::string_view name)
{
    auto matches = [&](std::string_view prefix)
    {
        return name.substr(0, prefix.size()) == prefix &&
               (name.size() == prefix.size() || name[prefix.size()] == ':');
    };
    if(matches("gfx942"))
        return Arch::gfx942;
    if(matches("gfx950"))
        return Arch::gfx950;
    return std::nullopt;
}

// All supported algorithms.
enum class Algorithm
{
    Grouped,
    Direct
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
HIPCONV_API std::vector<KernelConfig> get_valid_configs(Arch arch, const Conv2dParams& par);

// All valid configs for the given params and algorithm.
HIPCONV_API std::vector<KernelConfig>
get_valid_configs(Arch arch, const Conv2dParams& par, Algorithm algo);

// Best config, or nullopt if unsupported.
HIPCONV_API std::optional<KernelConfig> find_config(Arch arch, const Conv2dParams& par);

HIPCONV_API size_t get_workspace_size(Arch arch, KernelConfig cfg, const Conv2dParams& par);

HIPCONV_API void launch(Arch arch,
                        KernelConfig cfg,
                        const Conv2dParams& par,
                        const void* in,
                        const void* wei,
                        void* out,
                        void* workspace    = nullptr,
                        hipStream_t stream = nullptr);

HIPCONV_API void
get_tolerance(Arch arch, KernelConfig cfg, const Conv2dParams& par, float& atol, float& rtol);

} // namespace hipconv
