#include "launch_params.h"
#include "kernel_variant.h"
#include "grouped_conv.hpp"

#include "grouped_16c_fp16.h"
#include "grouped_4c_fp16.h"

using hipconv::Conv2dParams;

namespace
{
// Algorithm-level parameter filter.
bool is_applicable(const Conv2dParams& par)
{
    if(par.groups == 1)
    {
        return false;
    }

    if(par.dilation_h != 1 || par.dilation_w != 1)
    {
        return false;
    }

    return true;
}

constexpr KernelVariant variants[] = {
    grouped_16c::make_variant(),
    grouped_4c::make_variant(),
};

constexpr int NUM_VARIANTS = sizeof(variants) / sizeof(variants[0]);

} // anonymous namespace

namespace grouped
{

std::vector<AlgoConfig> get_valid_configs(const Conv2dParams& par)
{
    std::vector<AlgoConfig> result;
    if(is_applicable(par))
    {
        for(int v = 0; v < NUM_VARIANTS; ++v)
        {
            if(!variants[v].is_applicable(par))
                continue;
            for(int i = 0; i < variants[v].num_configs; ++i)
            {
                if(variants[v].config_is_compatible(par, i))
                    result.push_back({v, i});
            }
        }
    }
    return result;
}

void launch(AlgoConfig gcfg,
            const Conv2dParams& par,
            const void* in,
            const void* wei,
            void* out,
            hipStream_t stream)
{
    auto& v = variants[gcfg.kernel_variant];
    auto lp = v.get_launch_params(gcfg.config_idx, par);
    v.launch(gcfg.config_idx, lp, par, in, wei, out, stream);
}

void get_tolerance(AlgoConfig gcfg, const Conv2dParams& par, float& atol, float& rtol)
{
    // Absolute tolerance ATOL covers near-subnormal cancellation.
    atol = 1.5e-5f;

    // Relative tolerance RTOL (~1 fp16 ULP) covers normal values.
    rtol = 1e-3f;
}

} // namespace grouped
