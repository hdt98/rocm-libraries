#include "launch_params.h"
#include "kernel_variant.h"
#include "grouped_conv.hpp"

#include "grouped_32c_fp16.h"
#include "grouped_16c_fp16.h"
#include "grouped_16c_wgrad_fp16.h"
#include "grouped_8c_fp16.h"
#include "grouped_4c_fp16.h"

#include <exception>
#include <stdexcept>

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
    grouped_32c::make_variant(),
    grouped_16c::make_variant(),
    grouped_16c_wgrad::make_variant(),
    grouped_8c::make_variant(),
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
            void* workspace,
            hipStream_t stream)
{
    auto& v = variants[gcfg.kernel_variant];
    auto lp = v.get_launch_params(gcfg.config_idx, par);
    v.launch(gcfg.config_idx, lp, par, in, wei, out, workspace, stream);
}

size_t get_workspace_size(AlgoConfig gcfg, const Conv2dParams& par)
{
    auto& v = variants[gcfg.kernel_variant];
    return v.get_workspace_size(gcfg.config_idx, par);
}

// Return the unit roundoff error for a supported data-type.
float get_unit_roundoff(hipconv::DataType dtype)
{
    switch(dtype)
    {
    case hipconv::DataType::fp16:
        return 0x1p-11f;
    case hipconv::DataType::bf16:
        return 0x1p-8f;
    case hipconv::DataType::fp32:
        return 0x1p-24f;
    default:
        throw std::logic_error("Data-type not supported");
    }
}

// Return the depth of the accumulation in the given conv2d layer.
size_t get_accumulation_depth(const Conv2dParams& par)
{
    if(par.direction == hipconv::Direction::Wgrad)
    {
        return static_cast<size_t>(par.n) * par.p * par.q;
    }
    else if(par.direction == hipconv::Direction::Fprop)
    {
        return static_cast<size_t>(par.channels_per_group()) * par.kh * par.kw;
    }
    else if(par.direction == hipconv::Direction::Dgrad)
    {
        return static_cast<size_t>(par.filters_per_group()) * par.kh * par.kw;
    }
    else
    {
        throw std::runtime_error("Unexpected direction.");
    }
}

void get_tolerance(AlgoConfig gcfg, const Conv2dParams& par, float& atol, float& rtol)
{
    auto n      = get_accumulation_depth(par);
    auto u_low  = get_unit_roundoff(par.out_type);
    auto u_high = get_unit_roundoff(hipconv::DataType::fp32);

    rtol = 2 * u_low + n * u_high;
    atol = u_high;
}

} // namespace grouped
