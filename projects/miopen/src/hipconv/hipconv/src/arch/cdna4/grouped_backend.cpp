#include "grouped/grouped_arch_backends.h"
#include "tolerance.h"

#include "grouped/grouped_16c_fp16.h"
#include "grouped/grouped_16c_wgrad_fp16.h"
#include "grouped/grouped_32c_fp16.h"
#include "grouped/grouped_32c_wgrad_fp16.h"
#include "grouped/grouped_4c_fp16.h"
#include "grouped/grouped_4c_wgrad_fp16.h"
#include "grouped/grouped_8c_fp16.h"
#include "grouped/grouped_8c_wgrad_fp16.h"

#include <array>

using hipconv::Conv2dParams;

namespace grouped_cdna4_detail
{
bool is_applicable(const Conv2dParams& par)
{
    if(par.groups == 1)
        return false;
    return par.dilation_h == 1 && par.dilation_w == 1;
}

constexpr auto variants = std::array{
    grouped_32c::make_variant(),
    grouped_16c::make_variant(),
    grouped_16c_wgrad::make_variant(),
    grouped_32c_wgrad::make_variant(),
    grouped_8c::make_variant(),
    grouped_8c_wgrad::make_variant(),
    grouped_4c::make_variant(),
    grouped_4c_wgrad::make_variant(),
};

std::vector<AlgoConfig> do_get_valid_configs(const Conv2dParams& par)
{
    std::vector<AlgoConfig> result;
    if(!is_applicable(par))
        return result;
    for(int v = 0; v < static_cast<int>(variants.size()); ++v)
    {
        if(!variants[v].is_applicable(par))
            continue;
        for(int i = 0; i < variants[v].num_configs; ++i)
        {
            if(variants[v].config_is_compatible(par, i))
                result.push_back({v, i});
        }
    }
    return result;
}

void do_launch(AlgoConfig gcfg,
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

size_t do_get_workspace_size(AlgoConfig gcfg, const Conv2dParams& par)
{
    auto& v = variants[gcfg.kernel_variant];
    return v.get_workspace_size(gcfg.config_idx, par);
}

void do_get_tolerance(AlgoConfig, const Conv2dParams& par, float& atol, float& rtol)
{
    hipconv::get_grouped_tolerance(par, atol, rtol);
}
} // namespace grouped_cdna4_detail

// Host-only: the device linker must not see this struct or its function pointers.
#ifndef __HIP_DEVICE_COMPILE__
extern const ArchBackend grouped_backend_cdna4 = {
    grouped_cdna4_detail::do_get_valid_configs,
    grouped_cdna4_detail::do_launch,
    grouped_cdna4_detail::do_get_workspace_size,
    grouped_cdna4_detail::do_get_tolerance,
};
#endif
