#include "conv3d.hpp"
#include "conv3d_96c_fp16.h"
#include "conv3d_3c96k_fp16.h"
#include "../kernel_variant.h"
#include "../launch_params.h"
#include "hipconv/conv3d_params.hpp"

#include <vector>

// -----------------------------------------------------------------------
// 3D conv variant table
// -----------------------------------------------------------------------

namespace
{

// Common checks shared by all 3D variants.
bool common_checks(const hipconv::Conv3dParams& par)
{
    if(par.kd != 3 || par.kh != 3 || par.kw != 3)
        return false;
    if(par.pad_d != 0)
        return false;
    if((par.pad_h != 0 && par.pad_h != 1) || (par.pad_w != 0 && par.pad_w != 1))
        return false;
    if(par.stride_d != 1 || par.stride_h != 1 || par.stride_w != 1)
        return false;
    if(par.dilation_d != 1 || par.dilation_h != 1 || par.dilation_w != 1)
        return false;
    if(par.od <= 0 || par.oh <= 0 || par.ow <= 0)
        return false;
    return true;
}

// KernelVariant-style wrappers for 3D kernels.
// (A separate struct avoids coupling the 3D path to Conv2dParams.)
struct KernelVariant3d
{
    bool (*is_applicable)(const hipconv::Conv3dParams&);
    bool (*config_is_compatible)(const hipconv::Conv3dParams&, int);
    LaunchParams (*get_launch_params)(int, const hipconv::Conv3dParams&);
    void (*launch)(int,
                   const LaunchParams&,
                   const hipconv::Conv3dParams&,
                   const void*,
                   const void*,
                   void*,
                   hipStream_t);
    int num_configs;
};

// ---- Variant 0: C=K=96, pad_h/pad_w in {0,1}, groups=1 ----
bool conv3d_96c_is_applicable(const hipconv::Conv3dParams& par)
{
    if(!common_checks(par))
        return false;
    if(par.c != 96 || par.k != 96)
        return false;
    return true;
}

bool conv3d_96c_config_compatible(const hipconv::Conv3dParams& par, int idx)
{
    return conv3d_96c::is_valid_config(par, conv3d_96c::configs[idx]);
}

LaunchParams conv3d_96c_get_launch_params(int idx, const hipconv::Conv3dParams& par)
{
    return conv3d_96c::get_launch_params(idx, par);
}

void conv3d_96c_launch(int config_idx,
                       const LaunchParams& lp,
                       const hipconv::Conv3dParams& par,
                       const void* in,
                       const void* wei,
                       void* out,
                       hipStream_t stream)
{
    conv3d_96c::launch(config_idx, lp, par, in, wei, out, stream);
}

// ---- Variant 1: C=3, K=96, pad_h/pad_w in {0,1}, groups=1 ----
bool conv3d_3c96k_is_applicable(const hipconv::Conv3dParams& par)
{
    if(!common_checks(par))
        return false;
    if(par.c != 3 || par.k != 96)
        return false;
    return true;
}

bool conv3d_3c96k_config_compatible(const hipconv::Conv3dParams& par, int idx)
{
    return conv3d_3c96k::is_valid_config(par, conv3d_3c96k::configs[idx]);
}

LaunchParams conv3d_3c96k_get_launch_params(int idx, const hipconv::Conv3dParams& par)
{
    return conv3d_3c96k::get_launch_params(idx, par);
}

void conv3d_3c96k_launch(int config_idx,
                          const LaunchParams& lp,
                          const hipconv::Conv3dParams& par,
                          const void* in,
                          const void* wei,
                          void* out,
                          hipStream_t stream)
{
    conv3d_3c96k::launch(config_idx, lp, par, in, wei, out, stream);
}

// -----------------------------------------------------------------------
// Variant table (order = preference; first applicable wins).
// -----------------------------------------------------------------------
constexpr KernelVariant3d variants[] = {
    {
        conv3d_96c_is_applicable,
        conv3d_96c_config_compatible,
        conv3d_96c_get_launch_params,
        conv3d_96c_launch,
        conv3d_96c::NUM_CONFIGS,
    },
    {
        conv3d_3c96k_is_applicable,
        conv3d_3c96k_config_compatible,
        conv3d_3c96k_get_launch_params,
        conv3d_3c96k_launch,
        conv3d_3c96k::NUM_CONFIGS,
    },
};
constexpr int NUM_VARIANTS = sizeof(variants) / sizeof(variants[0]);

} // anonymous namespace


namespace conv3d
{

std::vector<AlgoConfig> get_valid_configs(const hipconv::Conv3dParams& par)
{
    std::vector<AlgoConfig> result;
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
    return result;
}

void launch(AlgoConfig gcfg,
            const hipconv::Conv3dParams& par,
            const void* in,
            const void* wei,
            void* out,
            hipStream_t stream)
{
    auto& v = variants[gcfg.kernel_variant];
    auto lp = v.get_launch_params(gcfg.config_idx, par);
    v.launch(gcfg.config_idx, lp, par, in, wei, out, stream);
}

void get_tolerance(AlgoConfig /*gcfg*/, const hipconv::Conv3dParams& /*par*/,
                   float& atol, float& rtol)
{
    atol = 1.5e-5f;
    rtol = 1e-3f;
}

} // namespace conv3d
