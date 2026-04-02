#include "hipconv/hipconv.hpp"
#include "grouped/grouped_conv.hpp"
#include "conv3d/conv3d.hpp"

#include <optional>
#include <stdexcept>
#include <vector>

namespace hipconv
{

namespace
{

constexpr const AlgorithmEntry* algorithms[] = {
    &grouped::algo_entry,
};

const AlgorithmEntry* find_algorithm(Algorithm algo)
{
    for(const auto* entry : algorithms)
    {
        if(entry->algorithm == algo)
            return entry;
    }
    return nullptr;
}

std::vector<KernelConfig> do_get_valid_configs(const Conv2dParams& par, const AlgorithmEntry& entry)
{
    auto acfgs = entry.get_valid_configs(par);
    std::vector<KernelConfig> result;
    result.reserve(acfgs.size());
    for(const auto& ac : acfgs)
        result.push_back({entry.algorithm, ac.kernel_variant, ac.config_idx});
    return result;
}

} // anonymous namespace


std::vector<KernelConfig> get_valid_configs(const Conv2dParams& par, Algorithm algo)
{
    auto* entry = find_algorithm(algo);
    if(!entry)
        return {};
    return do_get_valid_configs(par, *entry);
}

std::vector<KernelConfig> get_valid_configs(const Conv2dParams& par)
{
    std::vector<KernelConfig> result;
    for(const auto* entry : algorithms)
    {
        auto cfgs = do_get_valid_configs(par, *entry);
        result.insert(result.end(), cfgs.begin(), cfgs.end());
    }
    return result;
}

std::optional<KernelConfig> find_config(const Conv2dParams& par)
{
    auto cfgs = get_valid_configs(par);
    if(cfgs.empty())
        return std::nullopt;
    return cfgs.front();
}

void launch(KernelConfig cfg,
            const Conv2dParams& par,
            const void* in,
            const void* wei,
            void* out,
            hipStream_t stream)
{
    auto* entry = find_algorithm(cfg.algorithm);
    if(!entry)
        throw std::invalid_argument("unsupported algorithm");
    entry->launch({cfg.kernel_variant, cfg.config_idx}, par, in, wei, out, stream);
}

void get_tolerance(KernelConfig cfg, const Conv2dParams& par, float& atol, float& rtol)
{
    auto* entry = find_algorithm(cfg.algorithm);
    if(!entry)
        throw std::invalid_argument("unsupported algorithm");
    entry->get_tolerance({cfg.kernel_variant, cfg.config_idx}, par, atol, rtol);
}

// ---- 3D convolution dispatch ----

std::vector<KernelConfig> get_valid_configs(const Conv3dParams& par)
{
    auto acfgs = conv3d::get_valid_configs(par);
    std::vector<KernelConfig> result;
    result.reserve(acfgs.size());
    for(const auto& ac : acfgs)
        result.push_back({Algorithm::Direct3d, ac.kernel_variant, ac.config_idx});
    return result;
}

std::optional<KernelConfig> find_config(const Conv3dParams& par)
{
    auto cfgs = get_valid_configs(par);
    if(cfgs.empty())
        return std::nullopt;
    return cfgs.front();
}

void launch(KernelConfig cfg,
            const Conv3dParams& par,
            const void* in,
            const void* wei,
            void* out,
            hipStream_t stream)
{
    if(cfg.algorithm != Algorithm::Direct3d)
        throw std::invalid_argument("unsupported algorithm for Conv3dParams");
    conv3d::launch({cfg.kernel_variant, cfg.config_idx}, par, in, wei, out, stream);
}

void get_tolerance(KernelConfig cfg, const Conv3dParams& par, float& atol, float& rtol)
{
    if(cfg.algorithm != Algorithm::Direct3d)
        throw std::invalid_argument("unsupported algorithm for Conv3dParams");
    conv3d::get_tolerance({cfg.kernel_variant, cfg.config_idx}, par, atol, rtol);
}

} // namespace hipconv
