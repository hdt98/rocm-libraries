#include "hipconv/hipconv.hpp"
#include "grouped/grouped_conv.hpp"

#include <optional>
#include <exception>
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

size_t get_workspace_size(KernelConfig cfg, const Conv2dParams& par)
{
    auto* entry = find_algorithm(cfg.algorithm);
    if(!entry)
        throw std::invalid_argument("unsupported algorithm");
    return entry->get_workspace_size({cfg.kernel_variant, cfg.config_idx}, par);
}

void launch(KernelConfig cfg,
            const Conv2dParams& par,
            const void* in,
            const void* wei,
            void* out,
            void* workspace,
            hipStream_t stream)
{
    auto* entry = find_algorithm(cfg.algorithm);
    if(!entry)
        throw std::invalid_argument("unsupported algorithm");
    entry->launch({cfg.kernel_variant, cfg.config_idx}, par, in, wei, out, workspace, stream);
}

void get_tolerance(KernelConfig cfg, const Conv2dParams& par, float& atol, float& rtol)
{
    auto* entry = find_algorithm(cfg.algorithm);
    if(!entry)
        throw std::invalid_argument("unsupported algorithm");
    entry->get_tolerance({cfg.kernel_variant, cfg.config_idx}, par, atol, rtol);
}

} // namespace hipconv
