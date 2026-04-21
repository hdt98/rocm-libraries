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

std::vector<KernelConfig>
do_get_valid_configs(Arch arch, const Conv2dParams& par, const AlgorithmEntry& entry)
{
    auto acfgs = entry.get_valid_configs(arch, par);
    std::vector<KernelConfig> result;
    result.reserve(acfgs.size());
    for(const auto& ac : acfgs)
        result.push_back({entry.algorithm, ac.kernel_variant, ac.config_idx});
    return result;
}

} // anonymous namespace


bool has_arch_backend(Arch arch)
{
    switch(arch)
    {
#if HIPCONV_HAS_ARCH_CDNA3
    case Arch::gfx942:
        return true;
#endif
#if HIPCONV_HAS_ARCH_CDNA4
    case Arch::gfx950:
        return true;
#endif
    default:
        return false;
    }
}

std::vector<KernelConfig> get_valid_configs(Arch arch, const Conv2dParams& par, Algorithm algo)
{
    auto* entry = find_algorithm(algo);
    if(!entry)
        return {};
    return do_get_valid_configs(arch, par, *entry);
}

std::vector<KernelConfig> get_valid_configs(Arch arch, const Conv2dParams& par)
{
    std::vector<KernelConfig> result;
    for(const auto* entry : algorithms)
    {
        auto cfgs = do_get_valid_configs(arch, par, *entry);
        result.insert(result.end(), cfgs.begin(), cfgs.end());
    }
    return result;
}

std::optional<KernelConfig> find_config(Arch arch, const Conv2dParams& par)
{
    auto cfgs = get_valid_configs(arch, par);
    if(cfgs.empty())
        return std::nullopt;
    return cfgs.front();
}

size_t get_workspace_size(Arch arch, KernelConfig cfg, const Conv2dParams& par)
{
    auto* entry = find_algorithm(cfg.algorithm);
    if(!entry)
        throw std::invalid_argument("unsupported algorithm");
    return entry->get_workspace_size(arch, {cfg.kernel_variant, cfg.config_idx}, par);
}

void launch(Arch arch,
            KernelConfig cfg,
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
    entry->launch(arch, {cfg.kernel_variant, cfg.config_idx}, par, in, wei, out, workspace, stream);
}

void get_tolerance(Arch arch, KernelConfig cfg, const Conv2dParams& par, float& atol, float& rtol)
{
    auto* entry = find_algorithm(cfg.algorithm);
    if(!entry)
        throw std::invalid_argument("unsupported algorithm");
    entry->get_tolerance(arch, {cfg.kernel_variant, cfg.config_idx}, par, atol, rtol);
}

} // namespace hipconv
