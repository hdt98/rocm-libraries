#include "grouped_conv.hpp"
#include "grouped/grouped_arch_backends.h"

#include <stdexcept>

using hipconv::Arch;
using hipconv::Conv2dParams;

namespace
{

const ArchBackend* find(Arch arch)
{
    switch(arch)
    {
#if HIPCONV_HAS_ARCH_CDNA4
    case Arch::gfx950:
        return &grouped_backend_cdna4;
#endif
    default:
        return nullptr;
    }
}

} // namespace

namespace grouped
{

std::vector<AlgoConfig> get_valid_configs(Arch arch, const Conv2dParams& par)
{
    auto* b = find(arch);
    return b ? b->get_valid_configs(par) : std::vector<AlgoConfig>{};
}

void launch(Arch arch,
            AlgoConfig gcfg,
            const Conv2dParams& par,
            const void* in,
            const void* wei,
            void* out,
            void* workspace,
            hipStream_t stream)
{
    auto* b = find(arch);
    if(!b)
        throw std::runtime_error("no grouped backend for requested arch");
    b->launch(gcfg, par, in, wei, out, workspace, stream);
}

size_t get_workspace_size(Arch arch, AlgoConfig gcfg, const Conv2dParams& par)
{
    auto* b = find(arch);
    return b ? b->get_workspace_size(gcfg, par) : 0;
}

void get_tolerance(Arch arch, AlgoConfig gcfg, const Conv2dParams& par, float& atol, float& rtol)
{
    auto* b = find(arch);
    if(!b)
        throw std::runtime_error("no grouped backend for requested arch");
    b->get_tolerance(gcfg, par, atol, rtol);
}

} // namespace grouped
