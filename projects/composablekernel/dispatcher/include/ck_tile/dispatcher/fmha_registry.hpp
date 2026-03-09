// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/dispatcher/base_registry.hpp"
#include "ck_tile/dispatcher/fmha_kernel_instance.hpp"

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace ck_tile {
namespace dispatcher {

class FmhaRegistry : public BaseRegistry<FmhaRegistry, std::string, FmhaKernelInstance>
{
    using Base = BaseRegistry<FmhaRegistry, std::string, FmhaKernelInstance>;

    public:
    using Priority = ck_tile::dispatcher::Priority;

    FmhaRegistry() = default;

    bool register_kernel(FmhaKernelInstancePtr instance, Priority priority = Priority::Normal);

    [[nodiscard]] FmhaKernelInstancePtr lookup(const std::string& identifier) const;
    [[nodiscard]] FmhaKernelInstancePtr lookup(const FmhaKernelKey& key) const;
    [[nodiscard]] std::vector<FmhaKernelInstancePtr> get_all() const;

    [[nodiscard]] std::vector<FmhaKernelInstancePtr>
    filter(std::function<bool(const FmhaKernelInstance&)> predicate) const;

    [[nodiscard]] std::string export_json(bool include_statistics = true) const;
    bool export_json_to_file(const std::string& filename, bool include_statistics = true) const;

    std::size_t filter_by_arch(const std::string& gpu_arch);

    static FmhaRegistry& instance();
};

using FmhaRegistryPtr = std::shared_ptr<FmhaRegistry>;

inline FmhaRegistryPtr make_fmha_registry(const std::string& name = "")
{
    auto reg = std::make_shared<FmhaRegistry>();
    if(!name.empty())
    {
        reg->set_name(name);
    }
    return reg;
}

} // namespace dispatcher
} // namespace ck_tile
