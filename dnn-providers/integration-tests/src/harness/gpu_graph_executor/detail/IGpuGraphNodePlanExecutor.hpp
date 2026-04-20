// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <unordered_map>

namespace hipdnn_integration_tests::gpu_graph_executor::detail
{

// Implementations don't access `this`, but the methods must remain non-static
// because they are virtual overrides.
// NOLINTBEGIN(readability-convert-member-functions-to-static)
class IGpuGraphNodePlanExecutor
{
public:
    virtual ~IGpuGraphNodePlanExecutor() = default;

    virtual void execute(const std::unordered_map<int64_t, void*>& variantPack) = 0;
};
// NOLINTEND(readability-convert-member-functions-to-static)

} // namespace hipdnn_integration_tests::gpu_graph_executor::detail
