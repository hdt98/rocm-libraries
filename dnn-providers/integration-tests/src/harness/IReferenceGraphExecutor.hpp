// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_map>

namespace hipdnn_integration_tests
{

// Implementations may not access `this`, but the methods must remain non-static
// because they are virtual overrides.
// NOLINTBEGIN(readability-convert-member-functions-to-static)
class IReferenceGraphExecutor
{
public:
    virtual ~IReferenceGraphExecutor() = default;

    virtual void execute(void* graphBuffer,
                         size_t size,
                         const std::unordered_map<int64_t, void*>& variantPack) = 0;

    /// Returns true if the executor expects device (GPU) pointers in the variant pack.
    /// When false, the executor expects host pointers.
    virtual bool requiresDeviceMemory() const = 0;
};
// NOLINTEND(readability-convert-member-functions-to-static)

} // namespace hipdnn_integration_tests
