// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_plugin_sdk/interfaces/IPlan.hpp>

#include "SdpaKernelHandle.hpp"
#include "SdpaKernelSettings.hpp"

namespace sdpa_kernel_provider
{

/**
* @brief SDPA kernel plan.
*/
class SdpaKernelPlan : public hipdnn_plugin_sdk::IPlan<SdpaKernelHandle>
{
public:
    /**
     * @brief Construct a plan with kernel module and precomputed metadata.
     *
     * @param workspaceSize Precomputed LSE buffer size in bytes
     */
    explicit SdpaKernelPlan(size_t workspaceSize);

    ~SdpaKernelPlan() override = default;

    size_t getWorkspaceSize(const SdpaKernelHandle& handle) const override;

    void execute(const SdpaKernelHandle& handle,
                 const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 uint32_t numDeviceBuffers,
                 void* workspace = nullptr) const override;

private:
    size_t _workspaceSize;  ///< LSE buffer size: B * H_q * S_q * sizeof(float)
};

}
