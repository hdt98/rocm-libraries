// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "SdpaKernelPlan.hpp"
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

namespace sdpa_kernel_provider
{

SdpaKernelPlan::SdpaKernelPlan(size_t workspaceSize)
    : _workspaceSize(workspaceSize)
{
}

size_t SdpaKernelPlan::getWorkspaceSize(const SdpaKernelHandle& /*handle*/) const
{
    return _workspaceSize;
}

void SdpaKernelPlan::execute(const SdpaKernelHandle& /*handle*/,
                             const hipdnnPluginDeviceBuffer_t* /*deviceBuffers*/,
                             uint32_t /*numDeviceBuffers*/,
                             void* /*workspace*/) const
{
    HIPDNN_PLUGIN_LOG_ERROR("SdpaKernelPlan::execute not implemented");
}

}
