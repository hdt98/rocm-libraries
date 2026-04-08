// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_plugin_sdk/interfaces/IPlanBuilder.hpp>

#include "HipKernelContext.hpp"
#include "HipKernelHandle.hpp"
#include "HipKernelSettings.hpp"
#include "IDevicePropertyProvider.hpp"
#include "hip/IKernelCompiler.hpp"
<<<<<<< HEAD
#include "hipdnn_flatbuffers_sdk/flatbuffer_utilities/EngineConfigWrapper.hpp"
=======
#include "hipdnn_data_sdk/flatbuffer_utilities/EngineConfigWrapper.hpp"
>>>>>>> d9e199e220 (merge b-shi branch)

namespace hip_kernel_provider::rmsnorm
{

class RMSnormPlanBuilder
    : public hipdnn_plugin_sdk::IPlanBuilder<HipKernelHandle, HipKernelSettings, HipKernelContext>
{
public:
    RMSnormPlanBuilder(const IKernelCompiler& kernelCompiler,
                       const IDevicePropertyProvider& devicePropertyProvider);
    ~RMSnormPlanBuilder() override = default;

    // Disallow copy and assignment
    RMSnormPlanBuilder(const RMSnormPlanBuilder&) = delete;
    RMSnormPlanBuilder& operator=(const RMSnormPlanBuilder&) = delete;

<<<<<<< HEAD
    bool isApplicable(
        const HipKernelHandle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const override;

    size_t getMaxWorkspaceSize(const HipKernelHandle& handle,
                               const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
=======
    bool isApplicable(const HipKernelHandle& handle,
                      const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph) const override;

    size_t getMaxWorkspaceSize(const HipKernelHandle& handle,
                               const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
>>>>>>> d9e199e220 (merge b-shi branch)
                               const HipKernelSettings& executionSettings) const override;

    void initializeExecutionSettings(
        const HipKernelHandle& handle,
<<<<<<< HEAD
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
        HipKernelSettings& executionSettings) const override;

    void buildPlan(const HipKernelHandle& handle,
                   const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                   const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
                   HipKernelContext& executionContext) const override;

    std::vector<hipdnn_flatbuffers_sdk::data_objects::KnobT> getCustomKnobs(
        const HipKernelHandle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const override;
=======
        const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
        const hipdnn_data_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
        HipKernelSettings& executionSettings) const override;

    void buildPlan(const HipKernelHandle& handle,
                   const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph,
                   const hipdnn_data_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
                   HipKernelContext& executionContext) const override;

    std::vector<hipdnn_data_sdk::data_objects::KnobT>
        getCustomKnobs(const HipKernelHandle& handle,
                       const hipdnn_data_sdk::flatbuffer_utilities::IGraph& opGraph) const override;
>>>>>>> d9e199e220 (merge b-shi branch)

private:
    const IKernelCompiler& _kernelCompiler;
    const IDevicePropertyProvider& _devicePropertyProvider;
};

}
