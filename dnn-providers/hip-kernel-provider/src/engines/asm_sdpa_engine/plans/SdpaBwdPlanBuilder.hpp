// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "HipKernelContext.hpp"
#include "HipKernelHandle.hpp"
#include "HipKernelSettings.hpp"

#include <hipdnn_plugin_sdk/interfaces/IPlanBuilder.hpp>

#include <string>

namespace asm_sdpa_engine
{

class SdpaBwdPlanBuilder
    : public hipdnn_plugin_sdk::IPlanBuilder<HipKernelHandle, HipKernelSettings, HipKernelContext>
{
public:
    bool isApplicable(
        const HipKernelHandle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const override;

    size_t getMaxWorkspaceSize(const HipKernelHandle& handle,
                               const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                               const HipKernelSettings& executionSettings) const override;

    void initializeExecutionSettings(
        const HipKernelHandle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
        HipKernelSettings& executionSettings) const override;

    void buildPlan(const HipKernelHandle& handle,
                   const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                   const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
                   HipKernelContext& executionContext) const override;

    std::vector<hipdnn_flatbuffers_sdk::data_objects::KnobT>
        // NOLINTNEXTLINE(portability-template-virtual-member-function)
        getCustomKnobs(
            const HipKernelHandle& handle,
            const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const override;
};

namespace bwd_dispatch
{

// Pipeline-stage tag used by the registry-lookup helper to select which
// CSV-derived registry to walk.
enum class PipelineStage
{
    ODO,
    DQDKDV,
    DQ_CONVERT
};

// Resolves a registry row matching the supplied dispatch tuple. Returns the
// composite key (arch + knl_name) on a hit, or an empty string when no row
// in the chosen registry matches. The bf16Cvt argument is the integer the
// CSV stores for that column (0/1/2 for the bf16 rounding modes; 3 for fp16).
// Exposed for unit testing of the dispatch logic.
std::string lookupKernelNameKey(PipelineStage stage,
                                const std::string& archId,
                                const std::string& dataType,
                                int hdimQ,
                                int hdimV,
                                int mask,
                                int atomic32,
                                int pssk,
                                int pddv,
                                int mode,
                                int bf16Cvt);

// pssk = "padded sequence length K": 1 when seqLenKv is not a clean multiple
// of the K/V tile size for the chosen kernel, else 0. tsKv == 0 returns 1
// (treat unresolved tile as needing the padded path).
int computePssk(unsigned int seqLenKv, unsigned int tsKv);

// pddv = "padded D_v": 1 when headDimV does not match the kernel's fast-path
// alignment (128 in AITER's selection logic), else 0.
int computePddv(unsigned int headDimV);

} // namespace bwd_dispatch

} // namespace asm_sdpa_engine
