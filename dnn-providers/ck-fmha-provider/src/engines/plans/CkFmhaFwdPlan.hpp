// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_plugin_sdk/interfaces/IPlan.hpp>

#include "CkFmhaHandle.hpp"
#include "engines/CkFmhaParamParser.hpp"

namespace ck_fmha_plugin {

class CkFmhaFwdPlan : public hipdnn_plugin_sdk::IPlan<CkFmhaHandle> {
   public:
    CkFmhaFwdPlan(ParsedFwdParams params, ck_tile::dispatcher::FmhaExecutionPlan exec_plan);

    size_t getWorkspaceSize(const CkFmhaHandle& handle) const override;

    void execute(const CkFmhaHandle& handle, const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 uint32_t numDeviceBuffers, void* workspace) const override;

   private:
    ParsedFwdParams params_;
    ck_tile::dispatcher::FmhaExecutionPlan exec_plan_;

    static void* findBuffer(int64_t uid, const hipdnnPluginDeviceBuffer_t* bufs, uint32_t count);
};

}  // namespace ck_fmha_plugin
