// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_plugin_sdk/interfaces/IPlan.hpp>

#include "CkFmhaHandle.hpp"
#include "engines/CkFmhaParamParser.hpp"

namespace ck_fmha_plugin {

class CkFmhaBwdPlan : public hipdnn_plugin_sdk::IPlan<CkFmhaHandle> {
   public:
    CkFmhaBwdPlan(ParsedBwdParams params, ck_tile::dispatcher::FmhaExecutionPlan exec_plan,
                  ck_tile::dispatcher::FmhaBwdWorkspaceInfo ws_info);

    size_t getWorkspaceSize(const CkFmhaHandle& handle) const override;

    void execute(const CkFmhaHandle& handle, const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 uint32_t numDeviceBuffers, void* workspace) const override;

   private:
    ParsedBwdParams params_;
    ck_tile::dispatcher::FmhaExecutionPlan exec_plan_;
    ck_tile::dispatcher::FmhaBwdWorkspaceInfo ws_info_;

    static void* findBuffer(int64_t uid, const hipdnnPluginDeviceBuffer_t* bufs, uint32_t count);
};

}  // namespace ck_fmha_plugin
