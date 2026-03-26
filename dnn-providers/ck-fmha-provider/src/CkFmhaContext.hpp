// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_plugin_sdk/ExecutionContextBase.hpp>
#include <hipdnn_plugin_sdk/PluginBaseTypes.hpp>

#include "CkFmhaSettings.hpp"

namespace ck_fmha_plugin {

struct CkFmhaHandle;

}  // namespace ck_fmha_plugin

struct CkFmhaContext : HipdnnEnginePluginExecutionContext,
                       hipdnn_plugin_sdk::ExecutionContextBase<ck_fmha_plugin::CkFmhaHandle,
                                                               ck_fmha_plugin::CkFmhaSettings> {};
