// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "CkFmhaContainer.hpp"
#include "CkFmhaContext.hpp"
#include "CkFmhaHandle.hpp"

#define HIPDNN_PLUGIN_NAME "ck_fmha_provider_plugin"
#define HIPDNN_PLUGIN_VERSION "1.0.0"
#define HIPDNN_PLUGIN_API_VERSION "0.0.1"
#define HIPDNN_PLUGIN_CONTAINER_TYPE ck_fmha_plugin::CkFmhaContainer
#define HIPDNN_PLUGIN_HANDLE_TYPE CkFmhaHandle
#define HIPDNN_PLUGIN_CONTEXT_TYPE CkFmhaContext

#include <hipdnn_plugin_sdk/EnginePluginImpl.inl>
