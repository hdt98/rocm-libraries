// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <flatbuffers/flatbuffers.h>

#include <ck_tile/dispatcher_fmha.hpp>
#include <hipdnn_plugin_sdk/EngineManager.hpp>
#include <hipdnn_plugin_sdk/PluginBaseTypes.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include "CkFmhaContext.hpp"
#include "CkFmhaSettings.hpp"

namespace ck_fmha_plugin {

class CkFmhaContainer;

}  // namespace ck_fmha_plugin

struct CkFmhaHandle : HipdnnEnginePluginHandle {
    CkFmhaHandle();
    ~CkFmhaHandle() override;

    void setStream(hipStream_t stream) {
        stream_ = stream;
    }
    hipStream_t getStream() const {
        return stream_;
    }

    std::shared_ptr<ck_fmha_plugin::CkFmhaContainer> container;

    hipdnn_plugin_sdk::EngineManager<CkFmhaHandle, ck_fmha_plugin::CkFmhaSettings, CkFmhaContext>&
    getEngineManager();

    void storeEngineDetailsDetachedBuffer(const void* ptr,
                                          std::unique_ptr<flatbuffers::DetachedBuffer> buffer) {
        engine_details_buffers_[ptr] = std::move(buffer);
    }

    void removeEngineDetailsDetachedBuffer(const void* ptr) {
        engine_details_buffers_.erase(ptr);
    }

    const std::string& gfxArch() const {
        return gfx_arch_;
    }
    ck_tile::dispatcher::FmhaDispatcher* dispatcher() const {
        return dispatcher_.get();
    }
    ck_tile::dispatcher::FmhaRegistry* registry() const {
        return registry_.get();
    }

    const ck_tile::dispatcher::FmhaExecutionPlan* getCachedPlan(const std::string& key) const;
    void cachePlan(const std::string& key, ck_tile::dispatcher::FmhaExecutionPlan plan);

    void loadSupplementalKernels(const std::string& dir_path);

   private:
    hipStream_t stream_ = nullptr;
    std::string gfx_arch_;
    ck_tile::dispatcher::FmhaRegistryPtr registry_;
    std::shared_ptr<ck_tile::dispatcher::FmhaDispatcher> dispatcher_;

    mutable std::mutex plan_cache_mutex_;
    std::unordered_map<std::string, ck_tile::dispatcher::FmhaExecutionPlan> plan_cache_;

    std::unordered_map<const void*, std::unique_ptr<flatbuffers::DetachedBuffer>>
        engine_details_buffers_;

    std::vector<void*> supplemental_lib_handles_;
};
