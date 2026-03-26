// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <optional>
#include <string>

namespace ck_fmha_plugin {

struct CkFmhaSettings {
    enum class SelectionMode { Builtin, Heuristic };

    SelectionMode selectionMode() const {
        return selection_mode_;
    }
    void setSelectionMode(SelectionMode mode) {
        selection_mode_ = mode;
    }

    const std::string& forcedKernelId() const {
        return forced_kernel_id_;
    }
    void setForcedKernelId(const std::string& id) {
        forced_kernel_id_ = id;
    }

    std::optional<int> receiptFilter() const {
        return receipt_filter_;
    }
    void setReceiptFilter(int receipt) {
        receipt_filter_ = receipt;
    }

    const std::string& cachePath() const {
        return cache_path_;
    }
    void setCachePath(const std::string& path) {
        cache_path_ = path;
    }

    const std::string& kernelLibPath() const {
        return kernel_lib_path_;
    }
    void setKernelLibPath(const std::string& path) {
        kernel_lib_path_ = path;
    }

   private:
    SelectionMode selection_mode_ = SelectionMode::Builtin;
    std::string forced_kernel_id_;
    std::optional<int> receipt_filter_;
    std::string cache_path_;
    std::string kernel_lib_path_;
};

}  // namespace ck_fmha_plugin
