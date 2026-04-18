// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdlib>
#include <vector>
#include <string>
#include "ck/host/types.hpp"
#include "ck/host/device_fmha_fwd/operation.hpp"
#include "ck/host/device_fmha_fwd_splitkv/problem.hpp"

namespace ck {
namespace host {
namespace device_fmha_fwd_splitkv {

using device_fmha_fwd::TileConfig;
using device_fmha_fwd::TileMap;

struct Operation
{
    TileConfig tile = {};
    std::string pipeline = "qr_splitkv";
    bool is_causal     = false;
    bool is_v_rowmajor = true;
    bool has_bias      = false;
    DataType dtype     = DataType::Half;
    bool pad_m = true, pad_n = true, pad_k = true, pad_o = true;
    bool has_lse       = false;
    std::size_t max_splits_log2 = 3; // log2(8)

    static std::vector<Operation> CreateOperations(const Problem& prob, const std::string& arch);
    Solution ToSolution() const;
};

} // namespace device_fmha_fwd_splitkv
} // namespace host
} // namespace ck
