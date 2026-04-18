// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdlib>
#include <map>
#include <utility>
#include <vector>
#include <string>
#include "ck/host/types.hpp"
#include "ck/host/device_fmha_fwd/operation.hpp" // reuse TileConfig, PipelineConfig, TileMap
#include "ck/host/device_fmha_fwd_pagedkv/problem.hpp"

namespace ck {
namespace host {
namespace device_fmha_fwd_pagedkv {

// Reuse TileConfig + PipelineConfig definitions from device_fmha_fwd.
// The paged-KV pipeline has a superset of forward's tile constraints
// plus page_block_size alignment, but for Phase 5 we reuse the forward
// tile tables and let ck_tile's internals emit static_asserts when
// the combination is invalid at hipRTC time.
using device_fmha_fwd::TileConfig;
using device_fmha_fwd::PipelineConfig;
using device_fmha_fwd::TileMap;

struct Operation
{
    TileConfig tile = {};
    std::string pipeline = "qr_pagedkv";
    bool is_causal     = false;
    bool is_v_rowmajor = true;
    bool has_bias      = false;
    DataType dtype     = DataType::Half;
    bool pad_m = true;
    bool pad_n = true;
    bool pad_k = true;
    bool pad_o = true;
    std::size_t page_block_size = 64;

    static std::vector<Operation> CreateOperations(const Problem& prob, const std::string& arch);
    Solution ToSolution() const;
};

} // namespace device_fmha_fwd_pagedkv
} // namespace host
} // namespace ck
