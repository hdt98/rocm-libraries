// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <ck_tile/dispatcher_fmha.hpp>
#include <cstdint>
#include <hipdnn_data_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <string>

namespace ck_fmha_plugin {

struct ParsedFwdParams {
    std::string data_type;
    int64_t batch = 0;
    int64_t nhead_q = 0;
    int64_t nhead_k = 0;
    int64_t seqlen_q = 0;
    int64_t seqlen_k = 0;
    int64_t hdim_q = 0;
    int64_t hdim_v = 0;

    int mask_type = 0;
    int bias_type = 0;
    bool has_lse = false;
    bool has_dropout = false;
    float scale = 0.0f;
    int64_t window_left = -1;
    int64_t window_right = -1;

    // Tensor UIDs for execute-time buffer lookup
    int64_t q_uid = -1;
    int64_t k_uid = -1;
    int64_t v_uid = -1;
    int64_t o_uid = -1;
    int64_t bias_uid = -1;
    int64_t lse_uid = -1;
    int64_t seed_uid = -1;
    int64_t offset_uid = -1;

    // Strides (BHSD convention: stride = hdim, nhead_stride = seqlen*hdim, etc.)
    bool is_bhsd_layout = true;
};

struct ParsedBwdParams {
    std::string data_type;
    int64_t batch = 0;
    int64_t nhead_q = 0;
    int64_t nhead_k = 0;
    int64_t seqlen_q = 0;
    int64_t seqlen_k = 0;
    int64_t hdim_q = 0;
    int64_t hdim_v = 0;

    int mask_type = 0;
    int bias_type = 0;
    bool has_dbias = false;
    bool has_dropout = false;
    float scale = 0.0f;
    int64_t window_left = -1;
    int64_t window_right = -1;

    // Tensor UIDs
    int64_t q_uid = -1;
    int64_t k_uid = -1;
    int64_t v_uid = -1;
    int64_t o_uid = -1;
    int64_t do_uid = -1;
    int64_t stats_uid = -1;
    int64_t dq_uid = -1;
    int64_t dk_uid = -1;
    int64_t dv_uid = -1;
    int64_t bias_uid = -1;
    int64_t dbias_uid = -1;
    int64_t seed_uid = -1;
    int64_t offset_uid = -1;

    bool is_bhsd_layout = true;
};

namespace CkFmhaParamParser {

bool isFwdSdpaGraph(const hipdnn_data_sdk::flatbuffer_utilities::IGraph& graph);
bool isBwdSdpaGraph(const hipdnn_data_sdk::flatbuffer_utilities::IGraph& graph);

ParsedFwdParams parseFwdGraph(const hipdnn_data_sdk::flatbuffer_utilities::IGraph& graph);
ParsedBwdParams parseBwdGraph(const hipdnn_data_sdk::flatbuffer_utilities::IGraph& graph);

ck_tile::dispatcher::FmhaProblem buildFwdProblem(const ParsedFwdParams& params,
                                                 const std::string& gfx_arch);
ck_tile::dispatcher::FmhaProblem buildBwdProblem(const ParsedBwdParams& params,
                                                 const std::string& gfx_arch);

}  // namespace CkFmhaParamParser
}  // namespace ck_fmha_plugin
