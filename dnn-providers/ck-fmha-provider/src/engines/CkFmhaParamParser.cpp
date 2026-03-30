// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "CkFmhaParamParser.hpp"

#include <hipdnn_data_sdk/data_objects/graph_generated.h>

#include <cmath>
#include <stdexcept>

namespace ck_fmha_plugin {
namespace CkFmhaParamParser {

namespace {

namespace fb = hipdnn_data_sdk::data_objects;

std::string mapDataType(fb::DataType dt) {
    switch (dt) {
        case fb::DataType::HALF:
            return "fp16";
        case fb::DataType::BFLOAT16:
            return "bf16";
        default:
            return "";
    }
}

const fb::TensorAttributes* lookupTensor(const hipdnn_data_sdk::flatbuffer_utilities::IGraph& graph,
                                         int64_t uid) {
    const auto& tensor_map = graph.getTensorMap();
    auto it = tensor_map.find(uid);
    return (it != tensor_map.end()) ? it->second : nullptr;
}

// CK mask_enum values: 0=no_mask, 1=mask_top_left, 2=mask_bottom_right, 3=window_generic
int mapMask(const fb::SdpaAttributes* attr) {
    if (attr->left_bound().has_value() || attr->right_bound().has_value())
        return 3;  // window_generic
    if (attr->causal_mask_bottom_right()) return 2;
    if (attr->causal_mask()) return 1;
    return 0;
}

int mapMaskBwd(const fb::SdpaBackwardAttributes* attr) {
    if (attr->left_bound().has_value() || attr->right_bound().has_value()) return 3;
    if (attr->causal_mask_bottom_right()) return 2;
    if (attr->causal_mask()) return 1;
    return 0;
}

// CK bias_enum values: 0=no_bias, 1=elementwise_bias, 2=alibi
int mapBias(const fb::SdpaAttributes* attr) {
    if (attr->alibi_mask()) return 2;
    if (attr->attn_mask_tensor_uid().value_or(0) != 0) return 1;
    return 0;
}

int mapBiasBwd(const fb::SdpaBackwardAttributes* attr) {
    if (attr->alibi_mask()) return 2;
    if (attr->attn_mask_tensor_uid().value_or(0) != 0) return 1;
    return 0;
}

}  // namespace

bool isFwdSdpaGraph(const hipdnn_data_sdk::flatbuffer_utilities::IGraph& graph) {
    if (graph.nodeCount() != 1) return false;
    return graph.getNode(0).attributes_type() == fb::NodeAttributes::SdpaAttributes;
}

bool isBwdSdpaGraph(const hipdnn_data_sdk::flatbuffer_utilities::IGraph& graph) {
    if (graph.nodeCount() != 1) return false;
    return graph.getNode(0).attributes_type() == fb::NodeAttributes::SdpaBackwardAttributes;
}

ParsedFwdParams parseFwdGraph(const hipdnn_data_sdk::flatbuffer_utilities::IGraph& graph) {
    const auto& node = graph.getNode(0);
    const auto* attr = node.attributes_as_SdpaAttributes();

    ParsedFwdParams p;
    p.q_uid = attr->q_tensor_uid();
    p.k_uid = attr->k_tensor_uid();
    p.v_uid = attr->v_tensor_uid();
    p.o_uid = attr->o_tensor_uid();
    p.bias_uid = attr->attn_mask_tensor_uid().value_or(0);
    p.seed_uid = attr->seed_tensor_uid().value_or(0);
    p.offset_uid = attr->offset_tensor_uid().value_or(0);

    if (attr->stats_tensor_uid().value_or(0) != 0) {
        p.lse_uid = attr->stats_tensor_uid().value_or(0);
        p.has_lse = true;
    }
    if (attr->generate_stats()) p.has_lse = true;

    const auto* q_tensor = lookupTensor(graph, p.q_uid);
    const auto* k_tensor = lookupTensor(graph, p.k_uid);
    const auto* v_tensor = lookupTensor(graph, p.v_uid);

    if (q_tensor == nullptr || k_tensor == nullptr || v_tensor == nullptr)
        throw std::runtime_error("CkFmhaParamParser: missing Q/K/V tensors");

    p.data_type = mapDataType(q_tensor->data_type());
    if (p.data_type.empty()) throw std::runtime_error("CkFmhaParamParser: unsupported data type");

    const auto* q_dims = q_tensor->dims();
    const auto* k_dims = k_tensor->dims();
    const auto* v_dims = v_tensor->dims();

    // Rank-4 tensors: [B, H, S, D] (BHSD) or [B, S, H, D] (BSHD)
    // Detect layout from strides: BHSD if stride[1] > stride[2]
    const auto* q_strides = q_tensor->strides();
    if (q_strides != nullptr && q_strides->size() == 4)
        p.is_bhsd_layout = (q_strides->Get(1) > q_strides->Get(2));

    if (p.is_bhsd_layout) {
        p.batch = q_dims->Get(0);
        p.nhead_q = q_dims->Get(1);
        p.seqlen_q = q_dims->Get(2);
        p.hdim_q = q_dims->Get(3);
        p.nhead_k = k_dims->Get(1);
        p.seqlen_k = k_dims->Get(2);
        p.hdim_v = v_dims->Get(3);
    } else {
        p.batch = q_dims->Get(0);
        p.seqlen_q = q_dims->Get(1);
        p.nhead_q = q_dims->Get(2);
        p.hdim_q = q_dims->Get(3);
        p.seqlen_k = k_dims->Get(1);
        p.nhead_k = k_dims->Get(2);
        p.hdim_v = v_dims->Get(3);
    }

    p.mask_type = mapMask(attr);
    p.bias_type = mapBias(attr);

    if (attr->left_bound().has_value()) p.window_left = attr->left_bound().value();
    if (attr->right_bound().has_value()) p.window_right = attr->right_bound().value();

    p.has_dropout = (p.seed_uid != 0 && p.offset_uid != 0);

    p.scale = attr->attn_scale_value().has_value()
                  ? attr->attn_scale_value().value()
                  : (p.hdim_q > 0 ? 1.0f / std::sqrt(static_cast<float>(p.hdim_q)) : 1.0f);

    return p;
}

ParsedBwdParams parseBwdGraph(const hipdnn_data_sdk::flatbuffer_utilities::IGraph& graph) {
    const auto& node = graph.getNode(0);
    const auto* attr = node.attributes_as_SdpaBackwardAttributes();

    ParsedBwdParams p;
    p.q_uid = attr->q_tensor_uid();
    p.k_uid = attr->k_tensor_uid();
    p.v_uid = attr->v_tensor_uid();
    p.o_uid = attr->o_tensor_uid();
    p.do_uid = attr->do_tensor_uid();
    p.stats_uid = attr->stats_tensor_uid();
    p.dq_uid = attr->dq_tensor_uid();
    p.dk_uid = attr->dk_tensor_uid();
    p.dv_uid = attr->dv_tensor_uid();
    p.bias_uid = attr->attn_mask_tensor_uid().value_or(0);
    p.dbias_uid = attr->dbias_tensor_uid().value_or(0);
    p.seed_uid = attr->seed_tensor_uid().value_or(0);
    p.offset_uid = attr->offset_tensor_uid().value_or(0);

    p.has_dbias = (p.dbias_uid != 0);
    p.has_dropout = (p.seed_uid != 0 && p.offset_uid != 0);
    if (attr->dropout_probability().has_value())
        p.dropout_probability = attr->dropout_probability().value();

    const auto* q_tensor = lookupTensor(graph, p.q_uid);
    const auto* k_tensor = lookupTensor(graph, p.k_uid);
    const auto* v_tensor = lookupTensor(graph, p.v_uid);

    if (q_tensor == nullptr || k_tensor == nullptr || v_tensor == nullptr)
        throw std::runtime_error("CkFmhaParamParser(bwd): missing Q/K/V tensors");

    p.data_type = mapDataType(q_tensor->data_type());
    if (p.data_type.empty())
        throw std::runtime_error("CkFmhaParamParser(bwd): unsupported data type");

    const auto* q_dims = q_tensor->dims();
    const auto* k_dims = k_tensor->dims();
    const auto* v_dims = v_tensor->dims();

    const auto* q_strides = q_tensor->strides();
    if (q_strides != nullptr && q_strides->size() == 4)
        p.is_bhsd_layout = (q_strides->Get(1) > q_strides->Get(2));

    if (p.is_bhsd_layout) {
        p.batch = q_dims->Get(0);
        p.nhead_q = q_dims->Get(1);
        p.seqlen_q = q_dims->Get(2);
        p.hdim_q = q_dims->Get(3);
        p.nhead_k = k_dims->Get(1);
        p.seqlen_k = k_dims->Get(2);
        p.hdim_v = v_dims->Get(3);
    } else {
        p.batch = q_dims->Get(0);
        p.seqlen_q = q_dims->Get(1);
        p.nhead_q = q_dims->Get(2);
        p.hdim_q = q_dims->Get(3);
        p.seqlen_k = k_dims->Get(1);
        p.nhead_k = k_dims->Get(2);
        p.hdim_v = v_dims->Get(3);
    }

    p.mask_type = mapMaskBwd(attr);
    p.bias_type = mapBiasBwd(attr);

    if (attr->left_bound().has_value()) p.window_left = attr->left_bound().value();
    if (attr->right_bound().has_value()) p.window_right = attr->right_bound().value();

    p.scale = attr->attn_scale_value().has_value()
                  ? attr->attn_scale_value().value()
                  : (p.hdim_q > 0 ? 1.0f / std::sqrt(static_cast<float>(p.hdim_q)) : 1.0f);

    return p;
}

ck_tile::dispatcher::FmhaProblem buildFwdProblem(const ParsedFwdParams& params,
                                                 const std::string& gfx_arch) {
    return ck_tile::dispatcher::FmhaProblemBuilder()
        .api_family(ck_tile::dispatcher::FmhaApiFamily::Fwd)
        .gfx_arch(gfx_arch)
        .data_type(params.data_type)
        .dims(params.hdim_q, params.hdim_v, params.batch, params.seqlen_q, params.seqlen_k)
        .nheads(params.nhead_q, params.nhead_k)
        .mask_type(params.mask_type)
        .bias_type(params.bias_type)
        .lse(params.has_lse)
        .dropout(params.has_dropout)
        .window(params.window_left, params.window_right)
        .build();
}

ck_tile::dispatcher::FmhaProblem buildBwdProblem(const ParsedBwdParams& params,
                                                 const std::string& gfx_arch) {
    return ck_tile::dispatcher::FmhaProblemBuilder()
        .api_family(ck_tile::dispatcher::FmhaApiFamily::Bwd)
        .gfx_arch(gfx_arch)
        .data_type(params.data_type)
        .dims(params.hdim_q, params.hdim_v, params.batch, params.seqlen_q, params.seqlen_k)
        .nheads(params.nhead_q, params.nhead_k)
        .mask_type(params.mask_type)
        .bias_type(params.bias_type)
        .lse(true)  // backward always requires LSE from forward
        .dropout(params.has_dropout)
        .bwd_flags(params.has_dbias, false,
                   false)  // is_store_randval=false, is_deterministic=false (schema gaps)
        .window(params.window_left, params.window_right)
        .build();
}

}  // namespace CkFmhaParamParser
}  // namespace ck_fmha_plugin
