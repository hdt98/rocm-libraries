// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <hipdnn_data_sdk/data_objects/sdpa_backward_attributes_generated.h>
#include <hipdnn_data_sdk/utilities/json/Common.hpp>

namespace hipdnn_data_sdk::data_objects
{

// NOLINTNEXTLINE(readability-identifier-naming)
inline void to_json(nlohmann::json& sdpaJson, const SdpaBackwardAttributes& sdpa)
{
    auto& inputs = sdpaJson["inputs"] = {};
    auto& outputs = sdpaJson["outputs"] = {};
    auto& attributes = sdpaJson["attributes"] = {};

    // Required input tensor UIDs
    inputs["q_tensor_uid"] = sdpa.q_tensor_uid();
    inputs["k_tensor_uid"] = sdpa.k_tensor_uid();
    inputs["v_tensor_uid"] = sdpa.v_tensor_uid();
    inputs["o_tensor_uid"] = sdpa.o_tensor_uid();
    inputs["do_tensor_uid"] = sdpa.do_tensor_uid();
    inputs["stats_tensor_uid"] = sdpa.stats_tensor_uid();

    // Optional input tensor UIDs
    inputs["scale_tensor_uid"] = sdpa.scale_tensor_uid();
    inputs["attn_mask_tensor_uid"] = sdpa.attn_mask_tensor_uid();
    inputs["seq_len_q_tensor_uid"] = sdpa.seq_len_q_tensor_uid();
    inputs["seq_len_kv_tensor_uid"] = sdpa.seq_len_kv_tensor_uid();
    inputs["seed_tensor_uid"] = sdpa.seed_tensor_uid();
    inputs["offset_tensor_uid"] = sdpa.offset_tensor_uid();
    inputs["dropout_mask_tensor_uid"] = sdpa.dropout_mask_tensor_uid();
    inputs["dropout_scale_tensor_uid"] = sdpa.dropout_scale_tensor_uid();
    inputs["dropout_scale_inv_tensor_uid"] = sdpa.dropout_scale_inv_tensor_uid();

    // Required output tensor UIDs
    outputs["dq_tensor_uid"] = sdpa.dq_tensor_uid();
    outputs["dk_tensor_uid"] = sdpa.dk_tensor_uid();
    outputs["dv_tensor_uid"] = sdpa.dv_tensor_uid();

    // Optional output tensor UIDs
    outputs["dbias_tensor_uid"] = sdpa.dbias_tensor_uid();

    // Boolean flags
    attributes["alibi_mask"] = sdpa.alibi_mask();
    attributes["padding_mask"] = sdpa.padding_mask();
    attributes["causal_mask"] = sdpa.causal_mask();
    attributes["causal_mask_bottom_right"] = sdpa.causal_mask_bottom_right();

    // Scalar attributes
    attributes["dropout_probability"] = sdpa.dropout_probability();
    attributes["attn_scale_value"] = sdpa.attn_scale_value();
    attributes["left_bound"] = sdpa.left_bound();
    attributes["right_bound"] = sdpa.right_bound();

    // Enum attributes
    attributes["diagonal_alignment"] = sdpa.diagonal_alignment();
}

}
namespace hipdnn_data_sdk::json
{
template <>
inline auto to<data_objects::SdpaBackwardAttributes>(flatbuffers::FlatBufferBuilder& builder,
                                                     const nlohmann::json& entry)
{
    auto& inputs = entry.at("inputs");
    auto& outputs = entry.at("outputs");
    auto& attributes = entry.at("attributes");

    return data_objects::CreateSdpaBackwardAttributes(
        builder,
        // Required input tensor UIDs
        inputs.at("q_tensor_uid").get<int64_t>(),
        inputs.at("k_tensor_uid").get<int64_t>(),
        inputs.at("v_tensor_uid").get<int64_t>(),
        inputs.at("o_tensor_uid").get<int64_t>(),
        inputs.at("do_tensor_uid").get<int64_t>(),
        inputs.at("stats_tensor_uid").get<int64_t>(),
        // Required output tensor UIDs
        outputs.at("dq_tensor_uid").get<int64_t>(),
        outputs.at("dk_tensor_uid").get<int64_t>(),
        outputs.at("dv_tensor_uid").get<int64_t>(),
        // Optional input tensor UIDs
        inputs.at("scale_tensor_uid").get<std::optional<int64_t>>(),
        inputs.at("attn_mask_tensor_uid").get<std::optional<int64_t>>(),
        inputs.at("seq_len_q_tensor_uid").get<std::optional<int64_t>>(),
        inputs.at("seq_len_kv_tensor_uid").get<std::optional<int64_t>>(),
        inputs.at("seed_tensor_uid").get<std::optional<int64_t>>(),
        inputs.at("offset_tensor_uid").get<std::optional<int64_t>>(),
        inputs.at("dropout_mask_tensor_uid").get<std::optional<int64_t>>(),
        inputs.at("dropout_scale_tensor_uid").get<std::optional<int64_t>>(),
        inputs.at("dropout_scale_inv_tensor_uid").get<std::optional<int64_t>>(),
        // Optional output tensor UIDs
        outputs.at("dbias_tensor_uid").get<std::optional<int64_t>>(),
        // Boolean flags
        attributes.at("alibi_mask").get<bool>(),
        attributes.at("padding_mask").get<bool>(),
        attributes.at("causal_mask").get<bool>(),
        attributes.at("causal_mask_bottom_right").get<bool>(),
        // Scalar attributes
        attributes.at("dropout_probability").get<std::optional<float>>(),
        attributes.at("attn_scale_value").get<std::optional<float>>(),
        attributes.at("left_bound").get<std::optional<int64_t>>(),
        attributes.at("right_bound").get<std::optional<int64_t>>(),
        // Enum attributes
        attributes.at("diagonal_alignment").get<data_objects::DiagonalAlignment>());
}

}
