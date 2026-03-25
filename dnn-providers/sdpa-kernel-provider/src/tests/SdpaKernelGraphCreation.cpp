// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "SdpaKernelGraphCreation.hpp"
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>

namespace sdpa_kernel_provider
{

flatbuffers::FlatBufferBuilder
    createValidSdpaFpropGraph(const std::vector<int64_t>& dims,
                              hipdnn_data_sdk::data_objects::DataType dataType,
                              hipdnn_data_sdk::data_objects::DataType computeType,
                              bool withAttnMask,
                              bool withScale,
                              bool withStats,
                              bool alibiMask,
                              bool paddingMask,
                              bool causalMask)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_data_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    int64_t uid = 1;

    auto strides = hipdnn_data_sdk::utilities::generateStrides(dims);

    const auto& qDims = dims;
    const auto& qStrides = strides;
    const auto& kDims = dims;
    const auto& kStrides = strides;
    const auto& vDims = dims;
    const auto& vStrides = strides;
    const auto& oDims = dims;
    const auto& oStrides = strides;

    const auto qUid = uid++;
    tensorAttributes.push_back(hipdnn_data_sdk::data_objects::CreateTensorAttributesDirect(
        builder, qUid, "q", dataType, &qStrides, &qDims));

    const auto kUid = uid++;
    tensorAttributes.push_back(hipdnn_data_sdk::data_objects::CreateTensorAttributesDirect(
        builder, kUid, "k", dataType, &kStrides, &kDims));

    const auto vUid = uid++;
    tensorAttributes.push_back(hipdnn_data_sdk::data_objects::CreateTensorAttributesDirect(
        builder, vUid, "v", dataType, &vStrides, &vDims));

    const auto oUid = uid++;
    tensorAttributes.push_back(hipdnn_data_sdk::data_objects::CreateTensorAttributesDirect(
        builder, oUid, "o", dataType, &oStrides, &oDims));

    flatbuffers::Optional<int64_t> attnMaskUid = flatbuffers::nullopt;
    if(withAttnMask)
    {
        // attn_mask: [batch, num_heads, seq_q, seq_kv]
        std::vector<int64_t> attnMaskDims = {qDims[0], qDims[1], qDims[2], kDims[2]};
        std::vector<int64_t> attnMaskStrides
            = {qDims[1] * qDims[2] * kDims[2], qDims[2] * kDims[2], kDims[2], 1};
        const auto maskUid = uid++;
        tensorAttributes.push_back(hipdnn_data_sdk::data_objects::CreateTensorAttributesDirect(
            builder, maskUid, "attn_mask", dataType, &attnMaskStrides, &attnMaskDims));
        attnMaskUid = flatbuffers::Optional<int64_t>(maskUid);
    }

    flatbuffers::Optional<int64_t> scaleUid = flatbuffers::nullopt;
    if(withScale)
    {
        std::vector<int64_t> passByValueDims = {1};
        hipdnn_data_sdk::data_objects::Float32Value scaleVal(1.0f);
        const auto sUid = uid++;
        tensorAttributes.push_back(hipdnn_data_sdk::data_objects::CreateTensorAttributesDirect(
            builder,
            sUid,
            "scale",
            hipdnn_data_sdk::data_objects::DataType::FLOAT,
            &passByValueDims,
            &passByValueDims,
            false,
            hipdnn_data_sdk::data_objects::TensorValue::Float32Value,
            builder.CreateStruct(scaleVal).Union()));
        scaleUid = flatbuffers::Optional<int64_t>(sUid);
    }

    flatbuffers::Optional<int64_t> statsUid = flatbuffers::nullopt;
    if(withStats)
    {
        // stats: [batch, num_heads, seq_q, 1]
        std::vector<int64_t> statsDims = {qDims[0], qDims[1], qDims[2], 1};
        std::vector<int64_t> statsStrides = {qDims[1] * qDims[2], qDims[2], 1, 1};
        const auto stUid = uid++;
        tensorAttributes.push_back(hipdnn_data_sdk::data_objects::CreateTensorAttributesDirect(
            builder, stUid, "stats", dataType, &statsStrides, &statsDims));
        statsUid = flatbuffers::Optional<int64_t>(stUid);
    }

    auto sdpaAttributes = hipdnn_data_sdk::data_objects::CreateSdpaAttributes(
        builder,
        qUid,
        kUid,
        vUid,
        oUid,
        attnMaskUid,
        scaleUid,
        flatbuffers::nullopt, // seq_len_q_tensor_uid
        flatbuffers::nullopt, // seq_len_kv_tensor_uid
        flatbuffers::nullopt, // seed_tensor_uid
        flatbuffers::nullopt, // offset_tensor_uid
        flatbuffers::nullopt, // dropout_mask_tensor_uid
        flatbuffers::nullopt, // dropout_scale_tensor_uid
        flatbuffers::nullopt, // page_table_k_tensor_uid
        flatbuffers::nullopt, // page_table_v_tensor_uid
        flatbuffers::nullopt, // block_mask_tensor_uid
        flatbuffers::nullopt, // sink_token_tensor_uid
        flatbuffers::nullopt, // descale_q_tensor_uid
        flatbuffers::nullopt, // descale_k_tensor_uid
        flatbuffers::nullopt, // descale_v_tensor_uid
        flatbuffers::nullopt, // descale_s_tensor_uid
        flatbuffers::nullopt, // scale_s_tensor_uid
        flatbuffers::nullopt, // scale_o_tensor_uid
        statsUid,
        flatbuffers::nullopt, // max_tensor_uid
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        alibiMask,
        paddingMask,
        causalMask);

    std::vector<::flatbuffers::Offset<hipdnn_data_sdk::data_objects::Node>> nodes;
    nodes.push_back(hipdnn_data_sdk::data_objects::CreateNodeDirect(
        builder,
        "sdpa_fprop",
        computeType,
        hipdnn_data_sdk::data_objects::NodeAttributes::SdpaAttributes,
        sdpaAttributes.Union()));

    auto graphOffset = hipdnn_data_sdk::data_objects::CreateGraphDirect(
        builder,
        "test",
        computeType,
        hipdnn_data_sdk::data_objects::DataType::HALF,
        hipdnn_data_sdk::data_objects::DataType::BFLOAT16,
        &tensorAttributes,
        &nodes);
    builder.Finish(graphOffset);
    return builder;
}
}
