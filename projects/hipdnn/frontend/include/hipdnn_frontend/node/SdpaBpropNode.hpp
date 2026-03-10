// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "Node.hpp"
#include <hipdnn_data_sdk/data_objects/graph_generated.h>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_frontend/attributes/SdpaBackwardAttributes.hpp>
#include <hipdnn_frontend/node/detail/Utilities.hpp>

namespace hipdnn_frontend::graph
{

/**
 * @class SdpaBpropNode
 * @brief Node implementing scaled dot-product attention backward pass
 *
 * Computes gradients dQ, dK, dV given:
 * - Forward inputs Q, K, V, O
 * - Upstream gradient dO
 * - Softmax statistics (logsumexp) from the forward pass
 *
 * The backward pass uses the flash attention algorithm to compute:
 *   dV = softmax(Q * K^T / sqrt(d_k))^T * dO
 *   dP = dO * V^T         (where P = softmax(Q * K^T / sqrt(d_k)))
 *   dS = P * (dP - rowsum(dO * O))
 *   dQ = dS * K / sqrt(d_k)
 *   dK = dS^T * Q / sqrt(d_k)
 */
class SdpaBpropNode : public BaseNode<SdpaBpropNode>
{
public:
    SdpaBackwardAttributes attributes;

    SdpaBpropNode(SdpaBackwardAttributes&& sdpaAttrs, const GraphAttributes& graphAttrs)
        : BaseNode(graphAttrs)
        , attributes(std::move(sdpaAttrs))
    {
    }

    Error pre_validate_node() const override
    {
        const auto q = attributes.get_q();
        const auto k = attributes.get_k();
        const auto v = attributes.get_v();
        const auto o = attributes.get_o();
        const auto dOut = attributes.get_do();
        const auto stats = attributes.get_stats();
        const auto dq = attributes.get_dq();
        const auto dk = attributes.get_dk();
        const auto dv = attributes.get_dv();

        // Validate required input tensors
        HIPDNN_RETURN_IF_FALSE(
            q, ErrorCode::ATTRIBUTE_NOT_SET, std::string("SdpaBpropNode missing Q input"));
        HIPDNN_RETURN_IF_FALSE(
            k, ErrorCode::ATTRIBUTE_NOT_SET, std::string("SdpaBpropNode missing K input"));
        HIPDNN_RETURN_IF_FALSE(
            v, ErrorCode::ATTRIBUTE_NOT_SET, std::string("SdpaBpropNode missing V input"));
        HIPDNN_RETURN_IF_FALSE(
            o, ErrorCode::ATTRIBUTE_NOT_SET, std::string("SdpaBpropNode missing O input"));
        HIPDNN_RETURN_IF_FALSE(
            dOut, ErrorCode::ATTRIBUTE_NOT_SET, std::string("SdpaBpropNode missing dO input"));
        HIPDNN_RETURN_IF_FALSE(
            stats, ErrorCode::ATTRIBUTE_NOT_SET, std::string("SdpaBpropNode missing Stats input"));

        // Validate required output tensors
        HIPDNN_RETURN_IF_FALSE(
            dq, ErrorCode::ATTRIBUTE_NOT_SET, std::string("SdpaBpropNode missing dQ output"));
        HIPDNN_RETURN_IF_FALSE(
            dk, ErrorCode::ATTRIBUTE_NOT_SET, std::string("SdpaBpropNode missing dK output"));
        HIPDNN_RETURN_IF_FALSE(
            dv, ErrorCode::ATTRIBUTE_NOT_SET, std::string("SdpaBpropNode missing dV output"));

        const auto& qDims = q->get_dim();
        const auto& kDims = k->get_dim();
        const auto& vDims = v->get_dim();

        // Rule 1: Q, K, V must be exactly rank-4
        const size_t reqRank = 4;
        HIPDNN_RETURN_IF_NE(qDims.size(),
                            reqRank,
                            ErrorCode::INVALID_VALUE,
                            "SdpaBpropNode: Query tensor Q must be rank-4, got rank="
                                + std::to_string(qDims.size()));
        HIPDNN_RETURN_IF_NE(kDims.size(),
                            reqRank,
                            ErrorCode::INVALID_VALUE,
                            "SdpaBpropNode: Key tensor K must be rank-4, got rank="
                                + std::to_string(kDims.size()));
        HIPDNN_RETURN_IF_NE(vDims.size(),
                            reqRank,
                            ErrorCode::INVALID_VALUE,
                            "SdpaBpropNode: Value tensor V must be rank-4, got rank="
                                + std::to_string(vDims.size()));

        // Rule 2: batch size consistency
        HIPDNN_RETURN_IF_NE(qDims[0],
                            kDims[0],
                            ErrorCode::INVALID_VALUE,
                            "SdpaBpropNode: batch size mismatch between Q and K: "
                                + std::to_string(qDims[0]) + " vs " + std::to_string(kDims[0]));
        HIPDNN_RETURN_IF_NE(qDims[0],
                            vDims[0],
                            ErrorCode::INVALID_VALUE,
                            "SdpaBpropNode: batch size mismatch between Q and V: "
                                + std::to_string(qDims[0]) + " vs " + std::to_string(vDims[0]));

        // Rule 3: head_dim: Q[-1] == K[-1]
        const auto headDimQ = qDims[3];
        const auto headDimK = kDims[3];
        HIPDNN_RETURN_IF_NE(headDimQ,
                            headDimK,
                            ErrorCode::INVALID_VALUE,
                            "SdpaBpropNode: head_dim mismatch between Q and K: "
                                + std::to_string(headDimQ) + " vs " + std::to_string(headDimK));

        // Rule 4: seq_kv: K[-2] == V[-2]
        const auto seqKvK = kDims[2];
        const auto seqKvV = vDims[2];
        HIPDNN_RETURN_IF_NE(seqKvK,
                            seqKvV,
                            ErrorCode::INVALID_VALUE,
                            "SdpaBpropNode: seq_kv mismatch between K and V: "
                                + std::to_string(seqKvK) + " vs " + std::to_string(seqKvV));

        // Rule 5: num_kv_heads: K[-3] == V[-3]; num_heads % num_kv_heads == 0 (GQA/MQA)
        const auto numHeads = qDims[1];
        const auto numKvHeadsK = kDims[1];
        const auto numKvHeadsV = vDims[1];
        HIPDNN_RETURN_IF_NE(numKvHeadsK,
                            numKvHeadsV,
                            ErrorCode::INVALID_VALUE,
                            "SdpaBpropNode: num_kv_heads mismatch between K and V: "
                                + std::to_string(numKvHeadsK) + " vs "
                                + std::to_string(numKvHeadsV));
        HIPDNN_RETURN_IF_TRUE(numHeads % numKvHeadsK != 0,
                              ErrorCode::INVALID_VALUE,
                              "SdpaBpropNode: num_heads must be divisible by num_kv_heads for "
                              "GQA/MQA. num_heads="
                                  + std::to_string(numHeads)
                                  + ", num_kv_heads=" + std::to_string(numKvHeadsK));

        // Rule 6: dO must match O shape [batch, num_heads, seq_q, head_dim_v]
        const auto& oDims = o->get_dim();
        if(!oDims.empty())
        {
            const auto& dOutDims = dOut->get_dim();
            if(!dOutDims.empty())
            {
                HIPDNN_RETURN_IF_NE(dOutDims.size(),
                                    reqRank,
                                    ErrorCode::INVALID_VALUE,
                                    "SdpaBpropNode: dO must be rank-4, got rank="
                                        + std::to_string(dOutDims.size()));
                HIPDNN_RETURN_IF_NE(dOutDims[0],
                                    oDims[0],
                                    ErrorCode::INVALID_VALUE,
                                    "SdpaBpropNode: dO batch mismatch with O");
                HIPDNN_RETURN_IF_NE(dOutDims[1],
                                    oDims[1],
                                    ErrorCode::INVALID_VALUE,
                                    "SdpaBpropNode: dO num_heads mismatch with O");
                HIPDNN_RETURN_IF_NE(dOutDims[2],
                                    oDims[2],
                                    ErrorCode::INVALID_VALUE,
                                    "SdpaBpropNode: dO seq_q mismatch with O");
                HIPDNN_RETURN_IF_NE(dOutDims[3],
                                    oDims[3],
                                    ErrorCode::INVALID_VALUE,
                                    "SdpaBpropNode: dO head_dim mismatch with O");
            }
        }

        // Rule 7: Optional attention scale must be a scalar tensor
        const auto scale = attributes.get_scale();
        if(scale)
        {
            HIPDNN_CHECK_ERROR(detail::validateScalarParameter(scale, "SCALE tensor"));
        }

        return {};
    }

    Error infer_properties_node() override
    {
        const auto q = attributes.get_q();
        const auto k = attributes.get_k();
        const auto v = attributes.get_v();
        const auto dq = attributes.get_dq();
        const auto dk = attributes.get_dk();
        const auto dv = attributes.get_dv();

        HIPDNN_RETURN_IF_FALSE(
            q, ErrorCode::ATTRIBUTE_NOT_SET, std::string("SdpaBpropNode missing Q input"));
        HIPDNN_RETURN_IF_FALSE(
            k, ErrorCode::ATTRIBUTE_NOT_SET, std::string("SdpaBpropNode missing K input"));
        HIPDNN_RETURN_IF_FALSE(
            v, ErrorCode::ATTRIBUTE_NOT_SET, std::string("SdpaBpropNode missing V input"));
        HIPDNN_RETURN_IF_FALSE(
            dq, ErrorCode::ATTRIBUTE_NOT_SET, std::string("SdpaBpropNode missing dQ output"));
        HIPDNN_RETURN_IF_FALSE(
            dk, ErrorCode::ATTRIBUTE_NOT_SET, std::string("SdpaBpropNode missing dK output"));
        HIPDNN_RETURN_IF_FALSE(
            dv, ErrorCode::ATTRIBUTE_NOT_SET, std::string("SdpaBpropNode missing dV output"));

        HIPDNN_CHECK_ERROR(attributes.fill_from_context(graph_attributes));

        // dQ has same shape as Q: [batch, num_heads, seq_q, head_dim]
        if(dq->get_dim().empty())
        {
            dq->set_dim(q->get_dim());
        }
        if(dq->get_stride().empty())
        {
            dq->set_stride(hipdnn_data_sdk::utilities::generateStrides(dq->get_dim()));
        }

        // dK has same shape as K: [batch, num_kv_heads, seq_kv, head_dim]
        if(dk->get_dim().empty())
        {
            dk->set_dim(k->get_dim());
        }
        if(dk->get_stride().empty())
        {
            dk->set_stride(hipdnn_data_sdk::utilities::generateStrides(dk->get_dim()));
        }

        // dV has same shape as V: [batch, num_kv_heads, seq_kv, head_dim_v]
        if(dv->get_dim().empty())
        {
            dv->set_dim(v->get_dim());
        }
        if(dv->get_stride().empty())
        {
            dv->set_stride(hipdnn_data_sdk::utilities::generateStrides(dv->get_dim()));
        }

        return {};
    }

    flatbuffers::Offset<hipdnn_data_sdk::data_objects::Node>
        pack_node(flatbuffers::FlatBufferBuilder& builder) const override
    {
        return hipdnn_data_sdk::data_objects::CreateNodeDirect(
            builder,
            attributes.get_name().c_str(),
            toSdkType(attributes.compute_data_type),
            hipdnn_data_sdk::data_objects::NodeAttributes::SdpaBackwardAttributes,
            attributes.pack_attributes(builder).Union());
    }
};
} // namespace hipdnn_frontend::graph
