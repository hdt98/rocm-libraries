// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

/**
 * @file SdpaBackwardAttributes.hpp
 * @brief Attributes for scaled dot-product attention (SDPA) backward pass
 *
 * This file defines the SdpaBackwardAttributes class used to configure
 * the backward pass (gradient computation) of scaled dot-product attention.
 */

#pragma once

#include "Attributes.hpp"
#include "TensorAttributes.hpp"
#include <hipdnn_data_sdk/data_objects/sdpa_backward_attributes_generated.h>
#include <hipdnn_frontend/Types.hpp>
#include <memory>
#include <optional>
#include <unordered_map>

namespace hipdnn_frontend::graph
{

/**
 * @class SdpaBackwardAttributes
 * @brief Configuration attributes for scaled dot-product attention backward pass
 *
 * SdpaBackwardAttributes configures the backward pass of scaled dot-product
 * attention, computing gradients with respect to Q, K, and V:
 * @code
 * Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
 * @endcode
 *
 * **Required inputs:**
 * - Q: Query tensor from forward pass [B, H, S_q, D]
 * - K: Key tensor from forward pass [B, H, S_kv, D]
 * - V: Value tensor from forward pass [B, H, S_kv, D_v]
 * - O: Output tensor from forward pass [B, H, S_q, D_v]
 * - dO: Gradient of loss w.r.t. output [B, H, S_q, D_v]
 * - Stats: Softmax statistics from forward pass [B, H, S_q, 1]
 *
 * **Outputs:**
 * - dQ: Gradient w.r.t. query [B, H, S_q, D]
 * - dK: Gradient w.r.t. key [B, H, S_kv, D]
 * - dV: Gradient w.r.t. value [B, H, S_kv, D_v]
 *
 * **Optional features:**
 * - Causal masking and diagonal band bounds
 * - Additive attention bias gradient (dBias)
 * - Dropout (probability + seed/offset tensors, or explicit mask)
 * - ALiBi positional encoding
 * - Attention scale override (attn_scale_value)
 *
 * @code{.cpp}
 * SdpaBackwardAttributes attr;
 * attr.set_attn_scale_value(1.0f / std::sqrt(static_cast<float>(d_k)))
 *     .set_causal_mask(true);
 *
 * auto [dq, dk, dv] = graph.sdpa_backward(q, k, v, o, do_, stats, attr);
 * @endcode
 *
 * @see SdpaAttributes for the forward pass
 */
class SdpaBackwardAttributes : public Attributes<SdpaBackwardAttributes>
{
public:
    enum class InputNames
    {
        Q = 0,
        K = 1,
        V = 2,
        O = 3,
        DO = 4, // Gradient of output (dO)
        STATS = 5, // Softmax statistics from forward pass
        SCALE = 6, // Attention scale tensor (Attn_scale)
        ATTN_MASK = 7, // Additive attention bias (Bias)
        SEQ_LEN_Q = 8,
        SEQ_LEN_KV = 9,
        SEED = 10, // Dropout seed
        OFFSET = 11, // Dropout offset
        DROPOUT_MASK = 12,
        DROPOUT_SCALE = 13,
        DROPOUT_SCALE_INV = 14,
    };
    typedef InputNames input_names; // NOLINT(readability-identifier-naming)

    enum class OutputNames
    {
        DQ = 0, // Gradient w.r.t. query
        DK = 1, // Gradient w.r.t. key
        DV = 2, // Gradient w.r.t. value
        DBIAS = 3, // Gradient w.r.t. additive attention bias (optional)
    };
    typedef OutputNames output_names; // NOLINT(readability-identifier-naming)

    std::unordered_map<InputNames, std::shared_ptr<TensorAttributes>> inputs;
    std::unordered_map<OutputNames, std::shared_ptr<TensorAttributes>> outputs;

    // NOLINTBEGIN(readability-identifier-naming)
    // Boolean flags
    bool alibi_mask = false;
    bool padding_mask = false;
    bool causal_mask = false; // Deprecated
    bool causal_mask_bottom_right = false; // Deprecated

    // Scalar attributes
    std::optional<float> dropout_probability;
    std::optional<float> attn_scale_value;
    std::optional<int64_t> left_bound;
    std::optional<int64_t> right_bound;

    // Enum attributes
    DiagonalAlignment diagonal_alignment = DiagonalAlignment::TOP_LEFT;
    // NOLINTEND(readability-identifier-naming)

    // -- Input tensor getters --

    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_q() const
    {
        return getInput(InputNames::Q);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_k() const
    {
        return getInput(InputNames::K);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_v() const
    {
        return getInput(InputNames::V);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_o() const
    {
        return getInput(InputNames::O);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_do() const
    {
        return getInput(InputNames::DO);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_stats() const
    {
        return getInput(InputNames::STATS);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_scale() const
    {
        return getInput(InputNames::SCALE);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_attn_mask() const
    {
        return getInput(InputNames::ATTN_MASK);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_seq_len_q() const
    {
        return getInput(InputNames::SEQ_LEN_Q);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_seq_len_kv() const
    {
        return getInput(InputNames::SEQ_LEN_KV);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_seed() const
    {
        return getInput(InputNames::SEED);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_offset() const
    {
        return getInput(InputNames::OFFSET);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_dropout_mask() const
    {
        return getInput(InputNames::DROPOUT_MASK);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_dropout_scale() const
    {
        return getInput(InputNames::DROPOUT_SCALE);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_dropout_scale_inv() const
    {
        return getInput(InputNames::DROPOUT_SCALE_INV);
    }

    // -- Output tensor getters --

    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_dq() const
    {
        return getOutput(OutputNames::DQ);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_dk() const
    {
        return getOutput(OutputNames::DK);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_dv() const
    {
        return getOutput(OutputNames::DV);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_dbias() const
    {
        return getOutput(OutputNames::DBIAS);
    }

    // -- Input tensor setters --

    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_q(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::Q, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_q(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::Q, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_k(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::K, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_k(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::K, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_v(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::V, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_v(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::V, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_o(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::O, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_o(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::O, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_do(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::DO, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_do(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::DO, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_stats(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::STATS, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_stats(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::STATS, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_scale(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::SCALE, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_scale(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::SCALE, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_attn_mask(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::ATTN_MASK, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_attn_mask(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::ATTN_MASK, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_seq_len_q(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::SEQ_LEN_Q, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_seq_len_q(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::SEQ_LEN_Q, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_seq_len_kv(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::SEQ_LEN_KV, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_seq_len_kv(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::SEQ_LEN_KV, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_seed(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::SEED, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_seed(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::SEED, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_offset(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::OFFSET, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_offset(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::OFFSET, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_dropout_mask(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::DROPOUT_MASK, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_dropout_mask(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::DROPOUT_MASK, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_dropout_scale(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::DROPOUT_SCALE, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_dropout_scale(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::DROPOUT_SCALE, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_dropout_scale_inv(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::DROPOUT_SCALE_INV, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_dropout_scale_inv(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::DROPOUT_SCALE_INV, std::move(value));
    }

    // -- Output tensor setters --

    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_dq(const std::shared_ptr<TensorAttributes>& value)
    {
        return setOutput(OutputNames::DQ, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_dq(std::shared_ptr<TensorAttributes>&& value)
    {
        return setOutput(OutputNames::DQ, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_dk(const std::shared_ptr<TensorAttributes>& value)
    {
        return setOutput(OutputNames::DK, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_dk(std::shared_ptr<TensorAttributes>&& value)
    {
        return setOutput(OutputNames::DK, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_dv(const std::shared_ptr<TensorAttributes>& value)
    {
        return setOutput(OutputNames::DV, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_dv(std::shared_ptr<TensorAttributes>&& value)
    {
        return setOutput(OutputNames::DV, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_dbias(const std::shared_ptr<TensorAttributes>& value)
    {
        return setOutput(OutputNames::DBIAS, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_dbias(std::shared_ptr<TensorAttributes>&& value)
    {
        return setOutput(OutputNames::DBIAS, std::move(value));
    }

    // -- Scalar/flag setters --

    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_alibi_mask(bool value)
    {
        alibi_mask = value;
        return *this;
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_padding_mask(bool value)
    {
        padding_mask = value;
        return *this;
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_causal_mask(bool value)
    {
        causal_mask = value;
        return *this;
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_causal_mask_bottom_right(bool value)
    {
        causal_mask_bottom_right = value;
        return *this;
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_dropout(float probability,
                                        const std::shared_ptr<TensorAttributes>& seed,
                                        const std::shared_ptr<TensorAttributes>& offset)
    {
        dropout_probability = probability;
        setInput(InputNames::SEED, seed);
        setInput(InputNames::OFFSET, offset);
        return *this;
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_attn_scale_value(float value)
    {
        attn_scale_value = value;
        return *this;
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_diagonal_band_left_bound(int64_t value)
    {
        left_bound = value;
        return *this;
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_diagonal_band_right_bound(int64_t value)
    {
        right_bound = value;
        return *this;
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    SdpaBackwardAttributes& set_diagonal_alignment(DiagonalAlignment value)
    {
        diagonal_alignment = value;
        return *this;
    }

    flatbuffers::Offset<hipdnn_data_sdk::data_objects::SdpaBackwardAttributes>
        pack_attributes(flatbuffers::FlatBufferBuilder& builder) const // NOLINT
    {
        const auto optUid
            = [](const std::shared_ptr<TensorAttributes>& t) -> flatbuffers::Optional<int64_t> {
            return t ? flatbuffers::Optional<int64_t>(t->get_uid()) : flatbuffers::nullopt;
        };

        return hipdnn_data_sdk::data_objects::CreateSdpaBackwardAttributes(
            builder,
            get_q()->get_uid(),
            get_k()->get_uid(),
            get_v()->get_uid(),
            get_o()->get_uid(),
            get_do()->get_uid(),
            get_stats()->get_uid(),
            get_dq()->get_uid(),
            get_dk()->get_uid(),
            get_dv()->get_uid(),
            optUid(get_scale()),
            optUid(get_attn_mask()),
            optUid(get_seq_len_q()),
            optUid(get_seq_len_kv()),
            optUid(get_seed()),
            optUid(get_offset()),
            optUid(get_dropout_mask()),
            optUid(get_dropout_scale()),
            optUid(get_dropout_scale_inv()),
            optUid(get_dbias()),
            alibi_mask,
            padding_mask,
            causal_mask,
            causal_mask_bottom_right,
            dropout_probability.has_value() ? flatbuffers::Optional<float>(*dropout_probability)
                                            : flatbuffers::nullopt,
            attn_scale_value.has_value() ? flatbuffers::Optional<float>(*attn_scale_value)
                                         : flatbuffers::nullopt,
            left_bound.has_value() ? flatbuffers::Optional<int64_t>(*left_bound)
                                   : flatbuffers::nullopt,
            right_bound.has_value() ? flatbuffers::Optional<int64_t>(*right_bound)
                                    : flatbuffers::nullopt,
            toSdkType(diagonal_alignment));
    }

    static SdpaBackwardAttributes fromFlatBuffer(
        const hipdnn_data_sdk::data_objects::SdpaBackwardAttributes* fb,
        const std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>>& tensorMap)
    {
        SdpaBackwardAttributes attr;

        attr.set_q(tensorMap.at(fb->q_tensor_uid()));
        attr.set_k(tensorMap.at(fb->k_tensor_uid()));
        attr.set_v(tensorMap.at(fb->v_tensor_uid()));
        attr.set_o(tensorMap.at(fb->o_tensor_uid()));
        attr.set_do(tensorMap.at(fb->do_tensor_uid()));
        attr.set_stats(tensorMap.at(fb->stats_tensor_uid()));
        attr.set_dq(tensorMap.at(fb->dq_tensor_uid()));
        attr.set_dk(tensorMap.at(fb->dk_tensor_uid()));
        attr.set_dv(tensorMap.at(fb->dv_tensor_uid()));

        if(fb->scale_tensor_uid().has_value())
        {
            attr.set_scale(tensorMap.at(fb->scale_tensor_uid().value()));
        }
        if(fb->attn_mask_tensor_uid().has_value())
        {
            attr.set_attn_mask(tensorMap.at(fb->attn_mask_tensor_uid().value()));
        }
        if(fb->seq_len_q_tensor_uid().has_value())
        {
            attr.set_seq_len_q(tensorMap.at(fb->seq_len_q_tensor_uid().value()));
        }
        if(fb->seq_len_kv_tensor_uid().has_value())
        {
            attr.set_seq_len_kv(tensorMap.at(fb->seq_len_kv_tensor_uid().value()));
        }
        if(fb->seed_tensor_uid().has_value())
        {
            attr.set_seed(tensorMap.at(fb->seed_tensor_uid().value()));
        }
        if(fb->offset_tensor_uid().has_value())
        {
            attr.set_offset(tensorMap.at(fb->offset_tensor_uid().value()));
        }
        if(fb->dropout_mask_tensor_uid().has_value())
        {
            attr.set_dropout_mask(tensorMap.at(fb->dropout_mask_tensor_uid().value()));
        }
        if(fb->dropout_scale_tensor_uid().has_value())
        {
            attr.set_dropout_scale(tensorMap.at(fb->dropout_scale_tensor_uid().value()));
        }
        if(fb->dropout_scale_inv_tensor_uid().has_value())
        {
            attr.set_dropout_scale_inv(tensorMap.at(fb->dropout_scale_inv_tensor_uid().value()));
        }
        if(fb->dbias_tensor_uid().has_value())
        {
            attr.set_dbias(tensorMap.at(fb->dbias_tensor_uid().value()));
        }

        attr.alibi_mask = fb->alibi_mask();
        attr.padding_mask = fb->padding_mask();
        attr.causal_mask = fb->causal_mask();
        attr.causal_mask_bottom_right = fb->causal_mask_bottom_right();

        if(fb->dropout_probability().has_value())
        {
            attr.dropout_probability = fb->dropout_probability().value();
        }
        if(fb->attn_scale_value().has_value())
        {
            attr.attn_scale_value = fb->attn_scale_value().value();
        }
        if(fb->left_bound().has_value())
        {
            attr.left_bound = fb->left_bound().value();
        }
        if(fb->right_bound().has_value())
        {
            attr.right_bound = fb->right_bound().value();
        }

        attr.diagonal_alignment = fromSdkType(fb->diagonal_alignment());

        return attr;
    }
};

typedef SdpaBackwardAttributes Sdpa_backward_attributes; // NOLINT(readability-identifier-naming)
} // namespace hipdnn_frontend::graph
