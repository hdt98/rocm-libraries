// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file RMSNormBackwardAttributes.hpp
 * @brief Attributes for RMSNormBackward operations
 *
 * This file defines the RMSNormBackwardAttributes class for configuring
 * RMSNormBackward operations in hipDNN computational graphs.
 */

#pragma once

#include "Attributes.hpp"
#include "TensorAttributes.hpp"
#include <hipdnn_data_sdk/data_objects/rmsnorm_backward_attributes_generated.h>
#include <memory>
#include <unordered_map>

namespace hipdnn_frontend::graph
{

/**
 * @class RMSNormBackwardAttributes
 * @brief Configuration for RMSNormBackward operations
 */
class RMSNormBackwardAttributes : public Attributes<RMSNormBackwardAttributes>
{
public:
    /// Input tensor identifiers
    enum class InputNames
    {
        DY = 0,
        X = 1,
        SCALE = 2,
        INV_RMS = 3
    };
    typedef InputNames input_names; ///< @brief Type alias for InputNames

    /// Output tensor identifiers
    enum class OutputNames
    {
        DX = 0,
        DSCALE = 1,
        DBIAS = 2
    };
    typedef OutputNames output_names; ///< @brief Type alias for OutputNames

    std::unordered_map<InputNames, std::shared_ptr<TensorAttributes>> inputs; ///< Input tensors
    std::unordered_map<OutputNames, std::shared_ptr<TensorAttributes>> outputs; ///< Output tensors

    /// @brief Get the dy input tensor
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_dy() const
    {
        return getInput(InputNames::DY);
    }
    /// @brief Get the x input tensor
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_x() const
    {
        return getInput(InputNames::X);
    }
    /// @brief Get the scale input tensor
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_scale() const
    {
        return getInput(InputNames::SCALE);
    }
    /// @brief Get the dx output tensor
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_dx() const
    {
        return getOutput(OutputNames::DX);
    }
    /// @brief Get the inv_rms input tensor (optional)
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_inv_rms() const
    {
        return getInput(InputNames::INV_RMS);
    }
    /// @brief Get the dscale output tensor
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_dscale() const
    {
        return getOutput(OutputNames::DSCALE);
    }
    /// @brief Get the dbias output tensor (optional)
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_dbias() const
    {
        return getOutput(OutputNames::DBIAS);
    }

    /// @brief Set the dy input tensor (move)
    // NOLINTNEXTLINE(readability-identifier-naming)
    RMSNormBackwardAttributes& set_dy(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::DY, std::move(value));
    }
    /// @brief Set the dy input tensor (copy)
    // NOLINTNEXTLINE(readability-identifier-naming)
    RMSNormBackwardAttributes& set_dy(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::DY, value);
    }
    /// @brief Set the x input tensor (move)
    // NOLINTNEXTLINE(readability-identifier-naming)
    RMSNormBackwardAttributes& set_x(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::X, std::move(value));
    }
    /// @brief Set the x input tensor (copy)
    // NOLINTNEXTLINE(readability-identifier-naming)
    RMSNormBackwardAttributes& set_x(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::X, value);
    }
    /// @brief Set the scale input tensor (move)
    // NOLINTNEXTLINE(readability-identifier-naming)
    RMSNormBackwardAttributes& set_scale(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::SCALE, std::move(value));
    }
    /// @brief Set the scale input tensor (copy)
    // NOLINTNEXTLINE(readability-identifier-naming)
    RMSNormBackwardAttributes& set_scale(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::SCALE, value);
    }
    /// @brief Set the dx output tensor (move)
    // NOLINTNEXTLINE(readability-identifier-naming)
    RMSNormBackwardAttributes& set_dx(std::shared_ptr<TensorAttributes>&& value)
    {
        return setOutput(OutputNames::DX, std::move(value));
    }
    /// @brief Set the dx output tensor (copy)
    // NOLINTNEXTLINE(readability-identifier-naming)
    RMSNormBackwardAttributes& set_dx(const std::shared_ptr<TensorAttributes>& value)
    {
        return setOutput(OutputNames::DX, value);
    }
    /// @brief Set the inv_rms input tensor (move)
    // NOLINTNEXTLINE(readability-identifier-naming)
    RMSNormBackwardAttributes& set_inv_rms(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::INV_RMS, std::move(value));
    }
    /// @brief Set the inv_rms input tensor (copy)
    // NOLINTNEXTLINE(readability-identifier-naming)
    RMSNormBackwardAttributes& set_inv_rms(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::INV_RMS, value);
    }
    /// @brief Set the dscale output tensor (move)
    // NOLINTNEXTLINE(readability-identifier-naming)
    RMSNormBackwardAttributes& set_dscale(std::shared_ptr<TensorAttributes>&& value)
    {
        return setOutput(OutputNames::DSCALE, std::move(value));
    }
    /// @brief Set the dscale output tensor (copy)
    // NOLINTNEXTLINE(readability-identifier-naming)
    RMSNormBackwardAttributes& set_dscale(const std::shared_ptr<TensorAttributes>& value)
    {
        return setOutput(OutputNames::DSCALE, value);
    }
    /// @brief Set the dbias output tensor (move)
    // NOLINTNEXTLINE(readability-identifier-naming)
    RMSNormBackwardAttributes& set_dbias(std::shared_ptr<TensorAttributes>&& value)
    {
        return setOutput(OutputNames::DBIAS, std::move(value));
    }
    /// @brief Set the dbias output tensor (copy)
    // NOLINTNEXTLINE(readability-identifier-naming)
    RMSNormBackwardAttributes& set_dbias(const std::shared_ptr<TensorAttributes>& value)
    {
        return setOutput(OutputNames::DBIAS, value);
    }

    flatbuffers::Offset<hipdnn_data_sdk::data_objects::RMSNormBackwardAttributes>
        pack_attributes(flatbuffers::FlatBufferBuilder& builder) const // NOLINT
    {
        auto invRms = get_inv_rms();
        auto dbias = get_dbias();
        return hipdnn_data_sdk::data_objects::CreateRMSNormBackwardAttributes(
            builder,
            get_dy()->get_uid(),
            get_x()->get_uid(),
            get_scale()->get_uid(),
            invRms ? flatbuffers::Optional<int64_t>(invRms->get_uid()) : flatbuffers::nullopt,
            get_dx()->get_uid(),
            get_dscale()->get_uid(),
            dbias ? flatbuffers::Optional<int64_t>(dbias->get_uid()) : flatbuffers::nullopt);
    }

    static RMSNormBackwardAttributes fromFlatBuffer(
        const hipdnn_data_sdk::data_objects::RMSNormBackwardAttributes* fb,
        const std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>>& tensorMap)
    {
        RMSNormBackwardAttributes attr;

        attr.set_dy(tensorMap.at(fb->dy_tensor_uid()));
        attr.set_x(tensorMap.at(fb->x_tensor_uid()));
        attr.set_scale(tensorMap.at(fb->scale_tensor_uid()));
        attr.set_dx(tensorMap.at(fb->dx_tensor_uid()));
        attr.set_dscale(tensorMap.at(fb->dscale_tensor_uid()));

        if(fb->inv_rms_tensor_uid().has_value())
        {
            attr.set_inv_rms(tensorMap.at(fb->inv_rms_tensor_uid().value()));
        }
        if(fb->dbias_tensor_uid().has_value())
        {
            attr.set_dbias(tensorMap.at(fb->dbias_tensor_uid().value()));
        }

        return attr;
    }
};
} // namespace hipdnn_frontend::graph
