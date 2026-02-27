// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include <hipdnn_frontend/detail/EngineOverrideConfig.hpp>
#include <hipdnn_frontend/node/ConvolutionDgradNode.hpp>
#include <hipdnn_frontend/node/ConvolutionFpropNode.hpp>
#include <hipdnn_frontend/node/ConvolutionWgradNode.hpp>
#include <hipdnn_frontend/node/Node.hpp>

#include <optional>

namespace hipdnn_frontend::engine_override
{

/// Information about an operation extracted from a graph, suitable for
/// constructing an OperationRule for engine override config persistence.
struct OperationInfo
{
    std::string op; ///< Operation name (e.g. "conv_fprop", "conv_dgrad", "conv_wgrad")
    std::vector<TensorPattern> tensors; ///< Tensor dim/stride patterns
};

/// Walk the graph using the node visitor to extract operation info (op name +
/// tensor dims/strides) from the first convolution node found.
/// Returns nullopt when no convolution node is present in the graph.
inline std::optional<OperationInfo> extractOperationInfo(const graph::INode& root)
{
    std::optional<OperationInfo> result;

    root.visit([&result](const graph::INode& node) {
        if(result.has_value())
        {
            return;
        }

        auto makeTensorPatterns
            = [](const std::vector<std::shared_ptr<graph::TensorAttributes>>& attrs) {
                  std::vector<TensorPattern> patterns;
                  for(const auto& attr : attrs)
                  {
                      TensorPattern pat;
                      pat.dim = attr->get_dim();
                      pat.stride = attr->get_stride();
                      patterns.push_back(std::move(pat));
                  }
                  return patterns;
              };

        if(const auto* fprop = dynamic_cast<const graph::ConvolutionFpropNode*>(&node))
        {
            OperationInfo info;
            info.op = "conv_fprop";
            info.tensors = makeTensorPatterns({fprop->attributes.get_x(), fprop->attributes.get_w()});
            result = std::move(info);
        }
        else if(const auto* dgrad = dynamic_cast<const graph::ConvolutionDgradNode*>(&node))
        {
            OperationInfo info;
            info.op = "conv_dgrad";
            info.tensors
                = makeTensorPatterns({dgrad->attributes.get_dy(), dgrad->attributes.get_w()});
            result = std::move(info);
        }
        else if(const auto* wgrad = dynamic_cast<const graph::ConvolutionWgradNode*>(&node))
        {
            OperationInfo info;
            info.op = "conv_wgrad";
            info.tensors
                = makeTensorPatterns({wgrad->attributes.get_x(), wgrad->attributes.get_dy()});
            result = std::move(info);
        }
    });

    return result;
}

/// Walk the graph using the node visitor to find the first convolution operation
/// and return the preferred engine ID from the lazily-loaded engine override config
/// (pointed to by HIPDNN_ENGINE_OVERRIDE_FILE).
///
/// Returns nullopt when:
/// - no convolution node is present in the graph,
/// - no rule in the config matches the convolution's tensors, or
/// - JSON support is compiled out (HIPDNN_FRONTEND_SKIP_JSON_LIB defined).
inline std::optional<int64_t> getPreferredIdFromOverrideConfig(const graph::INode& root)
{
    std::optional<int64_t> result;

    root.visit([&result](const graph::INode& node) {
        if(result.has_value())
        {
            return;
        }
        if(const auto* fprop = dynamic_cast<const graph::ConvolutionFpropNode*>(&node))
        {
            result = checkEngineOverride("conv_fprop",
                                         {fprop->attributes.get_x(), fprop->attributes.get_w()});
        }
        else if(const auto* dgrad = dynamic_cast<const graph::ConvolutionDgradNode*>(&node))
        {
            result = checkEngineOverride("conv_dgrad",
                                         {dgrad->attributes.get_dy(), dgrad->attributes.get_w()});
        }
        else if(const auto* wgrad = dynamic_cast<const graph::ConvolutionWgradNode*>(&node))
        {
            result = checkEngineOverride("conv_wgrad",
                                         {wgrad->attributes.get_x(), wgrad->attributes.get_dy()});
        }
    });

    return result;
}

} // namespace hipdnn_frontend::engine_override
