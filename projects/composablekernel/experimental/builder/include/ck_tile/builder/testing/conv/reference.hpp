// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/builder/testing/testing.hpp"
#include <stdexcept>
#include <vector>

/// This file contains the implementation details for invoking/testing
/// grouped convolution operations using the reference implementation.
/// The main item is the `run()` function, which is the primary way to
/// invoke the reference execution mechanism.
/// The implementation of this file mostly looks like `conv_fwd_ck.hpp`,
/// but its made specific to the reference implementation, which is
/// invoked in a slightly different way.

namespace ck_tile::builder::test {

namespace detail {

/// @brief Concept for checking whether this is the reference convolution
/// implementation.
///
/// This concept is used to tell whether a convolution implementation is
/// likely to be the reference implementation - that is, whether we should
/// invoke it like the reference kernel. This is mainly used with `run()` to
/// differentiate which implementation that should be invoked.
///
/// - SIGNATURE is the operation signature.
/// - Conv is a convolution instance created by the CK Builder API.
/// - InDataType, WeiDataType, OutDataType are the types of the respective tensors.
template <typename Conv,
          auto SIGNATURE,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType>
concept RefConvInstance = requires(Conv& conv,
                                   InDataType* input,
                                   WeiDataType* weight,
                                   OutDataType* output,
                                   int G,
                                   int N,
                                   int K,
                                   int C,
                                   std::vector<long_index_t> input_spatial,
                                   std::vector<long_index_t> filter_spatial,
                                   std::vector<long_index_t> strides,
                                   std::vector<long_index_t> dilations,
                                   std::vector<long_index_t> left_pads,
                                   std::vector<long_index_t> right_pads) {
    requires ValidConvSignature<SIGNATURE>;
    { conv.Run(input, weight, output, G, N, K, C, 
               input_spatial, filter_spatial, strides,
               dilations, left_pads, right_pads) };
};

/// @brief Generic `run` implementation for forward/backwards reference kernels.
///
/// @tparam SIGNATURE The signature of the operation to perform.
///
/// @return std::tuple<bool, float> - whether the problem is supported and
///         kernel execution time (0.0f for reference).
/// @see run()
template <auto SIGNATURE, typename InDataType, typename WeiDataType, typename OutDataType>
[[nodiscard]] RunResult
run(RefConvInstance<SIGNATURE, InDataType, WeiDataType, OutDataType> auto& conv,
    const Args<SIGNATURE>& args,
    InDataType* input,
    WeiDataType* weight,
    OutDataType* output)
{
    // Helper to convert FilterExtent to std::vector
    const auto to_vec = [](const auto& extent) {
        auto arr = extent.template to_array<long_index_t>();
        return std::vector<long_index_t>(arr.begin(), arr.end());
    };

    // TODO: The reference convolution is currently missing a few features.
    // Just throw for now, but regard these as TODO items that should be resolved
    // eventually.

    if(!args.make_input_descriptor().is_packed())
        return RunResult::not_supported("TODO: Support non-packed input tensor in reference conv");

    if(!args.make_weight_descriptor().is_packed())
        return RunResult::not_supported("TODO: Support non-packed weight tensor in reference conv");

    if(!args.make_output_descriptor().is_packed())
        return RunResult::not_supported("TODO: Support non-packed output tensor in reference conv");

    conv.Run(input,
             weight,
             output,
             static_cast<int>(args.lengths.groups),
             static_cast<int>(args.lengths.batch_size),
             static_cast<int>(args.lengths.output_channels),
             static_cast<int>(args.lengths.input_channels),
             to_vec(args.lengths.image),
             to_vec(args.lengths.filter),
             to_vec(args.filter_strides),      // strides (not output_spatial!)
             to_vec(args.filter_dilation),     // dilations
             to_vec(args.input_left_pad),      // left_pads
             to_vec(args.input_right_pad));    // right_pads (was missing!)
    
    return RunResult::from_runtime(0); // ref conv does not return a meaningful runtime.
}

} // namespace detail

/// @brief Concept for checking whether this is the reference convolution
/// forward implementation.
template <typename Conv, auto SIGNATURE>
concept RefConvFwdInstance =
    detail::RefConvInstance<Conv, SIGNATURE, const void*, const void*, void*> &&
    ConvDirectionIsForward<SIGNATURE>;

/// @brief `run()` specialization for forward convolution and the reference
/// forward implementation.
///
/// @tparam SIGNATURE The signature of the operation to perform. Must be forwards.
/// @returns RunResult about how the operation completed (or not).
///
/// @see run()
template <auto SIGNATURE>
    requires ValidConvSignature<SIGNATURE> &&
             // TODO: Maybe we can unify this implementation for bwd/weight too?
             // for now, just concern outselves with reference and see when the
             // rest of the bwd/weight plumbing is there.
             ConvDirectionIsForward<SIGNATURE>
[[nodiscard]] RunResult run(RefConvFwdInstance<SIGNATURE> auto& conv,
                            const Args<SIGNATURE>& args,
                            const Inputs<SIGNATURE>& inputs,
                            const Outputs<SIGNATURE>& outputs)
{
    return detail::run(conv, args, inputs.input, inputs.weight, outputs.output);
}

/// @brief Concept for checking whether this is the reference convolution
/// backward weight implementation.
template <typename Conv, auto SIGNATURE>
concept RefConvBwdWeightInstance =
    detail::RefConvInstance<Conv, SIGNATURE, const void*, void*, const void*> &&
    ConvDirectionIsBackwardWeight<SIGNATURE>;

/// @brief `run()` specialization for forward convolution and the reference
/// backward weight implementation.
///
/// @tparam SIGNATURE The signature of the operation to perform. Must be backwards weight.
/// @returns RunResult about how the operation completed (or not).
///
/// @see run()
template <auto SIGNATURE>
[[nodiscard]] RunResult run(RefConvBwdWeightInstance<SIGNATURE> auto& conv,
                            const Args<SIGNATURE>& args,
                            const Inputs<SIGNATURE>& inputs,
                            const Outputs<SIGNATURE>& outputs)
{
    return detail::run(conv, args, inputs.input, outputs.weight, inputs.output);
}

} // namespace ck_tile::builder::test
