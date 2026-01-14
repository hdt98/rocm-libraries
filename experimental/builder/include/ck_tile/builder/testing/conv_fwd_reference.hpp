// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/builder/testing/conv_fwd.hpp"
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
template <typename Conv, auto SIGNATURE>
concept RefConvInstance = requires(Conv& conv,
                                   const void* input,
                                   const void* weight,
                                   void* output,
                                   int G,
                                   int N,
                                   int K,
                                   int C,
                                   std::vector<long_index_t> dims) {
    {
        conv.Run(input,
                 weight,
                 output,
                 G,
                 N,
                 K,
                 C,
                 dims, // input_spatial
                 dims, // filter_spatial
                 dims, // output_spatial
                 dims, // strides
                 dims, // dilations
                 dims  // left_pads
        )
    };
};

/// @brief `run()` specialization for forward convolution and the reference
/// implementation.
///
/// @tparam SIGNATURE Forward convolution signature.
/// @throws std::runtime_error if the arguments weren't actually valid for the
/// operation. This should be caught and reported by the testing framework.
///
/// @see run()
template <auto SIGNATURE>
    requires ValidConvSignature<SIGNATURE> &&
             // TODO: Maybe we can unify this implementation for bwd/weight too?
             // for now, just concern outselves with reference and see when the
             // rest of the bwd/weight plumbing is there.
             ConvDirectionIsForward<SIGNATURE>
void run(RefConvInstance<SIGNATURE> auto& conv,
         const Args<SIGNATURE>& args,
         const Inputs<SIGNATURE>& inputs,
         const Outputs<SIGNATURE>& outputs)
{
    const auto problem = args.make_conv_problem();

    // TODO: The reference convolution is currently missing a few features.
    // Just throw for now, but regard these as TODO items that should be resolved
    // eventually.

    // Right pads are not supported right now for some reason.
    for(auto right_pad : problem.right_pads)
    {
        if(right_pad != 0)
            throw std::runtime_error("TODO: Support right pad in reference conv");
    }

    if(!args.make_input_descriptor().is_packed())
        throw std::runtime_error("TODO: Support non-packed input tensor in reference conv");
    if(!args.make_weight_descriptor().is_packed())
        throw std::runtime_error("TODO: Support non-packed weight tensor in reference conv");
    if(!args.make_output_descriptor().is_packed())
        throw std::runtime_error("TODO: Support non-packed output tensor in reference conv");

    const std::vector<long_index_t> input_spatial(problem.input_spatial.begin(),
                                                  problem.input_spatial.end());
    const std::vector<long_index_t> filter_spatial(problem.filter_spatial.begin(),
                                                   problem.filter_spatial.end());
    const std::vector<long_index_t> output_spatial(problem.output_spatial.begin(),
                                                   problem.output_spatial.end());
    const std::vector<long_index_t> conv_strides(problem.conv_strides.begin(),
                                                 problem.conv_strides.end());
    const std::vector<long_index_t> conv_dilations(problem.conv_dilations.begin(),
                                                   problem.conv_dilations.end());
    const std::vector<long_index_t> left_pads(problem.left_pads.begin(), problem.left_pads.end());

    conv.Run(inputs.input,
             inputs.weight,
             outputs.output,
             problem.G,
             problem.N,
             problem.K,
             problem.C,
             input_spatial,
             filter_spatial,
             output_spatial,
             conv_strides,
             conv_dilations,
             left_pads);
}

} // namespace ck_tile::builder::test
