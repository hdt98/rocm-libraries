// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>

#include <cstring>
#include <iostream>
#include <ostream>
#include <string>
#include <tuple>

#include "ck_tile/host.hpp"
#include "grouped_convolution_utils.hpp"
#include "grouped_convolution_backward_weight_streamk_invoker.hpp"
#include "run_grouped_convolution_bwd_weight_example.inc"

template <template <typename PrecType> typename ConvConfig, typename Invoker>
int run_grouped_conv_bwd_weight_example(ck_tile::ArgParser& arg_parser)
{
    std::string data_type  = arg_parser.get_str("prec");
    std::string in_layout  = arg_parser.get_str("in_layout");
    std::string wei_layout = arg_parser.get_str("wei_layout");
    std::string out_layout = arg_parser.get_str("out_layout");

    if(data_type == "fp16")
    {
        return run_grouped_conv_bwd_weight_example_prec_type<Invoker,
                                                             ConvConfig<ck_tile::half_t>,
                                                             ck_tile::half_t>(
            in_layout, wei_layout, out_layout, arg_parser);
    }
    else if(data_type == "bf16")
    {
        return run_grouped_conv_bwd_weight_example_prec_type<Invoker,
                                                             ConvConfig<ck_tile::bf16_t>,
                                                             ck_tile::bf16_t>(
            in_layout, wei_layout, out_layout, arg_parser);
    }
    else
    {
        throw std::runtime_error("Unsupported data type for this operation!");
    }
}

int main(int argc, char* argv[])
{
    auto [result, arg_parser] =
        create_args(argc,
                    argv,
                    {
                        {"streamk_reduction", "tree", "StreamK reduction strategy: linear or tree"},
                    });
    if(!result)
        return -1;

    try
    {
        const std::string reduction = arg_parser.get_str("streamk_reduction");

        if(reduction == "linear")
        {
            using Invoker = GroupedConvolutionBackwardWeightStreamKInvoker<
                ck_tile::StreamKReductionStrategy::Linear>;
            return !run_grouped_conv_bwd_weight_example<ConvConfigComputeV3, Invoker>(arg_parser);
        }
        else if(reduction == "tree")
        {
            using Invoker = GroupedConvolutionBackwardWeightStreamKInvoker<
                ck_tile::StreamKReductionStrategy::Tree>;
            return !run_grouped_conv_bwd_weight_example<ConvConfigComputeV3, Invoker>(arg_parser);
        }
        else
        {
            std::cerr << "Unknown streamk_reduction: " << reduction
                      << ". Use 'linear' or 'tree'.\n";
            return EXIT_FAILURE;
        }
    }
    catch(const std::runtime_error& e)
    {
        std::cerr << "Runtime error: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
