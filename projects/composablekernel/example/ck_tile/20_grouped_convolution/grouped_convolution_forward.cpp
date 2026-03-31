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
#include "grouped_convolution_forward_invoker.hpp"
#include "run_grouped_convolution_fwd_example.inc"

// ---------------------------------------------------------------------------
// Config variants differing only in UseTiledIm2Col.
// ConvConfigComputeV3 inherits UseTiledIm2Col from ConvConfigBase (currently
// true).  The two wrappers below let us select the path at runtime.
// ---------------------------------------------------------------------------

template <typename PrecType>
struct ConvConfigComputeV3_TiledIm2Col : public ConvConfigComputeV6<PrecType>
{
    static constexpr bool UseTiledIm2Col = true;
};

template <typename PrecType>
struct ConvConfigComputeV3_GenericIm2Col : public ConvConfigComputeV6<PrecType>
{
    static constexpr bool UseTiledIm2Col = false;
};

// ---------------------------------------------------------------------------

template <template <typename PrecType> typename ConvConfig>
int run_grouped_conv_fwd_example(int argc, char* argv[])
{
    using Invoker = GroupedConvolutionForwardInvoker;

    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return -1;

    std::string data_type  = arg_parser.get_str("prec");
    std::string in_layout  = arg_parser.get_str("in_layout");
    std::string wei_layout = arg_parser.get_str("wei_layout");
    std::string out_layout = arg_parser.get_str("out_layout");

    if(data_type == "fp16")
    {
        return run_grouped_conv_fwd_example_prec_type<Invoker,
                                                      ConvConfig<ck_tile::half_t>,
                                                      ck_tile::half_t>(
            in_layout, wei_layout, out_layout, argc, argv);
    }
    else if(data_type == "bf16")
    {
        return run_grouped_conv_fwd_example_prec_type<Invoker,
                                                      ConvConfig<ck_tile::bf16_t>,
                                                      ck_tile::bf16_t>(
            in_layout, wei_layout, out_layout, argc, argv);
    }
    else
    {
        throw std::runtime_error("Unsupported data type for this operation !!!");
    }
}

int main(int argc, char* argv[])
{
    try
    {
        // Parse args early to read --tiled_im2col before selecting the config.
        auto [result, arg_parser] = create_args(argc, argv);
        if(!result)
            return EXIT_FAILURE;

        const bool use_tiled_im2col = arg_parser.get_int("tiled_im2col") != 0;

        std::cout << "[grouped_conv_fwd] im2col mode: "
                  << (use_tiled_im2col ? "tiled (fast coordinate)" : "generic (transform chain)")
                  << '\n';

#if CK_TILE_USE_WMMA
        // WMMA path: tiled im2col not yet wired for WMMA configs — use generic only.
        (void)use_tiled_im2col;
        return !run_grouped_conv_fwd_example<ConvConfigComputeV3_WMMA>(argc, argv);
#else
        if(use_tiled_im2col)
            return !run_grouped_conv_fwd_example<ConvConfigComputeV3_TiledIm2Col>(argc, argv);
        else
            return !run_grouped_conv_fwd_example<ConvConfigComputeV3_GenericIm2Col>(argc, argv);
#endif
    }
    catch(const std::runtime_error& e)
    {
        std::cerr << "Runtime error: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
