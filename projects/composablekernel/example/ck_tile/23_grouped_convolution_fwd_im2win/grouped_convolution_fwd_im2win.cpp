// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// ═══════════════════════════════════════════════════════════════════════
// grouped_convolution_fwd_im2win.cpp
//
// Driver for the im2win forward grouped convolution example.
//
// The active kernel configuration is controlled by the ActiveIm2winConfig
// alias defined in im2win_conv_configs.hpp.  To benchmark a different
// tile shape or pipeline, edit that alias and recompile — no other file
// needs to change.
//
// Usage (defaults shown):
//   ./tile_example_grouped_conv_fwd_im2win \
//       --prec fp16                  \
//       --in_layout GNCHW            \
//       --wei_layout GKCYX           \
//       --out_layout NHWGK           \
//       -g 32 -n 1 -k 8 -c 8        \
//       -y 3 -x 3 -hi 16 -wi 16     \
//       --conv_stride_h 1 --conv_stride_w 1 \
//       --lpad_h 1 --lpad_w 1        \
//       --warmup 5 --repeat 20       \
//       -v 2                         (0=no verify, 1=CPU ref, 2=GPU ref)
// ═══════════════════════════════════════════════════════════════════════

#include <hip/hip_runtime.h>

#include <iostream>
#include <stdexcept>
#include <string>

#include "ck_tile/host.hpp"
#include "im2win_conv_configs.hpp"
#include "grouped_convolution_fwd_im2win_invoker.hpp"
#include "run_grouped_conv_fwd_im2win_example.inc"

// ── Top-level driver ──────────────────────────────────────────────────────────
// ConvConfig is a template-template parameter so that the data-type specialisation
// (ConvConfig<half_t> vs ConvConfig<bf16_t>) is resolved after the precision is
// read from the command line, mirroring the pattern in grouped_convolution_forward.cpp.
template <template <typename PrecType> typename ConvConfig>
int run_im2win_example(int argc, char* argv[])
{
    using Invoker = GroupedConvolutionFwdIm2winInvoker;

    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return -1;

    const std::string data_type  = arg_parser.get_str("prec");
    const std::string in_layout  = arg_parser.get_str("in_layout");
    const std::string wei_layout = arg_parser.get_str("wei_layout");
    const std::string out_layout = arg_parser.get_str("out_layout");

    if(data_type == "fp16")
    {
        return run_grouped_conv_fwd_im2win_example_prec_type<Invoker,
                                                              ConvConfig<ck_tile::half_t>,
                                                              ck_tile::half_t>(
            in_layout, wei_layout, out_layout, argc, argv);
    }
    else if(data_type == "bf16")
    {
        return run_grouped_conv_fwd_im2win_example_prec_type<Invoker,
                                                              ConvConfig<ck_tile::bf16_t>,
                                                              ck_tile::bf16_t>(
            in_layout, wei_layout, out_layout, argc, argv);
    }
    else
    {
        throw std::runtime_error("Unsupported precision: " + data_type +
                                 " (supported: fp16, bf16)");
    }
}

int main(int argc, char* argv[])
{
    try
    {
        // ── Active kernel configuration ───────────────────────────────────────
        // To benchmark a different config, change ActiveIm2winConfig in
        // im2win_conv_configs.hpp and recompile.  All available configs are
        // listed and documented there.
        return run_im2win_example<ActiveIm2winConfig>(argc, argv);
    }
    catch(const std::runtime_error& e)
    {
        std::cerr << "Runtime error: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
