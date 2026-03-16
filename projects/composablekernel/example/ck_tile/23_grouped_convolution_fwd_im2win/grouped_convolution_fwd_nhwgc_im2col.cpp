// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// ═══════════════════════════════════════════════════════════════════════
// grouped_convolution_fwd_nhwgc_im2col.cpp
//
// Driver for NHWGC im2col forward convolution — comparison baseline for
// the im2win group-merging approach.
//
// GEMM shape: M=N×Ho×Wo, N=K, K_gemm=C×Y×X  (standard im2col)
// Layout:     NHWGC input, GKYXC weight, NHWGK output
//
// The active config is selected at compile time via NHWGC_IM2COL_CONFIG_ID:
//   0  = NhwgcIm2colConfig_CV3_M16N64K64          (baseline, no group merging)
//   1  = NhwgcIm2colConfig_Mem_M128N32K64          (memory pipeline, no group merging)
//   2  = NhwgcIm2colConfig_CV3_M128N128K64         (larger tile, no group merging)
//   3  = NhwgcIm2colConfig_Mem_M128N32K64_Gm32     (memory, Gm=32 group merging)
//   4  = NhwgcIm2colConfig_CV3_M128N128K64_Gm32    (CV3, Gm=32, N=K×Gm=128)
//   5  = NhwgcIm2colConfig_CV3_M64N128K64_Gm32     (CV3, Gm=32, smaller M)
//   6  = NhwgcIm2colConfig_CV3_M16N128K64_Gm32     (CV3, Gm=32, N=K×Gm=128)
//
// Usage:
//   ./tile_example_nhwgc_im2col_<config> \
//       -prec fp16 -in_layout NHWGC -wei_layout GKYXC -out_layout NHWGK \
//       -g 32 -n 32 -k 4 -c 4 -y 3 -x 3 -h 200 -w 200 \
//       -lpad_h 1 -lpad_w 1 -rpad_h 1 -rpad_w 1 -v 2
// ═══════════════════════════════════════════════════════════════════════

#include <hip/hip_runtime.h>
#include <iostream>
#include <stdexcept>
#include <string>

#include "ck_tile/host.hpp"
#include "nhwgc_im2col_configs.hpp"
#include "grouped_convolution_fwd_nhwgc_im2col_invoker.hpp"
#include "run_grouped_conv_fwd_im2win_example.inc"

// ── Select active config from compile-time NHWGC_IM2COL_CONFIG_ID ────────────
// IDs 0-6:  Standard im2col (UseIm2Win=false)
// IDs 7-10: Im2win A descriptor variant (UseIm2Win=true, same tile shapes as 0-3)
#ifndef NHWGC_IM2COL_CONFIG_ID
#define NHWGC_IM2COL_CONFIG_ID 1   // default: Mem_M128N32K64
#endif

template <typename PrecType>
#if NHWGC_IM2COL_CONFIG_ID == 0
using ActiveNhwgcIm2colConfig = NhwgcIm2colConfig_CV3_M16N64K64<PrecType>;
#elif NHWGC_IM2COL_CONFIG_ID == 1
using ActiveNhwgcIm2colConfig = NhwgcIm2colConfig_Mem_M128N32K64<PrecType>;
#elif NHWGC_IM2COL_CONFIG_ID == 2
using ActiveNhwgcIm2colConfig = NhwgcIm2colConfig_CV3_M128N128K64<PrecType>;
#elif NHWGC_IM2COL_CONFIG_ID == 3
using ActiveNhwgcIm2colConfig = NhwgcIm2colConfig_CV3_M64N64K64<PrecType>;
#elif NHWGC_IM2COL_CONFIG_ID == 4
using ActiveNhwgcIm2colConfig = NhwgcIm2colConfig_Mem_M128N16K64<PrecType>;
#elif NHWGC_IM2COL_CONFIG_ID == 5
using ActiveNhwgcIm2colConfig = NhwgcIm2colConfig_Mem_M128N32K64_Occ2<PrecType>;
#elif NHWGC_IM2COL_CONFIG_ID == 6
using ActiveNhwgcIm2colConfig = NhwgcIm2colConfig_CV3_M64N32K64<PrecType>;
// ── UseIm2Win=true variants (IDs 7-10) ──────────────────────────────────────
#elif NHWGC_IM2COL_CONFIG_ID == 7
using ActiveNhwgcIm2colConfig = NhwgcIm2colConfig_IW_CV3_M16N64K64<PrecType>;
#elif NHWGC_IM2COL_CONFIG_ID == 8
using ActiveNhwgcIm2colConfig = NhwgcIm2colConfig_IW_Mem_M128N32K64<PrecType>;
#elif NHWGC_IM2COL_CONFIG_ID == 9
using ActiveNhwgcIm2colConfig = NhwgcIm2colConfig_IW_CV3_M128N128K64<PrecType>;
#elif NHWGC_IM2COL_CONFIG_ID == 10
using ActiveNhwgcIm2colConfig = NhwgcIm2colConfig_IW_CV3_M64N64K64<PrecType>;
#else
#error "Unknown NHWGC_IM2COL_CONFIG_ID — valid range: 0..10"
#endif

template <template <typename PrecType> typename ConvConfig>
int run_nhwgc_im2col_example(int argc, char* argv[])
{
    using Invoker = GroupedConvolutionFwdNhwgcIm2colInvoker;
    using NHWGC = ck_tile::tensor_layout::convolution::NHWGC;
    using GKYXC = ck_tile::tensor_layout::convolution::GKYXC;
    using NHWGK = ck_tile::tensor_layout::convolution::NHWGK;

    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return -1;

    const std::string data_type  = arg_parser.get_str("prec");
    const std::string in_layout  = arg_parser.get_str("in_layout");
    const std::string wei_layout = arg_parser.get_str("wei_layout");
    const std::string out_layout = arg_parser.get_str("out_layout");

    // NHWGC im2col only supports channels-last layout.
    if(in_layout != "NHWGC" || wei_layout != "GKYXC" || out_layout != "NHWGK")
    {
        std::cerr << "nhwgc_im2col only supports NHWGC/GKYXC/NHWGK layout.\n"
                  << "Pass -in_layout=NHWGC -wei_layout=GKYXC -out_layout=NHWGK\n";
        return 1;
    }

    // Directly dispatch to NHWGC/GKYXC/NHWGK path — avoids compiler instantiation
    // of unsupported GNCHW layouts (which GroupedConvolutionForwardKernel does not
    // support when compiled with NHWGC traits).
    if(data_type == "fp16")
    {
        return run_grouped_conv_fwd_im2win_example_with_layouts<ck_tile::number<2>{},
                                                                 ConvConfig<ck_tile::half_t>,
                                                                 Invoker,
                                                                 ck_tile::half_t,
                                                                 ck_tile::half_t,
                                                                 ck_tile::half_t>(
            argc, argv, NHWGC{}, GKYXC{}, NHWGK{});
    }
    else if(data_type == "bf16")
    {
        return run_grouped_conv_fwd_im2win_example_with_layouts<ck_tile::number<2>{},
                                                                 ConvConfig<ck_tile::bf16_t>,
                                                                 Invoker,
                                                                 ck_tile::bf16_t,
                                                                 ck_tile::bf16_t,
                                                                 ck_tile::bf16_t>(
            argc, argv, NHWGC{}, GKYXC{}, NHWGK{});
    }
    else
    {
        std::cerr << "Unsupported precision: " << data_type << " (supported: fp16, bf16)\n";
        return 1;
    }
}

int main(int argc, char* argv[])
{
    try
    {
        return run_nhwgc_im2col_example<ActiveNhwgcIm2colConfig>(argc, argv);
    }
    catch(const std::runtime_error& e)
    {
        std::cerr << "Runtime error: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
