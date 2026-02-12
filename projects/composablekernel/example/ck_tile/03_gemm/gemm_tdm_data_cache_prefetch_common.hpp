// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "gemm_utils.hpp"
#include "run_gemm_example.inc"
#include "run_gemm_example_common.hpp"
#include "universal_gemm_invoker.hpp"
#include "ck_tile/core/utility/gemm_validation.hpp"

// Template function to run GEMM with optional prefetch comparison
// GemmConfig: The GEMM configuration struct (e.g., GemmConfigTDMV1Prefetch or
// GemmConfigTDMV2Prefetch)
template <template <typename, bool, bool> class GemmConfig,
          typename ADataType,
          typename... BCAccDataTypes>
bool run_gemm_with_prefetch_comparison(const std::string& a_layout,
                                       const std::string& b_layout,
                                       ck_tile::ArgParser& arg_parser,
                                       bool compare_with_non_prefetch,
                                       bool prefetch_to_l1)
{
    using Invoker = UniversalInvoker;

    std::cout << "\n=== Running with DataCache Prefetch ENABLED (";
    std::cout << (prefetch_to_l1 ? "L1" : "L2") << ") ===\n" << std::endl;

    bool pass_prefetch;
    if(prefetch_to_l1)
    {
        pass_prefetch =
            run_gemm_example_prec_type<GemmConfig<ADataType, true, true>,
                                       Invoker,
                                       ADataType,
                                       BCAccDataTypes...>(a_layout, b_layout, arg_parser);
    }
    else
    {
        pass_prefetch =
            run_gemm_example_prec_type<GemmConfig<ADataType, true, false>,
                                       Invoker,
                                       ADataType,
                                       BCAccDataTypes...>(a_layout, b_layout, arg_parser);
    }

    if(compare_with_non_prefetch)
    {
        std::cout << "\n=== Running with DataCache Prefetch DISABLED ===\n" << std::endl;
        bool pass_no_prefetch =
            run_gemm_example_prec_type<GemmConfig<ADataType, false, false>,
                                       Invoker,
                                       ADataType,
                                       BCAccDataTypes...>(a_layout, b_layout, arg_parser);

        std::cout << "\n=== Comparison Summary ===" << std::endl;
        std::cout << "Note: Check the timing results above to compare performance." << std::endl;
        std::cout << "With prefetch vs without prefetch - speedup can be observed in the "
                     "timing outputs."
                  << std::endl;

        return pass_prefetch && pass_no_prefetch;
    }

    return pass_prefetch;
}

// Common GEMM example runner
template <template <typename, bool, bool> class GemmConfig>
int run_gemm_example_with_prefetch(ck_tile::ArgParser& arg_parser)
{
    std::string data_type = arg_parser.get_str("prec");
    std::string a_layout  = arg_parser.get_str("a_layout");
    std::string b_layout  = arg_parser.get_str("b_layout");
    std::string c_layout  = arg_parser.get_str("c_layout");

    std::tuple<ck_tile::index_t, ck_tile::index_t, ck_tile::index_t> gemm_sizes =
        parse_gemm_size(arg_parser);

    int m = std::get<0>(gemm_sizes);
    int n = std::get<1>(gemm_sizes);
    int k = std::get<2>(gemm_sizes);

    int stride_a = arg_parser.get_int("stride_a");
    int stride_b = arg_parser.get_int("stride_b");
    int stride_c = arg_parser.get_int("stride_c");

    bool compare_with_non_prefetch = arg_parser.get_int("compare") == 1;
    bool prefetch_to_l1            = arg_parser.get_int("prefetch_l1") == 1;

    ck_tile::validate_gemm_stride(
        a_layout, b_layout, c_layout, m, n, k, stride_a, stride_b, stride_c);

    if(data_type == "fp16")
    {
        return run_gemm_with_prefetch_comparison<GemmConfig, ck_tile::half_t, ck_tile::half_t>(
            a_layout, b_layout, arg_parser, compare_with_non_prefetch, prefetch_to_l1);
    }
    else if(data_type == "bf16")
    {
        return run_gemm_with_prefetch_comparison<GemmConfig, ck_tile::bf16_t, ck_tile::bf16_t>(
            a_layout, b_layout, arg_parser, compare_with_non_prefetch, prefetch_to_l1);
    }
    else if(data_type == "fp8")
    {
        return run_gemm_with_prefetch_comparison<GemmConfig,
                                                 ck_tile::fp8_t,
                                                 ck_tile::fp8_t,
                                                 ck_tile::half_t>(
            a_layout, b_layout, arg_parser, compare_with_non_prefetch, prefetch_to_l1);
    }
    else if(data_type == "bf8")
    {
        return run_gemm_with_prefetch_comparison<GemmConfig,
                                                 ck_tile::bf8_t,
                                                 ck_tile::bf8_t,
                                                 ck_tile::half_t>(
            a_layout, b_layout, arg_parser, compare_with_non_prefetch, prefetch_to_l1);
    }
    else if(data_type == "i8")
    {
        return run_gemm_with_prefetch_comparison<GemmConfig,
                                                 ck_tile::int8_t,
                                                 ck_tile::int8_t,
                                                 int32_t>(
            a_layout, b_layout, arg_parser, compare_with_non_prefetch, prefetch_to_l1);
    }
    else
    {
        throw std::runtime_error("Unsupported data type for GEMM with prefetch!");
    }
}
