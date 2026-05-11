// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <tuple>

#include "../../experimental/builder/test/utils/conv_algorithm_type_utils.hpp"
#include "grouped_convolution_signatures.hpp"
#include "common.hpp"
#include "ck_tile/ref/naive_grouped_conv_fwd_gpu.hpp"

#include "ck_tile/builder/testing/filter_extent.hpp"
#include "ck_tile/builder/testing/conv/fwd.hpp"
#include "ck_tile/builder/testing/conv/ck_tile.hpp"
#include "ck_tile/builder/testing/conv/reference.hpp"
#include "ck_tile/builder/conv_builder.hpp"
#include "tile_profiler_common.hpp"

#define ENABLE_BUILDER_VALIDATE 1

namespace ck_tile::builder::profiling {

#include "../../experimental/grouped_convolution_tile_instances/instances/forward/grouped_convolution_forward_tile_nhwgc_fp32.inc"
#include "../../experimental/grouped_convolution_tile_instances/instances/forward/grouped_convolution_forward_tile_nhwgc_bf16.inc"
#include "../../experimental/grouped_convolution_tile_instances/instances/forward/grouped_convolution_forward_tile_nhwgc_fp16.inc"
#include "../../experimental/grouped_convolution_tile_instances/instances/forward/grouped_convolution_forward_tile_ndhwgc_fp32.inc"
#include "../../experimental/grouped_convolution_tile_instances/instances/forward/grouped_convolution_forward_tile_ndhwgc_bf16.inc"
#include "../../experimental/grouped_convolution_tile_instances/instances/forward/grouped_convolution_forward_tile_ndhwgc_fp16.inc"

/// @brief `run_grouped_conv_forward_tile_algs()` run all grouped conv fwd instances.
///
/// @tparam SIGNATURE Forward convolution signature.
///
/// @see run_grouped_conv_forward_tile_algs()
template <auto SIGNATURE>
std::tuple<bool, float, std::string>
run_grouped_conv_forward_tile_algs(const ckt::Args<SIGNATURE>& args,
                                   const ckt::Inputs<SIGNATURE>& inputs,
                                   const ckt::Outputs<SIGNATURE>& outputs,
                                   const ck_tile::stream_config& s_conf)
{
    using DataType = DeduceDataType<SIGNATURE>;

    // Run first instance as dummy to get proper time from the first instance
    bool dummy_run_executed = false;
    float best_avg_time     = std::numeric_limits<float>::max();
    std::string best_op_name, op_name;
    bool is_supported;
    float avg_time;
    bool valid = true;

    auto reference = compute_reference<SIGNATURE>(args, inputs);

    auto run_alg = [&](auto&& run_alg_func) {
        std::tie(is_supported, avg_time, op_name) = run_alg_func(args, inputs, outputs, s_conf);
        if(is_supported)
        {
            if((s_conf.time_kernel_ || s_conf.flush_cache_) && !dummy_run_executed)
            {
                // Run first instance twice
                std::tie(is_supported, avg_time, op_name) =
                    run_alg_func(args, inputs, outputs, s_conf);
                dummy_run_executed = true;
            }
            best_avg_time = std::min(best_avg_time, avg_time);
            best_op_name  = best_avg_time < avg_time ? best_op_name : op_name;
            std::cout << "Perf: " << std::setw(10) << avg_time << " ms," << " " << op_name
                      << std::endl;

            if(!validate_and_report<SIGNATURE, ConvBuffer::Output>(
                   args,
                   outputs,
                   reference.get(),
                   ck::profiler::get_rtol<DataType>(),
                   ck::profiler::get_atol<DataType>()))
            {
                valid = false;
            }
        }
        else
        {
            std::cout << " " << op_name << std::endl;
        }
    };

    if constexpr(SIGNATURE == SIGNATURE_NHWGC_FP16_FWD)
    {
#include "../../experimental/grouped_convolution_tile_instances/instances/forward/grouped_convolution_forward_tile_nhwgc_fp16_calls.inc"
    }
    else if constexpr(SIGNATURE == SIGNATURE_NHWGC_BF16_FWD)
    {
#include "../../experimental/grouped_convolution_tile_instances/instances/forward/grouped_convolution_forward_tile_nhwgc_bf16_calls.inc"
    }
    else if constexpr(SIGNATURE == SIGNATURE_NHWGC_FP32_FWD)
    {
#include "../../experimental/grouped_convolution_tile_instances/instances/forward/grouped_convolution_forward_tile_nhwgc_fp32_calls.inc"
    }
    else if constexpr(SIGNATURE == SIGNATURE_NDHWGC_FP16_FWD)
    {
#include "../../experimental/grouped_convolution_tile_instances/instances/forward/grouped_convolution_forward_tile_ndhwgc_fp16_calls.inc"
    }
    else if constexpr(SIGNATURE == SIGNATURE_NDHWGC_BF16_FWD)
    {
#include "../../experimental/grouped_convolution_tile_instances/instances/forward/grouped_convolution_forward_tile_ndhwgc_bf16_calls.inc"
    }
    else if constexpr(SIGNATURE == SIGNATURE_NDHWGC_FP32_FWD)
    {
#include "../../experimental/grouped_convolution_tile_instances/instances/forward/grouped_convolution_forward_tile_ndhwgc_fp32_calls.inc"
    }
    else
    {
        std::cout << "Signature not supported" << std::endl;
        return std::make_tuple(false, best_avg_time, best_op_name);
    }
    return std::make_tuple(valid, best_avg_time, best_op_name);
}

} // namespace ck_tile::builder::profiling
