// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Dispatcher-based backward weight profiler header.
// Drop-in replacement for grouped_convolution_backward_weight_tile_algs.hpp
// that uses the CK Dispatcher registry instead of CK Builder .inc files.

#pragma once

#include <iostream>
#include <tuple>
#include <stdexcept>

#include "../../experimental/builder/test/utils/conv_algorithm_type_utils.hpp"
#include "grouped_convolution_signatures.hpp"
#include "ck_tile/ref/naive_grouped_conv_bwd_weight_gpu.hpp"
#include "ck_tile/builder/testing/filter_extent.hpp"
#include "ck_tile/builder/testing/conv/fwd.hpp"
#include "ck_tile/builder/testing/conv/ck_tile.hpp"
#include "ck_tile/builder/testing/conv/reference.hpp"
#include "ck_tile/builder/conv_builder.hpp"
#include "tile_profiler_utils.hpp"

// Dispatcher headers
#include "ck_tile/dispatcher/grouped_conv_registry.hpp"
#include "ck_tile/dispatcher/grouped_conv_problem.hpp"
#include "ck_tile/dispatcher/backends/generated_conv_backend.hpp"

// Suppress warnings from generated kernel headers
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wheader-hygiene"
#pragma clang diagnostic ignored "-Wunused-parameter"

// Include ALL generated backward weight kernels
#include "include_all_grouped_conv_bwd_weight_kernels.hpp"

// Include registration header
#include "register_all_grouped_conv_kernels.hpp"

#pragma clang diagnostic pop

namespace ck_tile::builder::profiling {

namespace ckb = ck_tile::builder;
namespace ckt = ck_tile::builder::test;

/// Get dispatcher dtype string from SIGNATURE
template <auto SIGNATURE>
constexpr const char* get_dtype_string()
{
    if constexpr(SIGNATURE.data_type == ckb::DataType::FP32)
        return "fp32";
    else if constexpr(SIGNATURE.data_type == ckb::DataType::FP16)
        return "fp16";
    else
        return "bf16";
}

/// Convert builder Args to dispatcher GroupedConvProblem
template <auto SIGNATURE>
inline ck_tile::dispatcher::GroupedConvProblem
args_to_problem(const ckt::Args<SIGNATURE>& args, int k_batch = 1)
{
    const auto conv_param = args.to_ck_tile_conv_param();
    ck_tile::dispatcher::GroupedConvProblem problem;

    problem.N  = conv_param.N_;
    problem.C  = conv_param.C_;
    problem.K  = conv_param.K_;
    problem.G  = conv_param.G_;
    problem.op = ck_tile::dispatcher::GroupedConvOp::BackwardWeight;
    problem.split_k = k_batch;

    constexpr int ndim = SIGNATURE.spatial_dim;

    // Fill spatial dims (3D array, pad with 1 for 2D)
    if constexpr(ndim == 2)
    {
        problem.input_spatial  = {1, conv_param.input_spatial_lengths_[0],
                                  conv_param.input_spatial_lengths_[1]};
        problem.filter_spatial = {1, conv_param.filter_spatial_lengths_[0],
                                  conv_param.filter_spatial_lengths_[1]};
        problem.output_spatial = {1, conv_param.output_spatial_lengths_[0],
                                  conv_param.output_spatial_lengths_[1]};
        problem.stride   = {1, conv_param.conv_filter_strides_[0],
                            conv_param.conv_filter_strides_[1]};
        problem.padding  = {0, conv_param.input_left_pads_[0],
                            conv_param.input_left_pads_[1]};
        problem.dilation = {1, conv_param.conv_filter_dilations_[0],
                            conv_param.conv_filter_dilations_[1]};
    }
    else if constexpr(ndim == 3)
    {
        problem.input_spatial  = {conv_param.input_spatial_lengths_[0],
                                  conv_param.input_spatial_lengths_[1],
                                  conv_param.input_spatial_lengths_[2]};
        problem.filter_spatial = {conv_param.filter_spatial_lengths_[0],
                                  conv_param.filter_spatial_lengths_[1],
                                  conv_param.filter_spatial_lengths_[2]};
        problem.output_spatial = {conv_param.output_spatial_lengths_[0],
                                  conv_param.output_spatial_lengths_[1],
                                  conv_param.output_spatial_lengths_[2]};
        problem.stride   = {conv_param.conv_filter_strides_[0],
                            conv_param.conv_filter_strides_[1],
                            conv_param.conv_filter_strides_[2]};
        problem.padding  = {conv_param.input_left_pads_[0],
                            conv_param.input_left_pads_[1],
                            conv_param.input_left_pads_[2]};
        problem.dilation = {conv_param.conv_filter_dilations_[0],
                            conv_param.conv_filter_dilations_[1],
                            conv_param.conv_filter_dilations_[2]};
    }

    return problem;
}

template <auto SIGNATURE>
void run_cpu_validation(const ckt::Args<SIGNATURE>& args,
                        const ckt::Outputs<SIGNATURE>& outputs,
                        const ckt::Outputs<SIGNATURE>& reference)
{
    using DataType =
        std::conditional_t<SIGNATURE.data_type == ckb::DataType::FP32,
                           float,
                           std::conditional_t<SIGNATURE.data_type == ckb::DataType::FP16,
                                              ck_tile::half_t,
                                              ck_tile::bfloat16_t>>;
    const auto conv_param = args.to_ck_tile_conv_param();

    const std::size_t weight_bytes_num = conv_param.template GetWeightByte<DataType>();
    std::vector<DataType> wei(weight_bytes_num / sizeof(DataType));
    std::vector<DataType> ref(weight_bytes_num / sizeof(DataType));
    HIP_CHECK_ERROR(
        hipMemcpy(&ref.data()[0], reference.weight, weight_bytes_num, hipMemcpyDeviceToHost));
    HIP_CHECK_ERROR(
        hipMemcpy(&wei.data()[0], outputs.weight, weight_bytes_num, hipMemcpyDeviceToHost));
    ck_tile::check_err(wei, ref, "\tError: Incorrect results!");
}

std::string get_runtime_arch_name()
{
    hipDeviceProp_t props{};
    int device = 0;
    ck_tile::hip_check_error(hipGetDevice(&device));
    ck_tile::hip_check_error(hipGetDeviceProperties(&props, device));
    // Extract base arch name (e.g. "gfx950" from "gfx950:sramecc+:xnack-")
    std::string name(props.gcnArchName);
    auto pos = name.find(':');
    if(pos != std::string::npos)
        name = name.substr(0, pos);
    return name;
}

/// @brief Dispatcher-based `run_grouped_conv_backward_weight_tile_algs()`.
/// Iterates all registered dispatcher kernels instead of builder-generated .inc files.
template <auto SIGNATURE>
std::tuple<bool, float, std::string, int>
run_grouped_conv_backward_weight_tile_algs(const ckt::Args<SIGNATURE>& args,
                                           const std::string& split_k,
                                           const ckt::Inputs<SIGNATURE>& inputs,
                                           const ckt::Outputs<SIGNATURE>& outputs,
                                           const ck_tile::stream_config& s_conf)
{
    bool dummy_run_executed = false;
    float best_avg_time     = std::numeric_limits<float>::max();
    std::string best_op_name, op_name;
    int best_split_k = 1;
    bool is_supported;
    float avg_time{0};
    bool all_instances_valid = true;

    using DataType =
        std::conditional_t<SIGNATURE.data_type == ckb::DataType::FP32,
                           float,
                           std::conditional_t<SIGNATURE.data_type == ckb::DataType::FP16,
                                              ck_tile::half_t,
                                              ck_tile::bfloat16_t>>;

    // Compute reference using builder reference implementation
    auto reference = ckt::alloc_outputs(args);
    using ReferenceInstance =
        typename ckb::ConvBuilder<SIGNATURE, ckt::ConvAlgorithm_Reference{}>::Instance;
    auto ref_conv   = ReferenceInstance{};
    auto ref_result = ckt::run(ref_conv, args, inputs, reference.get());

    const auto conv_param = args.to_ck_tile_conv_param();

    // Get max possible value in the output for tolerance calculation
    const std::size_t weight_bytes_num = conv_param.template GetWeightByte<DataType>();
    std::vector<DataType> ref(weight_bytes_num / sizeof(DataType));
    ck_tile::hip_check_error(
        hipMemcpy(&ref.data()[0], reference.get().weight, weight_bytes_num, hipMemcpyDeviceToHost));
    const float max_accumulated_value = *std::max_element(ref.begin(), ref.end());
    const index_t num_accums = std::accumulate(std::begin(conv_param.output_spatial_lengths_),
                                               std::end(conv_param.output_spatial_lengths_),
                                               static_cast<std::size_t>(1),
                                               std::multiplies<std::size_t>()) *
                               conv_param.N_;
    const auto split_k_values = get_split_k_values(split_k);

    // Register all generated backward weight kernels
    static bool kernels_registered = false;
    if(!kernels_registered)
    {
        const auto arch_name = get_runtime_arch_name();
        std::cout << "Runtime arch: " << arch_name << std::endl;
        ck_tile::dispatcher::register_all_grouped_conv_bwd_weight_kernels(arch_name);
        kernels_registered = true;
    }

    // Get backward weight kernels matching data type and spatial dims
    constexpr const char* dtype_str = get_dtype_string<SIGNATURE>();
    constexpr int ndim = SIGNATURE.spatial_dim;
    auto& registry = ck_tile::dispatcher::GroupedConvRegistry::instance();
    auto all_kernels = registry.filter([](const ck_tile::dispatcher::GroupedConvKernelInstance& k) {
        return k.key().op == ck_tile::dispatcher::GroupedConvOp::BackwardWeight &&
               k.key().dtype_in == dtype_str &&
               k.key().ndim_spatial == ndim;
    });

    // Set up thread-local buffer context
    auto& ctx       = ck_tile::dispatcher::g_conv_dispatch_buffers;
    ctx.input_ptr   = inputs.input;
    ctx.weight_ptr  = inputs.output; // For bwd_weight: "output" gradient = dY
    ctx.output_ptr  = outputs.weight; // dW being computed
    ctx.warmup      = s_conf.cold_niters_;
    ctx.repeat      = s_conf.nrepeat_;
    ctx.benchmarking = s_conf.time_kernel_;

    // For verification purposes, we use the instance string as the op_name.
    // This allows us to compare the tile based dispatcher output to the CK builder based output.
    constexpr bool use_instance_string = true;

    // Iterate all kernels × split-K values
    for(const auto* kernel : all_kernels)
    {
        for(auto& k_batch : split_k_values)
        {
            auto problem  = args_to_problem<SIGNATURE>(args, k_batch);
            ctx.split_k   = k_batch;

            op_name       = kernel->name(use_instance_string);

            // Check support before launching
            is_supported = kernel->is_supported(problem);
            if(!is_supported)
            {
                std::cout << "[Not supported] " << op_name << ", SplitK " << k_batch << std::endl;
                continue;
            }

            try
            {
                avg_time     = kernel->run(problem, nullptr);
                is_supported = true;
            }
            catch(const std::runtime_error& e)
            {
                std::cerr << "[Exception] " << op_name << " SplitK=" << k_batch
                          << " : " << e.what() << std::endl;
                ck_tile::hip_check_error(hipDeviceSynchronize());
                ck_tile::hip_check_error(hipGetLastError());
                is_supported = false;
            }

            if((s_conf.time_kernel_ || s_conf.flush_cache_) && !dummy_run_executed)
            {
                // Run first instance twice when profiling to stabilize timing
                try
                {
                    avg_time = kernel->run(problem, nullptr);
                }
                catch(const std::runtime_error& e)
                {
                    std::cerr << "[Exception] " << op_name << " SplitK=" << k_batch << " : " << e.what() << std::endl;
                    ck_tile::hip_check_error(hipDeviceSynchronize());
                    ck_tile::hip_check_error(hipGetLastError());
                    is_supported = false;
                }
                dummy_run_executed = true;
            }

            if(is_supported)
            {
                ckt::ValidationReport report;
                auto&& [rtol, atol] = get_rtol_atol<SIGNATURE>(num_accums, k_batch, max_accumulated_value);
                ckt::Outputs<SIGNATURE>::reflect(
                    args,
                    [&](std::string_view name,
                        const auto& desc,
                        void* ckt::Outputs<SIGNATURE>::*ptr) {
                        report.check(name, desc, outputs.*ptr, reference.get().*ptr, rtol, atol);
                    });

                const bool valid = report.get_errors().empty();
                best_avg_time    = std::min(best_avg_time, avg_time);
                best_op_name     = best_avg_time < avg_time ? best_op_name : op_name;
                best_split_k     = best_avg_time < avg_time ? best_split_k : k_batch;
                if(valid)
                {
                    std::cout << "[Valid] Perf: " << std::setw(10) << avg_time << " ms,"
                                << " " << op_name << ", SplitK " << k_batch << std::endl;
                }
                else
                {
                    std::cout << "[Error] " << op_name << ", SplitK " << k_batch << std::endl;
                    for(const auto& error : report.get_errors())
                    {
                        std::cout << "\tNumber of incorrect values: " << error.wrong_elements
                                    << " Is all zero:" << error.is_all_zero()
                                    << " max err: " << error.max_error << std::endl;
                        ckt::Args<SIGNATURE> args_k_batch = args;
                        args_k_batch.k_batch              = k_batch;
                        run_cpu_validation<SIGNATURE>(args_k_batch, outputs, reference.get());
                    }
                    all_instances_valid = false;
                }
            }
        }
    }

    return std::make_tuple(all_instances_valid, best_avg_time, best_op_name, best_split_k);
}

} // namespace ck_tile::builder::profiling
