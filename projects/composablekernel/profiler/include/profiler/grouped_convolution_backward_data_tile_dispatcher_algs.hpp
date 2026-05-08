// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Dispatcher-based backward data profiler header.
// Drop-in replacement for grouped_convolution_backward_data_tile_algs.hpp
// that uses the CK Dispatcher registry instead of CK Builder .inc files.

#pragma once

#include <iostream>
#include <tuple>
#include <stdexcept>

#include "../../experimental/builder/test/utils/conv_algorithm_type_utils.hpp"
#include "grouped_convolution_signatures.hpp"
#include "ck_tile/ref/naive_grouped_conv_bwd_data_gpu.hpp"
#include "ck_tile/builder/testing/filter_extent.hpp"
#include "ck_tile/builder/testing/conv/ck_tile.hpp"
#include "ck_tile/builder/testing/conv/reference.hpp"
#include "ck_tile/builder/conv_builder.hpp"
#include "tile_profiler_utils.hpp"

// Dispatcher headers
#include "ck_tile/dispatcher/grouped_conv_registry.hpp"
#include "ck_tile/dispatcher/grouped_conv_problem.hpp"

// Forward declaration of registration function
#include "ck_tile/dispatcher/register_all_grouped_conv_kernels.hpp"

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
    problem.op = ck_tile::dispatcher::GroupedConvOp::BackwardData;
    problem.split_k = k_batch;

    constexpr int ndim = SIGNATURE.spatial_dim;

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

    const std::size_t input_bytes_num = conv_param.template GetInputByte<DataType>();
    std::vector<DataType> in(input_bytes_num / sizeof(DataType));
    std::vector<DataType> ref(input_bytes_num / sizeof(DataType));
    HIP_CHECK_ERROR(
        hipMemcpy(&ref.data()[0], reference.input, input_bytes_num, hipMemcpyDeviceToHost));
    HIP_CHECK_ERROR(
        hipMemcpy(&in.data()[0], outputs.input, input_bytes_num, hipMemcpyDeviceToHost));
    ck_tile::check_err(in, ref, "\tError: Incorrect results!");
}

inline std::string get_runtime_arch_name()
{
    hipDeviceProp_t props{};
    int device = 0;
    ck_tile::hip_check_error(hipGetDevice(&device));
    ck_tile::hip_check_error(hipGetDeviceProperties(&props, device));
    std::string name(props.gcnArchName);
    auto pos = name.find(':');
    if(pos != std::string::npos)
        name = name.substr(0, pos);
    return name;
}

/// @brief Dispatcher-based `run_grouped_conv_backward_data_tile_algs()`.
/// Iterates all registered dispatcher kernels instead of builder-generated .inc files.
template <auto SIGNATURE>
std::tuple<bool, float, std::string, int, int>
run_grouped_conv_backward_data_tile_algs(const ckt::Args<SIGNATURE>& args,
                                         const std::string& split_k,
                                         const index_t instance_index,
                                         const ckt::Inputs<SIGNATURE>& inputs,
                                         const ckt::Outputs<SIGNATURE>& outputs,
                                         const ck_tile::stream_config& s_conf)
{
    bool dummy_run_executed = false;
    float best_avg_time     = std::numeric_limits<float>::max();
    std::string best_op_name, op_name;
    int best_split_k                = 0;
    ck::index_t best_instance_index = -1;
    bool is_supported;
    float avg_time{0};
    bool all_instances_valid = true;

    using DataType =
        std::conditional_t<SIGNATURE.data_type == ckb::DataType::FP32,
                           float,
                           std::conditional_t<SIGNATURE.data_type == ckb::DataType::FP16,
                                              ck_tile::half_t,
                                              ck_tile::bfloat16_t>>;

    auto reference = ckt::alloc_outputs(args);
    using ReferenceInstance =
        typename ckb::ConvBuilder<SIGNATURE, ckt::ConvAlgorithm_Reference{}>::Instance;
    auto ref_conv   = ReferenceInstance{};
    auto ref_result = ckt::run(ref_conv, args, inputs, reference.get());

    const auto conv_param = args.to_ck_tile_conv_param();

    // Get max possible value in the output
    const std::size_t input_bytes_num = conv_param.template GetInputByte<DataType>();
    std::vector<DataType> ref(input_bytes_num / sizeof(DataType));
    HIP_CHECK_ERROR(
        hipMemcpy(&ref.data()[0], reference.get().input, input_bytes_num, hipMemcpyDeviceToHost));
    const float max_accumulated_value = *std::max_element(ref.begin(), ref.end());

    const index_t num_accums = conv_param.K_;

    // BWD data doesn't support split-K autodeduce value -1
    auto split_k_values = get_split_k_values(split_k);
    split_k_values.erase(std::remove(split_k_values.begin(), split_k_values.end(), -1),
                         split_k_values.end());

    // Register all generated backward data kernels
    static bool kernels_registered = false;
    if(!kernels_registered)
    {
        const auto arch_name = get_runtime_arch_name();
        ck_tile::dispatcher::register_all_grouped_conv_bwd_data_kernels(arch_name);
        kernels_registered = true;
    }

    // Get backward data kernels matching data type and spatial dims
    constexpr const char* dtype_str = get_dtype_string<SIGNATURE>();
    constexpr int ndim = SIGNATURE.spatial_dim;
    auto& registry = ck_tile::dispatcher::GroupedConvRegistry::instance();
    auto all_kernels = registry.filter([](const ck_tile::dispatcher::GroupedConvKernelInstance& k) {
        return k.key().op == ck_tile::dispatcher::GroupedConvOp::BackwardData &&
               k.key().dtype_in == dtype_str &&
               k.key().ndim_spatial == ndim;
    });

    // Set up thread-local buffer context
    // For bwd_data: inputs.output = dY, inputs.weight = W, outputs.input = dX
    // Backend mapping: ctx.input_ptr -> dY, ctx.weight_ptr -> W, ctx.output_ptr -> dX
    auto& ctx       = ck_tile::dispatcher::g_conv_dispatch_buffers;
    ctx.input_ptr   = inputs.output;  // dY (gradient from next layer)
    ctx.weight_ptr  = inputs.weight;  // W
    ctx.output_ptr  = outputs.input;  // dX (being computed)
    ctx.warmup      = s_conf.cold_niters_;
    ctx.repeat      = s_conf.nrepeat_;
    ctx.benchmarking = s_conf.time_kernel_;

    constexpr bool use_instance_string = true;

    index_t num_kernel = 0;
    for(const auto* kernel : all_kernels)
    {
        num_kernel++;
        // Skip if a specific instance was requested and this isn't it
        const bool running_specific_instance = (instance_index != -1);
        const bool current_is_target         = (num_kernel - 1 == instance_index);
        if(running_specific_instance && !current_is_target)
        {
            continue;
        }

        for(auto& k_batch : split_k_values)
        {
            auto problem  = args_to_problem<SIGNATURE>(args, k_batch);
            ctx.split_k   = k_batch;

            op_name       = kernel->name(use_instance_string);

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
                try
                {
                    avg_time = kernel->run(problem, nullptr);
                }
                catch(const std::runtime_error& e)
                {
                    std::cerr << "[Exception] " << op_name << " SplitK=" << k_batch
                              << " : " << e.what() << std::endl;
                    ck_tile::hip_check_error(hipDeviceSynchronize());
                    ck_tile::hip_check_error(hipGetLastError());
                    is_supported = false;
                }
                dummy_run_executed = true;
            }

            if(is_supported)
            {
                ckt::ValidationReport report;
                auto&& [rtol, atol] =
                    get_rtol_atol<SIGNATURE>(num_accums, k_batch, max_accumulated_value);
                ckt::Outputs<SIGNATURE>::reflect(
                    args,
                    [&](std::string_view name,
                        const auto& desc,
                        void* ckt::Outputs<SIGNATURE>::*ptr) {
                        report.check(name, desc, outputs.*ptr, reference.get().*ptr, rtol, atol);
                    });

                const bool valid = report.get_errors().empty();
                if(valid)
                {
                    if(avg_time < best_avg_time)
                    {
                        best_instance_index = num_kernel - 1;
                    }
                    best_avg_time = std::min(best_avg_time, avg_time);
                    best_op_name  = best_avg_time < avg_time ? best_op_name : op_name;
                    best_split_k  = best_avg_time < avg_time ? best_split_k : k_batch;
                    std::cout << "[Valid] Perf: " << std::setw(10) << avg_time << " ms," << " "
                              << op_name << " (instance " << num_kernel - 1 << "), SplitK "
                              << k_batch << std::endl;
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

    return std::make_tuple(
        all_instances_valid, best_avg_time, best_op_name, best_split_k, best_instance_index);
}

} // namespace ck_tile::builder::profiling
