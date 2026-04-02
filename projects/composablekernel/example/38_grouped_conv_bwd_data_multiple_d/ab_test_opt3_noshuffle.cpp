// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_data_multiple_d_xdl_cshuffle_v1.hpp"
#include "common.hpp"

using OutDataType      = BF16;
using WeiDataType      = BF16;
using AccDataType      = FP32;
using CShuffleDataType = BF16;
using DsDataType       = ck::Tuple<>;
using InDataType       = BF16;

using OutLayout = ck::tensor_layout::convolution::GNHWK;
using WeiLayout = ck::tensor_layout::convolution::GKYXC;
using DsLayout  = ck::Tuple<>;
using InLayout  = ck::tensor_layout::convolution::GNHWC;

using OutElementOp = PassThrough;
using WeiElementOp = PassThrough;
using InElementOp  = PassThrough;

template <typename DeviceConvPtr>
float run_single_instance(DeviceConvPtr& conv_ptr,
                          const ck::utils::conv::ConvParam& conv_params,
                          const HostTensorDescriptor& out_g_n_k_wos_desc,
                          const HostTensorDescriptor& wei_g_k_c_xs_desc,
                          const HostTensorDescriptor& in_g_n_c_wis_desc,
                          const OutElementOp& out_element_op,
                          const WeiElementOp& wei_element_op,
                          const InElementOp& in_element_op,
                          bool do_verify)
{
    Tensor<OutDataType> out(out_g_n_k_wos_desc);
    Tensor<WeiDataType> wei(wei_g_k_c_xs_desc);
    Tensor<InDataType> in_device(in_g_n_c_wis_desc);

    out.GenerateTensorValue(GeneratorTensor_2<OutDataType>{-5, 5});
    wei.GenerateTensorValue(GeneratorTensor_2<WeiDataType>{-5, 5});

    DeviceMem out_device_buf(sizeof(OutDataType) * out.mDesc.GetElementSpaceSize());
    DeviceMem wei_device_buf(sizeof(WeiDataType) * wei.mDesc.GetElementSpaceSize());
    DeviceMem in_device_buf(sizeof(InDataType) * in_device.mDesc.GetElementSpaceSize());

    out_device_buf.ToDevice(out.mData.data());
    wei_device_buf.ToDevice(wei.mData.data());
    in_device_buf.SetZero();

    std::array<ck::index_t, NDimSpatial + 3> a_g_n_k_wos_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> a_g_n_k_wos_strides{};
    std::array<ck::index_t, NDimSpatial + 3> b_g_k_c_xs_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> b_g_k_c_xs_strides{};
    std::array<ck::index_t, NDimSpatial + 3> e_g_n_c_wis_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> e_g_n_c_wis_strides{};
    std::array<ck::index_t, NDimSpatial> conv_filter_strides{};
    std::array<ck::index_t, NDimSpatial> conv_filter_dilations{};
    std::array<ck::index_t, NDimSpatial> input_left_pads{};
    std::array<ck::index_t, NDimSpatial> input_right_pads{};

    auto copy = [](auto& x, auto& y) { ck::ranges::copy(x, y.begin()); };

    copy(out_g_n_k_wos_desc.GetLengths(), a_g_n_k_wos_lengths);
    copy(out_g_n_k_wos_desc.GetStrides(), a_g_n_k_wos_strides);
    copy(wei_g_k_c_xs_desc.GetLengths(), b_g_k_c_xs_lengths);
    copy(wei_g_k_c_xs_desc.GetStrides(), b_g_k_c_xs_strides);
    copy(in_g_n_c_wis_desc.GetLengths(), e_g_n_c_wis_lengths);
    copy(in_g_n_c_wis_desc.GetStrides(), e_g_n_c_wis_strides);
    copy(conv_params.conv_filter_strides_, conv_filter_strides);
    copy(conv_params.conv_filter_dilations_, conv_filter_dilations);
    copy(conv_params.input_left_pads_, input_left_pads);
    copy(conv_params.input_right_pads_, input_right_pads);

    auto invoker  = conv_ptr->MakeInvokerPointer();
    auto argument = conv_ptr->MakeArgumentPointer(
        out_device_buf.GetDeviceBuffer(),
        wei_device_buf.GetDeviceBuffer(),
        std::array<const void*, 0>{},
        in_device_buf.GetDeviceBuffer(),
        a_g_n_k_wos_lengths,
        a_g_n_k_wos_strides,
        b_g_k_c_xs_lengths,
        b_g_k_c_xs_strides,
        std::array<std::array<ck::index_t, NDimSpatial + 3>, 0>{},
        std::array<std::array<ck::index_t, NDimSpatial + 3>, 0>{},
        e_g_n_c_wis_lengths,
        e_g_n_c_wis_strides,
        conv_filter_strides,
        conv_filter_dilations,
        input_left_pads,
        input_right_pads,
        out_element_op,
        wei_element_op,
        in_element_op);

    if(!conv_ptr->IsSupportedArgument(argument.get()))
    {
        return -1.0f;
    }

    float ave_time = invoker->Run(argument.get(), StreamConfig{nullptr, true});

    if(do_verify)
    {
        Tensor<InDataType> in_host(in_g_n_c_wis_desc);
        auto ref_conv = ck::tensor_operation::host::ReferenceConvBwdData<NDimSpatial,
                                                                         InDataType,
                                                                         WeiDataType,
                                                                         OutDataType,
                                                                         PassThrough,
                                                                         WeiElementOp,
                                                                         OutElementOp>();
        auto ref_invoker  = ref_conv.MakeInvoker();
        auto ref_argument = ref_conv.MakeArgument(in_host,
                                                  wei,
                                                  out,
                                                  conv_params.conv_filter_strides_,
                                                  conv_params.conv_filter_dilations_,
                                                  conv_params.input_left_pads_,
                                                  conv_params.input_right_pads_,
                                                  PassThrough{},
                                                  wei_element_op,
                                                  out_element_op);
        ref_invoker.Run(ref_argument);
        in_device_buf.FromDevice(in_device.mData.data());
        bool pass = ck::utils::check_err(in_device.mData, in_host.mData);
        std::cout << "  Verification: " << (pass ? "PASS" : "FAIL") << std::endl;
    }

    return ave_time;
}

using DeviceOp = ck::tensor_operation::device::DeviceGroupedConvBwdDataMultipleD<
    NDimSpatial,
    OutLayout,
    WeiLayout,
    DsLayout,
    InLayout,
    OutDataType,
    WeiDataType,
    DsDataType,
    InDataType,
    OutElementOp,
    WeiElementOp,
    InElementOp>;

template <typename Instance>
void add_instance(std::vector<std::unique_ptr<DeviceOp>>& instances)
{
    instances.push_back(std::make_unique<Instance>());
}

template <typename Tuple, std::size_t... Is>
void add_tuple_instances_impl(std::vector<std::unique_ptr<DeviceOp>>& instances,
                              std::index_sequence<Is...>)
{
    (add_instance<std::tuple_element_t<Is, Tuple>>(instances), ...);
}

template <typename Tuple>
void add_tuple_instances(std::vector<std::unique_ptr<DeviceOp>>& instances)
{
    add_tuple_instances_impl<Tuple>(instances,
                                    std::make_index_sequence<std::tuple_size_v<Tuple>>{});
}

// Include instance definitions
#include "ck/library/tensor_operation_instance/gpu/grouped_conv_bwd_data/device_grouped_conv_bwd_data_xdl_instance.hpp"

namespace inst = ck::tensor_operation::device::instance;

int main(int argc, char* argv[])
{
    namespace ctc = ck::tensor_layout::convolution;

    ExecutionConfig config;
    ck::utils::conv::ConvParam conv_params = DefaultConvParams;

    if(!parse_cmd_args(argc, argv, config, conv_params))
    {
        return EXIT_FAILURE;
    }

    const auto in_element_op  = InElementOp{};
    const auto wei_element_op = WeiElementOp{};
    const auto out_element_op = OutElementOp{};

    if(conv_params.num_dim_spatial_ != NDimSpatial)
    {
        std::cerr << "unsupported # of spatials dimensions" << std::endl;
        return EXIT_FAILURE;
    }

    const auto out_g_n_k_wos_desc =
        ck::utils::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<OutLayout>(
            conv_params);
    const auto wei_g_k_c_xs_desc =
        ck::utils::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<WeiLayout>(conv_params);
    const auto in_g_n_c_wis_desc =
        ck::utils::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<InLayout>(conv_params);

    std::cout << "Shape: G=" << conv_params.G_ << " N=" << conv_params.N_
              << " K=" << conv_params.K_ << " C=" << conv_params.C_
              << " Y=" << conv_params.filter_spatial_lengths_[0]
              << " X=" << conv_params.filter_spatial_lengths_[1]
              << " Hi=" << conv_params.input_spatial_lengths_[0]
              << " Wi=" << conv_params.input_spatial_lengths_[1]
              << " Ho=" << conv_params.output_spatial_lengths_[0]
              << " Wo=" << conv_params.output_spatial_lengths_[1]
              << std::endl;

    std::vector<std::unique_ptr<DeviceOp>> instances;

    // BF16 noshuffle instances (SPV=1) - Default specialization
    add_tuple_instances<inst::device_grouped_conv_bwd_data_xdl_bf16_noshuffle_instances<
        NDimSpatial, OutLayout, WeiLayout, DsLayout, InLayout, inst::ConvBwdDataDefault>>(instances);

    // BF16 noshuffle instances (SPV=1) - Filter1x1 specialization
    add_tuple_instances<inst::device_grouped_conv_bwd_data_xdl_bf16_noshuffle_instances<
        NDimSpatial, OutLayout, WeiLayout, DsLayout, InLayout, inst::ConvBwdDataFilter1x1Stride1Pad0>>(instances);

    // Also include other BF16 instances for comparison
    add_tuple_instances<inst::device_grouped_conv_bwd_data_xdl_bf16_generic_instances<
        NDimSpatial, OutLayout, WeiLayout, DsLayout, InLayout, inst::ConvBwdDataDefault>>(instances);
    add_tuple_instances<inst::device_grouped_conv_bwd_data_xdl_bf16_instances<
        NDimSpatial, OutLayout, WeiLayout, DsLayout, InLayout, inst::ConvBwdDataDefault>>(instances);
    add_tuple_instances<inst::device_grouped_conv_bwd_data_xdl_bf16_16_16_instances<
        NDimSpatial, OutLayout, WeiLayout, DsLayout, InLayout, inst::ConvBwdDataDefault>>(instances);
    add_tuple_instances<inst::device_grouped_conv_bwd_data_xdl_bf16_optimized_loads_instances<
        NDimSpatial, OutLayout, WeiLayout, DsLayout, InLayout, inst::ConvBwdDataDefault>>(instances);
    add_tuple_instances<inst::device_grouped_conv_bwd_data_xdl_bf16_nongrouped_match_instances<
        NDimSpatial, OutLayout, WeiLayout, DsLayout, InLayout, inst::ConvBwdDataDefault>>(instances);
    add_tuple_instances<inst::device_grouped_conv_bwd_data_xdl_bf16_instances<
        NDimSpatial, OutLayout, WeiLayout, DsLayout, InLayout, inst::ConvBwdDataFilter1x1Stride1Pad0>>(instances);
    add_tuple_instances<inst::device_grouped_conv_bwd_data_xdl_bf16_16_16_instances<
        NDimSpatial, OutLayout, WeiLayout, DsLayout, InLayout, inst::ConvBwdDataFilter1x1Stride1Pad0>>(instances);
    add_tuple_instances<inst::device_grouped_conv_bwd_data_xdl_bf16_optimized_loads_instances<
        NDimSpatial, OutLayout, WeiLayout, DsLayout, InLayout, inst::ConvBwdDataFilter1x1Stride1Pad0>>(instances);
    add_tuple_instances<inst::device_grouped_conv_bwd_data_xdl_bf16_nongrouped_match_instances<
        NDimSpatial, OutLayout, WeiLayout, DsLayout, InLayout, inst::ConvBwdDataFilter1x1Stride1Pad0>>(instances);

    std::cout << "\nTotal instances: " << instances.size() << std::endl;
    std::cout << "Verify: " << (config.do_verification ? "YES" : "NO") << std::endl;
    std::cout << "Time kernel: " << (config.time_kernel ? "YES" : "NO") << std::endl;
    std::cout << "\n--- Running all instances ---\n" << std::endl;

    float best_time  = std::numeric_limits<float>::max();
    int best_idx     = -1;
    int supported    = 0;

    for(int i = 0; i < static_cast<int>(instances.size()); ++i)
    {
        std::cout << "[Instance " << i << "] " << instances[i]->GetTypeString() << std::endl;

        float time = run_single_instance(instances[i],
                                         conv_params,
                                         out_g_n_k_wos_desc,
                                         wei_g_k_c_xs_desc,
                                         in_g_n_c_wis_desc,
                                         out_element_op,
                                         wei_element_op,
                                         in_element_op,
                                         config.do_verification);

        if(time < 0)
        {
            std::cout << "  -> Not supported for this shape" << std::endl;
            continue;
        }

        supported++;
        std::size_t flop      = conv_params.GetFlops();
        std::size_t num_btype = conv_params.GetByte<InDataType, WeiDataType, OutDataType>();
        float tflops          = static_cast<float>(flop) / 1.E9 / time;
        float gb_per_sec      = num_btype / 1.E6 / time;

        std::cout << "  -> Time: " << time << " ms, " << tflops << " TFlops, " << gb_per_sec
                  << " GB/s" << std::endl;

        if(time < best_time)
        {
            best_time = time;
            best_idx  = i;
        }
    }

    std::cout << "\n=== SUMMARY ===" << std::endl;
    std::cout << "Supported instances: " << supported << " / " << instances.size() << std::endl;
    if(best_idx >= 0)
    {
        std::size_t flop      = conv_params.GetFlops();
        std::size_t num_btype = conv_params.GetByte<InDataType, WeiDataType, OutDataType>();
        float tflops          = static_cast<float>(flop) / 1.E9 / best_time;
        float gb_per_sec      = num_btype / 1.E6 / best_time;

        std::cout << "Best: Instance " << best_idx << std::endl;
        std::cout << "  " << instances[best_idx]->GetTypeString() << std::endl;
        std::cout << "  Time: " << best_time << " ms, " << tflops << " TFlops, " << gb_per_sec
                  << " GB/s" << std::endl;
    }

    return 0;
}
