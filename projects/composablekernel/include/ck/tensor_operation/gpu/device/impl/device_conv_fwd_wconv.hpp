// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <sstream>

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_conv_fwd.hpp"
#include "ck/tensor_operation/gpu/device/convolution_forward_specialization.hpp"
#include "ck/tensor_operation/gpu/device/matrix_padder.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_utils.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_conv_fwd_wconv.hpp"
#include "ck/tensor_operation/operator_transform/transform_conv_fwd_to_wconv.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"
#include "ck/utility/loop_scheduler.hpp"

template <typename T>
struct Debug;

namespace ck {
namespace tensor_operation {
namespace device {

template <index_t NDimSpatial,
          typename InLayout,
          typename WeiLayout,
          typename AccLayout,
          typename InDataType,
          typename WeiDataType,
          typename AccDataType,
          typename InElementwiseOperation,
          typename WeiElementwiseOperation,
          typename AccElementwiseOperation,
          ConvolutionForwardSpecialization ConvForwardSpecialization,
          index_t NumPrefetch,
          index_t BlockSize,
          index_t HPerBlock,
          index_t WPerBlock,
          index_t CPerBlock,
          index_t KPerBlock,
          index_t HRepeat,
          index_t WRepeat,
          index_t HPerWconv,
          index_t WPerWconv,
          index_t FilterSize,
          index_t DilationX,
          index_t DilationY,
          typename InBlockTransferThreadClusterLengths,
          index_t InBlockTransferSrcScalarPerVector,
          index_t InBlockTransferDstScalarPerVector,
          bool InEnableLds,
          bool InBlockLdsAddExtraM,
          typename WeiBlockTransferThreadClusterLengths,
          index_t WeiBlockTransferSrcScalarPerVector,
          index_t WeiBlockTransferDstScalarPerVector,
          bool WeiEnableLds,
          bool WeiBlockLdsAddExtraM,
          typename AccBlockTransferClusterLengths,
          index_t AccBlockTransferScalarPerVector,
          bool AccEnableLds,
          bool EnableAsync,
          bool EnableWaveGroup>
struct DeviceConvWconv : public DeviceGroupedConvFwd<NDimSpatial,
                                                     InLayout,
                                                     WeiLayout,
                                                     AccLayout,
                                                     InDataType,
                                                     WeiDataType,
                                                     AccDataType,
                                                     InElementwiseOperation,
                                                     WeiElementwiseOperation,
                                                     AccElementwiseOperation>
{
    using DeviceOp = DeviceConvWconv;
    static constexpr auto conv_to_wconv_transformer =
        TransformConvFwdToWconv<NDimSpatial, ConvForwardSpecialization>{};

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};
    static constexpr auto I5 = Number<5>{};
    static constexpr auto I6 = Number<6>{};

    static constexpr index_t GridFilterSize = (FilterSize == 2) ? 1 : FilterSize;
    static constexpr index_t GridHPerBlock  = (FilterSize == 2) ? HPerBlock / 2 : HPerBlock;
    static constexpr index_t GridWPerBlock  = (FilterSize == 2) ? WPerBlock / 2 : WPerBlock;
    static constexpr index_t GridCPerBlock  = (FilterSize == 2) ? CPerBlock * 4 : CPerBlock;
    static constexpr index_t GridHRepeat    = (FilterSize == 2) ? HRepeat / 2 : HRepeat;
    static constexpr index_t GridWRepeat    = (FilterSize == 2) ? WRepeat / 2 : WRepeat;

    // Describe how data read from Global memory
    static auto
    MakeInGridDescriptor(const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_lengths,
                         const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_strides,
                         const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_lengths,
                         const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_strides,
                         const std::array<index_t, NDimSpatial + 3>& e_g_n_k_wos_lengths,
                         const std::array<index_t, NDimSpatial + 3>& e_g_n_k_wos_strides,
                         const std::array<index_t, NDimSpatial>& conv_filter_strides,
                         const std::array<index_t, NDimSpatial>& conv_filter_dilations,
                         const std::array<index_t, NDimSpatial>& input_left_pads,
                         const std::array<index_t, NDimSpatial>& input_right_pads)
    {
        // H W C
        const auto in_data_raw_desc =
            conv_to_wconv_transformer.template MakeADescriptor_H_W_C<InLayout>(
                a_g_n_c_wis_lengths,
                a_g_n_c_wis_strides,
                b_g_k_c_xs_lengths,
                b_g_k_c_xs_strides,
                e_g_n_k_wos_lengths,
                e_g_n_k_wos_strides,
                conv_filter_strides,
                conv_filter_dilations,
                input_left_pads,
                input_right_pads);

        // H W C with pad
        const auto in_data_desc =
            PadTensorDescriptor(in_data_raw_desc,
                                make_tuple(GridHPerBlock, GridWPerBlock, GridCPerBlock),
                                Sequence<true, true, true>{});
        return in_data_desc;
    }

    static auto
    MakeWeiGridDescriptor(const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_lengths,
                          const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_strides)
    {
        const auto wei_data_raw_desc =
            conv_to_wconv_transformer.template MakeBDescriptor_K_YX_C<WeiLayout>(
                b_g_k_c_xs_lengths, b_g_k_c_xs_strides);

        const auto wei_data_desc = PadTensorDescriptor(wei_data_raw_desc,
                                                       make_tuple(KPerBlock, 1, GridCPerBlock),
                                                       Sequence<true, false, true>{});
        return wei_data_desc;
    }

    static auto
    MakeAccGridDescriptor(const std::array<index_t, NDimSpatial + 3>& e_g_n_k_wos_lengths,
                          const std::array<index_t, NDimSpatial + 3>& e_g_n_k_wos_strides)
    {
        const auto acc_data_raw_desc =
            conv_to_wconv_transformer.template MakeCDescriptor_H_W_K<AccLayout>(
                e_g_n_k_wos_lengths, e_g_n_k_wos_strides);

        const auto acc_data_desc =
            PadTensorDescriptor(acc_data_raw_desc,
                                make_tuple(GridHPerBlock, GridWPerBlock, KPerBlock),
                                Sequence<true, true, true>{});

        return acc_data_desc;
    }

    // desc for problem definition
    using InGridDesc =
        decltype(DeviceOp::MakeInGridDescriptor({}, {}, {}, {}, {}, {}, {}, {}, {}, {}));
    using WeiGridDesc = decltype(DeviceOp::MakeWeiGridDescriptor({}, {}));
    using AccGridDesc = remove_cvref_t<decltype(MakeAccGridDescriptor({}, {}))>;

    // GridwiseConv
    using GridwiseConv = GridwiseConv_Wconv<BlockSize,
                                            InDataType,
                                            WeiDataType,
                                            AccDataType,
                                            InGridDesc,
                                            WeiGridDesc,
                                            AccGridDesc,
                                            InElementwiseOperation,
                                            WeiElementwiseOperation,
                                            AccElementwiseOperation,
                                            GridHPerBlock,
                                            GridWPerBlock,
                                            GridCPerBlock,
                                            KPerBlock,
                                            GridHRepeat,
                                            GridWRepeat,
                                            HPerWconv,
                                            WPerWconv,
                                            GridFilterSize,
                                            DilationX,
                                            DilationY,
                                            InBlockTransferThreadClusterLengths,
                                            InBlockTransferSrcScalarPerVector,
                                            InBlockTransferDstScalarPerVector,
                                            InEnableLds,
                                            InBlockLdsAddExtraM,
                                            WeiBlockTransferThreadClusterLengths,
                                            WeiBlockTransferSrcScalarPerVector,
                                            WeiBlockTransferDstScalarPerVector,
                                            WeiEnableLds,
                                            WeiBlockLdsAddExtraM,
                                            AccBlockTransferClusterLengths,
                                            AccBlockTransferScalarPerVector,
                                            AccEnableLds,
                                            EnableAsync,
                                            NumPrefetch,
                                            EnableWaveGroup>;

    // Argument
    struct Argument : public BaseArgument
    {
        Argument(const void* p_in,
                 const void* p_wei,
                 void* p_acc,
                 const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_lengths,
                 const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_strides,
                 const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_lengths,
                 const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_strides,
                 const std::array<index_t, NDimSpatial + 3>& e_g_n_k_wos_lengths,
                 const std::array<index_t, NDimSpatial + 3>& e_g_n_k_wos_strides,
                 const std::array<index_t, NDimSpatial>& conv_filter_strides,
                 const std::array<index_t, NDimSpatial>& conv_filter_dilations,
                 const std::array<index_t, NDimSpatial>& input_left_pads,
                 const std::array<index_t, NDimSpatial>& input_right_pads,
                 const InElementwiseOperation& in_element_op,
                 const WeiElementwiseOperation& wei_element_op,
                 const AccElementwiseOperation& acc_element_op)
            : p_in_grid_{static_cast<const InDataType*>(p_in)},
              p_wei_grid_{static_cast<const WeiDataType*>(p_wei)},
              p_acc_grid_{static_cast<AccDataType*>(p_acc)},
              num_group_{a_g_n_c_wis_lengths[0]},
              acc_grid_desc_{
                  DeviceOp::MakeAccGridDescriptor(e_g_n_k_wos_lengths, e_g_n_k_wos_strides)},
              in_grid_desc_{DeviceOp::MakeInGridDescriptor(a_g_n_c_wis_lengths,
                                                           a_g_n_c_wis_strides,
                                                           b_g_k_c_xs_lengths,
                                                           b_g_k_c_xs_strides,
                                                           e_g_n_k_wos_lengths,
                                                           e_g_n_k_wos_strides,
                                                           conv_filter_strides,
                                                           conv_filter_dilations,
                                                           input_left_pads,
                                                           input_right_pads)},
              wei_grid_desc_{
                  DeviceOp::MakeWeiGridDescriptor(b_g_k_c_xs_lengths, b_g_k_c_xs_strides)},
              block_2_etile_map_{GridwiseConv::MakeDefaultBlock2CTileMap(acc_grid_desc_, 1, 1)},
              compute_ptr_offset_of_batch_{},
              in_element_op_{in_element_op},
              wei_element_op_{wei_element_op},
              acc_element_op_{acc_element_op},
              a_g_n_c_wis_lengths_{a_g_n_c_wis_lengths},
              a_g_n_c_wis_strides_{a_g_n_c_wis_strides},
              b_g_k_c_xs_lengths_{b_g_k_c_xs_lengths},
              b_g_k_c_xs_strides_{b_g_k_c_xs_strides},
              e_g_n_k_wos_lengths_{e_g_n_k_wos_lengths},
              e_g_n_k_wos_strides_{e_g_n_k_wos_strides},
              conv_filter_strides_{conv_filter_strides},
              conv_filter_dilations_{conv_filter_dilations},
              input_left_pads_{input_left_pads},
              input_right_pads_{input_right_pads}
        {
            // A/B/E Batch Stride
            compute_ptr_offset_of_batch_.BatchStrideA_ = a_g_n_c_wis_strides[0];
            compute_ptr_offset_of_batch_.BatchStrideB_ = b_g_k_c_xs_strides[0];
            compute_ptr_offset_of_batch_.BatchStrideE_ = e_g_n_k_wos_strides[0];
        }

        void Print() const
        {
            std::cout << "In: " << in_grid_desc_ << std::endl;
            std::cout << "Wei: " << wei_grid_desc_ << std::endl;
            std::cout << "Acc: " << acc_grid_desc_ << std::endl;
        }

        // pointers
        const InDataType* p_in_grid_;
        const WeiDataType* p_wei_grid_;
        AccDataType* p_acc_grid_;

        // tensor descriptors for problem definiton
        index_t num_group_;
        AccGridDesc acc_grid_desc_;

        // tensor descriptors for block/thread-wise copy
        InGridDesc in_grid_desc_;
        WeiGridDesc wei_grid_desc_;

        // block-to-e-tile map
        typename GridwiseConv::DefaultBlock2CTileMap block_2_etile_map_;

        // for computing batch offset
        ComputePtrOffsetOfStridedBatch<I1, I1, I1> compute_ptr_offset_of_batch_;

        // element-wise op
        InElementwiseOperation in_element_op_;
        WeiElementwiseOperation wei_element_op_;
        AccElementwiseOperation acc_element_op_;

        // for checking IsSupportedArgument()
        std::array<index_t, NDimSpatial + 3> a_g_n_c_wis_lengths_;
        std::array<index_t, NDimSpatial + 3> a_g_n_c_wis_strides_;
        std::array<index_t, NDimSpatial + 3> b_g_k_c_xs_lengths_;
        std::array<index_t, NDimSpatial + 3> b_g_k_c_xs_strides_;
        std::array<index_t, NDimSpatial + 3> e_g_n_k_wos_lengths_;
        std::array<index_t, NDimSpatial + 3> e_g_n_k_wos_strides_;
        std::array<index_t, NDimSpatial> conv_filter_strides_;
        std::array<index_t, NDimSpatial> conv_filter_dilations_;
        std::array<index_t, NDimSpatial> input_left_pads_;
        std::array<index_t, NDimSpatial> input_right_pads_;
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        using Argument = DeviceOp::Argument;

        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            if(stream_config.log_level_ > 0)
            {
                arg.Print();
            }

            const index_t grid_size =
                arg.block_2_etile_map_.CalculateGridSize(arg.acc_grid_desc_) * arg.num_group_;

            auto launch_kernel = [&](auto has_main_block_loop) {
                constexpr bool has_main_loop = has_main_block_loop.value;

                if constexpr(EnableWaveGroup)
                {
                    const auto kernel = kernel_grouped_conv_wconv_wavegroup<
                        GridwiseConv,
                        InDataType,
                        WeiDataType,
                        AccDataType,
                        InElementwiseOperation,
                        WeiElementwiseOperation,
                        AccElementwiseOperation,
                        DeviceOp::InGridDesc,
                        DeviceOp::WeiGridDesc,
                        DeviceOp::AccGridDesc,
                        remove_reference_t<typename GridwiseConv::DefaultBlock2CTileMap>,
                        ComputePtrOffsetOfStridedBatch<I1, I1, I1>,
                        has_main_loop>;

                    return launch_and_time_kernel(stream_config,
                                                  kernel,
                                                  dim3(grid_size),
                                                  dim3(BlockSize),
                                                  0,
                                                  arg.p_in_grid_,
                                                  arg.p_wei_grid_,
                                                  arg.p_acc_grid_,
                                                  arg.in_element_op_,
                                                  arg.wei_element_op_,
                                                  arg.acc_element_op_,
                                                  arg.a_g_n_c_wis_lengths_[0], // Group count
                                                  arg.in_grid_desc_,
                                                  arg.wei_grid_desc_,
                                                  arg.acc_grid_desc_,
                                                  arg.block_2_etile_map_,
                                                  arg.compute_ptr_offset_of_batch_);
                }
                else
                {
                    const auto kernel = kernel_grouped_conv_wconv<
                        GridwiseConv,
                        InDataType,
                        WeiDataType,
                        AccDataType,
                        InElementwiseOperation,
                        WeiElementwiseOperation,
                        AccElementwiseOperation,
                        DeviceOp::InGridDesc,
                        DeviceOp::WeiGridDesc,
                        DeviceOp::AccGridDesc,
                        remove_reference_t<typename GridwiseConv::DefaultBlock2CTileMap>,
                        ComputePtrOffsetOfStridedBatch<I1, I1, I1>,
                        has_main_loop>;

                    return launch_and_time_kernel(stream_config,
                                                  kernel,
                                                  dim3(grid_size),
                                                  dim3(BlockSize),
                                                  0,
                                                  arg.p_in_grid_,
                                                  arg.p_wei_grid_,
                                                  arg.p_acc_grid_,
                                                  arg.in_element_op_,
                                                  arg.wei_element_op_,
                                                  arg.acc_element_op_,
                                                  arg.a_g_n_c_wis_lengths_[0], // Group count
                                                  arg.in_grid_desc_,
                                                  arg.wei_grid_desc_,
                                                  arg.acc_grid_desc_,
                                                  arg.block_2_etile_map_,
                                                  arg.compute_ptr_offset_of_batch_);
                }
            };

            const auto C = arg.in_grid_desc_.GetLength(I2);

            if(GridwiseConv::CalculateHasMainBlockLoop(C))
            {
                return launch_kernel(integral_constant<bool, true>{});
            }
            else
            {
                return launch_kernel(integral_constant<bool, false>{});
            }
        }

        float Run(const BaseArgument* p_arg,
                  const StreamConfig& stream_config = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), stream_config);
        }
    };

    static constexpr bool IsValidCompilationParameter()
    {
        // TODO: properly implement this check
        return true;
    }

    static bool IsSupportedArgument(const Argument& arg)
    {
        namespace ctc = ck::tensor_layout::convolution;

        if(1) // ck::is_gfx13_supported())
        {
            if constexpr(!(is_same_v<AccDataType, float> || is_same_v<AccDataType, ck::half_t> ||
                           is_same_v<AccDataType, ck::bhalf_t> || is_same_v<AccDataType, int32_t>))
            {
                printf("DeviceOp err: AccDataType");
                return false;
            }
        }
        else
        {
            printf("DeviceOp err: Arch");
            return false;
        }

        // check ConvolutionForwardSpecialization
        if constexpr(ConvForwardSpecialization ==
                     ConvolutionForwardSpecialization::Filter1x1Stride1Pad0)
        {
            // check if it's 1x1, stride=1 conv
            for(index_t i = 0; i < NDimSpatial; ++i)
            {
                const index_t X          = arg.b_g_k_c_xs_lengths_[i + 3];
                const index_t ConvStride = arg.conv_filter_strides_[i];
                const index_t LeftPad    = arg.input_left_pads_[i];
                const index_t RightPad   = arg.input_right_pads_[i];

                if(!(X == 1 && ConvStride == 1 && LeftPad == 0 && RightPad == 0))
                {
                    return false;
                }
            }
        }
        else if constexpr((ConvForwardSpecialization ==
                           ConvolutionForwardSpecialization::Filter2x2Stride2Pad0) ||
                          (ConvForwardSpecialization ==
                           ConvolutionForwardSpecialization::Filter2x2Stride2OddHWPad0))
        {
            // check if it's 1x1, stride=1 conv
            for(index_t i = 0; i < NDimSpatial; ++i)
            {
                const index_t X          = arg.b_g_k_c_xs_lengths_[i + 3];
                const index_t ConvStride = arg.conv_filter_strides_[i];
                const index_t LeftPad    = arg.input_left_pads_[i];
                const index_t RightPad   = arg.input_right_pads_[i];

                if(!(X == 2 && ConvStride == 2 && LeftPad == 0 && RightPad == 0))
                {
                    return false;
                }
            }
        }
        else if constexpr((ConvForwardSpecialization ==
                           ConvolutionForwardSpecialization::Filter3x3Stride1Pad0) ||
                          (ConvForwardSpecialization ==
                           ConvolutionForwardSpecialization::Filter3x3Stride1MultiLayerPad0))
        {
            for(index_t i = 0; i < NDimSpatial; ++i)
            {
                const index_t X          = arg.b_g_k_c_xs_lengths_[i + 3];
                const index_t ConvStride = arg.conv_filter_strides_[i];
                const index_t LeftPad    = arg.input_left_pads_[i];
                const index_t RightPad   = arg.input_right_pads_[i];
                const index_t Dilation   = arg.conv_filter_dilations_[i];
                if(!(X == 3 && ConvStride == 1 && (Dilation < 3) && LeftPad == Dilation &&
                     RightPad == Dilation))
                {
                    return false;
                }
            }
        }
        else
        {
            return false;
        }

        // check vector access of InData
        // FIXME: layout
        if constexpr(is_same_v<InLayout, ctc::G_NW_C> || is_same_v<InLayout, ctc::G_NHW_C> ||
                     is_same_v<InLayout, ctc::G_NDHW_C> || is_same_v<InLayout, ctc::GNWC> ||
                     is_same_v<InLayout, ctc::GNHWC> || is_same_v<InLayout, ctc::GNDHWC> ||
                     is_same_v<InLayout, ctc::NWGC> || is_same_v<InLayout, ctc::NHWGC> ||
                     is_same_v<InLayout, ctc::NDHWGC>)
        {
            const index_t C = arg.a_g_n_c_wis_lengths_[2];

            if(!(C % InBlockTransferSrcScalarPerVector == 0))
            {
                return false;
            }
        }
        else
        {
            return false;
        }

        // check vector access of WeiData
        // FIXME: layout
        if constexpr(is_same_v<WeiLayout, ctc::G_K_X_C> || is_same_v<WeiLayout, ctc::G_K_YX_C> ||
                     is_same_v<WeiLayout, ctc::G_K_ZYX_C> || is_same_v<WeiLayout, ctc::GKXC> ||
                     is_same_v<WeiLayout, ctc::GKYXC> || is_same_v<WeiLayout, ctc::GKZYXC> ||
                     is_same_v<WeiLayout, ctc::KXGC> || is_same_v<WeiLayout, ctc::KYXGC> ||
                     is_same_v<WeiLayout, ctc::KZYXGC>)

        {
            const index_t C = arg.b_g_k_c_xs_lengths_[2];

            if(!(C % WeiBlockTransferSrcScalarPerVector == 0))
            {
                return false;
            }
        }
        else
        {
            return false;
        }

        // check vector access of Accum
        if constexpr(is_same_v<AccLayout, ctc::G_NW_K> || is_same_v<AccLayout, ctc::G_NHW_K> ||
                     is_same_v<AccLayout, ctc::G_NDHW_K> || is_same_v<AccLayout, ctc::GNWK> ||
                     is_same_v<AccLayout, ctc::GNHWK> || is_same_v<AccLayout, ctc::GNDHWK> ||
                     is_same_v<AccLayout, ctc::NWGK> || is_same_v<AccLayout, ctc::NHWGK> ||
                     is_same_v<AccLayout, ctc::NDHWGK>)
        {
            const index_t K = arg.e_g_n_k_wos_lengths_[2];

            if(!(K % AccBlockTransferScalarPerVector == 0))
            {
                return false;
            }
        }
        else
        {
            return false;
        }

        // check Gridwise Conv
        return GridwiseConv::CheckValidity(
            arg.in_grid_desc_, arg.wei_grid_desc_, arg.acc_grid_desc_, arg.block_2_etile_map_);
        return true;
    }

    // polymorphic
    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto MakeArgument(const void* p_in,
                             const void* p_wei,
                             void* p_acc,
                             const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_lengths,
                             const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_strides,
                             const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_lengths,
                             const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_strides,
                             const std::array<index_t, NDimSpatial + 3>& e_g_n_k_wos_lengths,
                             const std::array<index_t, NDimSpatial + 3>& e_g_n_k_wos_strides,
                             const std::array<index_t, NDimSpatial>& conv_filter_strides,
                             const std::array<index_t, NDimSpatial>& conv_filter_dilations,
                             const std::array<index_t, NDimSpatial>& input_left_pads,
                             const std::array<index_t, NDimSpatial>& input_right_pads,
                             const InElementwiseOperation& in_element_op,
                             const WeiElementwiseOperation& wei_element_op,
                             const AccElementwiseOperation& acc_element_op)
    {
        return Argument{p_in,
                        p_wei,
                        p_acc,
                        a_g_n_c_wis_lengths,
                        a_g_n_c_wis_strides,
                        b_g_k_c_xs_lengths,
                        b_g_k_c_xs_strides,
                        e_g_n_k_wos_lengths,
                        e_g_n_k_wos_strides,
                        conv_filter_strides,
                        conv_filter_dilations,
                        input_left_pads,
                        input_right_pads,
                        in_element_op,
                        wei_element_op,
                        acc_element_op};
    }

    static auto MakeInvoker() { return Invoker{}; }

    // polymorphic
    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_in,
                        const void* p_wei,
                        void* p_acc,
                        const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_lengths,
                        const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_strides,
                        const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_lengths,
                        const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_strides,
                        const std::array<index_t, NDimSpatial + 3>& e_g_n_k_wos_lengths,
                        const std::array<index_t, NDimSpatial + 3>& e_g_n_k_wos_strides,
                        const std::array<index_t, NDimSpatial>& conv_filter_strides,
                        const std::array<index_t, NDimSpatial>& conv_filter_dilations,
                        const std::array<index_t, NDimSpatial>& input_left_pads,
                        const std::array<index_t, NDimSpatial>& input_right_pads,
                        const InElementwiseOperation& in_element_op,
                        const WeiElementwiseOperation& wei_element_op,
                        const AccElementwiseOperation& acc_element_op) override
    {
        return std::make_unique<Argument>(p_in,
                                          p_wei,
                                          p_acc,
                                          a_g_n_c_wis_lengths,
                                          a_g_n_c_wis_strides,
                                          b_g_k_c_xs_lengths,
                                          b_g_k_c_xs_strides,
                                          e_g_n_k_wos_lengths,
                                          e_g_n_k_wos_strides,
                                          conv_filter_strides,
                                          conv_filter_dilations,
                                          input_left_pads,
                                          input_right_pads,
                                          in_element_op,
                                          wei_element_op,
                                          acc_element_op);
    }

    // polymorphic
    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    // polymorphic
    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        // clang-format off
        str << "DeviceConvWconv"
            << "<"
            << BlockSize << ", "
            << HPerBlock << ", "
            << WPerBlock << ", "
            << CPerBlock << ", "
            << KPerBlock << ", "
            << HRepeat << ", "
            << WRepeat << ", "
            << HPerWconv << ", "
            << WPerWconv << ", "
            << FilterSize << ", "
            << DilationX << ", "
            << DilationY << ">"
            << " InEnableLds: "
            << InEnableLds << ", "
            << "WeiEnableLds: "
            << WeiEnableLds << ", "
            << "NumPrefetch: "
            << NumPrefetch << ", "
            << "EnableWaveGroup: "
            << EnableWaveGroup;
        // clang-format on

        return str.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
