// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/ck.hpp"
#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/convolution_forward_specialization.hpp"

namespace ck {
namespace tensor_operation {

template <
        index_t N_,
        index_t Hi_,
        index_t Wi_,
        index_t Ho_,
        index_t Wo_,
        index_t Y_,
        index_t X_,
        index_t K_,
        index_t C_,
        index_t HiStride_,
        index_t WiStride_,
        index_t HoStride_,
        index_t WoStride_,
        index_t XStride_,
        index_t CStrideTensorA_,
        index_t CStrideTensorB_,
        index_t KStrideTensorB_,
        index_t KStrideTensorC_,
        index_t NStrideTensorA_,
        index_t NStrideTensorC_,
        index_t GStrideTensorA_,
        index_t GStrideTensorB_,
        index_t GStrideTensorC_,
        index_t ConvStrideH_,
        index_t ConvStrideW_,
        index_t ConvDilationH_,
        index_t ConvDilationW_,
        index_t InLeftPadH_,
        index_t InLeftPadW_,
        index_t InRightPadH_,
        index_t InRightPadW_,
        index_t ZYX_,
        index_t NumGroupsToMerge>
struct TransformConvFwdToGemm_V2
{
    private:
    static constexpr auto I1 = Number<1>{};

    public:

    __host__ __device__ consteval TransformConvFwdToGemm_V2() {}

    template <typename ALayout,
              typename ck::enable_if<
                                         (is_same_v<ALayout, tensor_layout::convolution::G_NHW_C> ||
                                          is_same_v<ALayout, tensor_layout::convolution::NHWGC> ||
                                          is_same_v<ALayout, tensor_layout::convolution::GNHWC>),
                                     bool>::type = false>
    __host__ __device__ consteval auto MakeADescriptor_M_K() const

    {
        if constexpr(NumGroupsToMerge == 1)
        {
            constexpr auto in_n_hi_wi_c_desc = make_naive_tensor_descriptor(
                make_tuple(N_, Hi_, Wi_, C_),
                make_tuple(NStrideTensorA_, HiStride_, WiStride_, CStrideTensorA_));

            constexpr auto in_n_hip_wip_c_desc = transform_tensor_descriptor(
                in_n_hi_wi_c_desc,
                make_tuple(make_pass_through_transform(N_),
                            make_pad_transform(Hi_, InLeftPadH_, InRightPadH_),
                            make_pad_transform(Wi_, InLeftPadW_, InRightPadW_),
                            make_pass_through_transform(C_)),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

            constexpr auto in_n_y_ho_x_wo_c_desc = transform_tensor_descriptor(
                in_n_hip_wip_c_desc,
                make_tuple(make_pass_through_transform(N_),
                            make_embed_transform(make_tuple(Y_, Ho_),
                                                make_tuple(ConvDilationH_, ConvStrideH_)),
                            make_embed_transform(make_tuple(X_, Wo_),
                                                make_tuple(ConvDilationW_, ConvStrideW_)),
                            make_pass_through_transform(C_)),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3, 4>{}, Sequence<5>{}));

            return transform_tensor_descriptor(
                in_n_y_ho_x_wo_c_desc,
                make_tuple(make_merge_transform(make_tuple(N_, Ho_, Wo_)),
                            make_merge_transform(make_tuple(Y_, X_, C_))),
                make_tuple(Sequence<0, 2, 4>{}, Sequence<1, 3, 5>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));
        }
        else
        {

            constexpr auto in_n_hi_wi_groups_c_desc = make_naive_tensor_descriptor(
                make_tuple(N_, Hi_, Wi_, NumGroupsToMerge, C_),
                make_tuple(
                    NStrideTensorA_, HiStride_, WiStride_, GStrideTensorA_, CStrideTensorA_));

            constexpr auto in_n_hip_wip_groups_c_desc = transform_tensor_descriptor(
                in_n_hi_wi_groups_c_desc,
                make_tuple(make_pass_through_transform(N_),
                            make_pad_transform(Hi_, InLeftPadH_, InRightPadH_),
                            make_pad_transform(Wi_, InLeftPadW_, InRightPadW_),
                            make_pass_through_transform(NumGroupsToMerge),
                            make_pass_through_transform(C_)),
                make_tuple(
                    Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
                make_tuple(
                    Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}));

            constexpr auto in_n_y_ho_x_wo_groups_c_desc = transform_tensor_descriptor(
                in_n_hip_wip_groups_c_desc,
                make_tuple(make_pass_through_transform(N_),
                            make_embed_transform(make_tuple(Y_, Ho_),
                                                make_tuple(ConvDilationH_, ConvStrideH_)),
                            make_embed_transform(make_tuple(X_, Wo_),
                                                make_tuple(ConvDilationW_, ConvStrideW_)),
                            make_pass_through_transform(NumGroupsToMerge),
                            make_pass_through_transform(C_)),
                make_tuple(
                    Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
                make_tuple(Sequence<0>{},
                            Sequence<1, 2>{},
                            Sequence<3, 4>{},
                            Sequence<5>{},
                            Sequence<6>{}));

            return transform_tensor_descriptor(
                in_n_y_ho_x_wo_groups_c_desc,
                make_tuple(make_merge_transform(make_tuple(N_, Ho_, Wo_, NumGroupsToMerge)),
                            make_merge_transform(make_tuple(Y_, X_, C_))),
                make_tuple(Sequence<0, 2, 4, 5>{}, Sequence<1, 3, 6>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));
        }
    }

    template <typename BLayout,
              typename ck::enable_if<is_same_v<BLayout, tensor_layout::convolution::GKXC> ||
                                         is_same_v<BLayout, tensor_layout::convolution::GKYXC> ||
                                         is_same_v<BLayout, tensor_layout::convolution::GKZYXC>,
                                     bool>::type = false>
    __host__ __device__ consteval auto MakeBDescriptor_N_K() const
    {
        if constexpr(NumGroupsToMerge == 1)
        {
            return make_naive_tensor_descriptor_packed(make_tuple(K_, ZYX_ * C_));
        }
        else
        {
            constexpr auto wei_gemmn_groups_gemmk_desc = make_naive_tensor_descriptor(
                make_tuple(NumGroupsToMerge, K_, ZYX_ * C_),
                make_tuple(GStrideTensorB_, KStrideTensorB_, CStrideTensorB_));
            return transform_tensor_descriptor(
                wei_gemmn_groups_gemmk_desc,
                make_tuple(make_merge_transform(make_tuple(NumGroupsToMerge, K_)),
                            make_pass_through_transform(ZYX_ * C_)),
                make_tuple(Sequence<0, 1>{}, Sequence<2>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));
        }
    }

    template <typename CLayout,
              typename ck::enable_if<
                                         (is_same_v<CLayout, tensor_layout::convolution::G_NHW_K> ||
                                          is_same_v<CLayout, tensor_layout::convolution::NHWGK> ||
                                          is_same_v<CLayout, tensor_layout::convolution::GNHWK>),
                                     bool>::type = false>
    __host__ __device__ consteval auto MakeCDescriptor_M_N() const
    {
        constexpr index_t NDoHoWo = N_ * Ho_ * Wo_;
        if constexpr(NumGroupsToMerge == 1)
        {
            return make_naive_tensor_descriptor(make_tuple(NDoHoWo, K_),
                                                make_tuple(WoStride_, KStrideTensorC_));
        }
        else
        {
            constexpr auto nhwo_groups_k_1_desc =
                make_naive_tensor_descriptor(make_tuple(N_, Ho_, Wo_, NumGroupsToMerge, 1, K_),
                                             make_tuple(NStrideTensorC_,
                                                        HoStride_,
                                                        WoStride_,
                                                        GStrideTensorC_,
                                                        GStrideTensorC_,
                                                        KStrideTensorC_));
            // Padd 1 to NumGroupsToMerge
            constexpr auto padded_desc = transform_tensor_descriptor(
                nhwo_groups_k_1_desc,
                make_tuple(make_merge_transform(make_tuple(N_, Ho_, Wo_)),
                           make_pass_through_transform(NumGroupsToMerge),
                           make_pad_transform(1, 0, NumGroupsToMerge - 1),
                           make_pass_through_transform(K_)),
                make_tuple(Sequence<0, 1, 2>{}, Sequence<3>{}, Sequence<4>{}, Sequence<5>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));
            // We need only matrices from diagonal. X_or returns 0 for the same
            // values. So if matrices is not on diagonal then it will be stored in padding.
            // To avoid use of modulo after xor we assume that NumBatch to merge is power of 2.
            static_assert(NumGroupsToMerge == 1 || NumGroupsToMerge == 2 || NumGroupsToMerge == 4 ||
                          NumGroupsToMerge == 8 || NumGroupsToMerge == 16 ||
                          NumGroupsToMerge == 32 || NumGroupsToMerge == 64);
            constexpr auto unmerged_padded_desc = transform_tensor_descriptor(
                padded_desc,
                make_tuple(make_pass_through_transform(NDoHoWo),
                           make_xor_transform(make_tuple(NumGroupsToMerge, NumGroupsToMerge)),
                           make_pass_through_transform(K_)),
                make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3>{}),
                make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3>{}));

            // Merge To M, N
            return transform_tensor_descriptor(
                unmerged_padded_desc,
                make_tuple(make_merge_transform(make_tuple(NDoHoWo, NumGroupsToMerge)),
                           make_merge_transform(make_tuple(NumGroupsToMerge, K_))),
                make_tuple(Sequence<0, 1>{}, Sequence<2, 3>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));
        }
    }
};

} // namespace tensor_operation
} // namespace ck
