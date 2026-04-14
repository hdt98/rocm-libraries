// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/multi_index_transform_helper.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"

namespace ck {

// Shared grid descriptor builders for gridwise_gemm_xdl_cshuffle variants.
// These functions are identical across 19+ gridwise_gemm files and depend only
// on template parameters (ALayout/BLayout, GemmSpec, K1Value) and function
// arguments — no class state.

template <typename Layout,
          tensor_operation::device::GemmSpecialization GemmSpec,
          index_t K1Value,
          typename IndexType = index_t>
__host__ __device__ static auto MakeAGridDescriptor_AK0_M_AK1(
    IndexType M, IndexType MPad, IndexType K, IndexType KPad, IndexType StrideA, IndexType AK0)
{
    static constexpr auto I1 = Number<1>{};

    const auto a_grid_desc_mraw_kraw = [&]() {
        if constexpr(is_same_v<tensor_layout::gemm::RowMajor, Layout>)
        {
            return make_naive_tensor_descriptor(make_tuple(M, K), make_tuple(StrideA, I1));
        }
        else if constexpr(is_same_v<tensor_layout::gemm::ColumnMajor, Layout>)
        {
            return make_naive_tensor_descriptor(make_tuple(M, K), make_tuple(I1, StrideA));
        }
    }();

    using GemmSpecialization = tensor_operation::device::GemmSpecialization;

    constexpr auto AK1Value = Number<K1Value>{};

    if constexpr(GemmSpec == GemmSpecialization::MKPadding ||
                 GemmSpec == GemmSpecialization::MNKPadding)
    {
        // pad both M and K
        const auto a_grid_desc_m_k =
            transform_tensor_descriptor(a_grid_desc_mraw_kraw,
                                        make_tuple(make_right_pad_transform(M, MPad - M),
                                                   make_right_pad_transform(K, KPad - K)),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}));

        const auto a_grid_desc_ak0_m_ak1 = transform_tensor_descriptor(
            a_grid_desc_m_k,
            make_tuple(make_unmerge_transform(make_tuple(AK0, AK1Value)),
                       make_pass_through_transform(MPad)),
            make_tuple(Sequence<1>{}, Sequence<0>{}),
            make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

        return a_grid_desc_ak0_m_ak1;
    }
    else if constexpr(GemmSpec == GemmSpecialization::MPadding ||
                      GemmSpec == GemmSpecialization::MNPadding)
    {
        // pad M, but not K
        const auto a_grid_desc_ak0_m_ak1 = transform_tensor_descriptor(
            a_grid_desc_mraw_kraw,
            make_tuple(make_unmerge_transform(make_tuple(AK0, AK1Value)),
                       make_right_pad_transform(M, MPad - M)),
            make_tuple(Sequence<1>{}, Sequence<0>{}),
            make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

        return a_grid_desc_ak0_m_ak1;
    }
    else if constexpr(GemmSpec == GemmSpecialization::KPadding ||
                      GemmSpec == GemmSpecialization::NKPadding)
    {
        // pad K, but not M
        const auto a_grid_desc_m_k = transform_tensor_descriptor(
            a_grid_desc_mraw_kraw,
            make_tuple(make_pass_through_transform(M), make_right_pad_transform(K, KPad - K)),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        const auto a_grid_desc_ak0_m_ak1 = transform_tensor_descriptor(
            a_grid_desc_m_k,
            make_tuple(make_unmerge_transform(make_tuple(AK0, AK1Value)),
                       make_pass_through_transform(M)),
            make_tuple(Sequence<1>{}, Sequence<0>{}),
            make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

        return a_grid_desc_ak0_m_ak1;
    }
    else
    {
        // not pad M or K
        const auto a_grid_desc_ak0_m_ak1 = transform_tensor_descriptor(
            a_grid_desc_mraw_kraw,
            make_tuple(make_unmerge_transform(make_tuple(AK0, AK1Value)),
                       make_pass_through_transform(M)),
            make_tuple(Sequence<1>{}, Sequence<0>{}),
            make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

        return a_grid_desc_ak0_m_ak1;
    }
}

template <typename Layout,
          tensor_operation::device::GemmSpecialization GemmSpec,
          index_t K1Value,
          typename IndexType = index_t>
__host__ __device__ static auto MakeBGridDescriptor_BK0_N_BK1(
    IndexType K, IndexType KPad, IndexType N, IndexType NPad, IndexType StrideB, IndexType BK0)
{
    static constexpr auto I1 = Number<1>{};

    const auto b_grid_desc_nraw_kraw = [&]() {
        if constexpr(is_same<tensor_layout::gemm::RowMajor, Layout>::value)
        {
            return make_naive_tensor_descriptor(make_tuple(N, K), make_tuple(I1, StrideB));
        }
        else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, Layout>::value)
        {
            return make_naive_tensor_descriptor(make_tuple(N, K), make_tuple(StrideB, I1));
        }
    }();

    using GemmSpecialization = tensor_operation::device::GemmSpecialization;

    constexpr auto BK1Value = Number<K1Value>{};

    if constexpr(GemmSpec == GemmSpecialization::NKPadding ||
                 GemmSpec == GemmSpecialization::MNKPadding)
    {
        // pad both N and K
        const auto b_grid_desc_n_k =
            transform_tensor_descriptor(b_grid_desc_nraw_kraw,
                                        make_tuple(make_right_pad_transform(N, NPad - N),
                                                   make_right_pad_transform(K, KPad - K)),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}));

        const auto b_grid_desc_bk0_n_bk1 = transform_tensor_descriptor(
            b_grid_desc_n_k,
            make_tuple(make_unmerge_transform(make_tuple(BK0, BK1Value)),
                       make_pass_through_transform(NPad)),
            make_tuple(Sequence<1>{}, Sequence<0>{}),
            make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

        return b_grid_desc_bk0_n_bk1;
    }
    else if constexpr(GemmSpec == GemmSpecialization::NPadding ||
                      GemmSpec == GemmSpecialization::MNPadding)
    {
        // pad N, but not K
        const auto b_grid_desc_bk0_n_bk1 = transform_tensor_descriptor(
            b_grid_desc_nraw_kraw,
            make_tuple(make_unmerge_transform(make_tuple(BK0, BK1Value)),
                       make_right_pad_transform(N, NPad - N)),
            make_tuple(Sequence<1>{}, Sequence<0>{}),
            make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

        return b_grid_desc_bk0_n_bk1;
    }
    else if constexpr(GemmSpec == GemmSpecialization::KPadding ||
                      GemmSpec == GemmSpecialization::MKPadding)
    {
        // pad K, but not N
        const auto b_grid_desc_n_k = transform_tensor_descriptor(
            b_grid_desc_nraw_kraw,
            make_tuple(make_pass_through_transform(N), make_right_pad_transform(K, KPad - K)),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        const auto b_grid_desc_bk0_n_bk1 = transform_tensor_descriptor(
            b_grid_desc_n_k,
            make_tuple(make_unmerge_transform(make_tuple(BK0, BK1Value)),
                       make_pass_through_transform(N)),
            make_tuple(Sequence<1>{}, Sequence<0>{}),
            make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

        return b_grid_desc_bk0_n_bk1;
    }
    else
    {
        // not pad N or K
        const auto b_grid_desc_bk0_n_bk1 = transform_tensor_descriptor(
            b_grid_desc_nraw_kraw,
            make_tuple(make_unmerge_transform(make_tuple(BK0, BK1Value)),
                       make_pass_through_transform(N)),
            make_tuple(Sequence<1>{}, Sequence<0>{}),
            make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

        return b_grid_desc_bk0_n_bk1;
    }
}

} // namespace ck
