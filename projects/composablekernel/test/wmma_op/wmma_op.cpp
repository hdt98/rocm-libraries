// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <tuple>
#include <vector>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "test/wmma_op/wmma_op_util.hpp"

template <typename SrcType, typename DstType, typename GPUAccType, typename CPUAccType>
bool run_test()
{
    if(ck::is_gfx13_supported()) // gfx13 uses another test
        return true;
    using Row         = ck::tensor_layout::gemm::RowMajor;
    using Col         = ck::tensor_layout::gemm::ColumnMajor;
    using PassThrough = ck::tensor_operation::element_wise::PassThrough;
    bool pass         = true;

    const auto matmul_default   = ck::wmma_op_util::matmul<SrcType, DstType, GPUAccType>;
    const auto matmul_swizzle_a = ck::wmma_op_util::matmul_swizzle_a<SrcType, DstType, GPUAccType>;

    const auto wmma_kernel_container = std::make_tuple(matmul_default, matmul_swizzle_a);

    ck::static_for<0, 2, 1>{}([&](auto i) {
        pass &=
            ck::wmma_op_util::TestWmma<decltype(std::get<ck::Number<i>{}>(wmma_kernel_container)),
                                       SrcType,
                                       SrcType,
                                       DstType,
                                       GPUAccType,
                                       CPUAccType,
                                       decltype(Row{}),
                                       decltype(Col{}),
                                       decltype(Row{}),
                                       PassThrough,
                                       PassThrough,
                                       PassThrough,
                                       1>{}(std::get<ck::Number<i>{}>(wmma_kernel_container));
    });

    return pass ? 1 : 0;
}

template <typename SrcAType,
          typename SrcBType,
          typename DstType,
          typename GPUAccType,
          typename CPUAccType,
          ck::index_t KMultiplier = 1>
bool run_test()
{
    if(!ck::is_gfx13_supported())
        return true;
    using Row         = ck::tensor_layout::gemm::RowMajor;
    using Col         = ck::tensor_layout::gemm::ColumnMajor;
    using PassThrough = ck::tensor_operation::element_wise::PassThrough;
    bool pass         = true;
    const auto matmul_default =
        ck::wmma_op_util::matmul<SrcAType, SrcBType, DstType, GPUAccType, KMultiplier>;
    const auto matmul_swizzle_a =
        ck::wmma_op_util::matmul_swizzle_a<SrcAType, SrcBType, DstType, GPUAccType, KMultiplier>;

    const auto wmma_kernel_container = std::make_tuple(matmul_default, matmul_swizzle_a);

    ck::static_for<0, 2, 1>{}([&](auto i) {
        pass &=
            ck::wmma_op_util::TestWmma<decltype(std::get<ck::Number<i>{}>(wmma_kernel_container)),
                                       SrcAType,
                                       SrcBType,
                                       DstType,
                                       GPUAccType,
                                       CPUAccType,
                                       decltype(Row{}),
                                       decltype(Col{}),
                                       decltype(Row{}),
                                       PassThrough,
                                       PassThrough,
                                       PassThrough,
                                       KMultiplier>{}(
                std::get<ck::Number<i>{}>(wmma_kernel_container));
    });

    return pass ? 1 : 0;
}

template <typename GPUSrc0Type,
          typename GPUSrc1Type,
          typename CPUSrc0Type,
          typename CPUSrc1Type,
          typename DstType,
          typename GPUAccType,
          typename CPUAccType,
          ck::index_t AScaleSel,
          ck::index_t BScaleSel>
bool run_mixedfp_test()
{
    using Row         = ck::tensor_layout::gemm::RowMajor;
    using Col         = ck::tensor_layout::gemm::ColumnMajor;
    using PassThrough = ck::tensor_operation::element_wise::PassThrough;
    bool pass         = true;
    if(!ck::is_gfx13_supported())
        return true;
    // will change to mixed format
    const auto matmul_default = ck::wmma_op_util::
        matmul_mixedfp<GPUSrc0Type, GPUSrc1Type, AScaleSel, BScaleSel, DstType, GPUAccType>;

    const auto wmma_kernel_container = std::make_tuple(matmul_default);
    ck::wmma_op_util::GemmParams gemmParams{16, 16, 64, 64, 64, 16, 1.f, 0.f};

    ck::static_for<0, 1, 1>{}([&](auto i) {
        pass &= ck::wmma_op_util::TestMixedFPWmma<
            decltype(std::get<ck::Number<i>{}>(wmma_kernel_container)),
            GPUSrc0Type,
            GPUSrc1Type,
            CPUSrc0Type,
            CPUSrc1Type,
            DstType,
            GPUAccType,
            CPUAccType,
            decltype(Row{}),
            decltype(Col{}),
            decltype(Row{}),
            PassThrough,
            PassThrough,
            PassThrough,
            AScaleSel,
            BScaleSel>{}(std::get<ck::Number<i>{}>(wmma_kernel_container), gemmParams);
    });

    return pass ? 1 : 0;
}

int main(int, char*[])
{
    bool pass = true;
    // clang-format off
    //              |SrcType     |DstType     |GPUAccType  |CPUAccType
    pass &= run_test<ck::half_t,  ck::half_t,  float,       float      >();
    pass &= run_test<ck::bhalf_t, ck::bhalf_t, float,       float      >();
    pass &= run_test<ck::half_t,  ck::half_t,  ck::half_t,  ck::half_t >();
    //# wmma_op crash on gfx12 CI for unknown reason. it looks it is a regression in clang,
    // disable it temporarily
    //pass &= run_test<ck::bhalf_t, ck::bhalf_t, ck::bhalf_t, float      >();
    pass &= run_test<int8_t,      int8_t,      int32_t,     int32_t    >();
    // the below are gfx13 only
    //               |SrcAType    |SrcBType,     |DstType     |GPUAccType  |CPUAccType      |KMultiplier
    pass &= run_test<ck::half_t,  ck::half_t,   float,       float,       float              >(); // V_WMMA_F32_16X16_F16
    pass &= run_test<ck::bhalf_t, ck::bhalf_t,  float,       float,       float              >(); // V_WMMA_F32_16X16_BF16
    pass &= run_test<ck::half_t,  ck::half_t,   ck::half_t,  ck::half_t,  ck::half_t         >(); // V_WMMA_F16_16X16_F16
    pass &= run_test<ck::bhalf_t, ck::bhalf_t,  ck::bhalf_t, ck::bhalf_t, float              >(); // V_WMMA_BF16_16X16_BF16
    pass &= run_test<int8_t,      int8_t,       int32_t,     int32_t,     int32_t            >(); // V_WMMA_I32_16X16_IU8
    pass &= run_test<int8_t,      int8_t,       int32_t,     int32_t,     int32_t,          2>(); // V_WMMA_I32_16X16_IU8
    pass &= run_test<int8_t,      int8_t,       float,       float,       float              >(); // V_WMMA_F32_16X16_IU8
    pass &= run_test<int8_t,      int8_t,       float,       float,       float,            2>(); // V_WMMA_F32_16X16_IU8
    pass &= run_test<ck::f8_t,    ck::f8_t,     float,       float,       float              >(); // V_WMMA_F32_16X16_FP8_FP8
    pass &= run_test<ck::f8_t,    ck::f8_t,     float,       float,       float,            2>(); // V_WMMA_F32_16X16_FP8_FP8
    pass &= run_test<ck::f8_t,    ck::bf8_t,    float,       float,       float              >(); // V_WMMA_F32_16X16_FP8_BF8
    pass &= run_test<ck::f8_t,    ck::bf8_t,    float,       float,       float,            2>();
    pass &= run_test<ck::bf8_t,   ck::f8_t,     float,       float,       float              >(); // V_WMMA_F32_16X16_BF8_FP8
    pass &= run_test<ck::bf8_t,   ck::f8_t,     float,       float,       float,            2>();
    pass &= run_test<ck::bf8_t,   ck::bf8_t,    float,       float,       float              >(); // V_WMMA_F32_16X16_BF8_BF8
    pass &= run_test<ck::bf8_t,   ck::bf8_t,    float,       float,       float,            2>();
    pass &= run_test<ck::f8_t,    ck::bf8_t,    ck::half_t,  ck::half_t,  ck::half_t         >(); // V_WMMA_F16_16X16_FP8_BF8
    pass &= run_test<ck::f8_t,    ck::bf8_t,    ck::half_t,  ck::half_t,  ck::half_t,       2>();
    pass &= run_test<ck::bf8_t,   ck::f8_t,     ck::half_t,  ck::half_t,  ck::half_t         >(); // V_WMMA_F16_16X16_BF8_FP8
    pass &= run_test<ck::bf8_t,   ck::f8_t,     ck::half_t,  ck::half_t,  ck::half_t,       2>();
    pass &= run_test<ck::bf8_t,   ck::bf8_t,    ck::half_t,  ck::half_t,  ck::half_t         >(); // V_WMMA_F16_16X16_BF8_BF8
    pass &= run_test<ck::bf8_t,   ck::bf8_t,    ck::half_t,  ck::half_t,  ck::half_t,       2>();
    pass &= run_test<ck::f8_t,    ck::f8_t,     ck::half_t,  ck::half_t,  ck::half_t         >(); // V_WMMA_F16_16X16_FP8_FP8
    pass &= run_test<ck::f8_t,    ck::f8_t,     ck::half_t,  ck::half_t,  ck::half_t,       2>();
    pass &= run_test<int8_t,      int8_t,       float,       int32_t,     int32_t            >();
    pass &= run_test<int8_t,      int8_t,       float,       int32_t,     int32_t,          2>();
    #ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
    pass &= run_test<ck::int4_t,  ck::int4_t,   int32_t,     int32_t,     int32_t            >();
    pass &= run_test<ck::int4_t,  ck::int4_t,   int32_t,     int32_t,     int32_t,          2>();
    pass &= run_test<ck::int4_t,  ck::int4_t,   int32_t,     int32_t,     int32_t,          4>();
    pass &= run_test<ck::int4_t,  ck::int4_t,   float,       float,       float              >();
    pass &= run_test<ck::int4_t,  ck::int4_t,   float,       float,       float,            2>();
    pass &= run_test<ck::int4_t,  ck::int4_t,   float,       float,       float,            4>();
    pass &= run_test<ck::int4_t,  ck::int4_t,   float,       int32_t,     int32_t            >();
    pass &= run_test<ck::int4_t,  ck::int4_t,   float,       int32_t,     int32_t,          2>();
    pass &= run_test<ck::int4_t,  ck::int4_t,   float,       int32_t,     int32_t,          4>();
    #endif
    pass &= run_mixedfp_test</*DeviceAType*/ck::MxType_t<ck::MTX_FMT::MTX_FMT_FP8_E4M3>,
                             /*DeviceBType*/ck::MxType_t<ck::MTX_FMT::MTX_FMT_FP8_E5M2>,
                             /*HostAType*/  float,
                             /*HostBType*/  float,
                             /*DstType*/    float,
                             /*GPUAccType*/ float,
                             /*CPUAccType*/ float,
                             /*AScaleSel*/  0,
                             /*BScaleSel*/  0>();

    pass &= run_mixedfp_test</*DeviceAType*/ck::MxType_t<ck::MTX_FMT::MTX_FMT_FP8_E5M2>,
                             /*DeviceBType*/ck::MxType_t<ck::MTX_FMT::MTX_FMT_FP8_E4M3>,
                             /*HostAType*/  float,
                             /*HostBType*/  float,
                             /*DstType*/    float,
                             /*GPUAccType*/ float,
                             /*CPUAccType*/ float,
                             /*AScaleSel*/  0,
                             /*BScaleSel*/  1>();

    pass &= run_mixedfp_test</*DeviceAType*/ck::MxType_t<ck::MTX_FMT::MTX_FMT_FP8_E4M3>,
                             /*DeviceBType*/ck::MxType_t<ck::MTX_FMT::MTX_FMT_FP8_E4M3>,
                             /*HostAType*/  float,
                             /*HostBType*/  float,
                             /*DstType*/    float,
                             /*GPUAccType*/ float,
                             /*CPUAccType*/ float,
                             /*AScaleSel*/  1,
                             /*BScaleSel*/  1>();
    
    pass &= run_mixedfp_test</*DeviceAType*/ck::MxType_t<ck::MTX_FMT::MTX_FMT_FP8_E4M3>,
                             /*DeviceBType*/ck::MxType_t<ck::MTX_FMT::MTX_FMT_FP8_E4M3>,
                             /*HostAType*/  float,
                             /*HostBType*/  float,
                             /*DstType*/    float,
                             /*GPUAccType*/ float,
                             /*CPUAccType*/ float,
                             /*AScaleSel*/  1,
                             /*BScaleSel*/  0>();

    pass &= run_mixedfp_test</*DeviceAType*/ck::MxType_t<ck::MTX_FMT::MTX_FMT_FP6_E2M3>,
                             /*DeviceBType*/ck::MxType_t<ck::MTX_FMT::MTX_FMT_FP6_E2M3>,
                             /*HostAType*/  float,
                             /*HostBType*/  float,
                             /*DstType*/    float,
                             /*GPUAccType*/ float,
                             /*CPUAccType*/ float,
                             /*AScaleSel*/  0,
                             /*BScaleSel*/  0>();

    pass &= run_mixedfp_test</*DeviceAType*/ck::MxType_t<ck::MTX_FMT::MTX_FMT_FP6_E3M2>,
                             /*DeviceBType*/ck::MxType_t<ck::MTX_FMT::MTX_FMT_FP4_E2M1>,
                             /*HostAType*/  float,
                             /*HostBType*/  float,
                             /*DstType*/    float,
                             /*GPUAccType*/ float,
                             /*CPUAccType*/ float,
                             /*AScaleSel*/  0,
                             /*BScaleSel*/  1>();

    pass &= run_mixedfp_test</*DeviceAType*/ck::MxType_t<ck::MTX_FMT::MTX_FMT_FP4_E2M1>,
                             /*DeviceBType*/ck::MxType_t<ck::MTX_FMT::MTX_FMT_FP4_E2M1>,
                             /*HostAType*/  float,
                             /*HostBType*/  float,
                             /*DstType*/    float,
                             /*GPUAccType*/ float,
                             /*CPUAccType*/ float,
                             /*AScaleSel*/  1,
                             /*BScaleSel*/  1>();
    
    pass &= run_mixedfp_test</*DeviceAType*/ck::MxType_t<ck::MTX_FMT::MTX_FMT_FP4_E2M1>,
                             /*DeviceBType*/ck::MxType_t<ck::MTX_FMT::MTX_FMT_FP6_E2M3>,
                             /*HostAType*/  float,
                             /*HostBType*/  float,
                             /*DstType*/    float,
                             /*GPUAccType*/ float,
                             /*CPUAccType*/ float,
                             /*AScaleSel*/  1,
                             /*BScaleSel*/  0>();
    // clang-format on

    std::cout << "TestGemm ..... " << (pass ? "SUCCESS" : "FAILURE") << std::endl;
    return pass ? 0 : 1;
}
