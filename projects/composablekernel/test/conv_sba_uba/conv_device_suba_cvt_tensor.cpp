// SPDX-License-Identifier: MIT
// Copyriconv_device_impl.hght (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.
#include "conv_device_suba_cvt_tensor_impl.h"

// clang-format on
ExecutionConfig config;

template <typename SrcType, typename GPUAccType, int LdsMode, int SbaMode, uint32_t TestMask>
bool run_test_fmt()
{
    if((config.test_mask & TestMask) == 0)
    {
        return true;
    }
    bool pass = true;

#ifdef ENABLE_WAVEGROUP
    constexpr bool WaveGroup = true;
#else
    constexpr bool WaveGroup = false;
#endif
    // TODO: verify cvtTensorScale != 0
    // clang-format off
    //                                                     |ShapeType  |FilterType |Dilation |Lds |WaveGroup | SbaMode | activeFun | CvtToTensor | TestMask
    if constexpr(std::is_same<GPUAccType, float>::value)
    { 
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_1X1, false, 0,       WaveGroup, SbaMode, 0, 1, TestMask | 0x100>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_1X1, false, 0,       WaveGroup, SbaMode, 0, 1, TestMask | 0x100>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_3X3, false, 0,       WaveGroup, SbaMode, 0, 1, TestMask | 0x200>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_3X3, true,  0,       WaveGroup, SbaMode, 0, 1, TestMask | 0x200>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_3X3, false, LdsMode, WaveGroup, SbaMode, 0, 1, TestMask | 0x400>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_3X3, false, 0,       WaveGroup, SbaMode, 1, 1, TestMask | 0x800>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_3X3, true,  0,       WaveGroup, SbaMode, 1, 1, TestMask | 0x800>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_1X1, false, 0,       WaveGroup, SbaMode, 1, 1, TestMask | 0x1000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_1X1, false, LdsMode, WaveGroup, SbaMode, 1, 1, TestMask | 0x2000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_3X3, false, LdsMode, WaveGroup, SbaMode, 1, 1, TestMask | 0x4000>();
    }
    else
    {
        // 1st issue@llvm: https://ontrack-internal.amd.com/browse/LWPSCGFX13-478 for v_scale_bias_activate_f16 which will impact the all the accType=half case which will impact 4x4
        // 2nd issue@ffm: https://github.amd.com/GFX-Modeling/shader_complex_ffm/issues/960 impact on i8_f16 for 4X4 and 8x4 
// 4X4 issue
#if 0
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X4, Filter_1X1, false, 0,       WaveGroup, SbaMode, 0, 1, TestMask | 0x100>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X4, Filter_3X3, false, 0,       WaveGroup, SbaMode, 0, 1, TestMask | 0x200>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X4, Filter_3X3, true,  0,       WaveGroup, SbaMode, 0, 1, TestMask | 0x200>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X4, Filter_1X1, false, LdsMode, WaveGroup, SbaMode, 0, 1, TestMask | 0x400>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X4, Filter_3X3, false, LdsMode, WaveGroup, SbaMode, 0, 1, TestMask | 0x800>();

        //ActivativeFun: 1
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X4, Filter_1X1, false, 0,       WaveGroup, SbaMode, 1, 1, TestMask | 0x1000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X4, Filter_3X3, false, 0,       WaveGroup, SbaMode, 1, 1, TestMask | 0x2000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X4, Filter_3X3, true,  0,       WaveGroup, SbaMode, 1, 1, TestMask | 0x2000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X4, Filter_1X1, false, LdsMode, WaveGroup, SbaMode, 1, 1, TestMask | 0x4000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X4, Filter_3X3, false, LdsMode, WaveGroup, SbaMode, 1, 1, TestMask | 0x8000>();
#endif
// 8x4
        //ActivativeFun: 0
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_8X4, Filter_1X1, false, 0,       WaveGroup, SbaMode, 0, 1, TestMask | 0x10000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_8X4, Filter_3X3, false, 0,       WaveGroup, SbaMode, 0, 1, TestMask | 0x20000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_8X4, Filter_3X3, true,  0,       WaveGroup, SbaMode, 0, 1, TestMask | 0x20000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_8X4, Filter_1X1, false, LdsMode, WaveGroup, SbaMode, 0, 1, TestMask | 0x40000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_8X4, Filter_3X3, false, LdsMode, WaveGroup, SbaMode, 0, 1, TestMask | 0x80000>();

        //ActivativeFun: 1
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_8X4, Filter_1X1, false, 0,       WaveGroup, SbaMode, 1, 1, TestMask | 0x100000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_8X4, Filter_3X3, false, 0,       WaveGroup, SbaMode, 1, 1, TestMask | 0x200000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_8X4, Filter_3X3, true,  0,       WaveGroup, SbaMode, 1, 1, TestMask | 0x200000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_8X4, Filter_1X1, false, LdsMode, WaveGroup, SbaMode, 1, 1, TestMask | 0x400000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_8X4, Filter_3X3, false, LdsMode, WaveGroup, SbaMode, 1, 1, TestMask | 0x800000>();
// 4X2
        //ActivativeFun: 0
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_1X1, false, 0,       WaveGroup, SbaMode, 0, 1, TestMask | 0x1000000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_3X3, false, 0,       WaveGroup, SbaMode, 0, 1, TestMask | 0x2000000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_3X3, true,  0,       WaveGroup, SbaMode, 0, 1, TestMask | 0x2000000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_1X1, false, LdsMode, WaveGroup, SbaMode, 0, 1, TestMask | 0x4000000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_3X3, false, LdsMode, WaveGroup, SbaMode, 0, 1, TestMask | 0x8000000>();

        //ActivativeFun: 1
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_1X1, false, 0,       WaveGroup, SbaMode, 1, 1, TestMask | 0x10000000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_3X3, false, 0,       WaveGroup, SbaMode, 1, 1, TestMask | 0x20000000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_3X3, true,  0,       WaveGroup, SbaMode, 1, 1, TestMask | 0x20000000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_1X1, false, LdsMode, WaveGroup, SbaMode, 1, 1, TestMask | 0x40000000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_3X3, false, LdsMode, WaveGroup, SbaMode, 1, 1, TestMask | 0x80000000>();
    }
    // clang-format on
    return pass;
}

using half_t  = ck::half_t;
using bhalf_t = ck::bhalf_t;
using f8_t    = ck::f8_t;
using bf8_t   = ck::bf8_t;
using int4_t  = ck::int4_t;
using uint4_t = ck::uint4_t;

int main(int argc, char* argv[])
{
    bool pass = true;
    if(parse_cmd_args(argc, argv, config) == false)
    {
        return -1;
    }

    // clang-format off
    // Ds keep same with acc currently
    //                |SrcType |GPUAccType |LdsMode | SbaMode |TestMask
    pass &= run_test_fmt<half_t,   float,   0x1f, 1, 0x1>();
#if !defined(ENABLE_WAVEGROUP)
    pass &= run_test_fmt<bhalf_t,  float,   0x17, 1, 0x2>();
    pass &= run_test_fmt<int8_t,   float,   0x17, 1, 0x4>();
    pass &= run_test_fmt<bf8_t,    float,   0x17, 1, 0x8>();
    pass &= run_test_fmt<f8_t,     float,   0x17, 1, 0x8>();
#endif
    pass &= run_test_fmt<f8_t,     half_t,  0x1f, 1, 0x10>();
#if !defined(ENABLE_WAVEGROUP)
    pass &= run_test_fmt<int8_t,   half_t,  0x1f, 1, 0x20>();
    pass &= run_test_fmt<half_t,   half_t,  0x1f, 1, 0x40>();
    pass &= run_test_fmt<bf8_t,    half_t,  0x1f, 1, 0x80>();
    pass &= run_test_fmt<bhalf_t,  bhalf_t, 0x1f, 1, 0x80>();
#endif
    // clang-format on
    std::cout << "conv_device_suba_cvt: ..... " << (pass ? "SUCCESS" : "FAILURE") << std::endl;
    return pass ? 0 : 1;
}
