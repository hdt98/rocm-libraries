// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#include "grouped_conv_fwd_wcnn_sba_cvt_impl.h"

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

    // clang-format off
    //                                                     |ShapeType  |FilterType |Dilation |Lds |WaveGroup | SbaMode | ActiveFun | CvtToTensor | TestMask
    if constexpr(std::is_same<GPUAccType, float>::value)
    {
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_1X1, false, 0,       WaveGroup, SbaMode, 0, 0, TestMask | 0x100>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_3X3, false, 0,       WaveGroup, SbaMode, 0, 0, TestMask | 0x200>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_3X3, true,  0,       WaveGroup, SbaMode, 0, 0, TestMask | 0x200>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_3X3, false, LdsMode, WaveGroup, SbaMode, 0, 0, TestMask | 0x400>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_3X3, false, 0,       WaveGroup, SbaMode, 1, 0, TestMask | 0x800>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_3X3, true,  0,       WaveGroup, SbaMode, 1, 0, TestMask | 0x800>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_1X1, false, 0,       WaveGroup, SbaMode, 1, 0, TestMask | 0x1000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_1X1, false, LdsMode, WaveGroup, SbaMode, 1, 0, TestMask | 0x2000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_3X3, false, LdsMode, WaveGroup, SbaMode, 1, 0, TestMask | 0x4000>();
        //Tan (only for sba/uba)
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_1X1, false, 0,       WaveGroup, SbaMode, 2, 0, TestMask | 0x10000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_3X3, false, 0,       WaveGroup, SbaMode, 2, 0, TestMask | 0x20000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_3X3, true,  0,       WaveGroup, SbaMode, 2, 0, TestMask | 0x20000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_1X1, false, LdsMode, WaveGroup, SbaMode, 2, 0, TestMask | 0x40000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_3X3, false, LdsMode, WaveGroup, SbaMode, 2, 0, TestMask | 0x80000>();
    }
    else
    {  
        // 1st issue@llvm: https://ontrack-internal.amd.com/browse/LWPSCGFX13-478 for v_scale_bias_activate_f16 which will impact the all the accType=half case which will impact 4x4
        // 2nd issue@ffm: https://github.amd.com/GFX-Modeling/shader_complex_ffm/issues/960 impact on i8_f16 for 4X4 and 8x4   
// 4X4 issue
#if 0
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X4, Filter_1X1, false, 0,       WaveGroup, SbaMode, 0, 0, TestMask | 0x100>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X4, Filter_3X3, false, 0,       WaveGroup, SbaMode, 0, 0, TestMask | 0x200>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X4, Filter_3X3, true,  0,       WaveGroup, SbaMode, 0, 0, TestMask | 0x200>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X4, Filter_1X1, false, LdsMode, WaveGroup, SbaMode, 0, 0, TestMask | 0x400>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X4, Filter_3X3, false, LdsMode, WaveGroup, SbaMode, 0, 0, TestMask | 0x400>();

        //ActivativeFun: 1
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X4, Filter_1X1, false, 0,       WaveGroup, SbaMode, 1, 0, TestMask | 0x800>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X4, Filter_3X3, false, 0,       WaveGroup, SbaMode, 1, 0, TestMask | 0x1000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X4, Filter_3X3, true,  0,       WaveGroup, SbaMode, 1, 0, TestMask | 0x1000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X4, Filter_1X1, false, LdsMode, WaveGroup, SbaMode, 1, 0, TestMask | 0x2000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X4, Filter_3X3, false, LdsMode, WaveGroup, SbaMode, 1, 0, TestMask | 0x2000>();

        //ActivativeFun: 2
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X4, Filter_1X1, false, 0,       WaveGroup, SbaMode, 2, 0, TestMask | 0x4000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X4, Filter_3X3, false, 0,       WaveGroup, SbaMode, 2, 0, TestMask | 0x8000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X4, Filter_3X3, true,  0,       WaveGroup, SbaMode, 2, 0, TestMask | 0x8000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X4, Filter_1X1, false, LdsMode, WaveGroup, SbaMode, 2, 0, TestMask | 0x2000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X4, Filter_3X3, false, LdsMode, WaveGroup, SbaMode, 2, 0, TestMask | 0x2000>();
#endif
// 8x4
        //ActivativeFun: 0
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_8X4, Filter_1X1, false, 0,       WaveGroup, SbaMode, 0, 0, TestMask | 0x10000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_8X4, Filter_3X3, false, 0,       WaveGroup, SbaMode, 0, 0, TestMask | 0x20000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_8X4, Filter_3X3, true,  0,       WaveGroup, SbaMode, 0, 0, TestMask | 0x20000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_8X4, Filter_1X1, false, LdsMode, WaveGroup, SbaMode, 0, 0, TestMask | 0x40000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_8X4, Filter_3X3, false, LdsMode, WaveGroup, SbaMode, 0, 0, TestMask | 0x40000>();
        //ActivativeFun: 1
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_8X4, Filter_1X1, false, 0,       WaveGroup, SbaMode, 1, 0, TestMask | 0x80000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_8X4, Filter_3X3, false, 0,       WaveGroup, SbaMode, 1, 0, TestMask | 0x100000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_8X4, Filter_3X3, true,  0,       WaveGroup, SbaMode, 1, 0, TestMask | 0x100000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_8X4, Filter_1X1, false, LdsMode, WaveGroup, SbaMode, 1, 0, TestMask | 0x200000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_8X4, Filter_3X3, false, LdsMode, WaveGroup, SbaMode, 1, 0, TestMask | 0x200000>();

        //ActivativeFun: 2
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_8X4, Filter_1X1, false, 0,       WaveGroup, SbaMode, 2, 0, TestMask | 0x400000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_8X4, Filter_3X3, false, 0,       WaveGroup, SbaMode, 2, 0, TestMask | 0x800000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_8X4, Filter_3X3, true,  0,       WaveGroup, SbaMode, 2, 0, TestMask | 0x800000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_8X4, Filter_1X1, false, LdsMode, WaveGroup, SbaMode, 2, 0, TestMask | 0x200000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_8X4, Filter_3X3, false, LdsMode, WaveGroup, SbaMode, 2, 0, TestMask | 0x200000>();
// 4X2
        //NoneActFun
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_1X1, false, 0,       WaveGroup, SbaMode, 0, 0, TestMask | 0x1000000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_3X3, false, 0,       WaveGroup, SbaMode, 0, 0, TestMask | 0x2000000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_3X3, true,  0,       WaveGroup, SbaMode, 0, 0, TestMask | 0x2000000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_2X2, false, 0,       WaveGroup, SbaMode, 0, 0, TestMask | 0x2000000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_1X1, false, LdsMode, WaveGroup, SbaMode, 0, 0, TestMask | 0x4000000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_3X3, false, LdsMode, WaveGroup, SbaMode, 0, 0, TestMask | 0x4000000>();

        //Relu
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_1X1, false, 0,       WaveGroup, SbaMode, 1, 0, TestMask | 0x8000000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_3X3, false, 0,       WaveGroup, SbaMode, 1, 0, TestMask | 0x10000000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_3X3, true,  0,       WaveGroup, SbaMode, 1, 0, TestMask | 0x10000000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_1X1, false, LdsMode, WaveGroup, SbaMode, 1, 0, TestMask | 0x20000000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_3X3, false, LdsMode, WaveGroup, SbaMode, 1, 0, TestMask | 0x20000000>();

        //Tan (only for sba/uba)
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_1X1, false, 0,       WaveGroup, SbaMode, 2, 0, TestMask | 0x40000000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_3X3, false, 0,       WaveGroup, SbaMode, 2, 0, TestMask | 0x80000000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_3X3, true,  0,       WaveGroup, SbaMode, 2, 0, TestMask | 0x80000000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_1X1, false, LdsMode, WaveGroup, SbaMode, 2, 0, TestMask | 0x20000000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_3X3, false, LdsMode, WaveGroup, SbaMode, 2, 0, TestMask | 0x20000000>();
    }
    // clang-format on
    return pass;
}

int main(int argc, char* argv[])
{
    // To refactor on only Acc=32bit supported on 4x2 and Acc=16bit supported on 4x2/4x4/8x4
    bool pass = true;
    if(parse_cmd_args(argc, argv, config) == false)
    {
        return -1;
    }

    // clang-format off
    // Ds keep same with acc currently
    //            |SrcType |GPUAccType |LdsMode |SbaMode |TestMask
    pass &= run_test_fmt<half_t,  float,   0x17, 0, 0x1>();
#if !defined(ENABLE_WAVEGROUP)
    pass &= run_test_fmt<half_t,  float,   0x17, 1, 0x2>();
    pass &= run_test_fmt<half_t,  float,   0x17, 2, 0x4>();
    pass &= run_test_fmt<half_t,  half_t,  0x1f, 0, 0x8>();
#endif
    pass &= run_test_fmt<half_t,  half_t,  0x1f, 1, 0x10>();
#if !defined(ENABLE_WAVEGROUP)
    pass &= run_test_fmt<half_t,  half_t,  0x1f, 2, 0x20>();
    pass &= run_test_fmt<bhalf_t, bhalf_t, 0x17, 0, 0x40>();
    pass &= run_test_fmt<bhalf_t, bhalf_t, 0x17, 1, 0x80>();
    pass &= run_test_fmt<bhalf_t, bhalf_t, 0x17, 2, 0x80>();
#endif
    // clang-format on
    std::cout << "conv_device_suba: ..... " << (pass ? "SUCCESS" : "FAILURE") << std::endl;
    return pass ? 0 : 1;
}
