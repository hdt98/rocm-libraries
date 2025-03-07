// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#include "grouped_conv_fwd_wcnn_fma_cvt_impl.h"

template <typename SrcType, typename GPUAccType, int LdsMode, uint32_t TestMask>
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
    //                                                          |ShapeType |Lds |WaveGroup |FmaMode | ActiveFunc | CvtToTensor | TestMask
    if constexpr(std::is_same<GPUAccType, float>::value)
    {
        pass &= run_test<SrcType, SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, 0,       WaveGroup, 1, 0, 0, TestMask | 0x10000>();
        pass &= run_test<SrcType, SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, LdsMode, WaveGroup, 1, 0, 0, TestMask | 0x20000>();
        pass &= run_test<SrcType, SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, 0,       WaveGroup, 0, 0, 0, TestMask | 0x40000>();
        pass &= run_test<SrcType, SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, 0,       WaveGroup, 0, 1, 0, TestMask | 0x80000>();
        pass &= run_test<SrcType, SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, 0,       WaveGroup, 1, 1, 0, TestMask | 0x100000>();
     }
    else
    {
        pass &= run_test<SrcType, SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, 0,       WaveGroup, 1, 0, 0, TestMask | 0x10000>();
        pass &= run_test<SrcType, SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, LdsMode, WaveGroup, 1, 0, 0, TestMask | 0x80000>();
        pass &= run_test<SrcType, SrcType, SrcType, GPUAccType, SrcType, Shape_4X4, 0,       WaveGroup, 1, 0, 0, TestMask | 0x20000>();
        pass &= run_test<SrcType, SrcType, SrcType, GPUAccType, SrcType, Shape_8X4, 0,       WaveGroup, 1, 0, 0, TestMask | 0x40000>();
        pass &= run_test<SrcType, SrcType, SrcType, GPUAccType, SrcType, Shape_4X4, LdsMode, WaveGroup, 1, 0, 0, TestMask | 0x100000>();
        pass &= run_test<SrcType, SrcType, SrcType, GPUAccType, SrcType, Shape_8X4, LdsMode, WaveGroup, 1, 0, 0, TestMask | 0x200000>();
     }
    // clang-format on
    return pass;
}

int main(int argc, char* argv[])
{
    bool pass = true;
    if(parse_cmd_args(argc, argv, config) == false)
    {
        return -1;
    }

    // clang-format off
    // Ds keep same with acc currently
    //                |SrcType |GPUAccType| LdsMode TestMask
    pass &= run_test_fmt<half_t,  float,   0x17, 0x1  >();
    pass &= run_test_fmt<bhalf_t, float,   0x1f, 0x2  >();
    pass &= run_test_fmt<f8_t,    float,   0x17, 0x4  >();
    pass &= run_test_fmt<bf8_t,   float,   0x1f, 0x8  >();
    pass &= run_test_fmt<int8_t,  float,   0x17, 0x10 >();

    pass &= run_test_fmt<half_t,  half_t,  0x1f, 0x40 >();
    pass &= run_test_fmt<bhalf_t, bhalf_t, 0x17, 0x80 >();
    pass &= run_test_fmt<f8_t,    half_t,  0x1f, 0x100>();
    pass &= run_test_fmt<bf8_t,   half_t,  0x17, 0x200>();
    pass &= run_test_fmt<int8_t,  half_t,  0x1f, 0x400>();
    //
    // clang-format on
    std::cout << "grouped_conv_fwd_wcnn_fma: ..... " << (pass ? "SUCCESS" : "FAILURE") << std::endl;
    return pass ? 0 : 1;
}
