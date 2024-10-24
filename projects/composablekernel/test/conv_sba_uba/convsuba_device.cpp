// SPDX-License-Identifier: MIT
// Copyriconv_device_impl.hght (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.
#include "convsuba_device_impl.h"
#include "conv_device_124.cpp"

// DsTensor ldsmode keep same with InTensor
// clang-format off
Extern_Test_Func(half_t,  float,   float,   0x17, 0, 0, 0x1);
Extern_Test_Func(half_t,  float,   float,   0x17, 1, 0, 0x1);
Extern_Test_Func(half_t,  float,   float,   0x17, 0, 1, 0x1);
Extern_Test_Func(half_t,  half_t,  half_t,  0x1f, 0, 0, 0x2);
Extern_Test_Func(half_t,  half_t,  half_t,  0x1f, 1, 0, 0x2);
Extern_Test_Func(half_t,  half_t,  half_t,  0x1f, 0, 1, 0x2);
Extern_Test_Func(bhalf_t, bhalf_t, half_t,  0x17, 0, 0, 0x4);
Extern_Test_Func(bhalf_t, bhalf_t, half_t,  0x17, 1, 0, 0x4);
Extern_Test_Func(bhalf_t, bhalf_t, half_t,  0x17, 0, 1, 0x4);

// clang-format on
ExecutionConfig config;

int main(int argc, char* argv[])
{
    bool pass = true;
    if(parse_cmd_args(argc, argv, config) == false)
    {
        return -1;
    }

    // clang-format off
    // Ds keep same with acc currently
    //                |SrcType |GPUAccType |CPUAccType| LdsMode| scaleBiasPacked| uniformScale| TestMask
    pass &= Call_Test_Func(half_t,  float,   float,   0x17, 0, 0, 0x1);
    pass &= Call_Test_Func(half_t,  float,   float,   0x17, 1, 0, 0x1);
    pass &= Call_Test_Func(half_t,  float,   float,   0x17, 0, 1, 0x1);
    pass &= Call_Test_Func(half_t,  half_t,  half_t,  0x1f, 0, 0, 0x2);
    pass &= Call_Test_Func(half_t,  half_t,  half_t,  0x1f, 1, 0, 0x2);
    pass &= Call_Test_Func(half_t,  half_t,  half_t,  0x1f, 0, 1, 0x2);
    pass &= Call_Test_Func(bhalf_t, bhalf_t, half_t,  0x17, 0, 0, 0x4);
    pass &= Call_Test_Func(bhalf_t, bhalf_t, half_t,  0x17, 1, 0, 0x4);
    pass &= Call_Test_Func(bhalf_t, bhalf_t, half_t,  0x17, 0, 1, 0x4);

    // clang-format on
    std::cout << "conv_device: ..... " << (pass ? "SUCCESS" : "FAILURE") << std::endl;
    return pass ? 0 : 1;
}
