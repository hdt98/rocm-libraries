// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.
#include "conv_device_impl.h"

// clang-format off
Extern_Test_Func(half_t,  float,   float,   0x7, 0x1);
Extern_Test_Func(bhalf_t, float,   float,   0xf, 0x2);
Extern_Test_Func(f8_t,    float,   float,   0x5, 0x4);
Extern_Test_Func(bf8_t,   float,   float,   0x6, 0x8);
Extern_Test_Func(int8_t,  float,   float,   0xd, 0x10);
Extern_Test_Func(int8_t,  int32_t, int32_t, 0xe, 0x20);
Extern_Test_Func(half_t,  half_t,  half_t,  0xf, 0x40);
Extern_Test_Func(bhalf_t, bhalf_t, half_t,  0x7, 0x80);
Extern_Test_Func(f8_t,    half_t,  half_t,  0x3, 0x100);
Extern_Test_Func(bf8_t,   bhalf_t, half_t,  0xb, 0x200);
Extern_Test_Func(int8_t,  half_t,  half_t,  0x9, 0x400);
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
    //                |SrcType |GPUAccType |CPUAccType| LdsMode |TestMask
    pass &= Call_Test_Func(half_t,  float,   float,   0x7, 0x1);
    pass &= Call_Test_Func(bhalf_t, float,   float,   0xf, 0x2);
    pass &= Call_Test_Func(f8_t,    float,   float,   0x5, 0x4);
    pass &= Call_Test_Func(bf8_t,   float,   float,   0x6, 0x8);
    pass &= Call_Test_Func(int8_t,  float,   float,   0xd, 0x10);
    pass &= Call_Test_Func(int8_t,  int32_t, int32_t, 0xe, 0x20);
    pass &= Call_Test_Func(half_t,  half_t,  half_t,  0xf, 0x40);
    pass &= Call_Test_Func(bhalf_t, bhalf_t, half_t,  0x7, 0x80);
    pass &= Call_Test_Func(f8_t,    half_t,  half_t,  0x3, 0x100);
    pass &= Call_Test_Func(bf8_t,   bhalf_t, half_t,  0xb, 0x200);
    pass &= Call_Test_Func(int8_t,  half_t,  half_t,  0x9, 0x400);
    // clang-format on

    std::cout << "conv_device: ..... " << (pass ? "SUCCESS" : "FAILURE") << std::endl;
    return pass ? 0 : 1;
}
