// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include "common.hpp"

using InDataType  = F16;
using OutDataType = F16;

// clang-format off
using DevicePermuteInstance = ck::tensor_operation::device::DevicePermuteImpl
// ######| NumDim|     InData|     OutData| Elementwise| Block|  NPer|  HPer|  WPer|   InBlock|      InBlockTransfer|           InBlockTransfer|       Src|       Dst|             Src|             Dst|
// ######|       |       Type|        Type|   Operation|  Size| Block| Block| Block| LdsExtraW| ThreadClusterLengths| ThreadClusterArrangeOrder| VectorDim| VectorDim| ScalarPerVector| ScalarPerVector|
// ######|       |           |            |            |      |      |      |      |          |                     |                          |          |          |                |                |
// ######|       |           |            |            |      |      |      |      |          |                     |                          |          |          |                |                |
         <      3, InDataType, OutDataType, PassThrough,   256,     1,    32,    32,         3,         S<1, 32,  8>,                S<0, 1, 2>,         2,         1,               2,               1>;
// clang-format on

#include "run_permute_element_example.inc"

int main(int argc, char* argv[])
{
    bool time_kernel = true;
    if(argc == 2)
    {
        time_kernel = atoi(argv[1]);
    }
    return !run_permute_element_example(time_kernel, {1, 32000, 80}, {0, 2, 1});
}
