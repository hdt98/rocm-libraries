// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include "common.hpp"

#include "gemm_xdl_v3_benchmark_instance.hpp"

using ReferenceGemmInstance = ck::tensor_operation::host::
    ReferenceGemm<ADataType, BDataType, CDataType, AccDataType, AElementOp, BElementOp, CElementOp>;

#include "run_gemm_example_v3.inc"

int main(int argc, char* argv[]) { return !run_gemm_splitk_example(argc, argv); }
