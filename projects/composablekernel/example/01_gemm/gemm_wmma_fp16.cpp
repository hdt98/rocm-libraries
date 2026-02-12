// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "common.hpp"

#include "ck/tensor_operation/gpu/device/impl/device_gemm_wmma.hpp"

#if defined(GEMM_CONFIG_1) || defined(GEMM_CONFIG_2) || defined(GEMM_CONFIG_3) || \
    defined(GEMM_CONFIG_4)
using ADataType = ck::f8_t;
using BDataType = ck::f8_t;
#else
using ADataType = ck::half_t;
using BDataType = ck::half_t;
#endif

using AccDataType      = float;
using CShuffleDataType = float;
using CDataType        = ck::half_t;

using ALayout = Row;
using BLayout = Col;
using CLayout = Row;

using AElementOp = PassThrough;
using BElementOp = PassThrough;
using CElementOp = PassThrough;

#ifdef GEMM_SPEC_DEFAULT
static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::Default;
#elif defined(GEMM_CONFIG_3)
static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::MPadding;
#else
static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::MNKPadding;
#endif

// Configuration 1: 256x256x128, BlockSize=512, ab_type=f8_t
#if defined GEMM_CONFIG_1
using DeviceGemmInstance = ck::tensor_operation::device::DeviceGemmWmma_CShuffle<
    ALayout,
    BLayout,
    CLayout,
    ADataType,
    BDataType,
    CDataType,
    AccDataType,
    CShuffleDataType,
    AElementOp,
    BElementOp,
    CElementOp,
    GemmDefault,
    1,               // Prefetch stage
    512,             // BlockSize
    256,             // MPerBlock
    256,             // NPerBlock
    128,             // KPerBlock
    16,              // K1
    16,              // MPerWmma
    16,              // NPerWmma
    4,               // M-Repeat
    4,               // N-Repeat
    S<8, 64, 1>,     // ABlockTransferThreadClusterLengths_K0_M_K1
    S<1, 0, 2>,      // ABlockTransferThreadClusterArrangeOrder
    S<1, 0, 2>,      // ABlockTransferSrcAccessOrder
    2,               // ABlockTransferSrcVectorDim
    16,              // ABlockTransferSrcScalarPerVector
    16,              // ABlockTransferDstScalarPerVector_K1
    true,            // AThreadTransferSrcResetCoordinateAfterRun
    S<8, 64, 1>,     // BBlockTransferThreadClusterLengths_K0_N_K1
    S<1, 0, 2>,      // BBlockTransferThreadClusterArrangeOrder
    S<1, 0, 2>,      // BBlockTransferSrcAccessOrder
    2,               // BBlockTransferSrcVectorDim
    16,              // BBlockTransferSrcScalarPerVector
    16,              // BBlockTransferDstScalarPerVector_K1
    true,            // BThreadTransferSrcResetCoordinateAfterRun
    2,               // CShuffleMRepeatPerStore
    4,               // CShuffleNRepeatPerStore
    S<1, 16, 1, 16>, // CBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
    8>;              // CBlockTransferScalarPerVector_NWaveNPerXdl
#endif

// Configuration 2: 256x256x128, BlockSize=256, ab_type=f8_t
#ifdef GEMM_CONFIG_2
using DeviceGemmInstance = ck::tensor_operation::device::DeviceGemmWmma_CShuffle<
    ALayout,
    BLayout,
    CLayout,
    ADataType,
    BDataType,
    CDataType,
    AccDataType,
    CShuffleDataType,
    AElementOp,
    BElementOp,
    CElementOp,
    GemmDefault,
    1,               // Prefetch stage
    256,             // BlockSize
    256,             // MPerBlock
    256,             // NPerBlock
    128,             // KPerBlock
    16,              // K1
    16,              // MPerWmma
    16,              // NPerWmma
    8,               // M-Repeat
    4,               // N-Repeat
    S<8, 32, 1>,     // ABlockTransferThreadClusterLengths_K0_M_K1
    S<1, 0, 2>,      // ABlockTransferThreadClusterArrangeOrder
    S<1, 0, 2>,      // ABlockTransferSrcAccessOrder
    2,               // ABlockTransferSrcVectorDim
    16,              // ABlockTransferSrcScalarPerVector
    16,              // ABlockTransferDstScalarPerVector_K1
    true,            // AThreadTransferSrcResetCoordinateAfterRun
    S<8, 32, 1>,     // BBlockTransferThreadClusterLengths_K0_N_K1
    S<1, 0, 2>,      // BBlockTransferThreadClusterArrangeOrder
    S<1, 0, 2>,      // BBlockTransferSrcAccessOrder
    2,               // BBlockTransferSrcVectorDim
    16,              // BBlockTransferSrcScalarPerVector
    16,              // BBlockTransferDstScalarPerVector_K1
    true,            // BThreadTransferSrcResetCoordinateAfterRun
    2,               // CShuffleMRepeatPerStore
    4,               // CShuffleNRepeatPerStore
    S<1, 16, 1, 16>, // CBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
    8>;              // CBlockTransferScalarPerVector_NWaveNPerXdl
#endif

// Configuration 3: 16x192x256, BlockSize=128, abtype=f8_t
#ifdef GEMM_CONFIG_3
using DeviceGemmInstance = ck::tensor_operation::device::DeviceGemmWmma_CShuffle<
    ALayout,
    BLayout,
    CLayout,
    ADataType,
    BDataType,
    CDataType,
    AccDataType,
    CShuffleDataType,
    AElementOp,
    BElementOp,
    CElementOp,
    GemmDefault,
    1,              // Prefetch stage
    128,            // BlockSize
    16,             // MPerBlock
    192,            // NPerBlock
    256,            // KPerBlock
    16,             // K1
    16,             // MPerWmma
    16,             // NPerWmma
    1,              // M-Repeat
    3,              // N-Repeat
    S<8, 16, 1>,    // ABlockTransferThreadClusterLengths_K0_M_K1
    S<1, 0, 2>,     // ABlockTransferThreadClusterArrangeOrder
    S<1, 0, 2>,     // ABlockTransferSrcAccessOrder
    2,              // ABlockTransferSrcVectorDim
    16,             // ABlockTransferSrcScalarPerVector
    16,             // ABlockTransferDstScalarPerVector_K1
    true,           // AThreadTransferSrcResetCoordinateAfterRun
    S<8, 16, 1>,    // BBlockTransferThreadClusterLengths_K0_N_K1
    S<1, 0, 2>,     // BBlockTransferThreadClusterArrangeOrder
    S<1, 0, 2>,     // BBlockTransferSrcAccessOrder
    2,              // BBlockTransferSrcVectorDim
    16,             // BBlockTransferSrcScalarPerVector
    16,             // BBlockTransferDstScalarPerVector_K1
    true,           // BThreadTransferSrcResetCoordinateAfterRun
    1,              // CShuffleMRepeatPerStore
    1,              // CShuffleNRepeatPerStore
    S<1, 16, 1, 8>, // CBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
    8>;             // CBlockTransferScalarPerVector_NWaveNPerXdl
#endif

// Configuration 4: 128x192x128, BlockSize=256, abtype=f8_t
#ifdef GEMM_CONFIG_4
using DeviceGemmInstance = ck::tensor_operation::device::DeviceGemmWmma_CShuffle<
    ALayout,
    BLayout,
    CLayout,
    ADataType,
    BDataType,
    CDataType,
    AccDataType,
    CShuffleDataType,
    AElementOp,
    BElementOp,
    CElementOp,
    GemmDefault,
    1,              // Prefetch stage
    256,            // BlockSize
    128,            // MPerBlock
    192,            // NPerBlock
    128,            // KPerBlock
    16,             // K1
    16,             // MPerWmma
    16,             // NPerWmma
    4,              // M-Repeat
    3,              // N-Repeat
    S<8, 32, 1>,    // ABlockTransferThreadClusterLengths_K0_M_K1
    S<1, 0, 2>,     // ABlockTransferThreadClusterArrangeOrder
    S<1, 0, 2>,     // ABlockTransferSrcAccessOrder
    2,              // ABlockTransferSrcVectorDim
    16,             // ABlockTransferSrcScalarPerVector
    16,             // ABlockTransferDstScalarPerVector_K1
    true,           // AThreadTransferSrcResetCoordinateAfterRun
    S<8, 32, 1>,    // BBlockTransferThreadClusterLengths_K0_N_K1
    S<1, 0, 2>,     // BBlockTransferThreadClusterArrangeOrder
    S<1, 0, 2>,     // BBlockTransferSrcAccessOrder
    2,              // BBlockTransferSrcVectorDim
    16,             // BBlockTransferSrcScalarPerVector
    16,             // BBlockTransferDstScalarPerVector_K1
    true,           // BThreadTransferSrcResetCoordinateAfterRun
    2,              // CShuffleMRepeatPerStore
    1,              // CShuffleNRepeatPerStore
    S<1, 32, 1, 8>, // CBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
    8>;             // CBlockTransferScalarPerVector_NWaveNPerXdl
#endif

// Configuration 5: 256x128x64, BlockSize=256, ab_type=half_t
#if defined(GEMM_CONFIG_5) ||                                                         \
    (!defined(GEMM_CONFIG_1) && !defined(GEMM_CONFIG_2) && !defined(GEMM_CONFIG_3) && \
     !defined(GEMM_CONFIG_4) && !defined(GEMM_CONFIG_5) && !defined(GEMM_CONFIG_6) && \
     !defined(GEMM_CONFIG_7) && !defined(GEMM_CONFIG_8) && !defined(GEMM_CONFIG_9))
using DeviceGemmInstance = ck::tensor_operation::device::DeviceGemmWmma_CShuffle<
    ALayout,
    BLayout,
    CLayout,
    ADataType,
    BDataType,
    CDataType,
    AccDataType,
    CShuffleDataType,
    AElementOp,
    BElementOp,
    CElementOp,
    GemmDefault,
    1,               // Prefetch stage
    256,             // BlockSize
    256,             // MPerBlock
    128,             // NPerBlock
    64,              // KPerBlock
    8,               // K1
    16,              // MPerWmma
    16,              // NPerWmma
    4,               // M-Repeat
    4,               // N-Repeat
    S<8, 32, 1>,     // ABlockTransferThreadClusterLengths_K0_M_K1
    S<1, 0, 2>,      // ABlockTransferThreadClusterArrangeOrder
    S<1, 0, 2>,      // ABlockTransferSrcAccessOrder
    2,               // ABlockTransferSrcVectorDim
    8,               // ABlockTransferSrcScalarPerVector
    8,               // ABlockTransferDstScalarPerVector_K1
    true,            // AThreadTransferSrcResetCoordinateAfterRun
    S<8, 32, 1>,     // BBlockTransferThreadClusterLengths_K0_N_K1
    S<1, 0, 2>,      // BBlockTransferThreadClusterArrangeOrder
    S<1, 0, 2>,      // BBlockTransferSrcAccessOrder
    2,               // BBlockTransferSrcVectorDim
    8,               // BBlockTransferSrcScalarPerVector
    8,               // BBlockTransferDstScalarPerVector_K1
    true,            // BThreadTransferSrcResetCoordinateAfterRun
    2,               // CShuffleMRepeatPerStore
    4,               // CShuffleNRepeatPerStore
    S<1, 16, 1, 16>, // CBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
    8>;              // CBlockTransferScalarPerVector_NWaveNPerXdl
#endif

// Configuration 6: 256x256x64, BlockSize=256, ab_type=half_t
#ifdef GEMM_CONFIG_6
using DeviceGemmInstance = ck::tensor_operation::device::DeviceGemmWmma_CShuffle<
    ALayout,
    BLayout,
    CLayout,
    ADataType,
    BDataType,
    CDataType,
    AccDataType,
    CShuffleDataType,
    AElementOp,
    BElementOp,
    CElementOp,
    GemmDefault,
    1,               // Prefetch stage
    256,             // BlockSize
    256,             // MPerBlock
    256,             // NPerBlock
    64,              // KPerBlock
    8,               // K1
    16,              // MPerWmma
    16,              // NPerWmma
    4,               // M-Repeat
    8,               // N-Repeat
    S<8, 32, 1>,     // ABlockTransferThreadClusterLengths_K0_M_K1
    S<1, 0, 2>,      // ABlockTransferThreadClusterArrangeOrder
    S<1, 0, 2>,      // ABlockTransferSrcAccessOrder
    2,               // ABlockTransferSrcVectorDim
    8,               // ABlockTransferSrcScalarPerVector
    8,               // ABlockTransferDstScalarPerVector_K1
    true,            // AThreadTransferSrcResetCoordinateAfterRun
    S<8, 32, 1>,     // BBlockTransferThreadClusterLengths_K0_N_K1
    S<1, 0, 2>,      // BBlockTransferThreadClusterArrangeOrder
    S<1, 0, 2>,      // BBlockTransferSrcAccessOrder
    2,               // BBlockTransferSrcVectorDim
    8,               // BBlockTransferSrcScalarPerVector
    8,               // BBlockTransferDstScalarPerVector_K1
    true,            // BThreadTransferSrcResetCoordinateAfterRun
    2,               // CShuffleMRepeatPerStore
    4,               // CShuffleNRepeatPerStore
    S<1, 16, 1, 16>, // CBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
    8>;              // CBlockTransferScalarPerVector_NWaveNPerXdl
#endif

// Configuration 7: 128x256x64, BlockSize=128, ab_type=half_t
#ifdef GEMM_CONFIG_7
using DeviceGemmInstance = ck::tensor_operation::device::DeviceGemmWmma_CShuffle<
    ALayout,
    BLayout,
    CLayout,
    ADataType,
    BDataType,
    CDataType,
    AccDataType,
    CShuffleDataType,
    AElementOp,
    BElementOp,
    CElementOp,
    GemmDefault,
    1,              // Prefetch stage
    128,            // BlockSize
    128,            // MPerBlock
    256,            // NPerBlock
    64,             // KPerBlock
    8,              // K1
    16,             // MPerWmma
    16,             // NPerWmma
    4,              // M-Repeat
    8,              // N-Repeat
    S<8, 16, 1>,    // ABlockTransferThreadClusterLengths_K0_M_K1
    S<1, 0, 2>,     // ABlockTransferThreadClusterArrangeOrder
    S<1, 0, 2>,     // ABlockTransferSrcAccessOrder
    2,              // ABlockTransferSrcVectorDim
    8,              // ABlockTransferSrcScalarPerVector
    8,              // ABlockTransferDstScalarPerVector_K1
    true,           // AThreadTransferSrcResetCoordinateAfterRun
    S<8, 16, 1>,    // BBlockTransferThreadClusterLengths_K0_N_K1
    S<1, 0, 2>,     // BBlockTransferThreadClusterArrangeOrder
    S<1, 0, 2>,     // BBlockTransferSrcAccessOrder
    2,              // BBlockTransferSrcVectorDim
    8,              // BBlockTransferSrcScalarPerVector
    8,              // BBlockTransferDstScalarPerVector_K1
    true,           // BThreadTransferSrcResetCoordinateAfterRun
    2,              // CShuffleMRepeatPerStore
    4,              // CShuffleNRepeatPerStore
    S<1, 8, 1, 16>, // CBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
    8>;             // CBlockTransferScalarPerVector_NWaveNPerXdl
#endif

// Configuration 8: 128x128x64, BlockSize=256, ab_type=half_t
#ifdef GEMM_CONFIG_8
using DeviceGemmInstance = ck::tensor_operation::device::DeviceGemmWmma_CShuffle<
    ALayout,
    BLayout,
    CLayout,
    ADataType,
    BDataType,
    CDataType,
    AccDataType,
    CShuffleDataType,
    AElementOp,
    BElementOp,
    CElementOp,
    GemmDefault,
    1,              // Prefetch stage
    256,            // BlockSize
    128,            // MPerBlock
    128,            // NPerBlock
    64,             // KPerBlock
    8,              // K1
    16,             // MPerWmma
    16,             // NPerWmma
    4,              // M-Repeat
    2,              // N-Repeat
    S<4, 64, 1>,    // ABlockTransferThreadClusterLengths_K0_M_K1
    S<1, 0, 2>,     // ABlockTransferThreadClusterArrangeOrder
    S<1, 0, 2>,     // ABlockTransferSrcAccessOrder
    2,              // ABlockTransferSrcVectorDim
    8,              // ABlockTransferSrcScalarPerVector
    8,              // ABlockTransferDstScalarPerVector_K1
    true,           // AThreadTransferSrcResetCoordinateAfterRun
    S<4, 64, 1>,    // BBlockTransferThreadClusterLengths_K0_N_K1
    S<1, 0, 2>,     // BBlockTransferThreadClusterArrangeOrder
    S<1, 0, 2>,     // BBlockTransferSrcAccessOrder
    2,              // BBlockTransferSrcVectorDim
    8,              // BBlockTransferSrcScalarPerVector
    8,              // BBlockTransferDstScalarPerVector_K1
    true,           // BThreadTransferSrcResetCoordinateAfterRun
    1,              // CShuffleMRepeatPerStore
    1,              // CShuffleNRepeatPerStore
    S<1, 32, 1, 8>, // CBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
    8>;             // CBlockTransferScalarPerVector_NWaveNPerXdl
#endif

// Configuration 9: 64x128x64, BlockSize=128, ab_type=half_t
#ifdef GEMM_CONFIG_9
using DeviceGemmInstance = ck::tensor_operation::device::DeviceGemmWmma_CShuffle<
    ALayout,
    BLayout,
    CLayout,
    ADataType,
    BDataType,
    CDataType,
    AccDataType,
    CShuffleDataType,
    AElementOp,
    BElementOp,
    CElementOp,
    GemmDefault,
    1,              // Prefetch stage
    128,            // BlockSize
    64,             // MPerBlock
    128,            // NPerBlock
    64,             // KPerBlock
    2,              // K1
    16,             // MPerWmma
    16,             // NPerWmma
    2,              // M-Repeat
    4,              // N-Repeat
    S<4, 32, 1>,    // ABlockTransferThreadClusterLengths_K0_M_K1
    S<1, 0, 2>,     // ABlockTransferThreadClusterArrangeOrder
    S<1, 0, 2>,     // ABlockTransferSrcAccessOrder
    2,              // ABlockTransferSrcVectorDim
    2,              // ABlockTransferSrcScalarPerVector
    2,              // ABlockTransferDstScalarPerVector_K1
    true,           // AThreadTransferSrcResetCoordinateAfterRun
    S<4, 32, 1>,    // BBlockTransferThreadClusterLengths_K0_N_K1
    S<1, 0, 2>,     // BBlockTransferThreadClusterArrangeOrder
    S<1, 0, 2>,     // BBlockTransferSrcAccessOrder
    2,              // BBlockTransferSrcVectorDim
    2,              // BBlockTransferSrcScalarPerVector
    2,              // BBlockTransferDstScalarPerVector_K1
    true,           // BThreadTransferSrcResetCoordinateAfterRun
    1,              // CShuffleMRepeatPerStore
    1,              // CShuffleNRepeatPerStore
    S<1, 32, 1, 4>, // CBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
    8>;             // CBlockTransferScalarPerVector_NWaveNPerXdl
#endif

using ReferenceGemmInstance = ck::tensor_operation::host::
    ReferenceGemm<ADataType, BDataType, CDataType, AccDataType, AElementOp, BElementOp, CElementOp>;

using ReferenceGemmInstanceGPU = ck::tensor_operation::device::ReferenceGemm<ALayout,
                                                                             BLayout,
                                                                             CLayout,
                                                                             ADataType,
                                                                             BDataType,
                                                                             CDataType,
                                                                             AccDataType,
                                                                             AElementOp,
                                                                             BElementOp,
                                                                             CElementOp>;

#include "run_gemm_example.inc"

int main(int argc, char* argv[]) { return !run_gemm_example(argc, argv); }
