// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_abd_xdl_waveletmodel_cshuffle_v3.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using namespace ck::tensor_layout::convolution;

using F16  = ck::half_t;
using BF16 = ck::bhalf_t;
using F32  = float;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using Empty_Tuple = ck::Tuple<>;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

static constexpr auto ConvFwdDefault =
    ck::tensor_operation::device::ConvolutionForwardSpecialization::Default;

// ---- Cluster type aliases (no commas in call sites) ----
// A/B block transfer cluster: S<AK0, TLoad/AK0, 1>
//   K=32 (AK0=4):
using AC_K32_TL64  = S<4, 16, 1>;
using AC_K32_TL128 = S<4, 32, 1>;
using AC_K32_TL256 = S<4, 64, 1>;
//   K=64 (AK0=8):
using AC_K64_TL64  = S<8, 8, 1>;
using AC_K64_TL128 = S<8, 16, 1>;
using AC_K64_TL256 = S<8, 32, 1>;
//   K=128 (AK0=16):
using AC_K128_TL64  = S<16, 4, 1>;
using AC_K128_TL128 = S<16, 8, 1>;
using AC_K128_TL256 = S<16, 16, 1>;

// CDE block transfer cluster: depends on wN and TMath.
//   wN=1 (ShufN=32): Csv=4, cluster=S<1,TMath/8,1,8>
using CDE_wN1_TM64  = S<1, 8, 1, 8>;
using CDE_wN1_TM128 = S<1, 16, 1, 8>;
using CDE_wN1_TM256 = S<1, 32, 1, 8>;
//   wN=2 (ShufN=64): Csv=8, cluster=S<1,TMath/8,1,8>  (same dims, different Csv passed separately)
using CDE_wN2_TM64  = S<1, 8, 1, 8>;
using CDE_wN2_TM128 = S<1, 16, 1, 8>;
using CDE_wN2_TM256 = S<1, 32, 1, 8>;
//   wN=4 (ShufN=128): Csv=8, cluster=S<1,TMath/16,1,16>
using CDE_wN4_TM64  = S<1, 4, 1, 16>;
using CDE_wN4_TM128 = S<1, 8, 1, 16>;
using CDE_wN4_TM256 = S<1, 16, 1, 16>;

// ---- Extra CDE aliases for MPerXDL=NPerXDL=16 ----
// With NPerXDL=16: ShufN=wN*16.
//         Reuse CDE_wN{1,2}_TM* shapes (same S<1,TMath/8,1,8>), pass Csv=8 at call site.

// Wavelet conv forward F16 instances.
// LaunchBlockSize=TLoad+TMath; BlockSize=TMath (MFMA assignment).
// K1=8, SrcVectorDim=2, DstScalar=8 (128-bit LDS stores), SrcScalar=8 (128-bit loads).
// AddExtraM/N=true: LDS bank-conflict padding.

// clang-format off
// Macro args contain no commas — S<...> types pre-aliased above.
// MPerXDL=NPerXDL=32 instances:
#define WAVELET_INST(TLoad, TMath, M, N, K, MXdl, NXdl, AC, CDE, Csv) \
    DeviceGroupedConvFwdMultipleABD_WaveletModel_Xdl_CShuffle_V3< \
        NDimSpatial, ALayout, BLayout, Empty_Tuple, ELayout, \
        F16, F16, F32, F16, Empty_Tuple, F16, \
        PassThrough, PassThrough, PassThrough, ConvSpec, \
        TLoad, TMath, M, N, K, 8, 32, 32, MXdl, NXdl, \
        AC, S<1,0,2>, S<1,0,2>, 2, 8, 8, true, \
        AC, S<1,0,2>, S<1,0,2>, 2, 8, 8, true, \
        1, 1, CDE, Csv>
template <ck::index_t NDimSpatial,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          ck::tensor_operation::device::ConvolutionForwardSpecialization ConvSpec>
using device_grouped_conv_fwd_wavelet_xdl_c_shuffle_f16_instances = std::tuple<

// ============================================================
// M=128, N=64, K=32  (AK0=4, LDS≈12KB)
// (MXdl=2,NXdl=1) → wM=2,wN=2,TMath=256 | Csv=8
// WAVELET_INST( 64, 256, 128,  64, 32, 2, 1, AC_K32_TL64,  CDE_wN2_TM256, 8),
// WAVELET_INST(128, 256, 128,  64, 32, 2, 1, AC_K32_TL128, CDE_wN2_TM256, 8),
// WAVELET_INST(256, 256, 128,  64, 32, 2, 1, AC_K32_TL256, CDE_wN2_TM256, 8),
// (MXdl=2,NXdl=2) → wM=2,wN=1,TMath=128 | Csv=4
// WAVELET_INST( 64, 128, 128,  64, 32, 2, 2, AC_K32_TL64,  CDE_wN1_TM128, 4),
// WAVELET_INST(128, 128, 128,  64, 32, 2, 2, AC_K32_TL128, CDE_wN1_TM128, 4),
// WAVELET_INST(256, 128, 128,  64, 32, 2, 2, AC_K32_TL256, CDE_wN1_TM128, 4),
// (MXdl=1,NXdl=2) → wM=4,wN=1,TMath=256 | Csv=4
// WAVELET_INST( 64, 256, 128,  64, 32, 1, 2, AC_K32_TL64,  CDE_wN1_TM256, 4),
// WAVELET_INST(128, 256, 128,  64, 32, 1, 2, AC_K32_TL128, CDE_wN1_TM256, 4),
// WAVELET_INST(256, 256, 128,  64, 32, 1, 2, AC_K32_TL256, CDE_wN1_TM256, 4),
// (MXdl=4,NXdl=1) → wM=1,wN=2,TMath=128 | Csv=8
// WAVELET_INST( 64, 128, 128,  64, 32, 4, 1, AC_K32_TL64,  CDE_wN2_TM128, 8),
// WAVELET_INST(128, 128, 128,  64, 32, 4, 1, AC_K32_TL128, CDE_wN2_TM128, 8),
// WAVELET_INST(256, 128, 128,  64, 32, 4, 1, AC_K32_TL256, CDE_wN2_TM128, 8),
// (MXdl=4,NXdl=2) → wM=1,wN=1,TMath=64  | Csv=4
// WAVELET_INST( 64,  64, 128,  64, 32, 4, 2, AC_K32_TL64,  CDE_wN1_TM64,  4),
// WAVELET_INST(128,  64, 128,  64, 32, 4, 2, AC_K32_TL128, CDE_wN1_TM64,  4),
// WAVELET_INST(256,  64, 128,  64, 32, 4, 2, AC_K32_TL256, CDE_wN1_TM64,  4),

// ============================================================
// M=128, N=64, K=64  (AK0=8, LDS≈24KB)
// (MXdl=2,NXdl=1) → wM=2,wN=2,TMath=256 | Csv=8
// WAVELET_INST( 64, 256, 128,  64, 64, 2, 1, AC_K64_TL64,  CDE_wN2_TM256, 8),
WAVELET_INST(128, 256, 128,  64, 64, 2, 1, AC_K64_TL128, CDE_wN2_TM256, 8),
// WAVELET_INST(256, 256, 128,  64, 64, 2, 1, AC_K64_TL256, CDE_wN2_TM256, 8),
// (MXdl=2,NXdl=2) → wM=2,wN=1,TMath=128 | Csv=4
// WAVELET_INST( 64, 128, 128,  64, 64, 2, 2, AC_K64_TL64,  CDE_wN1_TM128, 4),
// WAVELET_INST(128, 128, 128,  64, 64, 2, 2, AC_K64_TL128, CDE_wN1_TM128, 4),
// WAVELET_INST(256, 128, 128,  64, 64, 2, 2, AC_K64_TL256, CDE_wN1_TM128, 4),
// (MXdl=1,NXdl=2) → wM=4,wN=1,TMath=256 | Csv=4
// WAVELET_INST( 64, 256, 128,  64, 64, 1, 2, AC_K64_TL64,  CDE_wN1_TM256, 4),
WAVELET_INST(128, 256, 128,  64, 64, 1, 2, AC_K64_TL128, CDE_wN1_TM256, 4),
WAVELET_INST(256, 256, 128,  64, 64, 1, 2, AC_K64_TL256, CDE_wN1_TM256, 4),
// (MXdl=4,NXdl=1) → wM=1,wN=2,TMath=128 | Csv=8
// WAVELET_INST( 64, 128, 128,  64, 64, 4, 1, AC_K64_TL64,  CDE_wN2_TM128, 8),
// WAVELET_INST(128, 128, 128,  64, 64, 4, 1, AC_K64_TL128, CDE_wN2_TM128, 8),
// WAVELET_INST(256, 128, 128,  64, 64, 4, 1, AC_K64_TL256, CDE_wN2_TM128, 8),
// (MXdl=4,NXdl=2) → wM=1,wN=1,TMath=64  | Csv=4
// WAVELET_INST( 64,  64, 128,  64, 64, 4, 2, AC_K64_TL64,  CDE_wN1_TM64,  4),
// WAVELET_INST(128,  64, 128,  64, 64, 4, 2, AC_K64_TL128, CDE_wN1_TM64,  4),
// WAVELET_INST(256,  64, 128,  64, 64, 4, 2, AC_K64_TL256, CDE_wN1_TM64,  4),

// ============================================================
// M=128, N=128, K=32  (AK0=4, LDS≈16KB)
// (MXdl=2,NXdl=2) → wM=2,wN=2,TMath=256 | Csv=8
// WAVELET_INST( 64, 256, 128, 128, 32, 2, 2, AC_K32_TL64,  CDE_wN2_TM256, 8),
// WAVELET_INST(128, 256, 128, 128, 32, 2, 2, AC_K32_TL128, CDE_wN2_TM256, 8),
// WAVELET_INST(256, 256, 128, 128, 32, 2, 2, AC_K32_TL256, CDE_wN2_TM256, 8),
// (MXdl=1,NXdl=4) → wM=4,wN=1,TMath=256 | Csv=4
// WAVELET_INST( 64, 256, 128, 128, 32, 1, 4, AC_K32_TL64,  CDE_wN1_TM256, 4),
// WAVELET_INST(128, 256, 128, 128, 32, 1, 4, AC_K32_TL128, CDE_wN1_TM256, 4),
// WAVELET_INST(256, 256, 128, 128, 32, 1, 4, AC_K32_TL256, CDE_wN1_TM256, 4),
// (MXdl=2,NXdl=4) → wM=2,wN=1,TMath=128 | Csv=4
// WAVELET_INST( 64, 128, 128, 128, 32, 2, 4, AC_K32_TL64,  CDE_wN1_TM128, 4),
// WAVELET_INST(128, 128, 128, 128, 32, 2, 4, AC_K32_TL128, CDE_wN1_TM128, 4),
// WAVELET_INST(256, 128, 128, 128, 32, 2, 4, AC_K32_TL256, CDE_wN1_TM128, 4),
// (MXdl=4,NXdl=1) → wM=1,wN=4,TMath=256 | Csv=8
// WAVELET_INST( 64, 256, 128, 128, 32, 4, 1, AC_K32_TL64,  CDE_wN4_TM256, 8),
// WAVELET_INST(128, 256, 128, 128, 32, 4, 1, AC_K32_TL128, CDE_wN4_TM256, 8),
// WAVELET_INST(256, 256, 128, 128, 32, 4, 1, AC_K32_TL256, CDE_wN4_TM256, 8),
// (MXdl=4,NXdl=2) → wM=1,wN=2,TMath=128 | Csv=8
// WAVELET_INST( 64, 128, 128, 128, 32, 4, 2, AC_K32_TL64,  CDE_wN2_TM128, 8),
// WAVELET_INST(128, 128, 128, 128, 32, 4, 2, AC_K32_TL128, CDE_wN2_TM128, 8),
// WAVELET_INST(256, 128, 128, 128, 32, 4, 2, AC_K32_TL256, CDE_wN2_TM128, 8),
// (MXdl=4,NXdl=4) → wM=1,wN=1,TMath=64  | Csv=4
// WAVELET_INST( 64,  64, 128, 128, 32, 4, 4, AC_K32_TL64,  CDE_wN1_TM64,  4),
// WAVELET_INST(128,  64, 128, 128, 32, 4, 4, AC_K32_TL128, CDE_wN1_TM64,  4),
// WAVELET_INST(256,  64, 128, 128, 32, 4, 4, AC_K32_TL256, CDE_wN1_TM64,  4),

// ============================================================
// M=128, N=128, K=64  (AK0=8, LDS≈32KB)
// (MXdl=2,NXdl=2) → wM=2,wN=2,TMath=256 | Csv=8
// WAVELET_INST( 64, 256, 128, 128, 64, 2, 2, AC_K64_TL64,  CDE_wN2_TM256, 8),
// WAVELET_INST(128, 256, 128, 128, 64, 2, 2, AC_K64_TL128, CDE_wN2_TM256, 8),
// WAVELET_INST(256, 256, 128, 128, 64, 2, 2, AC_K64_TL256, CDE_wN2_TM256, 8),
// (MXdl=1,NXdl=4) → wM=4,wN=1,TMath=256 | Csv=4
// WAVELET_INST( 64, 256, 128, 128, 64, 1, 4, AC_K64_TL64,  CDE_wN1_TM256, 4),
// WAVELET_INST(128, 256, 128, 128, 64, 1, 4, AC_K64_TL128, CDE_wN1_TM256, 4),
// WAVELET_INST(256, 256, 128, 128, 64, 1, 4, AC_K64_TL256, CDE_wN1_TM256, 4),
// (MXdl=2,NXdl=4) → wM=2,wN=1,TMath=128 | Csv=4
// WAVELET_INST( 64, 128, 128, 128, 64, 2, 4, AC_K64_TL64,  CDE_wN1_TM128, 4),
// WAVELET_INST(128, 128, 128, 128, 64, 2, 4, AC_K64_TL128, CDE_wN1_TM128, 4),
// WAVELET_INST(256, 128, 128, 128, 64, 2, 4, AC_K64_TL256, CDE_wN1_TM128, 4),
// (MXdl=4,NXdl=1) → wM=1,wN=4,TMath=256 | Csv=8
// WAVELET_INST( 64, 256, 128, 128, 64, 4, 1, AC_K64_TL64,  CDE_wN4_TM256, 8),
// WAVELET_INST(128, 256, 128, 128, 64, 4, 1, AC_K64_TL128, CDE_wN4_TM256, 8),
// WAVELET_INST(256, 256, 128, 128, 64, 4, 1, AC_K64_TL256, CDE_wN4_TM256, 8),
// (MXdl=4,NXdl=2) → wM=1,wN=2,TMath=128 | Csv=8
// WAVELET_INST( 64, 128, 128, 128, 64, 4, 2, AC_K64_TL64,  CDE_wN2_TM128, 8),
// WAVELET_INST(128, 128, 128, 128, 64, 4, 2, AC_K64_TL128, CDE_wN2_TM128, 8),
// WAVELET_INST(256, 128, 128, 128, 64, 4, 2, AC_K64_TL256, CDE_wN2_TM128, 8),
// (MXdl=4,NXdl=4) → wM=1,wN=1,TMath=64  | Csv=4
// WAVELET_INST( 64,  64, 128, 128, 64, 4, 4, AC_K64_TL64,  CDE_wN1_TM64,  4),
// WAVELET_INST(128,  64, 128, 128, 64, 4, 4, AC_K64_TL128, CDE_wN1_TM64,  4),
// WAVELET_INST(256,  64, 128, 128, 64, 4, 4, AC_K64_TL256, CDE_wN1_TM64,  4),

// ============================================================
// M=64, N=64, K=32  (AK0=4, LDS≈8KB)
// (MXdl=1,NXdl=1) → wM=2,wN=2,TMath=256 | Csv=8
// WAVELET_INST( 64, 256,  64,  64, 32, 1, 1, AC_K32_TL64,  CDE_wN2_TM256, 8),
// WAVELET_INST(128, 256,  64,  64, 32, 1, 1, AC_K32_TL128, CDE_wN2_TM256, 8),
// WAVELET_INST(256, 256,  64,  64, 32, 1, 1, AC_K32_TL256, CDE_wN2_TM256, 8),
// (MXdl=1,NXdl=2) → wM=2,wN=1,TMath=128 | Csv=4
// WAVELET_INST( 64, 128,  64,  64, 32, 1, 2, AC_K32_TL64,  CDE_wN1_TM128, 4),
// WAVELET_INST(128, 128,  64,  64, 32, 1, 2, AC_K32_TL128, CDE_wN1_TM128, 4),
// WAVELET_INST(256, 128,  64,  64, 32, 1, 2, AC_K32_TL256, CDE_wN1_TM128, 4),
// (MXdl=2,NXdl=1) → wM=1,wN=2,TMath=128 | Csv=8
// WAVELET_INST( 64, 128,  64,  64, 32, 2, 1, AC_K32_TL64,  CDE_wN2_TM128, 8),
// WAVELET_INST(128, 128,  64,  64, 32, 2, 1, AC_K32_TL128, CDE_wN2_TM128, 8),
// WAVELET_INST(256, 128,  64,  64, 32, 2, 1, AC_K32_TL256, CDE_wN2_TM128, 8),
// (MXdl=2,NXdl=2) → wM=1,wN=1,TMath=64  | Csv=4
// WAVELET_INST( 64,  64,  64,  64, 32, 2, 2, AC_K32_TL64,  CDE_wN1_TM64,  4),
// WAVELET_INST(128,  64,  64,  64, 32, 2, 2, AC_K32_TL128, CDE_wN1_TM64,  4),
// WAVELET_INST(256,  64,  64,  64, 32, 2, 2, AC_K32_TL256, CDE_wN1_TM64,  4),

// ============================================================
// M=64, N=64, K=64  (AK0=8, LDS≈16KB)
// (MXdl=1,NXdl=1) → wM=2,wN=2,TMath=256 | Csv=8
// WAVELET_INST( 64, 256,  64,  64, 64, 1, 1, AC_K64_TL64,  CDE_wN2_TM256, 8),
// WAVELET_INST(128, 256,  64,  64, 64, 1, 1, AC_K64_TL128, CDE_wN2_TM256, 8),
// WAVELET_INST(256, 256,  64,  64, 64, 1, 1, AC_K64_TL256, CDE_wN2_TM256, 8),
// (MXdl=1,NXdl=2) → wM=2,wN=1,TMath=128 | Csv=4
// WAVELET_INST( 64, 128,  64,  64, 64, 1, 2, AC_K64_TL64,  CDE_wN1_TM128, 4),
// WAVELET_INST(128, 128,  64,  64, 64, 1, 2, AC_K64_TL128, CDE_wN1_TM128, 4),
// WAVELET_INST(256, 128,  64,  64, 64, 1, 2, AC_K64_TL256, CDE_wN1_TM128, 4),
// (MXdl=2,NXdl=1) → wM=1,wN=2,TMath=128 | Csv=8
// WAVELET_INST( 64, 128,  64,  64, 64, 2, 1, AC_K64_TL64,  CDE_wN2_TM128, 8),
// WAVELET_INST(128, 128,  64,  64, 64, 2, 1, AC_K64_TL128, CDE_wN2_TM128, 8),
// WAVELET_INST(256, 128,  64,  64, 64, 2, 1, AC_K64_TL256, CDE_wN2_TM128, 8),
// (MXdl=2,NXdl=2) → wM=1,wN=1,TMath=64  | Csv=4
// WAVELET_INST( 64,  64,  64,  64, 64, 2, 2, AC_K64_TL64,  CDE_wN1_TM64,  4),
// WAVELET_INST(128,  64,  64,  64, 64, 2, 2, AC_K64_TL128, CDE_wN1_TM64,  4),
// WAVELET_INST(256,  64,  64,  64, 64, 2, 2, AC_K64_TL256, CDE_wN1_TM64,  4),

// ============================================================
// M=64, N=128, K=32  (AK0=4, LDS≈12KB)
// (MXdl=1,NXdl=2) → wM=2,wN=2,TMath=256 | Csv=8
// WAVELET_INST( 64, 256,  64, 128, 32, 1, 2, AC_K32_TL64,  CDE_wN2_TM256, 8),
// WAVELET_INST(128, 256,  64, 128, 32, 1, 2, AC_K32_TL128, CDE_wN2_TM256, 8),
// WAVELET_INST(256, 256,  64, 128, 32, 1, 2, AC_K32_TL256, CDE_wN2_TM256, 8),
// (MXdl=1,NXdl=4) → wM=2,wN=1,TMath=128 | Csv=4
// WAVELET_INST( 64, 128,  64, 128, 32, 1, 4, AC_K32_TL64,  CDE_wN1_TM128, 4),
// WAVELET_INST(128, 128,  64, 128, 32, 1, 4, AC_K32_TL128, CDE_wN1_TM128, 4),
// WAVELET_INST(256, 128,  64, 128, 32, 1, 4, AC_K32_TL256, CDE_wN1_TM128, 4),
// (MXdl=2,NXdl=1) → wM=1,wN=4,TMath=256 | Csv=8
// WAVELET_INST( 64, 256,  64, 128, 32, 2, 1, AC_K32_TL64,  CDE_wN4_TM256, 8),
// WAVELET_INST(128, 256,  64, 128, 32, 2, 1, AC_K32_TL128, CDE_wN4_TM256, 8),
// WAVELET_INST(256, 256,  64, 128, 32, 2, 1, AC_K32_TL256, CDE_wN4_TM256, 8),
// (MXdl=2,NXdl=2) → wM=1,wN=2,TMath=128 | Csv=8
// WAVELET_INST( 64, 128,  64, 128, 32, 2, 2, AC_K32_TL64,  CDE_wN2_TM128, 8),
// WAVELET_INST(128, 128,  64, 128, 32, 2, 2, AC_K32_TL128, CDE_wN2_TM128, 8),
// WAVELET_INST(256, 128,  64, 128, 32, 2, 2, AC_K32_TL256, CDE_wN2_TM128, 8),
// (MXdl=2,NXdl=4) → wM=1,wN=1,TMath=64  | Csv=4
// WAVELET_INST( 64,  64,  64, 128, 32, 2, 4, AC_K32_TL64,  CDE_wN1_TM64,  4),
// WAVELET_INST(128,  64,  64, 128, 32, 2, 4, AC_K32_TL128, CDE_wN1_TM64,  4),
// WAVELET_INST(256,  64,  64, 128, 32, 2, 4, AC_K32_TL256, CDE_wN1_TM64,  4),

// ============================================================
// M=64, N=128, K=64  (AK0=8, LDS≈24KB)
// (MXdl=1,NXdl=2) → wM=2,wN=2,TMath=256 | Csv=8
// WAVELET_INST( 64, 256,  64, 128, 64, 1, 2, AC_K64_TL64,  CDE_wN2_TM256, 8),
// WAVELET_INST(128, 256,  64, 128, 64, 1, 2, AC_K64_TL128, CDE_wN2_TM256, 8),
// WAVELET_INST(256, 256,  64, 128, 64, 1, 2, AC_K64_TL256, CDE_wN2_TM256, 8),
// (MXdl=1,NXdl=4) → wM=2,wN=1,TMath=128 | Csv=4
// WAVELET_INST( 64, 128,  64, 128, 64, 1, 4, AC_K64_TL64,  CDE_wN1_TM128, 4),
// WAVELET_INST(128, 128,  64, 128, 64, 1, 4, AC_K64_TL128, CDE_wN1_TM128, 4),
// WAVELET_INST(256, 128,  64, 128, 64, 1, 4, AC_K64_TL256, CDE_wN1_TM128, 4),
// (MXdl=2,NXdl=1) → wM=1,wN=4,TMath=256 | Csv=8
// WAVELET_INST( 64, 256,  64, 128, 64, 2, 1, AC_K64_TL64,  CDE_wN4_TM256, 8),
// WAVELET_INST(128, 256,  64, 128, 64, 2, 1, AC_K64_TL128, CDE_wN4_TM256, 8),
WAVELET_INST(256, 256,  64, 128, 64, 2, 1, AC_K64_TL256, CDE_wN4_TM256, 8),
// (MXdl=2,NXdl=2) → wM=1,wN=2,TMath=128 | Csv=8
// WAVELET_INST( 64, 128,  64, 128, 64, 2, 2, AC_K64_TL64,  CDE_wN2_TM128, 8),
// WAVELET_INST(128, 128,  64, 128, 64, 2, 2, AC_K64_TL128, CDE_wN2_TM128, 8),
// WAVELET_INST(256, 128,  64, 128, 64, 2, 2, AC_K64_TL256, CDE_wN2_TM128, 8),
// (MXdl=2,NXdl=4) → wM=1,wN=1,TMath=64  | Csv=4
// WAVELET_INST( 64,  64,  64, 128, 64, 2, 4, AC_K64_TL64,  CDE_wN1_TM64,  4),
// WAVELET_INST(128,  64,  64, 128, 64, 2, 4, AC_K64_TL128, CDE_wN1_TM64,  4),
// WAVELET_INST(256,  64,  64, 128, 64, 2, 4, AC_K64_TL256, CDE_wN1_TM64,  4),

// ============================================================
// M=256, N=64, K=32  (AK0=4, LDS≈20KB)
// (MXdl=2,NXdl=2) → wM=4,wN=1,TMath=256 | Csv=4
// WAVELET_INST( 64, 256, 256,  64, 32, 2, 2, AC_K32_TL64,  CDE_wN1_TM256, 4),
// WAVELET_INST(128, 256, 256,  64, 32, 2, 2, AC_K32_TL128, CDE_wN1_TM256, 4),
// WAVELET_INST(256, 256, 256,  64, 32, 2, 2, AC_K32_TL256, CDE_wN1_TM256, 4),
// (MXdl=4,NXdl=1) → wM=2,wN=2,TMath=256 | Csv=8
// WAVELET_INST( 64, 256, 256,  64, 32, 4, 1, AC_K32_TL64,  CDE_wN2_TM256, 8),
// WAVELET_INST(128, 256, 256,  64, 32, 4, 1, AC_K32_TL128, CDE_wN2_TM256, 8),
// WAVELET_INST(256, 256, 256,  64, 32, 4, 1, AC_K32_TL256, CDE_wN2_TM256, 8),
// (MXdl=4,NXdl=2) → wM=2,wN=1,TMath=128 | Csv=4
// WAVELET_INST( 64, 128, 256,  64, 32, 4, 2, AC_K32_TL64,  CDE_wN1_TM128, 4),
// WAVELET_INST(128, 128, 256,  64, 32, 4, 2, AC_K32_TL128, CDE_wN1_TM128, 4),
// WAVELET_INST(256, 128, 256,  64, 32, 4, 2, AC_K32_TL256, CDE_wN1_TM128, 4),
// (MXdl=8,NXdl=1) → wM=1,wN=2,TMath=128 | Csv=8
// WAVELET_INST( 64, 128, 256,  64, 32, 8, 1, AC_K32_TL64,  CDE_wN2_TM128, 8),
// WAVELET_INST(128, 128, 256,  64, 32, 8, 1, AC_K32_TL128, CDE_wN2_TM128, 8),
// WAVELET_INST(256, 128, 256,  64, 32, 8, 1, AC_K32_TL256, CDE_wN2_TM128, 8),
// (MXdl=8,NXdl=2) → wM=1,wN=1,TMath=64  | Csv=4
// WAVELET_INST( 64,  64, 256,  64, 32, 8, 2, AC_K32_TL64,  CDE_wN1_TM64,  4),
// WAVELET_INST(128,  64, 256,  64, 32, 8, 2, AC_K32_TL128, CDE_wN1_TM64,  4),
// WAVELET_INST(256,  64, 256,  64, 32, 8, 2, AC_K32_TL256, CDE_wN1_TM64,  4),

// ============================================================
// M=256, N=64, K=64  (AK0=8, LDS≈40KB)
// (MXdl=2,NXdl=2) → wM=4,wN=1,TMath=256 | Csv=4
// WAVELET_INST( 64, 256, 256,  64, 64, 2, 2, AC_K64_TL64,  CDE_wN1_TM256, 4),
WAVELET_INST(128, 256, 256,  64, 64, 2, 2, AC_K64_TL128, CDE_wN1_TM256, 4),
WAVELET_INST(256, 256, 256,  64, 64, 2, 2, AC_K64_TL256, CDE_wN1_TM256, 4),
// (MXdl=4,NXdl=1) → wM=2,wN=2,TMath=256 | Csv=8
// WAVELET_INST( 64, 256, 256,  64, 64, 4, 1, AC_K64_TL64,  CDE_wN2_TM256, 8),
// WAVELET_INST(128, 256, 256,  64, 64, 4, 1, AC_K64_TL128, CDE_wN2_TM256, 8),
// WAVELET_INST(256, 256, 256,  64, 64, 4, 1, AC_K64_TL256, CDE_wN2_TM256, 8),
// (MXdl=4,NXdl=2) → wM=2,wN=1,TMath=128 | Csv=4
// WAVELET_INST( 64, 128, 256,  64, 64, 4, 2, AC_K64_TL64,  CDE_wN1_TM128, 4),
// WAVELET_INST(128, 128, 256,  64, 64, 4, 2, AC_K64_TL128, CDE_wN1_TM128, 4),
// WAVELET_INST(256, 128, 256,  64, 64, 4, 2, AC_K64_TL256, CDE_wN1_TM128, 4),
// (MXdl=8,NXdl=1) → wM=1,wN=2,TMath=128 | Csv=8
// WAVELET_INST( 64, 128, 256,  64, 64, 8, 1, AC_K64_TL64,  CDE_wN2_TM128, 8),
// WAVELET_INST(128, 128, 256,  64, 64, 8, 1, AC_K64_TL128, CDE_wN2_TM128, 8),
// WAVELET_INST(256, 128, 256,  64, 64, 8, 1, AC_K64_TL256, CDE_wN2_TM128, 8),
// (MXdl=8,NXdl=2) → wM=1,wN=1,TMath=64  | Csv=4
// WAVELET_INST( 64,  64, 256,  64, 64, 8, 2, AC_K64_TL64,  CDE_wN1_TM64,  4),
// WAVELET_INST(128,  64, 256,  64, 64, 8, 2, AC_K64_TL128, CDE_wN1_TM64,  4),
// WAVELET_INST(256,  64, 256,  64, 64, 8, 2, AC_K64_TL256, CDE_wN1_TM64,  4),

// ============================================================
// M=256, N=128, K=32  (AK0=4, LDS≈24KB)
// (MXdl=2,NXdl=4) → wM=4,wN=1,TMath=256 | Csv=4
// WAVELET_INST( 64, 256, 256, 128, 32, 2, 4, AC_K32_TL64,  CDE_wN1_TM256, 4),
// WAVELET_INST(128, 256, 256, 128, 32, 2, 4, AC_K32_TL128, CDE_wN1_TM256, 4),
// WAVELET_INST(256, 256, 256, 128, 32, 2, 4, AC_K32_TL256, CDE_wN1_TM256, 4),
// (MXdl=4,NXdl=2) → wM=2,wN=2,TMath=256 | Csv=8
// WAVELET_INST( 64, 256, 256, 128, 32, 4, 2, AC_K32_TL64,  CDE_wN2_TM256, 8),
// WAVELET_INST(128, 256, 256, 128, 32, 4, 2, AC_K32_TL128, CDE_wN2_TM256, 8),
// WAVELET_INST(256, 256, 256, 128, 32, 4, 2, AC_K32_TL256, CDE_wN2_TM256, 8),
// (MXdl=4,NXdl=4) → wM=2,wN=1,TMath=128 | Csv=4
// WAVELET_INST( 64, 128, 256, 128, 32, 4, 4, AC_K32_TL64,  CDE_wN1_TM128, 4),
// WAVELET_INST(128, 128, 256, 128, 32, 4, 4, AC_K32_TL128, CDE_wN1_TM128, 4),
// WAVELET_INST(256, 128, 256, 128, 32, 4, 4, AC_K32_TL256, CDE_wN1_TM128, 4),
// (MXdl=8,NXdl=2) → wM=1,wN=2,TMath=128 | Csv=8
// WAVELET_INST( 64, 128, 256, 128, 32, 8, 2, AC_K32_TL64,  CDE_wN2_TM128, 8),
// WAVELET_INST(128, 128, 256, 128, 32, 8, 2, AC_K32_TL128, CDE_wN2_TM128, 8),
// WAVELET_INST(256, 128, 256, 128, 32, 8, 2, AC_K32_TL256, CDE_wN2_TM128, 8),
// (MXdl=8,NXdl=4) → wM=1,wN=1,TMath=64  | Csv=4
// WAVELET_INST( 64,  64, 256, 128, 32, 8, 4, AC_K32_TL64,  CDE_wN1_TM64,  4),
// WAVELET_INST(128,  64, 256, 128, 32, 8, 4, AC_K32_TL128, CDE_wN1_TM64,  4),
// WAVELET_INST(256,  64, 256, 128, 32, 8, 4, AC_K32_TL256, CDE_wN1_TM64,  4),

// ============================================================
// M=256, N=128, K=64  (AK0=8, LDS≈48KB)
// (MXdl=2,NXdl=4) → wM=4,wN=1,TMath=256 | Csv=4
// WAVELET_INST( 64, 256, 256, 128, 64, 2, 4, AC_K64_TL64,  CDE_wN1_TM256, 4),
// WAVELET_INST(128, 256, 256, 128, 64, 2, 4, AC_K64_TL128, CDE_wN1_TM256, 4),
// WAVELET_INST(256, 256, 256, 128, 64, 2, 4, AC_K64_TL256, CDE_wN1_TM256, 4),
// (MXdl=4,NXdl=2) → wM=2,wN=2,TMath=256 | Csv=8
// WAVELET_INST( 64, 256, 256, 128, 64, 4, 2, AC_K64_TL64,  CDE_wN2_TM256, 8),
// WAVELET_INST(128, 256, 256, 128, 64, 4, 2, AC_K64_TL128, CDE_wN2_TM256, 8),
// WAVELET_INST(256, 256, 256, 128, 64, 4, 2, AC_K64_TL256, CDE_wN2_TM256, 8),
// (MXdl=4,NXdl=4) → wM=2,wN=1,TMath=128 | Csv=4
// WAVELET_INST( 64, 128, 256, 128, 64, 4, 4, AC_K64_TL64,  CDE_wN1_TM128, 4),
// WAVELET_INST(128, 128, 256, 128, 64, 4, 4, AC_K64_TL128, CDE_wN1_TM128, 4),
// WAVELET_INST(256, 128, 256, 128, 64, 4, 4, AC_K64_TL256, CDE_wN1_TM128, 4),
// (MXdl=8,NXdl=2) → wM=1,wN=2,TMath=128 | Csv=8
// WAVELET_INST( 64, 128, 256, 128, 64, 8, 2, AC_K64_TL64,  CDE_wN2_TM128, 8),
// WAVELET_INST(128, 128, 256, 128, 64, 8, 2, AC_K64_TL128, CDE_wN2_TM128, 8),
// WAVELET_INST(256, 128, 256, 128, 64, 8, 2, AC_K64_TL256, CDE_wN2_TM128, 8),
// (MXdl=8,NXdl=4) → wM=1,wN=1,TMath=64  | Csv=4
// WAVELET_INST( 64,  64, 256, 128, 64, 8, 4, AC_K64_TL64,  CDE_wN1_TM64,  4),
// WAVELET_INST(128,  64, 256, 128, 64, 8, 4, AC_K64_TL128, CDE_wN1_TM64,  4),
// WAVELET_INST(256,  64, 256, 128, 64, 8, 4, AC_K64_TL256, CDE_wN1_TM64,  4),

// ============================================================
// M=128, N=64, K=128  (AK0=16, A+B LDS≈50KB, C_shuffle=16KB — fits in 64KB)
// (MXdl=2,NXdl=1) → wM=2,wN=2,TMath=256 | Csv=8
// WAVELET_INST( 64, 256, 128,  64, 128, 2, 1, AC_K128_TL64,  CDE_wN2_TM256, 8),
// WAVELET_INST(128, 256, 128,  64, 128, 2, 1, AC_K128_TL128, CDE_wN2_TM256, 8),
// WAVELET_INST(256, 256, 128,  64, 128, 2, 1, AC_K128_TL256, CDE_wN2_TM256, 8),
// (MXdl=2,NXdl=2) → wM=2,wN=1,TMath=128 | Csv=4
// WAVELET_INST( 64, 128, 128,  64, 128, 2, 2, AC_K128_TL64,  CDE_wN1_TM128, 4),
// WAVELET_INST(128, 128, 128,  64, 128, 2, 2, AC_K128_TL128, CDE_wN1_TM128, 4),
// WAVELET_INST(256, 128, 128,  64, 128, 2, 2, AC_K128_TL256, CDE_wN1_TM128, 4),
// (MXdl=1,NXdl=2) → wM=4,wN=1,TMath=256 | Csv=4
// WAVELET_INST( 64, 256, 128,  64, 128, 1, 2, AC_K128_TL64,  CDE_wN1_TM256, 4),
WAVELET_INST(128, 256, 128,  64, 128, 1, 2, AC_K128_TL128, CDE_wN1_TM256, 4),
// WAVELET_INST(256, 256, 128,  64, 128, 1, 2, AC_K128_TL256, CDE_wN1_TM256, 4),
// (MXdl=4,NXdl=1) → wM=1,wN=2,TMath=128 | Csv=8
// WAVELET_INST( 64, 128, 128,  64, 128, 4, 1, AC_K128_TL64,  CDE_wN2_TM128, 8),
// WAVELET_INST(128, 128, 128,  64, 128, 4, 1, AC_K128_TL128, CDE_wN2_TM128, 8),
// WAVELET_INST(256, 128, 128,  64, 128, 4, 1, AC_K128_TL256, CDE_wN2_TM128, 8),
// (MXdl=4,NXdl=2) → wM=1,wN=1,TMath=64  | Csv=4
// WAVELET_INST( 64,  64, 128,  64, 128, 4, 2, AC_K128_TL64,  CDE_wN1_TM64,  4),
// WAVELET_INST(128,  64, 128,  64, 128, 4, 2, AC_K128_TL128, CDE_wN1_TM64,  4),
// WAVELET_INST(256,  64, 128,  64, 128, 4, 2, AC_K128_TL256, CDE_wN1_TM64,  4),

// ============================================================
// M=64, N=64, K=128  (AK0=16, LDS≈24KB)
// (MXdl=1,NXdl=1) → wM=2,wN=2,TMath=256 | Csv=8
// WAVELET_INST( 64, 256,  64,  64, 128, 1, 1, AC_K128_TL64,  CDE_wN2_TM256, 8),
WAVELET_INST(128, 256,  64,  64, 128, 1, 1, AC_K128_TL128, CDE_wN2_TM256, 8),
WAVELET_INST(256, 256,  64,  64, 128, 1, 1, AC_K128_TL256, CDE_wN2_TM256, 8),
// (MXdl=1,NXdl=2) → wM=2,wN=1,TMath=128 | Csv=4
// WAVELET_INST( 64, 128,  64,  64, 128, 1, 2, AC_K128_TL64,  CDE_wN1_TM128, 4),
// WAVELET_INST(128, 128,  64,  64, 128, 1, 2, AC_K128_TL128, CDE_wN1_TM128, 4),
// WAVELET_INST(256, 128,  64,  64, 128, 1, 2, AC_K128_TL256, CDE_wN1_TM128, 4),
// (MXdl=2,NXdl=1) → wM=1,wN=2,TMath=128 | Csv=8
// WAVELET_INST( 64, 128,  64,  64, 128, 2, 1, AC_K128_TL64,  CDE_wN2_TM128, 8),
// WAVELET_INST(128, 128,  64,  64, 128, 2, 1, AC_K128_TL128, CDE_wN2_TM128, 8),
// WAVELET_INST(256, 128,  64,  64, 128, 2, 1, AC_K128_TL256, CDE_wN2_TM128, 8),
// (MXdl=2,NXdl=2) → wM=1,wN=1,TMath=64  | Csv=4
// WAVELET_INST( 64,  64,  64,  64, 128, 2, 2, AC_K128_TL64,  CDE_wN1_TM64,  4),
// WAVELET_INST(128,  64,  64,  64, 128, 2, 2, AC_K128_TL128, CDE_wN1_TM64,  4),
// WAVELET_INST(256,  64,  64,  64, 128, 2, 2, AC_K128_TL256, CDE_wN1_TM64,  4),

// ============================================================
// M=64, N=128, K=128  (AK0=16, A+B LDS≈50KB, C_shuffle=16KB — fits in 64KB)
// (MXdl=1,NXdl=2) → wM=2,wN=2,TMath=256 | Csv=8
// WAVELET_INST( 64, 256,  64, 128, 128, 1, 2, AC_K128_TL64,  CDE_wN2_TM256, 8),
// WAVELET_INST(128, 256,  64, 128, 128, 1, 2, AC_K128_TL128, CDE_wN2_TM256, 8),
// WAVELET_INST(256, 256,  64, 128, 128, 1, 2, AC_K128_TL256, CDE_wN2_TM256, 8),
// (MXdl=1,NXdl=4) → wM=2,wN=1,TMath=128 | Csv=4
// WAVELET_INST( 64, 128,  64, 128, 128, 1, 4, AC_K128_TL64,  CDE_wN1_TM128, 4),
// WAVELET_INST(128, 128,  64, 128, 128, 1, 4, AC_K128_TL128, CDE_wN1_TM128, 4),
// WAVELET_INST(256, 128,  64, 128, 128, 1, 4, AC_K128_TL256, CDE_wN1_TM128, 4),
// (MXdl=2,NXdl=1) → wM=1,wN=4,TMath=256 | Csv=8
// WAVELET_INST( 64, 256,  64, 128, 128, 2, 1, AC_K128_TL64,  CDE_wN4_TM256, 8),
WAVELET_INST(128, 256,  64, 128, 128, 2, 1, AC_K128_TL128, CDE_wN4_TM256, 8),
WAVELET_INST(256, 256,  64, 128, 128, 2, 1, AC_K128_TL256, CDE_wN4_TM256, 8)
// (MXdl=2,NXdl=2) → wM=1,wN=2,TMath=128 | Csv=8
// WAVELET_INST( 64, 128,  64, 128, 128, 2, 2, AC_K128_TL64,  CDE_wN2_TM128, 8),
// WAVELET_INST(128, 128,  64, 128, 128, 2, 2, AC_K128_TL128, CDE_wN2_TM128, 8),
// WAVELET_INST(256, 128,  64, 128, 128, 2, 2, AC_K128_TL256, CDE_wN2_TM128, 8),
// (MXdl=2,NXdl=4) → wM=1,wN=1,TMath=64  | Csv=4
// WAVELET_INST( 64,  64,  64, 128, 128, 2, 4, AC_K128_TL64,  CDE_wN1_TM64,  4),
// WAVELET_INST(128,  64,  64, 128, 128, 2, 4, AC_K128_TL128, CDE_wN1_TM64,  4),
// WAVELET_INST(256,  64,  64, 128, 128, 2, 4, AC_K128_TL256, CDE_wN1_TM64,  4)
    // clang-format on
    >;

#undef WAVELET_INST

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
