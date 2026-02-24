// SPDX-License-Identifier: MIT
// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.

#pragma once

#include <ck_tile/builder/types.hpp>

namespace miopen {
namespace conv {
namespace ck_builder {
namespace factories {
namespace grouped_conv_bwd_weight {

namespace ckb = ck_tile::builder;

// Data types
[[maybe_unused]] constexpr auto BF16 = ckb::DataType::BF16;
[[maybe_unused]] constexpr auto F16  = ckb::DataType::FP16;
[[maybe_unused]] constexpr auto F32  = ckb::DataType::FP32;
// TODO: TF32 compute type is not yet supported in ckb::DataType enum
// [[maybe_unused]] constexpr auto TF32 = ckb::DataType::TF32;
[[maybe_unused]] constexpr auto I8   = ckb::DataType::I8;
[[maybe_unused]] constexpr auto I32  = ckb::DataType::I32;
[[maybe_unused]] constexpr auto F8   = ckb::DataType::FP8;
[[maybe_unused]] constexpr auto BF8  = ckb::DataType::BF8;

// Elementwise operations
[[maybe_unused]] constexpr auto PassThrough = ckb::ElementwiseOperation::PASS_THROUGH;
[[maybe_unused]] constexpr auto Bilinear    = ckb::ElementwiseOperation::BILINEAR;
[[maybe_unused]] constexpr auto Scale       = ckb::ElementwiseOperation::SCALE;

// GEMM specializations
[[maybe_unused]] constexpr auto GemmDefault    = ckb::GemmSpecialization::Default;
[[maybe_unused]] constexpr auto GemmMNKPadding = ckb::GemmSpecialization::MNKPadding;

// Pipeline schedulers
[[maybe_unused]] constexpr auto Intrawave = ckb::PipelineScheduler::INTRAWAVE;
[[maybe_unused]] constexpr auto Interwave = ckb::PipelineScheduler::INTERWAVE;

// Pipeline versions
[[maybe_unused]] constexpr auto PipeV1 = ckb::PipelineVersion::V1;
[[maybe_unused]] constexpr auto PipeV2 = ckb::PipelineVersion::V2;
[[maybe_unused]] constexpr auto PipeV5 = ckb::PipelineVersion::V5;

// Backward weight convolution specializations
[[maybe_unused]] constexpr auto ConvBwdWeightDefault =
    ckb::ConvSpecialization::DEFAULT;
[[maybe_unused]] constexpr auto ConvBwdWeightFilter1x1Stride1Pad0 =
    ckb::ConvSpecialization::FILTER_1X1_STRIDE1_PAD0;

} // namespace grouped_conv_bwd_weight
} // namespace factories
} // namespace ck_builder
} // namespace conv
} // namespace miopen
