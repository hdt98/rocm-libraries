// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Maps DataType enum values to CK Tile C++ numeric types.
// Device-side only — requires CK Tile headers.

#pragma once

#include <rocm_ck/datatype_utils.hpp>

#include "ck_tile/core.hpp"

namespace rocm_ck {

/// Maps a DataType enum value to the corresponding CK Tile numeric type.
/// Primary template is intentionally undefined — only valid specializations compile.
template <DataType>
struct CkTypeMap;

template <>
struct CkTypeMap<DataType::FP32>
{
    using type = float;
};
template <>
struct CkTypeMap<DataType::FP16>
{
    using type = ck_tile::half_t;
};
template <>
struct CkTypeMap<DataType::BF16>
{
    using type = ck_tile::bf16_t;
};
template <>
struct CkTypeMap<DataType::FP8>
{
    using type = ck_tile::fp8_t;
};

} // namespace rocm_ck
