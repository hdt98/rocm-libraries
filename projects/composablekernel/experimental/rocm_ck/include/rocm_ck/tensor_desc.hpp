// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Role: types — ResolvedTensor struct. No runtime, no CK deps.
//
// Common vocabulary type for resolved tensor metadata.
// Captures dtype, name, rank, and layout — used as the internal intermediate
// between user-facing Signature structs and makeSpec().

#pragma once

#include <rocm_ck/datatype_utils.hpp>
#include <rocm_ck/layout.hpp>

#include <optional>
#include <string_view>

namespace rocm_ck {

/// Resolved quantization info carried from Signature through resolve().
/// Present on tensors that have .quantize set (e.g., INT4 weight tensors).
struct ResolvedQuantization
{
    std::string_view scale_name; // name of the scale tensor
    DataType scale_dtype;        // element type of the scale tensor
    int group_size;              // elements per quantization group
};

/// Resolved metadata for a single tensor operand.
/// Plain aggregate — no constructors, no methods.
/// std::string_view makes it non-structural (can't be NTTP), but that's fine:
/// ResolvedTensor is an internal intermediate, never used as a template parameter.
struct ResolvedTensor
{
    std::string_view name;                                      // "A", "bias", "query"
    DataType dtype;                                             // element type
    int rank                                     = 2;           // dimensions
    Layout layout                                = Layout::Row; // default: row-major rank-2
    std::optional<ResolvedQuantization> quantize = std::nullopt;
};

} // namespace rocm_ck
