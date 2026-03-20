// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Memory layout enum for matrix operations.
// Host-safe — no CK Tile dependency.

#pragma once

namespace rocm_ck {

/// Memory layout for matrix operands.
/// Row = row-major (stride is number of columns).
/// Col = column-major (stride is number of rows).
enum class Layout
{
    Row,
    Col
};

} // namespace rocm_ck
