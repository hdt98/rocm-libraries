// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/** @file value_based.hpp
 *  @brief Convenience include-all for value-based coordinate transform graphs.
 *
 *  Value-based transform graphs coexist with the type-based descriptor system.
 *  Both are production-ready for their respective use cases:
 *
 *  - **Value-based** (this module): Targets reduced compile time for static
 *    (fully constexpr) descriptors. All transform parameters are known at
 *    compile time and encoded as structural NTTP values. Ideal for LDS tile
 *    descriptors and other compile-time-known layouts.
 *
 *  - **Type-based** (tensor_adaptor.hpp / tensor_descriptor.hpp): Remains for
 *    runtime-dimension descriptors where M, N, K are kernel arguments.
 *    Supports mixed compile-time / runtime dimension lengths.
 */

#pragma once

#include "ck_tile/core/tensor/value_based/transform_type.hpp"
#include "ck_tile/core/tensor/value_based/magic_division.hpp"
#include "ck_tile/core/tensor/value_based/coordinate_transform.hpp"
#include "ck_tile/core/tensor/value_based/transform_impl.hpp"
#include "ck_tile/core/tensor/value_based/transform_graph.hpp"
#include "ck_tile/core/tensor/value_based/make_transform.hpp"
#include "ck_tile/core/tensor/value_based/make_graph.hpp"
