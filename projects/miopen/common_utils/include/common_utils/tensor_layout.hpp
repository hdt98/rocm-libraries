// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef GUARD_COMMON_UTILS_TENSOR_LAYOUT_HPP
#define GUARD_COMMON_UTILS_TENSOR_LAYOUT_HPP

// HOST-SIDE entry point for tensor layout utilities.
//
// The canonical implementation lives at src/include/miopen/tensor_layout.hpp.
// This wrapper centralizes the include path so driver and test code can use
// tensor_layout_to_strides and tensor_layout_get_default without directly
// including internal MIOpen headers.

#include "../../src/include/miopen/tensor_layout.hpp"

#endif // GUARD_COMMON_UTILS_TENSOR_LAYOUT_HPP
