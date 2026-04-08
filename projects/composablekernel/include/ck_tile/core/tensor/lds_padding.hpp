// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/integer.hpp"

namespace ck_tile {

// buffer_load_dwordx3 to LDS uses a fixed 16-byte per-thread stride,
// padding each 12-byte element to 16 bytes in LDS.
template <typename T>
CK_TILE_HOST_DEVICE constexpr index_t lds_padded_sizeof()
{
    return (sizeof(T) == 12) ? 16 : sizeof(T);
}

// Typed wrapper whose sizeof() == lds_padded_sizeof<T>().
// Using this for pointer arithmetic instead of raw char* keeps LLVM's
// typed GEP intact, preserving alias analysis and load coalescing.
template <typename T>
struct alignas(lds_padded_sizeof<T>()) lds_padded_element
{
    static_assert(sizeof(T) <= lds_padded_sizeof<T>());
    T value;
};

} // namespace ck_tile
