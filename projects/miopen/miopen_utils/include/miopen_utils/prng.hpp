// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Host-side PRNG utilities for dropout emulation.
// Canonical kernel-side definition: src/kernels/miopen_rocrand.hpp
#pragma once

#include <cstdint>
#include <rocrand/rocrand_xorwow.h>

namespace prng {

// based on splitmix64
inline constexpr uint64_t hash(uint64_t x)
{
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    x = x ^ (x >> 31);
    return x;
}

// borrowed from <rocrand/rocrand_uniform.h>
inline constexpr float uniform_distribution(unsigned int v)
{
    constexpr float rocrand_2pow32_inv = 2.3283064e-10f;
    return rocrand_2pow32_inv + (static_cast<float>(v) * rocrand_2pow32_inv);
}

inline constexpr float xorwow_uniform(rocrand_device::xorwow_engine* state)
{
    return uniform_distribution(state->next());
}

} // namespace prng
