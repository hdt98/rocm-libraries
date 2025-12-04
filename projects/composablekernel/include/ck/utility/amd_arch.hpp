// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"

namespace ck {

// Architecture tags
struct gfx9_t
{
};
struct gfx950_t
{
};
struct gfx103_t
{
};
struct gfx11_t
{
};
struct gfx12_t
{
};
struct gfx120_t
{
};
struct gfx125_t
{
};
struct gfx_invalid_t
{
};

static constexpr auto get_device_arch()
{
#if defined(__gfx950__)
    return gfx950_t{};
#elif defined(__gfx9__)
    return gfx9_t{};
#elif defined(__gfx10__)
    return gfx10_t{};
#elif defined(__gfx11__)
    return gfx11_t{};
#elif defined(__gfx125__)
    return gfx125_t{};
#elif defined(__gfx12__)
    return gfx120_t{};
#else
    return gfx_invalid_t{};
#endif
}

} // namespace ck
