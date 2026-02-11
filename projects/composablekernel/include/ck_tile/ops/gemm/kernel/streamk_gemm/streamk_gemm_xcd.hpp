// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include "ck_tile/core/arch/arch.hpp"
namespace ck_tile {

template <typename CompilerTarget, typename Enabler = void>
struct NumXCD
{
    static constexpr int num_xcds = 1;
};

/**template <typename CompilerTarget>
struct NumXCD<CompilerTarget,
                        core::arch::enable_if_target_id_t<CompilerTarget,
                                                          core::arch::amdgcn_target_id::GFX940>>
{
    static constexpr int num_xcds = 6;
};**/

template <typename CompilerTarget>
struct NumXCD<
    CompilerTarget,
    core::arch::enable_if_target_id_t<CompilerTarget, core::arch::amdgcn_target_id::GFX942>>
{
    static constexpr int num_xcds = 8;
};

} // namespace ck_tile
