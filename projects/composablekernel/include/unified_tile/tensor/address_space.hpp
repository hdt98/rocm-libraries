// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "../config.hpp"

namespace unified_tile {

/// @brief Unified address space enum for tensor views
enum class address_space
{
    global, ///< Global (device) memory
    shared, ///< Shared (LDS) memory
    vgpr    ///< Register (VGPR) memory
};

} // namespace unified_tile
