// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

/**
 * @file config.hpp
 * @brief Backend selection and configuration for unified tile interface
 *
 * This file defines the compile-time backend selection mechanism.
 * Users select the backend via CMake flags:
 *   -DUNIFIED_TILE_BACKEND_CK_TILE  -> Use CK_Tile backend
 *   -DUNIFIED_TILE_BACKEND_MINT     -> Use MINT backend
 */

namespace unified_tile {

/// @brief Available backend types
enum class backend_type {
    ck_tile,  ///< CK_Tile backend (tile_window based)
    mint      ///< MINT backend (polymorpher based)
};

// ============================================================================
// Backend Selection (Compile-time)
// ============================================================================

#if defined(UNIFIED_TILE_BACKEND_CK_TILE) && defined(UNIFIED_TILE_BACKEND_MINT)
    #error "Cannot define both UNIFIED_TILE_BACKEND_CK_TILE and UNIFIED_TILE_BACKEND_MINT"
#endif

#if !defined(UNIFIED_TILE_BACKEND_CK_TILE) && !defined(UNIFIED_TILE_BACKEND_MINT)
    // Default to CK_Tile if no backend specified
    #define UNIFIED_TILE_BACKEND_CK_TILE
    #warning "No backend specified, defaulting to CK_Tile (use -DUNIFIED_TILE_BACKEND_CK_TILE or -DUNIFIED_TILE_BACKEND_MINT)"
#endif

#ifdef UNIFIED_TILE_BACKEND_CK_TILE
    /// @brief Active backend at compile time
    inline constexpr backend_type active_backend = backend_type::ck_tile;
    #define UNIFIED_TILE_BACKEND_IS_CK_TILE 1
    #define UNIFIED_TILE_BACKEND_IS_MINT 0
#else
    /// @brief Active backend at compile time
    inline constexpr backend_type active_backend = backend_type::mint;
    #define UNIFIED_TILE_BACKEND_IS_CK_TILE 0
    #define UNIFIED_TILE_BACKEND_IS_MINT 1
#endif

// ============================================================================
// Configuration Constants
// ============================================================================

/// @brief Enable verbose debug output
#ifndef UNIFIED_TILE_DEBUG
    #define UNIFIED_TILE_DEBUG 0
#endif

/// @brief Enable runtime backend verification
#ifndef UNIFIED_TILE_RUNTIME_CHECKS
    #define UNIFIED_TILE_RUNTIME_CHECKS 0
#endif

// ============================================================================
// Host/Device Annotation Macros
// ============================================================================

#ifdef UNIFIED_TILE_BACKEND_CK_TILE
    #define UNIFIED_TILE_HOST_DEVICE CK_TILE_HOST_DEVICE
    #define UNIFIED_TILE_DEVICE CK_TILE_DEVICE
    #define UNIFIED_TILE_HOST CK_TILE_HOST
#else
    #define UNIFIED_TILE_HOST_DEVICE MINT_HOST_DEVICE
    #define UNIFIED_TILE_DEVICE MINT_DEVICE
    #define UNIFIED_TILE_HOST MINT_HOST
#endif

// ============================================================================
// Utility Macros
// ============================================================================

/// @brief Execute code only for CK_Tile backend
#define IF_CK_TILE(...) \
    if constexpr (unified_tile::active_backend == unified_tile::backend_type::ck_tile) { \
        __VA_ARGS__ \
    }

/// @brief Execute code only for MINT backend
#define IF_MINT(...) \
    if constexpr (unified_tile::active_backend == unified_tile::backend_type::mint) { \
        __VA_ARGS__ \
    }

/// @brief Compile-time backend dispatch
#define BACKEND_DISPATCH(ck_tile_expr, mint_expr) \
    [&]() { \
        if constexpr (unified_tile::active_backend == unified_tile::backend_type::ck_tile) { \
            return ck_tile_expr; \
        } else { \
            return mint_expr; \
        } \
    }()

} // namespace unified_tile
