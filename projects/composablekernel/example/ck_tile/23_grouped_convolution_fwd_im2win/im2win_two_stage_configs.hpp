// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// ═══════════════════════════════════════════════════════════════════════
// im2win_two_stage_configs.hpp
//
// Config structs for the two-stage im2win forward convolution:
//
//   Stage 1 — ImageToIm2win kernel:
//     Reads I[G, N, C, Hi, Wi] and writes I'[G, N, C, Ho, Wi_pad, Y].
//     Config controls the (M=N×Ho×Wi_pad) × (K=C×Y) tile size.
//
//   Stage 2 — Standard forward conv kernel:
//     Reads I' (as if it were a normal input with layout GNCHW_Im2win)
//     and produces O[G, N, K, Ho, Wo] using the standard im2col+GEMM.
//     Re-uses configs from im2win_conv_configs.hpp for the GEMM stage.
//
// The ACTIVE_IM2WIN_TRANSFORM_CONFIG macro selects the Stage-1 tile shape.
// ═══════════════════════════════════════════════════════════════════════

#include "ck_tile/core.hpp"
#include "ck_tile/ops/image_to_im2win.hpp"

// ── Stage-1 block shapes ─────────────────────────────────────────────────────

// Config 0: 256×32 tile (default — good for most problems)
//   M = 256 spatial rows per block, K = 32 channel×filter-row elements per block
//   Block = 64 threads (1 warp × 64 threads/warp), each thread handles 4M×4K = 16 elems
using Im2winTransformShape0 = ck_tile::TileImageToIm2winShape<
    ck_tile::sequence<4, 4>,    // ThreadTile: 4M × 4K per thread
    ck_tile::sequence<64, 4>,   // WarpTile:  64M × 4K per warp (64 threads/warp)
    ck_tile::sequence<256, 32>  // BlockTile: 256M × 32K per block
>;

// Config 1: 128×32 tile
using Im2winTransformShape1 = ck_tile::TileImageToIm2winShape<
    ck_tile::sequence<4, 4>,
    ck_tile::sequence<64, 4>,
    ck_tile::sequence<128, 32>
>;

// Config 2: 256×16 tile — narrow K (good when C×Y is small, e.g. C=4, Y=3 → K=12)
using Im2winTransformShape2 = ck_tile::TileImageToIm2winShape<
    ck_tile::sequence<4, 4>,
    ck_tile::sequence<64, 4>,
    ck_tile::sequence<256, 16>
>;

// Config 3: 128×16 tile
using Im2winTransformShape3 = ck_tile::TileImageToIm2winShape<
    ck_tile::sequence<4, 4>,
    ck_tile::sequence<64, 4>,
    ck_tile::sequence<128, 16>
>;

// ── Problem type alias ────────────────────────────────────────────────────────

template <typename DataType,
          typename BlockShape,
          int AlignIn  = 4,
          int AlignOut = 4>
using Im2winTransformProblem = ck_tile::BlockImageToIm2winProblem<
    DataType,   // InDataType
    DataType,   // OutDataType (same — no type cast)
    BlockShape,
    AlignIn,
    AlignOut>;

// ── Stage-1 kernel type alias ─────────────────────────────────────────────────

template <typename DataType, typename BlockShape>
using Im2winTransformKernel =
    ck_tile::ImageToIm2win<Im2winTransformProblem<DataType, BlockShape>>;

// ── Active config (compile-time selection) ────────────────────────────────────

#ifndef ACTIVE_IM2WIN_TRANSFORM_CONFIG
#define ACTIVE_IM2WIN_TRANSFORM_CONFIG 0
#endif

#if ACTIVE_IM2WIN_TRANSFORM_CONFIG == 0
template <typename DataType>
using ActiveIm2winTransformKernel = Im2winTransformKernel<DataType, Im2winTransformShape0>;
#elif ACTIVE_IM2WIN_TRANSFORM_CONFIG == 1
template <typename DataType>
using ActiveIm2winTransformKernel = Im2winTransformKernel<DataType, Im2winTransformShape1>;
#elif ACTIVE_IM2WIN_TRANSFORM_CONFIG == 2
template <typename DataType>
using ActiveIm2winTransformKernel = Im2winTransformKernel<DataType, Im2winTransformShape2>;
#elif ACTIVE_IM2WIN_TRANSFORM_CONFIG == 3
template <typename DataType>
using ActiveIm2winTransformKernel = Im2winTransformKernel<DataType, Im2winTransformShape3>;
#else
#error "Unknown ACTIVE_IM2WIN_TRANSFORM_CONFIG — valid range: 0..3"
#endif
