// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"

namespace ck_tile {

// ═══════════════════════════════════════════════════════════════════════
// ImageToIm2win
//
// Stage-1 kernel of the two-stage im2win convolution pipeline.
//
// Reads from (two layouts supported):
//   GNCHW: I[G, N, C, Hi, Wi]   — channels-first
//   NHWGC: I[N, Hi, Wi, G, C]   — channels-last
//
// Writes to: I'[G, N, C, Ho, Wi_pad, Y] (packed, Y innermost)
//
// The transformation is defined by (from im2win_transform.md):
//
//   I'[n, c, ho, wi_pad, y] = I_padded[n, c,  ho·Sy + y·Dy,  wi_pad]
//                            = I[n, c,  ho·Sy + y·Dy - LPH,  wi_pad - LPW]
//
// where:
//   Wi_pad = Wi + LPW + RPW
//   out-of-bounds reads (padding zone) yield zero
//
// This applies the HEIGHT windowing only (im2win I' definition).
// The X (width) sliding window is applied lazily by the follow-on GEMM
// descriptor in Stage 2.
//
// Output I' layout (identical regardless of input layout):
//   Packed [M=N×Ho×Wi_pad, K=C×Y] with Y innermost in K.
//   Physical offsets: (n,c,ho,wi_pad,y) → (n*Ho*Wi_pad + ho*Wi_pad + wi_pad)*(C*Y) + c*Y + y
//
// Grid layout:
//   blockIdx.x → tile over M = N × Ho × Wi_pad
//   blockIdx.y → tile over K = C × Y
//   blockIdx.z → group index g ∈ [0, G)
// ═══════════════════════════════════════════════════════════════════════

template <typename Problem_>
struct ImageToIm2win
{
    static constexpr auto I0 = number<0>{};
    static constexpr auto I1 = number<1>{};
    static constexpr auto I2 = number<2>{};
    static constexpr auto I3 = number<3>{};
    static constexpr auto I4 = number<4>{};

    using Problem = remove_cvref_t<Problem_>;

    using InDataType  = remove_cvref_t<typename Problem::InDataType>;
    using OutDataType = remove_cvref_t<typename Problem::OutDataType>;

    static constexpr index_t kMPerBlock = Problem::BlockShape::kMPerBlock;
    static constexpr index_t kKPerBlock = Problem::BlockShape::kKPerBlock;
    static constexpr index_t kBlockSize = Problem::BlockShape::kBlockSize;

    static constexpr index_t AlignIn  = Problem::AlignIn;
    static constexpr index_t AlignOut = Problem::AlignOut;

    // ── Runtime kernel arguments ──────────────────────────────────────────
    struct Kargs
    {
        const void* p_in;   ///< I — GNCHW: [G,N,C,Hi,Wi] or NHWGC: [N,Hi,Wi,G,C]
        void*       p_out;  ///< I'[G, N, C, Ho, Wi_pad, Y] (packed, Y innermost)

        index_t G;
        index_t N;
        index_t C;
        index_t Hi;
        index_t Wi;
        index_t Ho;   ///< = (Hi + LPH + RPH - (Y-1)*Dy - 1) / Sy + 1
        index_t Y;    ///< filter height
        index_t Wi_pad; ///< = Wi + LPW + RPW

        index_t conv_stride_h;
        index_t conv_stride_w;   // not used in Stage 1 (X not applied here)
        index_t conv_dilation_h;
        index_t conv_dilation_w; // not used in Stage 1
        index_t left_pad_h;
        index_t left_pad_w;
        index_t right_pad_h;
        index_t right_pad_w;

        // Layout flag at end to avoid struct padding issues between pointers and integers.
        index_t is_nhwgc; ///< 1 = NHWGC input, 0 = GNCHW input (stored as index_t)
    };

    // ── Factory ───────────────────────────────────────────────────────────
    CK_TILE_HOST static constexpr Kargs
    MakeKargs(const void* p_in,
              void*       p_out,
              index_t G, index_t N, index_t C,
              index_t Hi, index_t Wi, index_t Ho, index_t Y,
              index_t conv_stride_h,   index_t conv_stride_w,
              index_t conv_dilation_h, index_t conv_dilation_w,
              index_t left_pad_h,  index_t left_pad_w,
              index_t right_pad_h, index_t right_pad_w,
              bool    is_nhwgc = false)
    {
        const index_t Wi_pad = Wi + left_pad_w + right_pad_w;
        return Kargs{p_in, p_out,
                     G, N, C, Hi, Wi, Ho, Y, Wi_pad,
                     conv_stride_h, conv_stride_w,
                     conv_dilation_h, conv_dilation_w,
                     left_pad_h, left_pad_w,
                     right_pad_h, right_pad_w,
                     is_nhwgc ? 1 : 0};
    }

    // ── Grid / block sizing ───────────────────────────────────────────────
    /// M = N × Ho × Wi_pad,  K = C × Y,  batch = G
    CK_TILE_HOST static constexpr auto GridSize(const Kargs& kargs)
    {
        const index_t M = kargs.N * kargs.Ho * kargs.Wi_pad;
        const index_t K = kargs.C * kargs.Y;
        return dim3(integer_divide_ceil(M, kMPerBlock),
                    integer_divide_ceil(K, kKPerBlock),
                    kargs.G);
    }

    CK_TILE_HOST static constexpr auto BlockSize()
    {
        return Problem::BlockShape::kBlockSize;
    }

    // ── Tile distribution (same pattern as ImageToColumn) ─────────────────
    CK_TILE_DEVICE static constexpr auto MakeBlockTileDistribution()
    {
        using P = typename Problem::BlockShape;
        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<1>,
                tuple<sequence<P::kMWarpPerBlock, P::kMThreadPerWarp, P::kMPerThread>,
                      sequence<P::kKWarpPerBlock, P::kKThreadPerWarp, P::kKPerThread>>,
                tuple<sequence<1, 2>, sequence<1, 2>>,
                tuple<sequence<0, 0>, sequence<1, 1>>,
                sequence<1, 2>,
                sequence<2, 2>>{});
    }

    // ── Source descriptor: GNCHW path ─────────────────────────────────────
    // I[N,C,Hi,Wi] → I'[N,C,Ho,Wi_pad,Y] → [N×Ho×Wi_pad, C×Y]
    // K ordering: flat_k = c·Y + y  (C outermost, Y innermost)
    CK_TILE_DEVICE auto MakeSourceDescGnchw(const Kargs& kargs) const
    {
        // Physical I[N, C, Hi, Wi] (packed GNCHW, per group)
        const index_t NStride  = kargs.C * kargs.Hi * kargs.Wi;
        const index_t CStride  = kargs.Hi * kargs.Wi;
        const index_t HiStride = kargs.Wi;
        const index_t WiStride = 1;

        const auto in_desc = make_naive_tensor_descriptor(
            make_tuple(kargs.N, kargs.C, kargs.Hi, kargs.Wi),
            make_tuple(NStride, CStride, HiStride, WiStride),
            number<AlignIn>{}, I1);

        // Pad Hi and Wi
        const auto in_pad_desc = transform_tensor_descriptor(
            in_desc,
            make_tuple(make_pass_through_transform(kargs.N),
                       make_pass_through_transform(kargs.C),
                       make_pad_transform(kargs.Hi, kargs.left_pad_h, kargs.right_pad_h),
                       make_pad_transform(kargs.Wi, kargs.left_pad_w, kargs.right_pad_w)),
            make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}),
            make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}));

        // Height windowing: embed Hi_pad → (Ho, Y); Wi_pad and C pass through
        // Dims after: [N=0, C=1, Ho=2, Wi_pad=3, Y=4]
        const auto in_prime_desc = transform_tensor_descriptor(
            in_pad_desc,
            make_tuple(make_pass_through_transform(kargs.N),
                       make_pass_through_transform(kargs.C),
                       make_embed_transform(make_tuple(kargs.Ho, kargs.Y),
                                            make_tuple(kargs.conv_stride_h,
                                                       kargs.conv_dilation_h)),
                       make_pass_through_transform(kargs.Wi_pad)),
            make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}),
            make_tuple(sequence<0>{}, sequence<1>{}, sequence<2, 4>{}, sequence<3>{}));

        // Merge to 2D: M=N×Ho×Wi_pad, K=C×Y (C outermost, Y innermost)
        return transform_tensor_descriptor(
            in_prime_desc,
            make_tuple(
                make_merge_transform(make_tuple(kargs.N, kargs.Ho, kargs.Wi_pad)),
                make_merge_transform(make_tuple(kargs.C, kargs.Y))),
            make_tuple(sequence<0, 2, 3>{}, sequence<1, 4>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));
    }

    // ── Source descriptor: NHWGC path ─────────────────────────────────────
    // I[N,Hi,Wi,G,C] → I'[N,Ho,Wi_pad,C,Y] → [N×Ho×Wi_pad, C×Y]
    // K ordering: flat_k = c·Y + y  (C outermost, Y innermost) — same as GNCHW
    //
    // Advantage: C is unit-stride in NHWGC → vectorised reads along C
    // at each (n, hi, wi) position (C=256 contiguous elements per pixel).
    //
    // For G=1 the group pointer offset is already subtracted before call,
    // so this descriptor operates on I[N, Hi, Wi, C] (G stripped).
    CK_TILE_DEVICE auto MakeSourceDescNhwgc(const Kargs& kargs) const
    {
        // Physical I[N, Hi, Wi, G, C] — pointer base shifted by g*C for group g.
        // The full NHWGC strides must include G (all groups share N/H/W dimensions):
        //   NStride  = Hi * Wi * G * C
        //   HiStride = Wi * G * C
        //   WiStride = G * C   (NOT just C — adjacent pixels are G*C apart in memory)
        //   CStride  = 1      (C innermost → vectorised reads of C channels per pixel)
        const index_t NStride  = kargs.Hi * kargs.Wi * kargs.G * kargs.C;
        const index_t HiStride = kargs.Wi * kargs.G * kargs.C;
        const index_t WiStride = kargs.G * kargs.C;
        const index_t CStride  = 1; // C unit-stride → vectorised reads

        const auto in_desc = make_naive_tensor_descriptor(
            make_tuple(kargs.N, kargs.Hi, kargs.Wi, kargs.C),
            make_tuple(NStride, HiStride, WiStride, CStride),
            number<AlignIn>{}, I1);

        // Pad Hi and Wi
        const auto in_pad_desc = transform_tensor_descriptor(
            in_desc,
            make_tuple(make_pass_through_transform(kargs.N),
                       make_pad_transform(kargs.Hi, kargs.left_pad_h, kargs.right_pad_h),
                       make_pad_transform(kargs.Wi, kargs.left_pad_w, kargs.right_pad_w),
                       make_pass_through_transform(kargs.C)),
            make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}),
            make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}));
        // Shape: [N, Hi_pad, Wi_pad, C]

        // Height windowing: embed Hi_pad → (Ho, Y)
        // Dims after: [N=0, Ho=1, Y=2, Wi_pad=3, C=4]
        const auto in_prime_desc = transform_tensor_descriptor(
            in_pad_desc,
            make_tuple(make_pass_through_transform(kargs.N),
                       make_embed_transform(make_tuple(kargs.Ho, kargs.Y),
                                            make_tuple(kargs.conv_stride_h,
                                                       kargs.conv_dilation_h)),
                       make_pass_through_transform(kargs.Wi_pad),
                       make_pass_through_transform(kargs.C)),
            make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}),
            make_tuple(sequence<0>{}, sequence<1, 2>{}, sequence<3>{}, sequence<4>{}));
        // Dims: [N=0, Ho=1, Y=2, Wi_pad=3, C=4]

        // Merge to 2D: M=N×Ho×Wi_pad, K=C×Y (C outermost, Y innermost)
        //   M: dims [N=0, Ho=1, Wi_pad=3]
        //   K: dims [C=4, Y=2]
        return transform_tensor_descriptor(
            in_prime_desc,
            make_tuple(
                make_merge_transform(make_tuple(kargs.N, kargs.Ho, kargs.Wi_pad)),
                make_merge_transform(make_tuple(kargs.C, kargs.Y))),
            make_tuple(sequence<0, 1, 3>{}, sequence<4, 2>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));
    }

    // ── Destination descriptor: packed I'[N,C,Ho,Wi_pad,Y] ────────────────
    //
    // I' is stored with Y as the innermost dimension (stride=1),
    // enabling vectorised writes along Y.
    //
    // Strides:
    //   stride(N)      = C · Ho · Wi_pad · Y
    //   stride(C)      = Ho · Wi_pad · Y
    //   stride(Ho)     = Wi_pad · Y
    //   stride(Wi_pad) = Y
    //   stride(Y)      = 1
    //
    // Merged to [N×Ho×Wi_pad, C×Y] for the tile window.
    CK_TILE_DEVICE auto MakeDestDesc(const Kargs& kargs) const
    {
        const index_t M = kargs.N * kargs.Ho * kargs.Wi_pad;
        const index_t K = kargs.C * kargs.Y;

        // Packed row-major I'[N×Ho×Wi_pad, C×Y] with Y innermost → stride(K)=1
        const auto out_m_k_desc = make_naive_tensor_view<address_space_enum::global>(
            static_cast<OutDataType*>(nullptr), // placeholder — offset added in operator()
            make_tuple(M, K),
            make_tuple(K, 1),
            number<AlignOut>{},
            I1);
        return out_m_k_desc;
    }

    // ── Main kernel ───────────────────────────────────────────────────────
    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        const index_t iM     = amd_wave_read_first_lane(
            static_cast<index_t>(blockIdx.x) * kMPerBlock);
        const index_t iK     = amd_wave_read_first_lane(
            static_cast<index_t>(blockIdx.y) * kKPerBlock);
        const index_t iBatch = amd_wave_read_first_lane(
            static_cast<index_t>(blockIdx.z));

        // Per-group pointer offsets
        // GNCHW: I[G,N,C,Hi,Wi] — group stride = N*C*Hi*Wi
        // NHWGC: I[N,Hi,Wi,G,C] — group stride = C  (G is the fast-varying G dim)
        const long_index_t in_offset = (kargs.is_nhwgc != 0)
            ? static_cast<long_index_t>(iBatch) * kargs.C    // NHWGC: g*C
            : static_cast<long_index_t>(iBatch) * kargs.N * kargs.C * kargs.Hi * kargs.Wi;

        const long_index_t out_offset =
            static_cast<long_index_t>(iBatch) * kargs.N * kargs.C * kargs.Ho * kargs.Wi_pad * kargs.Y;

        const index_t M = kargs.N * kargs.Ho * kargs.Wi_pad;
        const index_t K = kargs.C * kargs.Y;

        // ── Destination: packed I' ────────────────────────────────────────
        const auto dst_view = make_naive_tensor_view<address_space_enum::global>(
            static_cast<OutDataType*>(kargs.p_out) + out_offset,
            make_tuple(M, K),
            make_tuple(K, 1),
            number<AlignOut>{},
            I1);

        const auto dst_padded = pad_tensor_view(
            dst_view,
            make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}),
            sequence<false, true>{});

        constexpr auto dstr = MakeBlockTileDistribution();

        auto dst_tile = make_tile_window(
            dst_padded,
            make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}),
            {iM, iK},
            dstr);

        // ── Source + load+store: dispatch by layout ───────────────────────
        // The two source descriptors have different C++ types (different transform
        // chains), so we cannot share an `auto` variable. Instead, duplicate the
        // tiled load-store logic for each layout path.
        if(kargs.is_nhwgc != 0)
        {
            const auto src_desc   = MakeSourceDescNhwgc(kargs);
            const auto src_view   = make_tensor_view<address_space_enum::global>(
                static_cast<const InDataType*>(kargs.p_in) + in_offset, src_desc);
            const auto src_padded = pad_tensor_view(
                src_view,
                make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}),
                sequence<false, true>{});
            const auto src_tile   = make_tile_window(
                src_padded,
                make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}),
                {iM, iK}, dstr);
            store_tile(dst_tile, load_tile(src_tile));
        }
        else
        {
            const auto src_desc   = MakeSourceDescGnchw(kargs);
            const auto src_view   = make_tensor_view<address_space_enum::global>(
                static_cast<const InDataType*>(kargs.p_in) + in_offset, src_desc);
            const auto src_padded = pad_tensor_view(
                src_view,
                make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}),
                sequence<false, true>{});
            const auto src_tile   = make_tile_window(
                src_padded,
                make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}),
                {iM, iK}, dstr);
            store_tile(dst_tile, load_tile(src_tile));
        }
    }
};

} // namespace ck_tile
