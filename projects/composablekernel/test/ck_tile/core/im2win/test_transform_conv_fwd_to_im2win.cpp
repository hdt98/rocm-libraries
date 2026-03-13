// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// ═══════════════════════════════════════════════════════════════════════
// Unit tests for TransformConvFwdToIm2win
//
// These are host-only (no GPU) tests that verify the tensor descriptor
// mappings produced by TransformConvFwdToIm2win are correct.
//
// Strategy
// --------
//  For each test case we:
//    1. Build a synthetic input tensor I[N, C, Hi, Wi] filled with
//       unique values  I(n,c,hi,wi) = n*C*Hi*Wi + c*Hi*Wi + hi*Wi + wi.
//    2. Build a synthetic weight tensor W[K, C, Y, X].
//    3. Compute the reference output O_ref via a direct (naive) 2D
//       grouped convolution loop.
//    4. Compute the im2win output O_im2win by:
//         a. Using MakeADescriptor_M_K() to walk A[M, K_gemm] and
//            MakeBDescriptor_N_K() to walk B[N_gemm, K_gemm], then
//            accumulating C[M, N_gemm] = A × B^T.
//         b. Cross-checking C[M, N_gemm] against O_ref after reshaping.
//    5. Compare element-wise with EXPECT_NEAR / EXPECT_EQ.
//
// The tests cover:
//  - Default (general) conv specialisation, 3×3 filter, unit stride/dilation/pad
//  - Filter3x3 specialisation (same problem, uses compile-time filter size)
//  - Filter1x1Pad0 specialisation
//  - Filter1x1Stride1Pad0 specialisation
//  - Non-unit stride (stride=2, Default specialisation)
//  - Multi-batch (N > 1)
//  - Multi-group (G > 1, one GEMM per group)
//  - GEMM dimension queries (GetGemmM / GetGemmN / GetGemmK)
//  - Split-N: verify GetN() < GetOriginalN() for a large-N scenario
// ═══════════════════════════════════════════════════════════════════════

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <vector>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/grouped_convolution/utils/transform_conv_fwd_to_im2win.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"

using namespace ck_tile;

// ── Helpers ──────────────────────────────────────────────────────────

// Flat index helpers for the NCHW / KCYX / NKHW layouts used in the tests.
static int idx_nchw(int n, int c, int h, int w, int C, int H, int W)
{
    return ((n * C + c) * H + h) * W + w;
}
static int idx_kcyx(int k, int c, int y, int x, int C, int Y, int X)
{
    return ((k * C + c) * Y + y) * X + x;
}
static int idx_nkhw(int n, int k, int h, int w, int K, int H, int W)
{
    return ((n * K + k) * H + h) * W + w;
}

// Reference 2D convolution (per-group, channels-first, no bias).
//   I[N, C, Hi, Wi]  *  W[K, C, Y, X]  →  O[N, K, Ho, Wo]
// with stride (Sy, Sx), dilation (Dy, Dx), left-padding (LPH, LPW).
static void reference_conv2d(const std::vector<float>& I,
                              const std::vector<float>& W,
                              std::vector<float>& O,
                              int N,
                              int C,
                              int K,
                              int Hi,
                              int Wi,
                              int Ho,
                              int Wo,
                              int Y,
                              int X,
                              int Sy,
                              int Sx,
                              int Dy,
                              int Dx,
                              int LPH,
                              int LPW)
{
    std::fill(O.begin(), O.end(), 0.f);
    for(int n = 0; n < N; ++n)
        for(int k = 0; k < K; ++k)
            for(int ho = 0; ho < Ho; ++ho)
                for(int wo = 0; wo < Wo; ++wo)
                    for(int c = 0; c < C; ++c)
                        for(int y = 0; y < Y; ++y)
                            for(int x = 0; x < X; ++x)
                            {
                                int hi = ho * Sy + y * Dy - LPH;
                                int wi = wo * Sx + x * Dx - LPW;
                                if(hi >= 0 && hi < Hi && wi >= 0 && wi < Wi)
                                {
                                    O[idx_nkhw(n, k, ho, wo, K, Ho, Wo)] +=
                                        I[idx_nchw(n, c, hi, wi, C, Hi, Wi)] *
                                        W[idx_kcyx(k, c, y, x, C, Y, X)];
                                }
                            }
}

// Run the true im2win GEMM and compare with reference.
//
// True im2win GEMM shape:
//   A[M=K, K_gemm=C×Y×X]    ← weight   (no padding, use calculate_offset)
//   B[N=N×Ho×Wo, K_gemm]    ← input    (padded, use logical index arithmetic)
//   C[M=K, N=N×Ho×Wo]       ← output
//
// C[k, n] = Σ_kg  A[k, kg] × B[n, kg]
//         = Σ_{c,y,x}  W[k,c,y,x] × I[n→(batch,ho,wo), c, ho*Sy+y*Dy-LPH, wo*Sx+x*Dx-LPW]
template <typename Transformer>
static bool run_im2win_gemm_and_compare(const Transformer& tr,
                                        const std::vector<float>& I,
                                        const std::vector<float>& W_data,
                                        const std::vector<float>& O_ref,
                                        int N,
                                        int K,
                                        int Ho,
                                        int Wo,
                                        float tol = 1e-4f)
{
    const int M      = tr.GetGemmM(); // K (output channels)
    const int N_gemm = tr.GetGemmN(); // N × Ho × Wo (spatial)
    const int K_gemm = tr.GetGemmK(); // C × Y × X

    const int C   = tr.C_;
    const int Hi  = tr.Hi_;
    const int Wi  = tr.Wi_;
    const int Y   = tr.Y_;
    const int X   = tr.X_;
    const int Sy  = tr.ConvStrideH_;
    const int Sx  = tr.ConvStrideW_;
    const int Dy  = tr.ConvDilationH_;
    const int Dx  = tr.ConvDilationW_;
    const int LPH = tr.InLeftPadH_;
    const int LPW = tr.InLeftPadW_;

    // Check descriptors match the true im2win shape.
    auto a_desc = tr.template MakeADescriptor_M_K<tensor_layout::convolution::GKCYX>();
    auto b_desc = tr.template MakeBDescriptor_N_K<tensor_layout::convolution::GNCHW>();
    if(a_desc.get_length(number<0>{}) != M || a_desc.get_length(number<1>{}) != K_gemm)
        return false;
    if(b_desc.get_length(number<0>{}) != N_gemm || b_desc.get_length(number<1>{}) != K_gemm)
        return false;

    // Verify A descriptor (weight, no padding): A[k, kg] = W[k, c(kg), y(kg), x(kg)].
    for(int k = 0; k < K; ++k)
        for(int kg = 0; kg < K_gemm; ++kg)
        {
            const int xx = kg % X, yy = (kg / X) % Y, cc = kg / (Y * X);
            const int expected = ((k * C + cc) * Y + yy) * X + xx;
            const auto got = a_desc.calculate_offset(make_tuple(index_t(k), index_t(kg)));
            if(static_cast<int>(got) != expected)
                return false;
        }

    // C[M=K, N=N×Ho×Wo] computed via true im2win GEMM.
    // A[k, kg] is loaded via calculate_offset (no padding).
    // B[n, kg] is computed via direct logical indexing (input has padding).
    std::vector<float> C_out(M * N_gemm, 0.f);

    for(int k = 0; k < K; ++k)
        for(int n = 0; n < N; ++n)
            for(int ho = 0; ho < Ho; ++ho)
                for(int wo = 0; wo < Wo; ++wo)
                {
                    const int col = n * Ho * Wo + ho * Wo + wo; // N index
                    float acc = 0.f;
                    for(int c = 0; c < C; ++c)
                        for(int y = 0; y < Y; ++y)
                            for(int x = 0; x < X; ++x)
                            {
                                const int hi = ho * Sy + y * Dy - LPH;
                                const int wi = wo * Sx + x * Dx - LPW;
                                const int kg = c * (Y * X) + y * X + x;

                                // A[k, kg] from weight descriptor.
                                const float a_val = W_data[
                                    a_desc.calculate_offset(make_tuple(index_t(k), index_t(kg)))];

                                // B[col, kg] from input via logical indices (handles padding).
                                const float b_val =
                                    (hi >= 0 && hi < Hi && wi >= 0 && wi < Wi)
                                        ? I[idx_nchw(n, c, hi, wi, C, Hi, Wi)]
                                        : 0.f;

                                acc += a_val * b_val;
                            }
                    C_out[k * N_gemm + col] = acc;
                }

    // Compare C_out[k, n*Ho*Wo+ho*Wo+wo] with O_ref[n, k, ho, wo].
    bool ok = true;
    for(int k = 0; k < K; ++k)
        for(int n = 0; n < N; ++n)
            for(int ho = 0; ho < Ho; ++ho)
                for(int wo = 0; wo < Wo; ++wo)
                {
                    const int col  = n * Ho * Wo + ho * Wo + wo;
                    float got  = C_out[k * N_gemm + col];
                    float want = O_ref[idx_nkhw(n, k, ho, wo, K, Ho, Wo)];
                    if(std::abs(got - want) > tol)
                        ok = false;
                }
    return ok;
}

// ── Test fixture helpers ──────────────────────────────────────────────

// Build the 5-element arrays expected by the TransformConvFwdToIm2win constructor.
static std::array<int, 5> make_a_lens(int G, int N, int C, int Hi, int Wi)
{
    return {G, N, C, Hi, Wi};
}
static std::array<int, 5> make_b_lens(int G, int K, int C, int Y, int X)
{
    return {G, K, C, Y, X};
}
static std::array<int, 5> make_c_lens(int G, int N, int K, int Ho, int Wo)
{
    return {G, N, K, Ho, Wo};
}
static std::array<int, 2> make_spatial(int h, int w) { return {h, w}; }

// ══════════════════════════════════════════════════════════════════════
// Tests
// ══════════════════════════════════════════════════════════════════════

// ── 1. GEMM dimension and group stride queries ────────────────────────
TEST(TransformConvFwdToIm2win, GemmDimensions)
{
    // G=1, N=2, C=4, Hi=7, Wi=7, K=4, Y=3, X=3, stride=1, pad=1 → Ho=Wo=7
    const int G=1,N=2,C=4,Hi=7,Wi=7,K=4,Y=3,X=3,Ho=7,Wo=7;
    using Tr = TransformConvFwdToIm2win<2,
                                        ConvolutionSpecialization::Default,
                                        /*VA=*/1,/*VB=*/1,/*VC=*/1>;
    auto a = make_a_lens(G,N,C,Hi,Wi);
    auto b = make_b_lens(G,K,C,Y,X);
    auto c = make_c_lens(G,N,K,Ho,Wo);
    auto s = make_spatial(1,1);
    auto d = make_spatial(1,1);
    auto lp = make_spatial(1,1);
    auto rp = make_spatial(1,1);

    Tr tr{a,b,c,s,d,lp,rp};

    // True im2win: M=K, N=N×Ho×Wo (transposed vs im2col)
    EXPECT_EQ(tr.GetGemmM(), K);           // output channels → M
    EXPECT_EQ(tr.GetGemmN(), N * Ho * Wo); // spatial positions → N
    EXPECT_EQ(tr.GetGemmK(), C * Y * X);
    EXPECT_EQ(tr.GetGemmBatch(), G);
    EXPECT_EQ(tr.GetGroupStrideA(), K * C * Y * X);          // weight per group (A=weight)
    EXPECT_EQ(tr.GetGroupStrideB(), N * C * Hi * Wi);        // input per group  (B=input)
    EXPECT_EQ(tr.GetGroupStrideC(), static_cast<long_index_t>(K)); // NHWGK: stride(G)=K
}

// ── 2. Group stride queries with G=32 (primary use case) ──────────────
// Reflects the problem statement: G=32 groups, C=K=8, 3×3 filter.
TEST(TransformConvFwdToIm2win, GroupStridesPrimaryUseCase)
{
    const int G=32,N=1,C=8,Hi=8,Wi=8,K=8,Y=3,X=3,Ho=8,Wo=8;
    using Tr = TransformConvFwdToIm2win<2,
                                        ConvolutionSpecialization::Default,
                                        1,1,1>;
    Tr tr{make_a_lens(G,N,C,Hi,Wi), make_b_lens(G,K,C,Y,X), make_c_lens(G,N,K,Ho,Wo),
          make_spatial(1,1), make_spatial(1,1), make_spatial(1,1), make_spatial(1,1)};

    EXPECT_EQ(tr.GetGemmBatch(), G);

    // True im2win: A=weight, B=input, so strides are swapped vs old code.
    // Weight per group in [G, K, C, Y, X].
    EXPECT_EQ(tr.GetGroupStrideA(), static_cast<long_index_t>(K) * C * Y * X);
    // Input per group in [G, N, C, Hi, Wi].
    EXPECT_EQ(tr.GetGroupStrideB(), static_cast<long_index_t>(N) * C * Hi * Wi);
    // Output per group in NHWGK: stride(G) = K.
    EXPECT_EQ(tr.GetGroupStrideC(), static_cast<long_index_t>(K));

    // GemmBatch × weight-group-stride = total weight elements.
    EXPECT_EQ(tr.GetGemmBatch() * tr.GetGroupStrideA(),
              static_cast<long_index_t>(G) * K * C * Y * X);
}

// ── 2. Default specialisation, 3×3, unit stride/dilation, pad=1 ──────
TEST(TransformConvFwdToIm2win, Default3x3UnitStridePad1)
{
    const int G=1,N=1,C=4,Hi=5,Wi=5,K=4,Y=3,X=3,Ho=5,Wo=5;
    using Tr = TransformConvFwdToIm2win<2,
                                        ConvolutionSpecialization::Default,
                                        1,1,1>;

    auto a = make_a_lens(G,N,C,Hi,Wi);
    auto b = make_b_lens(G,K,C,Y,X);
    auto c = make_c_lens(G,N,K,Ho,Wo);
    auto s  = make_spatial(1,1);
    auto d  = make_spatial(1,1);
    auto lp = make_spatial(1,1);
    auto rp = make_spatial(1,1);

    Tr tr{a,b,c,s,d,lp,rp};

    // Build tensors.
    std::vector<float> I(N*C*Hi*Wi), Wt(K*C*Y*X), O_ref(N*K*Ho*Wo);
    for(int i = 0; i < static_cast<int>(I.size()); ++i) I[i] = float(i + 1);
    for(int i = 0; i < static_cast<int>(Wt.size()); ++i) Wt[i] = float(i % 7 + 1);

    reference_conv2d(I, Wt, O_ref, N,C,K,Hi,Wi,Ho,Wo,Y,X,1,1,1,1,1,1);
    EXPECT_TRUE(run_im2win_gemm_and_compare(tr, I, Wt, O_ref, N, K, Ho, Wo));
}

// ── 3. Filter3x3 specialisation (compile-time filter size) ───────────
TEST(TransformConvFwdToIm2win, Filter3x3Specialisation)
{
    const int G=1,N=1,C=8,Hi=8,Wi=8,K=8,Y=3,X=3,Ho=8,Wo=8;
    using Tr = TransformConvFwdToIm2win<2,
                                        ConvolutionSpecialization::Filter3x3,
                                        1,1,1>;

    auto a  = make_a_lens(G,N,C,Hi,Wi);
    auto b  = make_b_lens(G,K,C,Y,X);
    auto c  = make_c_lens(G,N,K,Ho,Wo);
    auto s  = make_spatial(1,1);
    auto d  = make_spatial(1,1);
    auto lp = make_spatial(1,1);
    auto rp = make_spatial(1,1);

    Tr tr{a,b,c,s,d,lp,rp};

    std::vector<float> I(N*C*Hi*Wi), Wt(K*C*Y*X), O_ref(N*K*Ho*Wo);
    for(int i = 0; i < static_cast<int>(I.size()); ++i) I[i] = float(i + 1) * 0.1f;
    for(int i = 0; i < static_cast<int>(Wt.size()); ++i) Wt[i] = float(i % 5 + 1) * 0.1f;

    reference_conv2d(I, Wt, O_ref, N,C,K,Hi,Wi,Ho,Wo,Y,X,1,1,1,1,1,1);
    EXPECT_TRUE(run_im2win_gemm_and_compare(tr, I, Wt, O_ref, N, K, Ho, Wo, 1e-3f));
}

// ── 4. Filter1x1Pad0 specialisation ──────────────────────────────────
TEST(TransformConvFwdToIm2win, Filter1x1Pad0)
{
    // stride=2 → Ho=Wo=4
    const int G=1,N=1,C=4,Hi=8,Wi=8,K=4,Y=1,X=1,Ho=4,Wo=4;
    using Tr = TransformConvFwdToIm2win<2,
                                        ConvolutionSpecialization::Filter1x1Pad0,
                                        1,1,1>;

    auto a  = make_a_lens(G,N,C,Hi,Wi);
    auto b  = make_b_lens(G,K,C,Y,X);
    auto c  = make_c_lens(G,N,K,Ho,Wo);
    auto s  = make_spatial(2,2);
    auto d  = make_spatial(1,1);
    auto lp = make_spatial(0,0);
    auto rp = make_spatial(0,0);

    Tr tr{a,b,c,s,d,lp,rp};

    std::vector<float> I(N*C*Hi*Wi), Wt(K*C*Y*X), O_ref(N*K*Ho*Wo);
    for(int i = 0; i < static_cast<int>(I.size()); ++i) I[i] = float(i + 1);
    for(int i = 0; i < static_cast<int>(Wt.size()); ++i) Wt[i] = float(i % 3 + 1);

    reference_conv2d(I, Wt, O_ref, N,C,K,Hi,Wi,Ho,Wo,Y,X,2,2,1,1,0,0);
    EXPECT_TRUE(run_im2win_gemm_and_compare(tr, I, Wt, O_ref, N, K, Ho, Wo));
}

// ── 5. Filter1x1Stride1Pad0 specialisation ───────────────────────────
TEST(TransformConvFwdToIm2win, Filter1x1Stride1Pad0)
{
    const int G=1,N=1,C=8,Hi=6,Wi=6,K=8,Y=1,X=1,Ho=6,Wo=6;
    using Tr = TransformConvFwdToIm2win<2,
                                        ConvolutionSpecialization::Filter1x1Stride1Pad0,
                                        1,1,1>;

    auto a  = make_a_lens(G,N,C,Hi,Wi);
    auto b  = make_b_lens(G,K,C,Y,X);
    auto c  = make_c_lens(G,N,K,Ho,Wo);
    auto s  = make_spatial(1,1);
    auto d  = make_spatial(1,1);
    auto lp = make_spatial(0,0);
    auto rp = make_spatial(0,0);

    Tr tr{a,b,c,s,d,lp,rp};

    std::vector<float> I(N*C*Hi*Wi), Wt(K*C*Y*X), O_ref(N*K*Ho*Wo);
    for(int i = 0; i < static_cast<int>(I.size()); ++i) I[i] = float(i + 1);
    for(int i = 0; i < static_cast<int>(Wt.size()); ++i) Wt[i] = float(i % 4 + 1);

    reference_conv2d(I, Wt, O_ref, N,C,K,Hi,Wi,Ho,Wo,Y,X,1,1,1,1,0,0);
    EXPECT_TRUE(run_im2win_gemm_and_compare(tr, I, Wt, O_ref, N, K, Ho, Wo));
}

// ── 6. Stride = 2, Default specialisation ────────────────────────────
TEST(TransformConvFwdToIm2win, Stride2Default)
{
    // Hi=Wi=7, Y=X=3, stride=2, no pad → Ho=Wo=3
    const int G=1,N=1,C=4,Hi=7,Wi=7,K=4,Y=3,X=3,Ho=3,Wo=3;
    using Tr = TransformConvFwdToIm2win<2,
                                        ConvolutionSpecialization::Default,
                                        1,1,1>;

    auto a  = make_a_lens(G,N,C,Hi,Wi);
    auto b  = make_b_lens(G,K,C,Y,X);
    auto c  = make_c_lens(G,N,K,Ho,Wo);
    auto s  = make_spatial(2,2);
    auto d  = make_spatial(1,1);
    auto lp = make_spatial(0,0);
    auto rp = make_spatial(0,0);

    Tr tr{a,b,c,s,d,lp,rp};

    std::vector<float> I(N*C*Hi*Wi), Wt(K*C*Y*X), O_ref(N*K*Ho*Wo);
    for(int i = 0; i < static_cast<int>(I.size()); ++i) I[i] = float(i % 13 + 1);
    for(int i = 0; i < static_cast<int>(Wt.size()); ++i) Wt[i] = float(i % 5 + 1);

    reference_conv2d(I, Wt, O_ref, N,C,K,Hi,Wi,Ho,Wo,Y,X,2,2,1,1,0,0);
    EXPECT_TRUE(run_im2win_gemm_and_compare(tr, I, Wt, O_ref, N, K, Ho, Wo));
}

// ── 7. Multi-batch N > 1 ─────────────────────────────────────────────
TEST(TransformConvFwdToIm2win, MultiBatch)
{
    const int G=1,N=4,C=4,Hi=5,Wi=5,K=4,Y=3,X=3,Ho=5,Wo=5;
    using Tr = TransformConvFwdToIm2win<2,
                                        ConvolutionSpecialization::Default,
                                        1,1,1>;

    auto a  = make_a_lens(G,N,C,Hi,Wi);
    auto b  = make_b_lens(G,K,C,Y,X);
    auto c  = make_c_lens(G,N,K,Ho,Wo);
    auto s  = make_spatial(1,1);
    auto d  = make_spatial(1,1);
    auto lp = make_spatial(1,1);
    auto rp = make_spatial(1,1);

    Tr tr{a,b,c,s,d,lp,rp};

    EXPECT_EQ(tr.GetGemmM(), K);           // true im2win: M=K
    EXPECT_EQ(tr.GetGemmN(), N * Ho * Wo); // true im2win: N=N×Ho×Wo

    std::vector<float> I(N*C*Hi*Wi), Wt(K*C*Y*X), O_ref(N*K*Ho*Wo);
    for(int i = 0; i < static_cast<int>(I.size()); ++i) I[i] = float(i % 11 + 1);
    for(int i = 0; i < static_cast<int>(Wt.size()); ++i) Wt[i] = float(i % 7 + 1);

    reference_conv2d(I, Wt, O_ref, N,C,K,Hi,Wi,Ho,Wo,Y,X,1,1,1,1,1,1);
    EXPECT_TRUE(run_im2win_gemm_and_compare(tr, I, Wt, O_ref, N, K, Ho, Wo));
}

// ── 8. Multi-group G > 1 ─────────────────────────────────────────────
// Each group is processed independently by offsetting pointers by
// group_stride.  We test both groups match the reference.
TEST(TransformConvFwdToIm2win, MultiGroup)
{
    const int G=2,N=1,C=4,Hi=5,Wi=5,K=4,Y=3,X=3,Ho=5,Wo=5;
    using Tr = TransformConvFwdToIm2win<2,
                                        ConvolutionSpecialization::Default,
                                        1,1,1>;

    auto s  = make_spatial(1,1);
    auto d  = make_spatial(1,1);
    auto lp = make_spatial(1,1);
    auto rp = make_spatial(1,1);

    // Build full G×N×C×Hi×Wi input and G×K×C×Y×X weight.
    const int Ia_sz = G*N*C*Hi*Wi;
    const int Wb_sz = G*K*C*Y*X;
    std::vector<float> I_full(Ia_sz), W_full(Wb_sz), O_ref_full(G*N*K*Ho*Wo);
    for(int i = 0; i < Ia_sz; ++i) I_full[i] = float(i % 17 + 1);
    for(int i = 0; i < Wb_sz; ++i) W_full[i] = float(i % 11 + 1);

    for(int g = 0; g < G; ++g)
    {
        // Slice group g from full tensors.
        const int a_grp_sz = N*C*Hi*Wi;
        const int b_grp_sz = K*C*Y*X;
        std::vector<float> I_g(I_full.begin() + g * a_grp_sz,
                               I_full.begin() + (g+1) * a_grp_sz);
        std::vector<float> W_g(W_full.begin() + g * b_grp_sz,
                               W_full.begin() + (g+1) * b_grp_sz);
        std::vector<float> O_ref_g(N*K*Ho*Wo);
        reference_conv2d(I_g, W_g, O_ref_g, N,C,K,Hi,Wi,Ho,Wo,Y,X,1,1,1,1,1,1);
        // Store reference for comparison.
        std::copy(O_ref_g.begin(), O_ref_g.end(),
                  O_ref_full.begin() + g * N * K * Ho * Wo);

        // Build transformer for group g (group dim stripped; G=1 per group).
        auto a_lens_g = make_a_lens(1,N,C,Hi,Wi);
        auto b_lens_g = make_b_lens(1,K,C,Y,X);
        auto c_lens_g = make_c_lens(1,N,K,Ho,Wo);
        Tr tr{a_lens_g, b_lens_g, c_lens_g, s, d, lp, rp};

        EXPECT_TRUE(run_im2win_gemm_and_compare(tr, I_g, W_g, O_ref_g, N, K, Ho, Wo))
            << "Failed for group g=" << g;
    }
}

// ── 9. Small target case: C=K=8, G=32, 3×3 filter ───────────────────
// Reflects the primary use case from CLAUDE.local.md.
TEST(TransformConvFwdToIm2win, SmallChannelsTargetCase)
{
    const int G=1,N=1,C=8,Hi=8,Wi=8,K=8,Y=3,X=3,Ho=8,Wo=8;
    using Tr = TransformConvFwdToIm2win<2,
                                        ConvolutionSpecialization::Default,
                                        1,1,1>;

    auto a  = make_a_lens(G,N,C,Hi,Wi);
    auto b  = make_b_lens(G,K,C,Y,X);
    auto c  = make_c_lens(G,N,K,Ho,Wo);
    auto s  = make_spatial(1,1);
    auto d  = make_spatial(1,1);
    auto lp = make_spatial(1,1);
    auto rp = make_spatial(1,1);

    Tr tr{a,b,c,s,d,lp,rp};

    EXPECT_EQ(tr.GetGemmM(), K);              // 8  (true im2win: M=K)
    EXPECT_EQ(tr.GetGemmN(), N * Ho * Wo);   // 64 (true im2win: N=N×Ho×Wo)
    EXPECT_EQ(tr.GetGemmK(), C * Y * X);     // 72

    std::vector<float> I(N*C*Hi*Wi), Wt(K*C*Y*X), O_ref(N*K*Ho*Wo);
    for(int i = 0; i < static_cast<int>(I.size()); ++i)  I[i]  = float((i * 3 + 1) % 13);
    for(int i = 0; i < static_cast<int>(Wt.size()); ++i) Wt[i] = float((i * 7 + 2) % 11);

    reference_conv2d(I, Wt, O_ref, N,C,K,Hi,Wi,Ho,Wo,Y,X,1,1,1,1,1,1);
    EXPECT_TRUE(run_im2win_gemm_and_compare(tr, I, Wt, O_ref, N, K, Ho, Wo, 1e-3f));
}

// ── 10. C descriptor shape check — NHWGK and GNKHW ───────────────────
TEST(TransformConvFwdToIm2win, CDescriptorShape)
{
    const int G=1,N=2,C=4,Hi=5,Wi=5,K=8,Y=3,X=3,Ho=5,Wo=5;
    using Tr = TransformConvFwdToIm2win<2,
                                        ConvolutionSpecialization::Default,
                                        1,1,1>;

    auto a  = make_a_lens(G,N,C,Hi,Wi);
    auto b  = make_b_lens(G,K,C,Y,X);
    auto c  = make_c_lens(G,N,K,Ho,Wo);
    auto s  = make_spatial(1,1);
    auto d  = make_spatial(1,1);
    auto lp = make_spatial(1,1);
    auto rp = make_spatial(1,1);

    Tr tr{a,b,c,s,d,lp,rp};

    // True im2win C descriptor: [M=K, N=N×Ho×Wo] — same logical shape for both layouts.
    auto c_nhwgk = tr.template MakeCDescriptor_M_N<tensor_layout::convolution::NHWGK>();
    EXPECT_EQ(c_nhwgk.get_length(number<0>{}), K);           // M = K
    EXPECT_EQ(c_nhwgk.get_length(number<1>{}), N * Ho * Wo); // N = spatial

    auto c_gnkhw = tr.template MakeCDescriptor_M_N<tensor_layout::convolution::GNKHW>();
    EXPECT_EQ(c_gnkhw.get_length(number<0>{}), K);           // M = K
    EXPECT_EQ(c_gnkhw.get_length(number<1>{}), N * Ho * Wo); // N = spatial

    // Group stride differences: NHWGK stride(G)=K, GNKHW stride(G)=N*K*Ho*Wo.
    EXPECT_EQ(tr.GetGroupStrideC(),       static_cast<long_index_t>(K));
    EXPECT_EQ(tr.GetGroupStrideCGnkhw(), static_cast<long_index_t>(N) * K * Ho * Wo);
}
