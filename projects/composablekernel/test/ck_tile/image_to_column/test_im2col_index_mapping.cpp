// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Unit tests for im2col index mappings and address decomposition.
//
// Reference offsets are obtained from TransformConvFwdToGemm::MakeADescriptor_M_K(),
// which is the production im2col descriptor used by the conv-fwd pipeline.
//
// The decomposed functions under test implement the analysis from
// im2col_tile_analysis_general.md:
//
//   offset(m_gemm, k_gemm) = M_base(m_gemm) + K_offset(k_gemm)
//
// where M_base depends only on m_gemm and K_offset depends only on k_gemm.
//
// All tests are host-side integer arithmetic; no GPU execution needed.

#include <gtest/gtest.h>
#include <array>
#include <vector>

#include "ck_tile/core.hpp"
#include "ck_tile/core/tensor/tiled_im2col_coordinate.hpp"
#include "ck_tile/ops/grouped_convolution/utils/convolution_specialization.hpp"
#include "ck_tile/ops/grouped_convolution/utils/transform_conv_fwd_to_gemm.hpp"
#include "ck_tile/ops/grouped_convolution/utils/transform_conv_fwd_to_gemm_v2.hpp"

using namespace ck_tile;

// ---------------------------------------------------------------------------
// Convolution parameter struct (2D, NHWGC / GKYXC / NHWGK)
// ---------------------------------------------------------------------------

struct Conv2dParams
{
    int G, N, C, K;
    int Hi, Wi;     // input spatial
    int Ho, Wo;     // output spatial (computed from others)
    int Y, X;       // filter spatial
    int SH, SW;     // stride
    int DH, DW;     // dilation
    int PH, PW;     // left (= right) padding

    int Ho_compute() const { return (Hi + 2 * PH - DH * (Y - 1) - 1) / SH + 1; }
    int Wo_compute() const { return (Wi + 2 * PW - DW * (X - 1) - 1) / SW + 1; }

    // NHWGC input strides
    int NStride()  const { return Hi * Wi * G * C; }
    int HiStride() const { return Wi * G * C; }
    int WiStride() const { return G * C; }
    int GStride()  const { return C; }

    // GEMM dimensions
    int M_gemm() const { return N * Ho * Wo; }
    int K_gemm() const { return Y * X * C; }
    int N_gemm() const { return K; }
};

// ---------------------------------------------------------------------------
// Reference: use TransformConvFwdToGemm to compute the im2col offset
// ---------------------------------------------------------------------------
//
// The transformer instantiation uses:
//   - NDimSpatial = 2
//   - ConvolutionSpecialization::Default  (general case)
//   - VectorSizeA/B/C = 1                (doesn't affect calculate_offset)
//   - NumGroupsToMerge = 1, SplitN = false

using RefTransformer = TransformConvFwdToGemm<
    2,
    ConvolutionSpecialization::Default,
    /*VectorSizeA=*/1,
    /*VectorSizeB=*/1,
    /*VectorSizeC=*/1,
    /*NumGroupsToMerge=*/1,
    /*SplitN=*/false,
    float,
    float,
    int>;

// Build the transformer from a Conv2dParams (right padding = left padding)
RefTransformer make_transformer(const Conv2dParams& p)
{
    std::array<int, 5> a_lens = {p.G, p.N, p.C, p.Hi, p.Wi};
    std::array<int, 5> b_lens = {p.G, p.K, p.C, p.Y,  p.X};
    std::array<int, 5> c_lens = {p.G, p.N, p.K, p.Ho, p.Wo};
    std::array<int, 2> strides    = {p.SH, p.SW};
    std::array<int, 2> dilations  = {p.DH, p.DW};
    std::array<int, 2> left_pads  = {p.PH, p.PW};
    std::array<int, 2> right_pads = {p.PH, p.PW};
    return RefTransformer{a_lens, b_lens, c_lens, strides, dilations, left_pads, right_pads};
}

// Reference offset for (m_gemm, k_gemm) via the production descriptor.
// Returns -1 for padded (invalid) locations.
int reference_offset(const Conv2dParams& p, int m, int k)
{
    auto transformer = make_transformer(p);
    auto desc        = transformer.template MakeADescriptor_M_K<
        tensor_layout::convolution::NHWGC>();

    auto coord = make_tensor_coordinate(desc, make_multi_index(m, k));

    // Check validity (padded positions are invalid)
    if(!coordinate_has_valid_offset_assuming_top_index_is_valid(desc, coord))
        return -1;

    return coord.get_offset();
}

// ---------------------------------------------------------------------------
// Decomposed index mappings (the functions under test)
// ---------------------------------------------------------------------------

struct MIndex { int n_conv, ho, wo; };
struct KIndex { int y, x, c_conv; };

MIndex decode_m(const Conv2dParams& p, int m_gemm)
{
    int HoWo   = p.Ho * p.Wo;
    int n_conv = m_gemm / HoWo;
    int rem    = m_gemm % HoWo;
    return {n_conv, rem / p.Wo, rem % p.Wo};
}

KIndex decode_k(const Conv2dParams& p, int k_gemm)
{
    int XC = p.X * p.C;
    int y  = k_gemm / XC;
    int rem = k_gemm % XC;
    return {y, rem / p.C, rem % p.C};
}

// M_base: offset contribution from m_gemm only (g=0; g is an external pointer offset)
int M_base(const Conv2dParams& p, int m_gemm)
{
    auto [n, ho, wo] = decode_m(p, m_gemm);
    return n  * p.NStride()
         + ho * p.SH * p.HiStride()
         + wo * p.SW * p.WiStride()
         - p.PH * p.HiStride()
         - p.PW * p.WiStride();
}

// K_offset: offset contribution from k_gemm only
int K_offset(const Conv2dParams& p, int k_gemm)
{
    auto [y, x, c] = decode_k(p, k_gemm);
    return y * p.DH * p.HiStride()
         + x * p.DW * p.WiStride()
         + c;
}

// Validity check: does (m_gemm, k_gemm) map to a non-padded input location?
bool is_valid(const Conv2dParams& p, int m_gemm, int k_gemm)
{
    auto [n, ho, wo] = decode_m(p, m_gemm);
    auto [y, x, c]   = decode_k(p, k_gemm);
    int ih = ho * p.SH + y * p.DH - p.PH;
    int iw = wo * p.SW + x * p.DW - p.PW;
    return (ih >= 0 && ih < p.Hi) && (iw >= 0 && iw < p.Wi);
}

// ---------------------------------------------------------------------------
// Exhaustive verifier: M_base + K_offset must match TransformConvFwdToGemm
// for all valid (m, k), and validity must agree with is_valid().
// ---------------------------------------------------------------------------

void verify_against_reference(const Conv2dParams& p)
{
    for(int m = 0; m < p.M_gemm(); ++m)
    {
        for(int k = 0; k < p.K_gemm(); ++k)
        {
            int ref = reference_offset(p, m, k);

            if(ref == -1)
            {
                EXPECT_FALSE(is_valid(p, m, k))
                    << "m=" << m << " k=" << k << " should be invalid";
            }
            else
            {
                EXPECT_TRUE(is_valid(p, m, k))
                    << "m=" << m << " k=" << k << " should be valid";

                int decomposed = M_base(p, m) + K_offset(p, k);
                EXPECT_EQ(decomposed, ref)
                    << "Decomposition mismatch: m=" << m << " k=" << k;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Shared test parameter factory functions
// ---------------------------------------------------------------------------

static Conv2dParams make_unit_params()
{
    Conv2dParams p;
    p.G = 2; p.N = 4; p.C = 8; p.K = 8;
    p.Hi = 8; p.Wi = 8;
    p.Y = 3; p.X = 3;
    p.SH = 1; p.SW = 1; p.DH = 1; p.DW = 1; p.PH = 0; p.PW = 0;
    p.Ho = p.Ho_compute(); p.Wo = p.Wo_compute(); // 6, 6
    return p;
}

static Conv2dParams make_stride2_params()
{
    Conv2dParams p;
    p.G = 1; p.N = 2; p.C = 4; p.K = 4;
    p.Hi = 8; p.Wi = 8;
    p.Y = 3; p.X = 3;
    p.SH = 2; p.SW = 2; p.DH = 1; p.DW = 1; p.PH = 0; p.PW = 0;
    p.Ho = p.Ho_compute(); p.Wo = p.Wo_compute(); // 3, 3
    return p;
}

static Conv2dParams make_dilation2_params()
{
    Conv2dParams p;
    p.G = 1; p.N = 2; p.C = 4; p.K = 4;
    p.Hi = 9; p.Wi = 9;
    p.Y = 3; p.X = 3;
    p.SH = 1; p.SW = 1; p.DH = 2; p.DW = 2; p.PH = 0; p.PW = 0;
    p.Ho = p.Ho_compute(); p.Wo = p.Wo_compute(); // 5, 5
    return p;
}

static Conv2dParams make_padded_params()
{
    Conv2dParams p;
    p.G = 1; p.N = 2; p.C = 4; p.K = 4;
    p.Hi = 6; p.Wi = 6;
    p.Y = 3; p.X = 3;
    p.SH = 1; p.SW = 1; p.DH = 1; p.DW = 1; p.PH = 1; p.PW = 1;
    p.Ho = p.Ho_compute(); p.Wo = p.Wo_compute(); // 6, 6
    return p;
}

// ===========================================================================
// TEST SUITE 1: Index Mappings
// ===========================================================================

class Im2colIndexMapping : public ::testing::Test
{
    protected:
    Conv2dParams p = make_unit_params();
};

TEST_F(Im2colIndexMapping, MappingBounds)
{
    for(int m = 0; m < p.M_gemm(); ++m)
    {
        auto [n, ho, wo] = decode_m(p, m);
        EXPECT_GE(n,  0); EXPECT_LT(n,  p.N);
        EXPECT_GE(ho, 0); EXPECT_LT(ho, p.Ho);
        EXPECT_GE(wo, 0); EXPECT_LT(wo, p.Wo);
    }
    for(int k = 0; k < p.K_gemm(); ++k)
    {
        auto [y, x, c] = decode_k(p, k);
        EXPECT_GE(y, 0); EXPECT_LT(y, p.Y);
        EXPECT_GE(x, 0); EXPECT_LT(x, p.X);
        EXPECT_GE(c, 0); EXPECT_LT(c, p.C);
    }
}

TEST_F(Im2colIndexMapping, MappingBijection)
{
    // Every (n, ho, wo) index hit exactly once across m_gemm ∈ [0, M_gemm)
    std::vector<int> hit_m(p.M_gemm(), 0);
    for(int m = 0; m < p.M_gemm(); ++m)
    {
        auto [n, ho, wo] = decode_m(p, m);
        hit_m[n * p.Ho * p.Wo + ho * p.Wo + wo]++;
    }
    for(int i = 0; i < p.M_gemm(); ++i)
        EXPECT_EQ(hit_m[i], 1) << "index " << i << " not hit exactly once in M";

    // Every (y, x, c) index hit exactly once across k_gemm ∈ [0, K_gemm)
    std::vector<int> hit_k(p.K_gemm(), 0);
    for(int k = 0; k < p.K_gemm(); ++k)
    {
        auto [y, x, c] = decode_k(p, k);
        hit_k[y * p.X * p.C + x * p.C + c]++;
    }
    for(int i = 0; i < p.K_gemm(); ++i)
        EXPECT_EQ(hit_k[i], 1) << "index " << i << " not hit exactly once in K";
}

TEST_F(Im2colIndexMapping, MDecodeSpecificValues)
{
    { auto [n, ho, wo] = decode_m(p, 0);
      EXPECT_EQ(n, 0); EXPECT_EQ(ho, 0); EXPECT_EQ(wo, 0); }

    { auto [n, ho, wo] = decode_m(p, p.Wo);
      EXPECT_EQ(n, 0); EXPECT_EQ(ho, 1); EXPECT_EQ(wo, 0); }

    { auto [n, ho, wo] = decode_m(p, p.Ho * p.Wo);
      EXPECT_EQ(n, 1); EXPECT_EQ(ho, 0); EXPECT_EQ(wo, 0); }

    { auto [n, ho, wo] = decode_m(p, p.Ho * p.Wo - 1);
      EXPECT_EQ(n, 0); EXPECT_EQ(ho, p.Ho - 1); EXPECT_EQ(wo, p.Wo - 1); }
}

TEST_F(Im2colIndexMapping, KDecodeSpecificValues)
{
    { auto [y, x, c] = decode_k(p, 0);
      EXPECT_EQ(y, 0); EXPECT_EQ(x, 0); EXPECT_EQ(c, 0); }

    { auto [y, x, c] = decode_k(p, 1);
      EXPECT_EQ(y, 0); EXPECT_EQ(x, 0); EXPECT_EQ(c, 1); }     // c increments

    { auto [y, x, c] = decode_k(p, p.C);
      EXPECT_EQ(y, 0); EXPECT_EQ(x, 1); EXPECT_EQ(c, 0); }     // x increments

    { auto [y, x, c] = decode_k(p, p.X * p.C);
      EXPECT_EQ(y, 1); EXPECT_EQ(x, 0); EXPECT_EQ(c, 0); }     // y increments

    { auto [y, x, c] = decode_k(p, p.K_gemm() - 1);
      EXPECT_EQ(y, p.Y-1); EXPECT_EQ(x, p.X-1); EXPECT_EQ(c, p.C-1); }
}

// ===========================================================================
// TEST SUITE 2: Decomposition vs. TransformConvFwdToGemm reference
// ===========================================================================

class Im2colDecomposition : public ::testing::Test {};

TEST_F(Im2colDecomposition, UnitStrideZeroPad)
{
    verify_against_reference(make_unit_params());
}

TEST_F(Im2colDecomposition, Stride2ZeroPad)
{
    verify_against_reference(make_stride2_params());
}

TEST_F(Im2colDecomposition, Dilation2ZeroPad)
{
    verify_against_reference(make_dilation2_params());
}

TEST_F(Im2colDecomposition, WithPadding)
{
    verify_against_reference(make_padded_params());
}

TEST_F(Im2colDecomposition, MBaseMatchesDescriptorForK0)
{
    // For k_gemm=0 (y=0, x=0, c=0): K_offset=0, so M_base must equal reference
    Conv2dParams p = make_unit_params();
    for(int m = 0; m < p.M_gemm(); ++m)
    {
        int ref = reference_offset(p, m, 0);
        if(ref == -1) continue;
        EXPECT_EQ(M_base(p, m), ref) << "m=" << m;
    }
}

TEST_F(Im2colDecomposition, KOffsetMatchesDescriptorForM0)
{
    // For m_gemm=0 (n=0, ho=0, wo=0): M_base=0 (no padding), so K_offset = reference
    Conv2dParams p = make_unit_params(); // no padding
    int m_base_0 = M_base(p, 0);        // should be 0
    for(int k = 0; k < p.K_gemm(); ++k)
    {
        int ref = reference_offset(p, 0, k);
        if(ref == -1) continue;
        EXPECT_EQ(K_offset(p, k), ref - m_base_0) << "k=" << k;
    }
}

TEST_F(Im2colDecomposition, MBaseIndependentOfK)
{
    Conv2dParams p = make_unit_params();
    for(int m = 0; m < p.M_gemm(); ++m)
    {
        int base0 = M_base(p, m);
        // M_base must not change when k changes (only depends on m)
        for(int k = 1; k < p.K_gemm(); k += p.C) // spot-check at x-boundaries
            EXPECT_EQ(M_base(p, m), base0) << "m=" << m << " k=" << k;
    }
}

TEST_F(Im2colDecomposition, KOffsetIndependentOfM)
{
    Conv2dParams p = make_unit_params();
    for(int k = 0; k < p.K_gemm(); ++k)
    {
        int off0 = K_offset(p, k);
        // K_offset must not change when m changes (only depends on k)
        for(int m = 1; m < p.M_gemm(); m += p.Wo) // spot-check at ho-boundaries
            EXPECT_EQ(K_offset(p, k), off0) << "k=" << k << " m=" << m;
    }
}

// ===========================================================================
// TEST SUITE 3: M-tile Pattern
// ===========================================================================

class Im2colMTilePattern : public ::testing::Test
{
    protected:
    Conv2dParams p = make_unit_params();

    int count_ho_boundaries(int m_start, int M_tile) const
    {
        return (m_start % p.Wo + M_tile - 1) / p.Wo;
    }
};

TEST_F(Im2colMTilePattern, StepWAndWrapDeltaValues)
{
    // step_w = SW * WiStride:  M_base change per +1 in wo (within same ho row)
    // wrap_delta: M_base correction when crossing a ho-row boundary
    int step_w     = p.SW * p.WiStride();
    int wrap_delta = p.SH * p.HiStride() - p.Wo * p.SW * p.WiStride();

    EXPECT_EQ(step_w,    p.WiStride());
    EXPECT_EQ(wrap_delta, p.HiStride() - p.Wo * p.WiStride());
    EXPECT_GT(wrap_delta, 0) << "wrap_delta must be positive";
}

TEST_F(Im2colMTilePattern, SingleHoTileIsLinear)
{
    // When no ho-boundary is crossed: M_base[i] = M_base[0] + i * step_w
    int M_tile = 4;
    int step_w = p.SW * p.WiStride();
    bool tested_any = false;

    for(int block = 0; block * M_tile < p.M_gemm(); ++block)
    {
        int m_start = block * M_tile;
        if(count_ho_boundaries(m_start, M_tile) != 0) continue;

        int base0 = M_base(p, m_start);
        for(int i = 0; i < M_tile && m_start + i < p.M_gemm(); ++i)
        {
            EXPECT_EQ(M_base(p, m_start + i), base0 + i * step_w)
                << "block=" << block << " i=" << i;
        }
        tested_any = true;
    }
    EXPECT_TRUE(tested_any) << "No single-ho tile found; increase conv size";
}

TEST_F(Im2colMTilePattern, DualHoTileHasWrapDelta)
{
    // When one ho-boundary is crossed: M_base[i] includes a wrap_delta correction
    int M_tile     = 4;
    int step_w     = p.SW * p.WiStride();
    int wrap_delta = p.SH * p.HiStride() - p.Wo * p.SW * p.WiStride();
    bool tested_any = false;

    for(int block = 0; block * M_tile < p.M_gemm(); ++block)
    {
        int m_start  = block * M_tile;
        int wo_start = m_start % p.Wo;
        if(count_ho_boundaries(m_start, M_tile) != 1) continue;

        int boundary = p.Wo - wo_start; // row index of the first wrap
        int base0    = M_base(p, m_start);

        for(int i = 0; i < M_tile && m_start + i < p.M_gemm(); ++i)
        {
            int correction = (i >= boundary) ? wrap_delta : 0;
            EXPECT_EQ(M_base(p, m_start + i), base0 + i * step_w + correction)
                << "block=" << block << " i=" << i << " wo_start=" << wo_start;
        }
        tested_any = true;
    }
    EXPECT_TRUE(tested_any) << "No dual-ho tile found; increase conv size";
}

TEST_F(Im2colMTilePattern, BoundaryCountFormula)
{
    // Verify: B_total = floor((wo_start + M_tile - 1) / Wo)
    int M_tile = 4;
    for(int block = 0; block * M_tile < p.M_gemm(); ++block)
    {
        int m_start  = block * M_tile;
        int formula  = count_ho_boundaries(m_start, M_tile);

        // Count actual ho-row crossings
        int ho_prev = decode_m(p, m_start).ho;
        int actual  = 0;
        for(int i = 1; i < M_tile && m_start + i < p.M_gemm(); ++i)
        {
            int ho_cur = decode_m(p, m_start + i).ho;
            if(ho_cur != ho_prev) { actual++; ho_prev = ho_cur; }
        }
        EXPECT_EQ(formula, actual) << "block=" << block;
    }
}

TEST_F(Im2colMTilePattern, GeneralFormulaWithNWraps)
{
    // M_base[i] = M_base[0] + i*step_w + floor((wo_start+i)/Wo) * wrap_delta
    // holds for any number of boundary crossings
    int step_w     = p.SW * p.WiStride();
    int wrap_delta = p.SH * p.HiStride() - p.Wo * p.SW * p.WiStride();

    // Use a large M_tile to force multiple crossings
    int M_tile = p.Wo + 3; // guaranteed to cross at least one boundary

    // Verify the formula for m_start=0 (one case is sufficient)
    ASSERT_LE(M_tile, p.M_gemm()) << "M_tile too large for test params";
    {
        int m_start  = 0;
        int wo_start = m_start % p.Wo;
        int base0    = M_base(p, m_start);

        for(int i = 0; i < M_tile; ++i)
        {
            int n_wraps  = (wo_start + i) / p.Wo;
            int expected = base0 + i * step_w + n_wraps * wrap_delta;
            EXPECT_EQ(M_base(p, m_start + i), expected)
                << "m_start=" << m_start << " i=" << i;
        }
    }
}

TEST_F(Im2colMTilePattern, NConvConstantWithinTile)
{
    // n_conv should be constant within any tile when M_tile << Ho*Wo
    int M_tile = 4;
    for(int block = 0; block * M_tile + M_tile - 1 < p.M_gemm(); ++block)
    {
        int m_start = block * M_tile;
        int n0 = decode_m(p, m_start).n_conv;
        for(int i = 1; i < M_tile; ++i)
        {
            int n_cur = decode_m(p, m_start + i).n_conv;
            // n_conv changes only at a Ho*Wo boundary; very rare for small M_tile
            if(n_cur != n0)
            {
                // Verify that this crossing is at the expected boundary
                int boundary = p.Ho * p.Wo - (m_start % (p.Ho * p.Wo));
                EXPECT_EQ(i, boundary)
                    << "n_conv changed unexpectedly at block=" << block << " i=" << i;
            }
        }
    }
}

TEST_F(Im2colMTilePattern, BoundaryProbability)
{
    // P(B_total=0) = (Wo - M_tile + 1) / Wo  for M_tile ≤ Wo,
    // when wo_start is uniform over [0, Wo).  Verify by counting directly
    // over all distinct wo_start values.
    int M_tile = 4;
    ASSERT_LE(M_tile, p.Wo);

    int single_ho = 0;
    for(int wo_start = 0; wo_start < p.Wo; ++wo_start)
        if((wo_start + M_tile - 1) / p.Wo == 0) single_ho++;

    double expected = double(p.Wo - M_tile + 1) / p.Wo;
    double actual   = double(single_ho) / p.Wo;
    EXPECT_NEAR(actual, expected, 1e-9)
        << "single_ho=" << single_ho << " Wo=" << p.Wo;
}

// ===========================================================================
// TEST SUITE 4: K-tile Pattern
// ===========================================================================

class Im2colKTilePattern : public ::testing::Test
{
    protected:
    Conv2dParams p = make_unit_params();
};

TEST_F(Im2colKTilePattern, YConstantForKTileEqualC)
{
    // K_tile = C: exactly one y value, one x value per tile
    int K_tile = p.C;
    for(int kt = 0; kt * K_tile < p.K_gemm(); ++kt)
    {
        int k_start = kt * K_tile;
        auto [y0, x0, c0] = decode_k(p, k_start);
        for(int j = 0; j < K_tile; ++j)
        {
            EXPECT_EQ(decode_k(p, k_start + j).y, y0) << "kt=" << kt << " j=" << j;
            EXPECT_EQ(decode_k(p, k_start + j).x, x0) << "kt=" << kt << " j=" << j;
        }
    }
}

TEST_F(Im2colKTilePattern, YConstantForKTileEqual2C)
{
    // K_tile = 2*C keeps y constant only when the tile does not cross a y-boundary
    // (i.e., k_start is a multiple of X*C and K_tile ≤ X*C).
    // For tiles aligned to X*C boundaries with K_tile=2C < X*C=X*C:
    //   - y is constant within the tile
    //   - there are exactly 2 distinct x values (x_start and x_start+1)
    int K_tile = 2 * p.C;
    int XC     = p.X * p.C;
    if(K_tile > p.K_gemm()) GTEST_SKIP() << "K_tile exceeds K_gemm";
    if(K_tile >= XC) GTEST_SKIP() << "K_tile >= X*C; y may change within tile";

    // Only test tiles aligned to X*C boundaries so y is guaranteed constant
    for(int k_start = 0; k_start + K_tile <= p.K_gemm(); k_start += XC)
    {
        int y0        = decode_k(p, k_start).y;
        int x_prev    = -1;
        int x_changes = 0;

        for(int j = 0; j < K_tile; ++j)
        {
            auto [y, x, c] = decode_k(p, k_start + j);
            EXPECT_EQ(y, y0) << "k_start=" << k_start << " j=" << j;
            if(x != x_prev) { x_changes++; x_prev = x; }
        }
        EXPECT_EQ(x_changes, 2) << "k_start=" << k_start << " should have exactly 2 x-values";
    }
}

TEST_F(Im2colKTilePattern, KOffsetIncrementsByOnePerC)
{
    // Within one x-column (K_tile = C): K_offset increments by 1 (CStride) each step
    int K_tile = p.C;
    for(int kt = 0; kt * K_tile < p.K_gemm(); ++kt)
    {
        int k_start = kt * K_tile;
        int off0    = K_offset(p, k_start);
        for(int j = 0; j < K_tile; ++j)
            EXPECT_EQ(K_offset(p, k_start + j), off0 + j) << "kt=" << kt << " j=" << j;
    }
}

TEST_F(Im2colKTilePattern, KOffsetXTransitionStep)
{
    // At each pure x-transition (k = multiple of C but NOT multiple of X*C):
    // ΔK_offset = DW * WiStride - (C-1)   (x += 1, c resets from C-1 to 0)
    // Skip y-transitions (multiples of X*C) which have a different delta.
    int expected_delta = p.DW * p.WiStride() - (p.C - 1);
    int XC = p.X * p.C;
    for(int k = p.C; k < p.K_gemm(); k += p.C)
    {
        if(k % XC == 0) continue; // y-transition, not a pure x-transition
        int delta = K_offset(p, k) - K_offset(p, k - 1);
        EXPECT_EQ(delta, expected_delta) << "x-transition at k=" << k;
    }
}

TEST_F(Im2colKTilePattern, KOffsetYTransitionStep)
{
    // At each y-transition (k = multiple of X*C, k > 0):
    // ΔK_offset = DH*HiStride - (X-1)*DW*WiStride - (C-1)
    int expected_delta = p.DH * p.HiStride()
                       - (p.X - 1) * p.DW * p.WiStride()
                       - (p.C - 1);
    for(int k = p.X * p.C; k < p.K_gemm(); k += p.X * p.C)
    {
        int delta = K_offset(p, k) - K_offset(p, k - 1);
        EXPECT_EQ(delta, expected_delta) << "y-transition at k=" << k;
    }
}

// ===========================================================================
// TEST SUITE 5: Stride and Dilation — decomposition still holds
// ===========================================================================

class Im2colStrideDialation : public ::testing::Test {};

TEST_F(Im2colStrideDialation, Stride2DecompositionVsReference)
{
    verify_against_reference(make_stride2_params());
}

TEST_F(Im2colStrideDialation, Dilation2DecompositionVsReference)
{
    verify_against_reference(make_dilation2_params());
}

TEST_F(Im2colStrideDialation, Stride2StepW)
{
    // step_w = SW * WiStride = 2 * WiStride for SH=SW=2
    Conv2dParams p = make_stride2_params();
    int step_w = p.SW * p.WiStride();
    EXPECT_EQ(step_w, 2 * p.WiStride());

    // Verify linearity within a single-ho tile
    int M_tile = 2;
    for(int block = 0; block * M_tile < p.M_gemm(); ++block)
    {
        int m_start  = block * M_tile;
        if((m_start % p.Wo + M_tile - 1) / p.Wo != 0) continue;
        int base0 = M_base(p, m_start);
        for(int i = 0; i < M_tile; ++i)
            EXPECT_EQ(M_base(p, m_start + i), base0 + i * step_w)
                << "block=" << block << " i=" << i;
        break;
    }
}

TEST_F(Im2colStrideDialation, Dilation2KOffsetYStep)
{
    // y-step in K_offset = DH * HiStride = 2 * HiStride for DH=2
    Conv2dParams p = make_dilation2_params();
    int k_y1 = p.X * p.C; // first k with y=1
    int k_y0 = k_y1 - 1;  // last k with y=0
    int delta = K_offset(p, k_y1) - K_offset(p, k_y0);
    int expected = p.DH * p.HiStride() - (p.X - 1) * p.DW * p.WiStride() - (p.C - 1);
    EXPECT_EQ(delta, expected);
}

// ===========================================================================
// TEST SUITE 6: Padding — validity and decomposition
// ===========================================================================

class Im2colPadding : public ::testing::Test
{
    protected:
    Conv2dParams p = make_padded_params();
};

TEST_F(Im2colPadding, ValidityMatchesReference)
{
    for(int m = 0; m < p.M_gemm(); ++m)
        for(int k = 0; k < p.K_gemm(); ++k)
            EXPECT_EQ(is_valid(p, m, k), reference_offset(p, m, k) != -1)
                << "m=" << m << " k=" << k;
}

TEST_F(Im2colPadding, CornerElementsInvalid)
{
    // (ho=0, wo=0, y=0, x=0) maps to ih=-PH, iw=-PW → padded
    EXPECT_FALSE(is_valid(p, 0, 0));
    // reference agrees
    EXPECT_EQ(reference_offset(p, 0, 0), -1);
}

TEST_F(Im2colPadding, CenterFilterValid)
{
    // (ho=0, wo=0, y=PH, x=PW) maps to ih=0, iw=0 → valid
    int k_center = p.PH * p.X * p.C + p.PW * p.C; // y=PH, x=PW, c=0
    EXPECT_TRUE(is_valid(p, 0, k_center));
    EXPECT_NE(reference_offset(p, 0, k_center), -1);
}

TEST_F(Im2colPadding, ValidElementsDecomposeCorrectly)
{
    verify_against_reference(p);
}

// ===========================================================================
// TEST SUITE 7: Precomputed M_base tile array
// ===========================================================================

class Im2colPrecomputation : public ::testing::Test
{
    protected:
    Conv2dParams p = make_unit_params();

    // Precompute M_base[0..M_tile-1] for a tile starting at m_start
    // using the formula: M_base[i] = base0 + i*step_w + floor((wo_start+i)/Wo)*wrap_delta
    std::vector<int> precompute_M_base_array(int m_start, int M_tile) const
    {
        int wo_start   = m_start % p.Wo;
        int step_w     = p.SW * p.WiStride();
        int wrap_delta = p.SH * p.HiStride() - p.Wo * p.SW * p.WiStride();

        auto [n0, ho0, wo0] = decode_m(p, m_start);
        int base0 = n0 * p.NStride()
                  + ho0 * p.SH * p.HiStride()
                  + wo0 * p.SW * p.WiStride()
                  - p.PH * p.HiStride()
                  - p.PW * p.WiStride();

        std::vector<int> bases(M_tile);
        for(int i = 0; i < M_tile; ++i)
            bases[i] = base0 + i * step_w + ((wo_start + i) / p.Wo) * wrap_delta;
        return bases;
    }
};

TEST_F(Im2colPrecomputation, PrecomputedArrayMatchesMBase)
{
    int M_tile = 4;
    for(int block = 0; block * M_tile + M_tile - 1 < p.M_gemm(); ++block)
    {
        int m_start = block * M_tile;
        auto arr    = precompute_M_base_array(m_start, M_tile);
        for(int i = 0; i < M_tile; ++i)
            EXPECT_EQ(arr[i], M_base(p, m_start + i)) << "block=" << block << " i=" << i;
    }
}

TEST_F(Im2colPrecomputation, PrecomputedPlusKOffsetMatchesReference)
{
    // Full round-trip: precomputed M_base + K_offset == TransformConvFwdToGemm reference
    int M_tile = 4;
    for(int block = 0; block * M_tile + M_tile - 1 < p.M_gemm(); ++block)
    {
        int m_start = block * M_tile;
        auto arr    = precompute_M_base_array(m_start, M_tile);

        for(int i = 0; i < M_tile; ++i)
        {
            int m = m_start + i;
            for(int k = 0; k < p.K_gemm(); ++k)
            {
                int ref = reference_offset(p, m, k);
                if(ref == -1) continue;
                EXPECT_EQ(arr[i] + K_offset(p, k), ref)
                    << "block=" << block << " i=" << i << " k=" << k;
            }
        }
    }
}

// ===========================================================================
// TEST SUITE 8: TiledIm2ColMetadata and TiledIm2ColCoordinate
// ===========================================================================
//
// Tests for the new tile-aware coordinate types.  MakeATileMetadata() on the
// V2 transformer produces a TiledIm2ColMetadata struct.  TiledIm2ColCoordinate
// uses that metadata to compute offset(m, k) = M_base(m) + K_offset(k).
//
// These tests verify that:
//   1. MakeATileMetadata() produces correct constants.
//   2. TiledIm2ColCoordinate::init() gives the same offset as TransformConvFwdToGemm.
//   3. TiledIm2ColCoordinate::move_k() updates K_offset correctly.
//   4. Validity agrees with reference.

using V2Transformer = TransformConvFwdToGemm_V2<
    2,
    ConvolutionSpecialization::Default,
    /*VectorSizeA=*/1,
    /*VectorSizeB=*/1,
    /*VectorSizeC=*/1,
    /*NumGroupsToMerge=*/1,
    /*SplitN=*/false,
    float,
    float,
    int>;

V2Transformer make_v2_transformer(const Conv2dParams& p)
{
    std::array<int, 5> a_lens = {p.G, p.N, p.C, p.Hi, p.Wi};
    std::array<int, 5> b_lens = {p.G, p.K, p.C, p.Y,  p.X};
    std::array<int, 5> c_lens = {p.G, p.N, p.K, p.Ho, p.Wo};
    std::array<int, 2> strides    = {p.SH, p.SW};
    std::array<int, 2> dilations  = {p.DH, p.DW};
    std::array<int, 2> left_pads  = {p.PH, p.PW};
    std::array<int, 2> right_pads = {p.PH, p.PW};
    return V2Transformer{a_lens, b_lens, c_lens, strides, dilations, left_pads, right_pads};
}

class Im2colTiledCoordinate : public ::testing::Test
{
    protected:
    Conv2dParams p = make_unit_params(); // unit stride, no padding
};

// ---- 8a. MakeATileMetadata constants ---

TEST_F(Im2colTiledCoordinate, MetadataConstants)
{
    auto v2   = make_v2_transformer(p);
    auto meta = v2.MakeATileMetadata<tensor_layout::convolution::NHWGC>();

    // Verify derived constants match the analysis formulas
    EXPECT_EQ(meta.WiStride,    p.WiStride());
    EXPECT_EQ(meta.HiStride,    p.HiStride());
    EXPECT_EQ(meta.NStride,     p.NStride());
    EXPECT_EQ(meta.step_w,      p.SW * p.WiStride());
    EXPECT_EQ(meta.SH_HiStride, p.SH * p.HiStride());
    EXPECT_EQ(meta.wrap_delta,  p.SH * p.HiStride() - p.Wo * p.SW * p.WiStride());
    EXPECT_EQ(meta.pad_offset,  p.PH * p.HiStride() + p.PW * p.WiStride());
    EXPECT_EQ(meta.DH_HiStride, p.DH * p.HiStride());
    EXPECT_EQ(meta.DW_WiStride, p.DW * p.WiStride());
    EXPECT_EQ(meta.C,   p.C);
    EXPECT_EQ(meta.XC,  p.X * p.C);
    EXPECT_EQ(meta.Wo,  p.Wo);
    EXPECT_EQ(meta.HoWo, p.Ho * p.Wo);
    EXPECT_EQ(meta.Hi,  p.Hi);
    EXPECT_EQ(meta.Wi,  p.Wi);
    EXPECT_EQ(meta.SH,  p.SH);
    EXPECT_EQ(meta.SW,  p.SW);
    EXPECT_EQ(meta.DH,  p.DH);
    EXPECT_EQ(meta.DW,  p.DW);
    EXPECT_EQ(meta.PH,  p.PH);
    EXPECT_EQ(meta.PW,  p.PW);
}

TEST_F(Im2colTiledCoordinate, MetadataConstantsWithPadding)
{
    auto pp   = make_padded_params();
    auto v2   = make_v2_transformer(pp);
    auto meta = v2.MakeATileMetadata<tensor_layout::convolution::NHWGC>();

    EXPECT_EQ(meta.pad_offset, pp.PH * pp.HiStride() + pp.PW * pp.WiStride());
    EXPECT_EQ(meta.PH, pp.PH);
    EXPECT_EQ(meta.PW, pp.PW);
}

// ---- 8b. TiledIm2ColCoordinate::init() vs. reference ---

TEST_F(Im2colTiledCoordinate, InitOffsetMatchesReference)
{
    auto v2   = make_v2_transformer(p);
    auto meta = v2.MakeATileMetadata<tensor_layout::convolution::NHWGC>();

    for(int m = 0; m < p.M_gemm(); ++m)
    {
        for(int k = 0; k < p.K_gemm(); ++k)
        {
            int ref = reference_offset(p, m, k);
            if(ref == -1) continue; // padded

            TiledIm2ColCoordinate coord;
            coord.init(m, k, meta);

            EXPECT_EQ(coord.get_offset(), ref) << "m=" << m << " k=" << k;
            EXPECT_TRUE(coord.is_valid())      << "m=" << m << " k=" << k;
        }
    }
}

TEST_F(Im2colTiledCoordinate, InitValidityMatchesReference)
{
    auto pp   = make_padded_params();
    auto v2   = make_v2_transformer(pp);
    auto meta = v2.MakeATileMetadata<tensor_layout::convolution::NHWGC>();

    for(int m = 0; m < pp.M_gemm(); ++m)
    {
        for(int k = 0; k < pp.K_gemm(); ++k)
        {
            int ref = reference_offset(pp, m, k);

            TiledIm2ColCoordinate coord;
            coord.init(m, k, meta);

            EXPECT_EQ(coord.is_valid(), ref != -1) << "m=" << m << " k=" << k;
            if(ref != -1)
                EXPECT_EQ(coord.get_offset(), ref) << "m=" << m << " k=" << k;
        }
    }
}

TEST_F(Im2colTiledCoordinate, InitWithStride2)
{
    auto ps   = make_stride2_params();
    auto v2   = make_v2_transformer(ps);
    auto meta = v2.MakeATileMetadata<tensor_layout::convolution::NHWGC>();

    for(int m = 0; m < ps.M_gemm(); ++m)
        for(int k = 0; k < ps.K_gemm(); ++k)
        {
            int ref = reference_offset(ps, m, k);
            if(ref == -1) continue;
            TiledIm2ColCoordinate coord;
            coord.init(m, k, meta);
            EXPECT_EQ(coord.get_offset(), ref) << "m=" << m << " k=" << k;
        }
}

TEST_F(Im2colTiledCoordinate, InitWithDilation2)
{
    auto pd   = make_dilation2_params();
    auto v2   = make_v2_transformer(pd);
    auto meta = v2.MakeATileMetadata<tensor_layout::convolution::NHWGC>();

    for(int m = 0; m < pd.M_gemm(); ++m)
        for(int k = 0; k < pd.K_gemm(); ++k)
        {
            int ref = reference_offset(pd, m, k);
            if(ref == -1) continue;
            TiledIm2ColCoordinate coord;
            coord.init(m, k, meta);
            EXPECT_EQ(coord.get_offset(), ref) << "m=" << m << " k=" << k;
        }
}

// ---- 8c. TiledIm2ColCoordinate::move_k() ---

TEST_F(Im2colTiledCoordinate, MoveKUpdatesKOffset)
{
    // init at k=0, then move_k to successive k values and compare to reference
    auto v2   = make_v2_transformer(p);
    auto meta = v2.MakeATileMetadata<tensor_layout::convolution::NHWGC>();

    for(int m = 0; m < p.M_gemm(); m += p.Wo) // spot-check at ho-boundaries
    {
        TiledIm2ColCoordinate coord;
        coord.init(m, 0, meta);
        const int m_base = coord.M_base;

        for(int k = 0; k < p.K_gemm(); ++k)
        {
            coord.move_k(k, m, meta);
            int ref = reference_offset(p, m, k);
            if(ref == -1) continue;

            EXPECT_EQ(coord.M_base, m_base) // M_base must be unchanged
                << "M_base changed at m=" << m << " k=" << k;
            EXPECT_EQ(coord.get_offset(), ref)
                << "m=" << m << " k=" << k;
        }
    }
}

// ---- 8d. M_base is constant across all k (move_k must not touch M_base) ---

TEST_F(Im2colTiledCoordinate, MBaseConstantAcrossKMoves)
{
    auto v2   = make_v2_transformer(p);
    auto meta = v2.MakeATileMetadata<tensor_layout::convolution::NHWGC>();

    for(int m = 0; m < p.M_gemm(); ++m)
    {
        TiledIm2ColCoordinate coord;
        coord.init(m, 0, meta);
        const int m_base_ref = coord.M_base;

        for(int k = p.C; k < p.K_gemm(); k += p.C)
        {
            coord.move_k(k, m, meta);
            EXPECT_EQ(coord.M_base, m_base_ref) << "m=" << m << " k=" << k;
        }
    }
}
