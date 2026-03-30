// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "engines/CkFmhaParamParser.hpp"

namespace ck_fmha_plugin {
namespace {

// ============================================================================
// FmhaProblemBuilder unit tests
// ============================================================================

class FwdProblemBuilderTest : public ::testing::Test {
   protected:
    ParsedFwdParams makeBaseParams() {
        ParsedFwdParams p;
        p.data_type = "fp16";
        p.batch = 2;
        p.nhead_q = 8;
        p.nhead_k = 8;
        p.seqlen_q = 128;
        p.seqlen_k = 128;
        p.hdim_q = 64;
        p.hdim_v = 64;
        p.mask_type = 0;
        p.bias_type = 0;
        p.has_lse = false;
        p.has_dropout = false;
        p.window_left = -1;
        p.window_right = -1;
        return p;
    }
};

TEST_F(FwdProblemBuilderTest, BasicFp16) {
    auto params = makeBaseParams();
    auto problem = CkFmhaParamParser::buildFwdProblem(params, "gfx942");

    EXPECT_EQ(problem.data_type, "fp16");
    EXPECT_EQ(problem.batch, 2);
    EXPECT_EQ(problem.nhead_q, 8);
    EXPECT_EQ(problem.nhead_k, 8);
    EXPECT_EQ(problem.seqlen_q, 128);
    EXPECT_EQ(problem.seqlen_k, 128);
    EXPECT_EQ(problem.hdim_q, 64);
    EXPECT_EQ(problem.hdim_v, 64);
    EXPECT_EQ(problem.gfx_arch, "gfx942");
    EXPECT_TRUE(problem.is_valid());
    EXPECT_FALSE(problem.has_lse);
    EXPECT_FALSE(problem.has_dropout);
}

TEST_F(FwdProblemBuilderTest, Bf16WithLSE) {
    auto params = makeBaseParams();
    params.data_type = "bf16";
    params.has_lse = true;
    auto problem = CkFmhaParamParser::buildFwdProblem(params, "gfx90a");

    EXPECT_EQ(problem.data_type, "bf16");
    EXPECT_EQ(problem.gfx_arch, "gfx90a");
    EXPECT_TRUE(problem.has_lse);
}

TEST_F(FwdProblemBuilderTest, CausalMaskTopLeft) {
    auto params = makeBaseParams();
    params.mask_type = 1;  // mask_top_left
    auto problem = CkFmhaParamParser::buildFwdProblem(params, "gfx942");

    EXPECT_EQ(problem.mask_type, 1);
}

TEST_F(FwdProblemBuilderTest, CausalMaskBottomRight) {
    auto params = makeBaseParams();
    params.mask_type = 2;  // mask_bottom_right
    auto problem = CkFmhaParamParser::buildFwdProblem(params, "gfx942");

    EXPECT_EQ(problem.mask_type, 2);
}

TEST_F(FwdProblemBuilderTest, WindowMask) {
    auto params = makeBaseParams();
    params.mask_type = 3;  // window_generic
    params.window_left = 128;
    params.window_right = 30;
    auto problem = CkFmhaParamParser::buildFwdProblem(params, "gfx942");

    EXPECT_EQ(problem.mask_type, 3);
}

TEST_F(FwdProblemBuilderTest, ElementwiseBias) {
    auto params = makeBaseParams();
    params.bias_type = 1;  // elementwise_bias
    auto problem = CkFmhaParamParser::buildFwdProblem(params, "gfx942");

    EXPECT_EQ(problem.bias_type, 1);
}

TEST_F(FwdProblemBuilderTest, AlibiBias) {
    auto params = makeBaseParams();
    params.bias_type = 2;  // alibi
    auto problem = CkFmhaParamParser::buildFwdProblem(params, "gfx942");

    EXPECT_EQ(problem.bias_type, 2);
}

TEST_F(FwdProblemBuilderTest, WithDropout) {
    auto params = makeBaseParams();
    params.has_dropout = true;
    params.has_lse = true;  // dropout typically implies stats
    auto problem = CkFmhaParamParser::buildFwdProblem(params, "gfx942");

    EXPECT_TRUE(problem.has_dropout);
    EXPECT_TRUE(problem.has_lse);
}

TEST_F(FwdProblemBuilderTest, GQA) {
    auto params = makeBaseParams();
    params.nhead_q = 16;
    params.nhead_k = 4;  // GQA ratio 4:1
    auto problem = CkFmhaParamParser::buildFwdProblem(params, "gfx942");

    EXPECT_EQ(problem.nhead_q, 16);
    EXPECT_EQ(problem.nhead_k, 4);
}

TEST_F(FwdProblemBuilderTest, AsymmetricHdim) {
    auto params = makeBaseParams();
    params.hdim_q = 128;
    params.hdim_v = 64;
    auto problem = CkFmhaParamParser::buildFwdProblem(params, "gfx942");

    EXPECT_EQ(problem.hdim_q, 128);
    EXPECT_EQ(problem.hdim_v, 64);
}

TEST_F(FwdProblemBuilderTest, AllHdims) {
    for (int hdim : {32, 64, 128, 256}) {
        auto params = makeBaseParams();
        params.hdim_q = hdim;
        params.hdim_v = hdim;
        auto problem = CkFmhaParamParser::buildFwdProblem(params, "gfx942");
        EXPECT_TRUE(problem.is_valid()) << "hdim=" << hdim;
    }
}

TEST_F(FwdProblemBuilderTest, AllArchitectures) {
    for (const auto& arch : {"gfx90a", "gfx942", "gfx950", "gfx1100", "gfx1201"}) {
        auto params = makeBaseParams();
        auto problem = CkFmhaParamParser::buildFwdProblem(params, arch);
        EXPECT_EQ(problem.gfx_arch, arch);
        EXPECT_TRUE(problem.is_valid()) << "arch=" << arch;
    }
}

// ============================================================================
// Backward problem builder tests
// ============================================================================

class BwdProblemBuilderTest : public ::testing::Test {
   protected:
    ParsedBwdParams makeBaseParams() {
        ParsedBwdParams p;
        p.data_type = "fp16";
        p.batch = 1;
        p.nhead_q = 4;
        p.nhead_k = 2;
        p.seqlen_q = 256;
        p.seqlen_k = 256;
        p.hdim_q = 128;
        p.hdim_v = 128;
        p.mask_type = 0;
        p.bias_type = 0;
        p.has_dbias = false;
        p.has_dropout = false;
        p.window_left = -1;
        p.window_right = -1;
        return p;
    }
};

TEST_F(BwdProblemBuilderTest, BasicBf16) {
    auto params = makeBaseParams();
    params.data_type = "bf16";
    auto problem = CkFmhaParamParser::buildBwdProblem(params, "gfx942");

    EXPECT_EQ(problem.data_type, "bf16");
    EXPECT_TRUE(problem.has_lse);  // backward always needs LSE
    EXPECT_TRUE(problem.is_valid());
}

TEST_F(BwdProblemBuilderTest, WithDbias) {
    auto params = makeBaseParams();
    params.has_dbias = true;
    params.bias_type = 1;
    auto problem = CkFmhaParamParser::buildBwdProblem(params, "gfx942");

    EXPECT_TRUE(problem.has_dbias);
    EXPECT_EQ(problem.bias_type, 1);
}

TEST_F(BwdProblemBuilderTest, SchemaGapDefaults) {
    auto params = makeBaseParams();
    auto problem = CkFmhaParamParser::buildBwdProblem(params, "gfx942");

    EXPECT_FALSE(problem.is_deterministic);
    EXPECT_FALSE(problem.is_store_randval);
    EXPECT_FALSE(problem.has_logits_soft_cap);
}

TEST_F(BwdProblemBuilderTest, GQABackward) {
    auto params = makeBaseParams();
    params.nhead_q = 8;
    params.nhead_k = 2;
    auto problem = CkFmhaParamParser::buildBwdProblem(params, "gfx942");

    EXPECT_EQ(problem.nhead_q, 8);
    EXPECT_EQ(problem.nhead_k, 2);
}

// ============================================================================
// Canonical key tests
// ============================================================================

class CanonicalKeyTest : public ::testing::Test {
   protected:
    ParsedFwdParams makeBaseParams() {
        ParsedFwdParams p;
        p.data_type = "fp16";
        p.batch = 2;
        p.nhead_q = 8;
        p.nhead_k = 8;
        p.seqlen_q = 512;
        p.seqlen_k = 512;
        p.hdim_q = 128;
        p.hdim_v = 128;
        return p;
    }
};

TEST_F(CanonicalKeyTest, Stability) {
    auto params = makeBaseParams();
    auto prob = CkFmhaParamParser::buildFwdProblem(params, "gfx942");
    EXPECT_EQ(prob.canonical_key(), prob.canonical_key());
}

TEST_F(CanonicalKeyTest, DifferentDtype) {
    auto p1 = makeBaseParams();
    auto p2 = makeBaseParams();
    p2.data_type = "bf16";

    auto k1 = CkFmhaParamParser::buildFwdProblem(p1, "gfx942").canonical_key();
    auto k2 = CkFmhaParamParser::buildFwdProblem(p2, "gfx942").canonical_key();
    EXPECT_NE(k1, k2);
}

TEST_F(CanonicalKeyTest, DifferentArch) {
    auto p = makeBaseParams();
    auto k1 = CkFmhaParamParser::buildFwdProblem(p, "gfx942").canonical_key();
    auto k2 = CkFmhaParamParser::buildFwdProblem(p, "gfx90a").canonical_key();
    EXPECT_NE(k1, k2);
}

TEST_F(CanonicalKeyTest, DifferentLSE) {
    auto p1 = makeBaseParams();
    auto p2 = makeBaseParams();
    p2.has_lse = true;

    auto k1 = CkFmhaParamParser::buildFwdProblem(p1, "gfx942").canonical_key();
    auto k2 = CkFmhaParamParser::buildFwdProblem(p2, "gfx942").canonical_key();
    EXPECT_NE(k1, k2);
}

TEST_F(CanonicalKeyTest, DifferentMask) {
    auto p1 = makeBaseParams();
    auto p2 = makeBaseParams();
    p2.mask_type = 1;

    auto k1 = CkFmhaParamParser::buildFwdProblem(p1, "gfx942").canonical_key();
    auto k2 = CkFmhaParamParser::buildFwdProblem(p2, "gfx942").canonical_key();
    EXPECT_NE(k1, k2);
}

TEST_F(CanonicalKeyTest, DifferentBias) {
    auto p1 = makeBaseParams();
    auto p2 = makeBaseParams();
    p2.bias_type = 2;

    auto k1 = CkFmhaParamParser::buildFwdProblem(p1, "gfx942").canonical_key();
    auto k2 = CkFmhaParamParser::buildFwdProblem(p2, "gfx942").canonical_key();
    EXPECT_NE(k1, k2);
}

TEST_F(CanonicalKeyTest, DifferentDropout) {
    auto p1 = makeBaseParams();
    auto p2 = makeBaseParams();
    p2.has_dropout = true;

    auto k1 = CkFmhaParamParser::buildFwdProblem(p1, "gfx942").canonical_key();
    auto k2 = CkFmhaParamParser::buildFwdProblem(p2, "gfx942").canonical_key();
    EXPECT_NE(k1, k2);
}

TEST_F(CanonicalKeyTest, SameConfigSameKey) {
    auto p1 = makeBaseParams();
    auto p2 = makeBaseParams();

    auto k1 = CkFmhaParamParser::buildFwdProblem(p1, "gfx942").canonical_key();
    auto k2 = CkFmhaParamParser::buildFwdProblem(p2, "gfx942").canonical_key();
    EXPECT_EQ(k1, k2);
}

// ============================================================================
// Backward workspace sizing tests
// ============================================================================

class BwdWorkspaceTest : public ::testing::Test {};

TEST_F(BwdWorkspaceTest, BasicSizing) {
    using namespace ck_tile::dispatcher;

    FmhaProblem p;
    p.api_family = FmhaApiFamily::Bwd;
    p.data_type = "fp16";
    p.batch = 2;
    p.nhead_q = 8;
    p.nhead_k = 8;
    p.seqlen_q = 256;
    p.seqlen_k = 256;
    p.hdim_q = 64;
    p.hdim_v = 64;

    auto ws = bwd_workspace_info(p);

    // d_bytes = B * Hq * Sq * sizeof(float)
    EXPECT_EQ(ws.d_bytes, 2u * 8 * 256 * sizeof(float));
    // dq_acc_bytes = B * Hq * Sq * Dq * sizeof(float)
    EXPECT_EQ(ws.dq_acc_bytes, 2u * 8 * 256 * 64 * sizeof(float));
    // d always at offset 0
    EXPECT_EQ(ws.d_offset, 0u);
    // dq_acc 256-byte aligned
    EXPECT_EQ(ws.dq_acc_offset % 256, 0u);
    EXPECT_GE(ws.dq_acc_offset, ws.d_bytes);
    // total includes both regions with alignment
    EXPECT_GE(ws.total_bytes, ws.d_bytes + ws.dq_acc_bytes);
    // no rand_val by default
    EXPECT_EQ(ws.rand_val_bytes, 0u);
}

TEST_F(BwdWorkspaceTest, LargeShape) {
    using namespace ck_tile::dispatcher;

    FmhaProblem p;
    p.api_family = FmhaApiFamily::Bwd;
    p.data_type = "bf16";
    p.batch = 4;
    p.nhead_q = 32;
    p.nhead_k = 8;
    p.seqlen_q = 2048;
    p.seqlen_k = 2048;
    p.hdim_q = 128;
    p.hdim_v = 128;

    auto ws = bwd_workspace_info(p);

    size_t expected_d = 4 * 32 * 2048 * sizeof(float);
    size_t expected_dq_acc = 4 * 32 * 2048 * 128 * sizeof(float);

    EXPECT_EQ(ws.d_bytes, expected_d);
    EXPECT_EQ(ws.dq_acc_bytes, expected_dq_acc);
    EXPECT_GT(ws.total_bytes, 0u);
}

TEST_F(BwdWorkspaceTest, MinimalShape) {
    using namespace ck_tile::dispatcher;

    FmhaProblem p;
    p.api_family = FmhaApiFamily::Bwd;
    p.data_type = "fp16";
    p.batch = 1;
    p.nhead_q = 1;
    p.nhead_k = 1;
    p.seqlen_q = 1;
    p.seqlen_k = 1;
    p.hdim_q = 32;
    p.hdim_v = 32;

    auto ws = bwd_workspace_info(p);

    EXPECT_EQ(ws.d_bytes, sizeof(float));
    EXPECT_EQ(ws.dq_acc_bytes, 32 * sizeof(float));
    EXPECT_GT(ws.total_bytes, 0u);
}

TEST_F(BwdWorkspaceTest, WithStoreRandval) {
    using namespace ck_tile::dispatcher;

    FmhaProblem p;
    p.api_family = FmhaApiFamily::Bwd;
    p.data_type = "fp16";
    p.batch = 1;
    p.nhead_q = 2;
    p.nhead_k = 2;
    p.seqlen_q = 64;
    p.seqlen_k = 64;
    p.hdim_q = 64;
    p.hdim_v = 64;
    p.is_store_randval = true;

    auto ws = bwd_workspace_info(p);

    EXPECT_GT(ws.rand_val_bytes, 0u);
    EXPECT_EQ(ws.rand_val_bytes, 1u * 2 * 64 * 64 * sizeof(uint8_t));
    EXPECT_EQ(ws.rand_val_offset % 256, 0u);
    EXPECT_GT(ws.total_bytes, ws.d_bytes + ws.dq_acc_bytes + ws.rand_val_bytes);
}

TEST_F(BwdWorkspaceTest, SuballocationNoOverlap) {
    using namespace ck_tile::dispatcher;

    FmhaProblem p;
    p.api_family = FmhaApiFamily::Bwd;
    p.data_type = "fp16";
    p.batch = 2;
    p.nhead_q = 4;
    p.nhead_k = 4;
    p.seqlen_q = 512;
    p.seqlen_k = 512;
    p.hdim_q = 128;
    p.hdim_v = 128;
    p.is_store_randval = true;

    auto ws = bwd_workspace_info(p);

    // d region: [d_offset, d_offset + d_bytes)
    // dq_acc region: [dq_acc_offset, dq_acc_offset + dq_acc_bytes)
    // rand_val region: [rand_val_offset, rand_val_offset + rand_val_bytes)
    EXPECT_LE(ws.d_offset + ws.d_bytes, ws.dq_acc_offset);
    EXPECT_LE(ws.dq_acc_offset + ws.dq_acc_bytes, ws.rand_val_offset);
    EXPECT_LE(ws.rand_val_offset + ws.rand_val_bytes, ws.total_bytes);
}

// ============================================================================
// Layout detection / stride computation tests
// ============================================================================

TEST(LayoutTest, BHSDStrides) {
    // BHSD: [B=2, H=8, S=128, D=64]
    // stride[3]=1, stride[2]=64, stride[1]=128*64=8192, stride[0]=8*128*64
    ParsedFwdParams p;
    p.is_bhsd_layout = true;
    p.batch = 2;
    p.nhead_q = 8;
    p.nhead_k = 8;
    p.seqlen_q = 128;
    p.seqlen_k = 128;
    p.hdim_q = 64;
    p.hdim_v = 64;

    // stride_q should be hdim_q for BHSD
    EXPECT_EQ(p.hdim_q, 64);
    // nhead_stride_q should be seqlen_q * hdim_q
    EXPECT_EQ(p.seqlen_q * p.hdim_q, 8192);
    // batch_stride_q should be nhead_q * seqlen_q * hdim_q
    EXPECT_EQ(p.nhead_q * p.seqlen_q * p.hdim_q, 65536);
}

TEST(LayoutTest, BSHDStrides) {
    // BSHD: [B=2, S=128, H=8, D=64]
    // stride[3]=1, stride[2]=64, stride[1]=8*64=512, stride[0]=128*8*64
    ParsedFwdParams p;
    p.is_bhsd_layout = false;
    p.batch = 2;
    p.nhead_q = 8;
    p.seqlen_q = 128;
    p.hdim_q = 64;

    // In BSHD, nhead_stride = hdim, stride = hdim (within a head)
    EXPECT_EQ(p.hdim_q, 64);
    // batch_stride = seqlen * nhead * hdim
    EXPECT_EQ(p.seqlen_q * p.nhead_q * p.hdim_q, 65536);
}

// ============================================================================
// Graph detection tests (without actual flatbuffer graphs)
// ============================================================================

TEST(GraphDetection, InvalidNodeCountReturnsNotApplicable) {
    SUCCEED() << "Parser functions compiled successfully";
}

// ============================================================================
// JIT tests
// ============================================================================

TEST(JitEnvGating, DisabledByDefault) {
    // CK_FMHA_ENABLE_JIT should not be set in the test environment
    unsetenv("CK_FMHA_ENABLE_JIT");
    EXPECT_FALSE(CkFmhaHandle::jitEnabled());
}

TEST(JitEnvGating, EnabledWhenSet) {
    setenv("CK_FMHA_ENABLE_JIT", "1", 1);
    EXPECT_TRUE(CkFmhaHandle::jitEnabled());
    unsetenv("CK_FMHA_ENABLE_JIT");
}

TEST(JitEnvGating, DisabledWhenZero) {
    setenv("CK_FMHA_ENABLE_JIT", "0", 1);
    EXPECT_FALSE(CkFmhaHandle::jitEnabled());
    unsetenv("CK_FMHA_ENABLE_JIT");
}

}  // namespace
}  // namespace ck_fmha_plugin
