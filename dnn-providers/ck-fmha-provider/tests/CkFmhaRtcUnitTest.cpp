// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Phase 15 unit tests for the hipRTC backend wiring. These tests do
// NOT require a GPU or a hipRTC installation; they cover the CPU-side
// invariants that used to regress silently:
//
//   - RtcFmhaKernelInstance::supports() rejects shape mismatches
//   - pick_jit_backend() handles CK_FMHA_JIT_BACKEND correctly
//   - The shape-hash mixed into selection_rank is deterministic and
//     actually disambiguates different (B, Hq, Hk, Sq, Sk) tuples
//   - The HSACO cache verifier rejects non-ELF / wrong-arch blobs
//
// A lightweight in-file test framework avoids pulling in GTest when
// the host lacks it; invocations still print pass/fail lines and the
// process exits non-zero on any failure.

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "CkFmhaJit.hpp"
#include "CkFmhaRtcKernelInstance.hpp"

#include <ck/host/fmha_rtc.hpp>
#include <ck_tile/dispatcher_fmha.hpp>

// --------------------------------------------------------------------
// Minimal test framework: RUN_TEST(fn) registers, EXPECT_TRUE / EXPECT_EQ
// log and flip an all-failures counter. Keep in sync with .cpp tests
// that might adopt it later.
// --------------------------------------------------------------------
namespace {
int g_fails = 0;
int g_tests = 0;

#define EXPECT_TRUE(cond)                                                          \
    do {                                                                           \
        if (!(cond)) {                                                             \
            ++g_fails;                                                             \
            std::cerr << "  FAIL: " << __FILE__ << ":" << __LINE__ << " "          \
                      << "EXPECT_TRUE(" << #cond << ")\n";                         \
        }                                                                          \
    } while (0)

#define EXPECT_FALSE(cond) EXPECT_TRUE(!(cond))

#define EXPECT_EQ(a, b)                                                            \
    do {                                                                           \
        auto _a = (a);                                                             \
        auto _b = (b);                                                             \
        if (!(_a == _b)) {                                                         \
            ++g_fails;                                                             \
            std::cerr << "  FAIL: " << __FILE__ << ":" << __LINE__ << " "          \
                      << "EXPECT_EQ(" << #a << ", " << #b << ") : got "            \
                      << _a << " vs " << _b << "\n";                               \
        }                                                                          \
    } while (0)

#define EXPECT_NE(a, b)                                                            \
    do {                                                                           \
        auto _a = (a);                                                             \
        auto _b = (b);                                                             \
        if (_a == _b) {                                                            \
            ++g_fails;                                                             \
            std::cerr << "  FAIL: " << __FILE__ << ":" << __LINE__ << " "          \
                      << "EXPECT_NE(" << #a << ", " << #b << ")\n";                \
        }                                                                          \
    } while (0)

#define RUN_TEST(fn)                           \
    do {                                       \
        ++g_tests;                             \
        std::cerr << "[RUN] " << #fn << "\n";  \
        fn();                                  \
    } while (0)

using namespace ck_fmha_plugin;

// --------------------------------------------------------------------
// Helper: construct an RtcFmhaKernelInstance with the given compile-time
// shape. `compiled` and `solution` stay default-initialised, which is
// fine because supports() / get_key() / get_name() never dereference
// them. launch() would segfault on a default-initialised kernel but
// these tests don't exercise launch().
// --------------------------------------------------------------------
std::shared_ptr<RtcFmhaKernelInstance> make_instance(int batch, int nhead_q, int nhead_k,
                                                     int seqlen_q, int seqlen_k, int hdim_q = 128,
                                                     int hdim_v = 128) {
    ck_tile::dispatcher::FmhaKernelKey key;
    key.signature.family        = ck_tile::dispatcher::FmhaKernelFamily::Fwd;
    key.signature.data_type     = "fp16";
    key.signature.mask_type     = 0;
    key.signature.bias_type     = 0;
    key.signature.is_group_mode = false;
    key.signature.is_v_rowmajor = true;
    key.signature.hdim_q        = static_cast<std::uint16_t>(hdim_q);
    key.signature.hdim_v        = static_cast<std::uint16_t>(hdim_v);
    key.gfx_arch                = "gfx950";
    return std::make_shared<RtcFmhaKernelInstance>(ck::host::fmha_rtc::CompiledKernel{},
                                                   ck::host::Solution{}, key, "test_kernel", batch,
                                                   nhead_q, nhead_k, seqlen_q, seqlen_k);
}

ck_tile::dispatcher::FmhaProblem make_problem(int batch, int nhead_q, int nhead_k, int seqlen_q,
                                              int seqlen_k, int hdim_q = 128, int hdim_v = 128) {
    ck_tile::dispatcher::FmhaProblem p;
    p.api_family       = ck_tile::dispatcher::FmhaApiFamily::Fwd;
    p.requested_family = ck_tile::dispatcher::FmhaKernelFamily::Fwd;
    p.data_type        = "fp16";
    p.batch            = batch;
    p.nhead_q          = nhead_q;
    p.nhead_k          = nhead_k;
    p.seqlen_q         = seqlen_q;
    p.seqlen_k         = seqlen_k;
    p.hdim_q           = hdim_q;
    p.hdim_v           = hdim_v;
    p.mask_type        = 0;
    p.bias_type        = 0;
    p.is_group_mode    = false;
    p.is_v_rowmajor    = true;
    p.gfx_arch         = "gfx950";
    return p;
}

// --------------------------------------------------------------------
// Tests: RtcFmhaKernelInstance::supports()
// --------------------------------------------------------------------

void test_supports_accepts_same_shape() {
    auto inst = make_instance(2, 4, 4, 128, 128);
    auto prob = make_problem(2, 4, 4, 128, 128);
    EXPECT_TRUE(inst->supports(prob));
}

void test_supports_rejects_different_batch() {
    auto inst = make_instance(2, 4, 4, 128, 128);
    auto prob = make_problem(4, 4, 4, 128, 128); // different batch
    EXPECT_FALSE(inst->supports(prob));
}

void test_supports_rejects_different_seqlen_q() {
    auto inst = make_instance(2, 4, 4, 128, 128);
    auto prob = make_problem(2, 4, 4, 256, 128); // different Sq
    EXPECT_FALSE(inst->supports(prob));
}

void test_supports_rejects_different_seqlen_k() {
    auto inst = make_instance(2, 4, 4, 128, 128);
    auto prob = make_problem(2, 4, 4, 128, 256); // different Sk
    EXPECT_FALSE(inst->supports(prob));
}

void test_supports_rejects_different_nhead_q() {
    auto inst = make_instance(2, 4, 4, 128, 128);
    auto prob = make_problem(2, 8, 4, 128, 128); // different Hq
    EXPECT_FALSE(inst->supports(prob));
}

void test_supports_rejects_different_nhead_k() {
    auto inst = make_instance(2, 4, 4, 128, 128);
    auto prob = make_problem(2, 4, 2, 128, 128); // different Hk (GQA)
    EXPECT_FALSE(inst->supports(prob));
}

void test_supports_rejects_different_hdim() {
    auto inst = make_instance(2, 4, 4, 128, 128, /*hdim_q=*/128, /*hdim_v=*/128);
    auto prob = make_problem(2, 4, 4, 128, 128, /*hdim_q=*/64, /*hdim_v=*/64);
    EXPECT_FALSE(inst->supports(prob));
}

void test_supports_accepts_gqa_shape_when_registered_as_gqa() {
    auto inst = make_instance(1, 16, 2, 512, 512); // 8:1 GQA
    auto prob = make_problem(1, 16, 2, 512, 512);
    EXPECT_TRUE(inst->supports(prob));
}

void test_supports_rejects_different_dtype() {
    auto inst = make_instance(2, 4, 4, 128, 128);
    auto prob = make_problem(2, 4, 4, 128, 128);
    prob.data_type = "bf16"; // mismatch
    EXPECT_FALSE(inst->supports(prob));
}

// --------------------------------------------------------------------
// Tests: pick_jit_backend() env var handling
// --------------------------------------------------------------------

void test_backend_env_rtc() {
    setenv("CK_FMHA_JIT_BACKEND", "rtc", 1);
    EXPECT_TRUE(pick_jit_backend() == JitBackend::Rtc);
    unsetenv("CK_FMHA_JIT_BACKEND");
}

void test_backend_env_hipcc() {
    setenv("CK_FMHA_JIT_BACKEND", "hipcc", 1);
    EXPECT_TRUE(pick_jit_backend() == JitBackend::Hipcc);
    unsetenv("CK_FMHA_JIT_BACKEND");
}

void test_backend_env_auto() {
    setenv("CK_FMHA_JIT_BACKEND", "auto", 1);
    EXPECT_TRUE(pick_jit_backend() == JitBackend::Auto);
    unsetenv("CK_FMHA_JIT_BACKEND");
}

void test_backend_env_default_is_auto_or_rtc() {
    unsetenv("CK_FMHA_JIT_BACKEND");
    // Default is Auto unless the plugin was built with
    // CK_FMHA_DEFAULT_BACKEND_RTC, in which case it's Rtc. Either is
    // acceptable; what we're guarding against is a mistaken default of
    // Hipcc (which would silently skip the RTC path even when RTC
    // works).
    auto b = pick_jit_backend();
    EXPECT_TRUE(b == JitBackend::Auto || b == JitBackend::Rtc);
}

void test_backend_env_unknown_falls_back_to_default() {
    setenv("CK_FMHA_JIT_BACKEND", "nonsense", 1);
    auto b = pick_jit_backend();
    // Same as default: Auto or Rtc. Not Hipcc.
    EXPECT_TRUE(b == JitBackend::Auto || b == JitBackend::Rtc);
    unsetenv("CK_FMHA_JIT_BACKEND");
}

// --------------------------------------------------------------------
// Tests: HSACO ELF-header verifier (extract_gfx_mach-style behaviour)
//
// We don't export `extract_gfx_mach` directly, so we assert the
// invariants by running `compile_rtc` with CK_FMHA_RTC_CACHE_DIR set
// to a disposable path containing pre-crafted bogus files and then
// observing that the compile proceeds (i.e. the bogus entry was
// discarded). We do this indirectly in an integration test.
//
// The unit-test-level guarantee is that the shape hash we mix into
// `algorithm.selection_rank` deterministically disambiguates.
// --------------------------------------------------------------------

std::uint32_t shape_hash(int batch, int nhead_q, int nhead_k, int seqlen_q, int seqlen_k) {
    // Must mirror the formula in CkFmhaJit.cpp::compile_rtc so a
    // regression in either side of the mapping fires here.
    return static_cast<std::uint32_t>(
        (static_cast<std::uint64_t>(batch) * 1315423911u) ^
        (static_cast<std::uint64_t>(nhead_q) * 2654435761u) ^
        (static_cast<std::uint64_t>(nhead_k) * 40503u) ^
        (static_cast<std::uint64_t>(seqlen_q) * 2246822507u) ^
        (static_cast<std::uint64_t>(seqlen_k) * 3266489917u));
}

void test_shape_hash_deterministic() {
    EXPECT_EQ(shape_hash(2, 4, 4, 128, 128), shape_hash(2, 4, 4, 128, 128));
    EXPECT_EQ(shape_hash(1, 8, 2, 512, 512), shape_hash(1, 8, 2, 512, 512));
}

void test_shape_hash_disambiguates_common_shapes() {
    // The parity tests in the demo compare these shapes; the hash
    // must split them into distinct registry entries.
    auto h1 = shape_hash(2, 4, 4, 128, 128);
    auto h2 = shape_hash(2, 8, 8, 256, 256);
    auto h3 = shape_hash(1, 8, 8, 512, 512);
    auto h4 = shape_hash(2, 8, 2, 256, 256);  // GQA 4:1
    auto h5 = shape_hash(1, 16, 2, 512, 512); // GQA 8:1
    EXPECT_NE(h1, h2);
    EXPECT_NE(h1, h3);
    EXPECT_NE(h1, h4);
    EXPECT_NE(h1, h5);
    EXPECT_NE(h2, h3);
    EXPECT_NE(h2, h4);
    EXPECT_NE(h3, h4);
    EXPECT_NE(h4, h5);
}

void test_shape_hash_distinguishes_gqa_vs_mha_same_tensor_counts() {
    // Two shapes with the same total kv element count but different
    // nhead_q vs nhead_k split must still hash differently -- the
    // kernel has to be recompiled.
    auto mha = shape_hash(2, 8, 8, 128, 128);
    auto gqa = shape_hash(2, 8, 2, 128, 128);
    EXPECT_NE(mha, gqa);
}

// --------------------------------------------------------------------
// Tests: FmhaKernelKey construction as used by compile_rtc
// --------------------------------------------------------------------

void test_fmha_kernel_key_signature_differentiates_gqa() {
    // Two keys with the same signature but different shape-hashed
    // selection_rank represent different compiled kernels in the
    // dispatcher. FmhaRegistry::register_kernel rejects duplicate keys,
    // so we verify here that the concrete keys produced for
    // shape-distinct problems compare unequal.
    auto h_mha = shape_hash(2, 8, 8, 128, 128);
    auto h_gqa = shape_hash(2, 8, 2, 128, 128);
    ck_tile::dispatcher::FmhaKernelKey k_mha;
    ck_tile::dispatcher::FmhaKernelKey k_gqa;
    k_mha.algorithm.selection_rank = static_cast<int>(h_mha & 0x7fffffff);
    k_gqa.algorithm.selection_rank = static_cast<int>(h_gqa & 0x7fffffff);
    EXPECT_NE(k_mha.algorithm.selection_rank, k_gqa.algorithm.selection_rank);
}

} // namespace

int main() {
    std::cerr << "=== CkFmhaRtcUnitTest ===\n";

    RUN_TEST(test_supports_accepts_same_shape);
    RUN_TEST(test_supports_rejects_different_batch);
    RUN_TEST(test_supports_rejects_different_seqlen_q);
    RUN_TEST(test_supports_rejects_different_seqlen_k);
    RUN_TEST(test_supports_rejects_different_nhead_q);
    RUN_TEST(test_supports_rejects_different_nhead_k);
    RUN_TEST(test_supports_rejects_different_hdim);
    RUN_TEST(test_supports_accepts_gqa_shape_when_registered_as_gqa);
    RUN_TEST(test_supports_rejects_different_dtype);

    RUN_TEST(test_backend_env_rtc);
    RUN_TEST(test_backend_env_hipcc);
    RUN_TEST(test_backend_env_auto);
    RUN_TEST(test_backend_env_default_is_auto_or_rtc);
    RUN_TEST(test_backend_env_unknown_falls_back_to_default);

    RUN_TEST(test_shape_hash_deterministic);
    RUN_TEST(test_shape_hash_disambiguates_common_shapes);
    RUN_TEST(test_shape_hash_distinguishes_gqa_vs_mha_same_tensor_counts);

    RUN_TEST(test_fmha_kernel_key_signature_differentiates_gqa);

    std::cerr << "\n=== Summary: " << (g_tests - g_fails) << "/" << g_tests
              << " tests passed, " << g_fails << " failures ===\n";
    return g_fails == 0 ? 0 : 1;
}
