// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <ck_tile/dispatcher_fmha.hpp>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

#include "CkFmhaContainer.hpp"
#include "CkFmhaHandle.hpp"
#include "engines/CkFmhaParamParser.hpp"
#include "engines/plans/CkFmhaBwdPlan.hpp"
#include "engines/plans/CkFmhaFwdPlan.hpp"

namespace ck_fmha_plugin {
namespace {

#define HIP_CHECK(expr) ASSERT_EQ((expr), hipSuccess) << "HIP error: " << hipGetErrorString(expr)

class GpuBuffer {
   public:
    GpuBuffer() = default;
    explicit GpuBuffer(size_t bytes) : bytes_(bytes) {
        if (bytes > 0) hipMalloc(&ptr_, bytes);
    }
    ~GpuBuffer() {
        if (ptr_ != nullptr) hipFree(ptr_);
    }

    GpuBuffer(const GpuBuffer&) = delete;
    GpuBuffer& operator=(const GpuBuffer&) = delete;
    GpuBuffer(GpuBuffer&& o) noexcept : ptr_(o.ptr_), bytes_(o.bytes_) {
        o.ptr_ = nullptr;
        o.bytes_ = 0;
    }

    void* ptr() const {
        return ptr_;
    }
    size_t bytes() const {
        return bytes_;
    }

    template <typename T>
    void upload(const std::vector<T>& host) {
        hipMemcpy(ptr_, host.data(), host.size() * sizeof(T), hipMemcpyHostToDevice);
    }

    template <typename T>
    std::vector<T> download(size_t count) const {
        std::vector<T> host(count);
        hipMemcpy(host.data(), ptr_, count * sizeof(T), hipMemcpyDeviceToHost);
        return host;
    }

   private:
    void* ptr_ = nullptr;
    size_t bytes_ = 0;
};

class CkFmhaEndToEndTest : public ::testing::Test {
   protected:
    void SetUp() override {
        hipError_t err = hipGetDeviceCount(&device_count_);
        if (err != hipSuccess || device_count_ == 0) GTEST_SKIP() << "No HIP devices available";
        hipSetDevice(0);
    }

    static std::vector<__half> randomHalf(size_t n, float scale = 0.1f) {
        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.0f, scale);
        std::vector<__half> v(n);
        for (auto& x : v) x = __float2half(dist(rng));
        return v;
    }

    static bool hasNaN(const std::vector<__half>& v) {
        return std::any_of(v.begin(), v.end(), [](auto x) { return std::isnan(__half2float(x)); });
    }

    static bool allZero(const std::vector<__half>& v) {
        return std::all_of(v.begin(), v.end(), [](auto x) { return __half2float(x) == 0.0f; });
    }

    int device_count_ = 0;
};

// ============================================================================
// T2.1: Plugin lifecycle
// ============================================================================

TEST_F(CkFmhaEndToEndTest, HandleCreationAndDestruction) {
    CkFmhaHandle handle;

    EXPECT_FALSE(handle.gfxArch().empty());
    EXPECT_NE(handle.dispatcher(), nullptr);
    EXPECT_NE(handle.registry(), nullptr);
    EXPECT_GT(handle.registry()->get_all().size(), 0u)
        << "Registry should have precompiled kernels";
}

TEST_F(CkFmhaEndToEndTest, HandleDetectsArchitecture) {
    CkFmhaHandle handle;
    const auto& arch = handle.gfxArch();

    EXPECT_TRUE(arch.substr(0, 3) == "gfx") << "Expected gfx prefix, got: " << arch;
    EXPECT_EQ(arch.find(':'), std::string::npos)
        << "Arch should be truncated to base name, got: " << arch;
}

// ============================================================================
// T2.3: Forward execution sanity
// ============================================================================

TEST_F(CkFmhaEndToEndTest, ForwardBasicFp16_BHSD) {
    const int B = 1, H = 2, Sq = 64, Sk = 64, Dq = 64, Dv = 64;

    auto q_host = randomHalf(B * H * Sq * Dq);
    auto k_host = randomHalf(B * H * Sk * Dq);
    auto v_host = randomHalf(B * H * Sk * Dv);
    std::vector<__half> o_host(B * H * Sq * Dv, __float2half(0.0f));

    GpuBuffer q_buf(q_host.size() * sizeof(__half));
    GpuBuffer k_buf(k_host.size() * sizeof(__half));
    GpuBuffer v_buf(v_host.size() * sizeof(__half));
    GpuBuffer o_buf(o_host.size() * sizeof(__half));

    q_buf.upload(q_host);
    k_buf.upload(k_host);
    v_buf.upload(v_host);
    o_buf.upload(o_host);

    CkFmhaHandle handle;

    ParsedFwdParams params;
    params.data_type = "fp16";
    params.batch = B;
    params.nhead_q = H;
    params.nhead_k = H;
    params.seqlen_q = Sq;
    params.seqlen_k = Sk;
    params.hdim_q = Dq;
    params.hdim_v = Dv;
    params.is_bhsd_layout = true;
    params.scale = 1.0f / std::sqrt(static_cast<float>(Dq));
    params.q_uid = 1;
    params.k_uid = 2;
    params.v_uid = 3;
    params.o_uid = 4;

    auto problem = CkFmhaParamParser::buildFwdProblem(params, handle.gfxArch());
    auto kernel = handle.dispatcher()->select_kernel(problem);
    if (kernel == nullptr) GTEST_SKIP() << "No kernel for this architecture";

    auto plan = handle.dispatcher()->plan(problem);
    ASSERT_TRUE(plan.is_valid());

    CkFmhaFwdPlan fwd_plan(params, plan);

    hipdnnPluginDeviceBuffer_t buffers[] = {
        {1, q_buf.ptr()}, {2, k_buf.ptr()}, {3, v_buf.ptr()}, {4, o_buf.ptr()}};

    fwd_plan.execute(handle, buffers, 4, nullptr);
    HIP_CHECK(hipDeviceSynchronize());

    auto result = o_buf.download<__half>(B * H * Sq * Dv);
    EXPECT_FALSE(allZero(result)) << "Output O should not be all zeros";
    EXPECT_FALSE(hasNaN(result)) << "Output O should not contain NaN";
}

// ============================================================================
// T2.4: Backward execution sanity
// ============================================================================

TEST_F(CkFmhaEndToEndTest, BackwardBasicFp16_BHSD) {
    const int B = 1, Hq = 2, Hk = 2, Sq = 64, Sk = 64, Dq = 64, Dv = 64;

    // Forward pass first to get O and LSE
    auto q_host = randomHalf(B * Hq * Sq * Dq);
    auto k_host = randomHalf(B * Hk * Sk * Dq);
    auto v_host = randomHalf(B * Hk * Sk * Dv);
    std::vector<__half> o_host(B * Hq * Sq * Dv, __float2half(0.0f));

    GpuBuffer q_buf(q_host.size() * sizeof(__half));
    GpuBuffer k_buf(k_host.size() * sizeof(__half));
    GpuBuffer v_buf(v_host.size() * sizeof(__half));
    GpuBuffer o_buf(o_host.size() * sizeof(__half));
    GpuBuffer lse_buf(B * Hq * Sq * sizeof(float));

    q_buf.upload(q_host);
    k_buf.upload(k_host);
    v_buf.upload(v_host);
    o_buf.upload(o_host);

    CkFmhaHandle handle;

    // Check if backward is supported on this arch
    ParsedBwdParams bwd_params;
    bwd_params.data_type = "fp16";
    bwd_params.batch = B;
    bwd_params.nhead_q = Hq;
    bwd_params.nhead_k = Hk;
    bwd_params.seqlen_q = Sq;
    bwd_params.seqlen_k = Sk;
    bwd_params.hdim_q = Dq;
    bwd_params.hdim_v = Dv;

    auto bwd_problem = CkFmhaParamParser::buildBwdProblem(bwd_params, handle.gfxArch());
    auto bwd_exec = handle.dispatcher()->plan(bwd_problem);
    if (!bwd_exec.is_valid()) GTEST_SKIP() << "No backward kernel for this architecture";

    // Run forward with LSE
    ParsedFwdParams fwd_params;
    fwd_params.data_type = "fp16";
    fwd_params.batch = B;
    fwd_params.nhead_q = Hq;
    fwd_params.nhead_k = Hk;
    fwd_params.seqlen_q = Sq;
    fwd_params.seqlen_k = Sk;
    fwd_params.hdim_q = Dq;
    fwd_params.hdim_v = Dv;
    fwd_params.has_lse = true;
    fwd_params.is_bhsd_layout = true;
    fwd_params.scale = 1.0f / std::sqrt(static_cast<float>(Dq));
    fwd_params.q_uid = 1;
    fwd_params.k_uid = 2;
    fwd_params.v_uid = 3;
    fwd_params.o_uid = 4;
    fwd_params.lse_uid = 5;

    auto fwd_problem = CkFmhaParamParser::buildFwdProblem(fwd_params, handle.gfxArch());
    auto fwd_plan_exec = handle.dispatcher()->plan(fwd_problem);
    ASSERT_TRUE(fwd_plan_exec.is_valid());

    CkFmhaFwdPlan fwd_plan(fwd_params, fwd_plan_exec);
    hipdnnPluginDeviceBuffer_t fwd_bufs[] = {
        {1, q_buf.ptr()}, {2, k_buf.ptr()}, {3, v_buf.ptr()}, {4, o_buf.ptr()}, {5, lse_buf.ptr()}};
    fwd_plan.execute(handle, fwd_bufs, 5, nullptr);
    HIP_CHECK(hipDeviceSynchronize());

    // Backward pass
    auto do_host = randomHalf(B * Hq * Sq * Dv);
    GpuBuffer do_buf(do_host.size() * sizeof(__half));
    do_buf.upload(do_host);

    GpuBuffer dq_buf(q_host.size() * sizeof(__half));
    GpuBuffer dk_buf(k_host.size() * sizeof(__half));
    GpuBuffer dv_buf(v_host.size() * sizeof(__half));
    hipMemset(dq_buf.ptr(), 0, q_host.size() * sizeof(__half));
    hipMemset(dk_buf.ptr(), 0, k_host.size() * sizeof(__half));
    hipMemset(dv_buf.ptr(), 0, v_host.size() * sizeof(__half));

    auto ws_info = ck_tile::dispatcher::bwd_workspace_info(bwd_problem);
    GpuBuffer ws_buf(ws_info.total_bytes);

    bwd_params.is_bhsd_layout = true;
    bwd_params.scale = 1.0f / std::sqrt(static_cast<float>(Dq));
    bwd_params.q_uid = 1;
    bwd_params.k_uid = 2;
    bwd_params.v_uid = 3;
    bwd_params.o_uid = 4;
    bwd_params.do_uid = 6;
    bwd_params.stats_uid = 5;
    bwd_params.dq_uid = 7;
    bwd_params.dk_uid = 8;
    bwd_params.dv_uid = 9;

    CkFmhaBwdPlan bwd_plan(bwd_params, bwd_exec, ws_info);
    hipdnnPluginDeviceBuffer_t bwd_bufs[] = {
        {1, q_buf.ptr()},  {2, k_buf.ptr()},   {3, v_buf.ptr()},
        {4, o_buf.ptr()},  {5, lse_buf.ptr()}, {6, do_buf.ptr()},
        {7, dq_buf.ptr()}, {8, dk_buf.ptr()},  {9, dv_buf.ptr()}};

    bwd_plan.execute(handle, bwd_bufs, 9, ws_buf.ptr());
    HIP_CHECK(hipDeviceSynchronize());

    auto dq_result = dq_buf.download<__half>(q_host.size());
    auto dk_result = dk_buf.download<__half>(k_host.size());
    auto dv_result = dv_buf.download<__half>(v_host.size());

    EXPECT_FALSE(allZero(dq_result)) << "dQ should not be all zeros";
    EXPECT_FALSE(allZero(dk_result)) << "dK should not be all zeros";
    EXPECT_FALSE(allZero(dv_result)) << "dV should not be all zeros";
    EXPECT_FALSE(hasNaN(dq_result)) << "dQ should not contain NaN";
    EXPECT_FALSE(hasNaN(dk_result)) << "dK should not contain NaN";
    EXPECT_FALSE(hasNaN(dv_result)) << "dV should not contain NaN";
}

// ============================================================================
// Plan caching
// ============================================================================

TEST_F(CkFmhaEndToEndTest, PlanCacheHitOnRepeat) {
    CkFmhaHandle handle;

    ParsedFwdParams params;
    params.data_type = "fp16";
    params.batch = 1;
    params.nhead_q = 4;
    params.nhead_k = 4;
    params.seqlen_q = 128;
    params.seqlen_k = 128;
    params.hdim_q = 64;
    params.hdim_v = 64;

    auto problem = CkFmhaParamParser::buildFwdProblem(params, handle.gfxArch());
    auto key = problem.canonical_key();

    EXPECT_EQ(handle.getCachedPlan(key), nullptr) << "Cache should be empty initially";

    auto plan = handle.dispatcher()->plan(problem);
    if (!plan.is_valid()) GTEST_SKIP() << "No kernel for this config";

    handle.cachePlan(key, plan);

    auto* cached = handle.getCachedPlan(key);
    ASSERT_NE(cached, nullptr) << "Plan should be in cache after insertion";
    EXPECT_TRUE(cached->is_valid());
}

// ============================================================================
// Dispatcher direct smoke test
// ============================================================================

TEST_F(CkFmhaEndToEndTest, DispatcherKernelSelection) {
    CkFmhaHandle handle;

    struct TestConfig {
        std::string dtype;
        int hdim;
        int mask;
    };

    std::vector<TestConfig> configs = {
        {"fp16", 64, 0}, {"fp16", 128, 0}, {"bf16", 64, 0},  {"bf16", 128, 0},
        {"fp16", 64, 1}, {"fp16", 128, 2}, {"fp16", 256, 0}, {"bf16", 256, 0},
    };

    int supported = 0;
    for (const auto& cfg : configs) {
        ParsedFwdParams p;
        p.data_type = cfg.dtype;
        p.batch = 1;
        p.nhead_q = 4;
        p.nhead_k = 4;
        p.seqlen_q = 128;
        p.seqlen_k = 128;
        p.hdim_q = cfg.hdim;
        p.hdim_v = cfg.hdim;
        p.mask_type = cfg.mask;

        auto problem = CkFmhaParamParser::buildFwdProblem(p, handle.gfxArch());
        auto kernel = handle.dispatcher()->select_kernel(problem);
        if (kernel != nullptr) ++supported;
    }

    EXPECT_GT(supported, 0) << "At least some configs should have kernels";
}

}  // namespace
}  // namespace ck_fmha_plugin
