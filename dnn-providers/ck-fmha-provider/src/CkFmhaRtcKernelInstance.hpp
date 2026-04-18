// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Phase 9 follow-up: a concrete FmhaKernelInstance that owns an
// `rtc::kernel` (hipRTC-compiled, loaded hipModule_t + hipFunction_t)
// and translates dispatcher `FmhaInvocation::args` into the flat kernel
// signature that ck::host::fmha_rtc::compile_fwd() emits.
//
// Register an instance of this class into FmhaRegistry::instance() from
// CkFmhaJit::compile_rtc() so that subsequent `dispatcher->select_kernel()`
// calls find it and the plan builder executes it through the standard
// FmhaDispatcher pipeline.

#pragma once

#ifdef CK_FMHA_WITH_RTC

#include <ck/host/device_fmha_fwd/operation.hpp>
#include <ck/host/fmha_rtc.hpp>
#include <ck_tile/dispatcher/fmha_kernel_instance.hpp>
#include <ck_tile/dispatcher_fmha.hpp>
#include <memory>
#include <string>

namespace ck_fmha_plugin {

class RtcFmhaKernelInstance : public ck_tile::dispatcher::FmhaKernelInstance {
public:
    RtcFmhaKernelInstance(ck::host::fmha_rtc::CompiledKernel compiled,
                          ck::host::Solution solution,
                          ck_tile::dispatcher::FmhaKernelKey key,
                          std::string name, int batch, int nhead_q, int nhead_k,
                          int seqlen_q, int seqlen_k);

    [[nodiscard]] const ck_tile::dispatcher::FmhaKernelKey& get_key() const override {
        return key_;
    }
    /// The MGX kernel template bakes batch / nhead / seqlen into the
    /// device descriptor as `constexpr`, so a kernel compiled for
    /// (B, H, Sq, Sk) can **only** service that exact shape. The
    /// FmhaKernelKey alone isn't shape-specific enough to express
    /// that, so we also gate on the concrete compile-time shape
    /// captured at construction.
    [[nodiscard]] bool supports(const ck_tile::dispatcher::FmhaProblem& problem) const override;
    [[nodiscard]] std::string get_name() const override { return name_; }

    void launch(const ck_tile::dispatcher::FmhaInvocation& invocation,
                const ck_tile::stream_config& stream_config) const override;

private:
    ck::host::fmha_rtc::CompiledKernel compiled_;
    ck::host::Solution solution_;
    ck_tile::dispatcher::FmhaKernelKey key_;
    std::string name_;
    // Compile-time shape baked into `compiled_.kernel`. `supports()`
    // rejects any FmhaProblem whose runtime shape disagrees with these,
    // which forces a fresh compile for each new (B,H,Sq,Sk) quad.
    int compiled_batch_    = 0;
    int compiled_nhead_q_  = 0;
    int compiled_nhead_k_  = 0;
    int compiled_seqlen_q_ = 0;
    int compiled_seqlen_k_ = 0;
};

}  // namespace ck_fmha_plugin

#endif  // CK_FMHA_WITH_RTC
