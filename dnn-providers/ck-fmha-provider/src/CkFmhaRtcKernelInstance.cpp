// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Implementation of RtcFmhaKernelInstance.
//
// The kernel template that `ck::host::fmha_rtc::compile_fwd()` emits
// declares the following signature:
//
//     extern "C" __launch_bounds__(BlockSize, BlockPerCu)
//     __global__ void f(const T* q,
//                       const T* k,
//                       const T* v,
//                       const T* bias,
//                       T* o);
//
// with the tile/wave configuration and descriptor strides baked into
// the generated source. The runtime only passes raw device pointers;
// all kargs (seqlen, hdim, strides, scale) are pinned at JIT time via
// the FwdLaunchParams fed to compile_fwd().
//
// Grid/block dims are derived from the Solution's BM0, BN1, RMx, RNx,
// RKx template parameters the same way the codegen test does in
// codegen/test/fmha_fwd.cpp::get_launch_dims().

#include "CkFmhaRtcKernelInstance.hpp"

#ifdef CK_FMHA_WITH_RTC

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <sstream>
#include <stdexcept>
#include <variant>

namespace ck_fmha_plugin {

namespace {

// Warp size: all AMDGPU CDNA/RDNA parts CK Tile currently supports
// are 64-lane. Revisit when gfx11/gfx12 get added (which use 32-lane
// wavefronts).
constexpr unsigned int kWarpSize = 64;

unsigned int ceil_div(unsigned int a, unsigned int b) { return b == 0 ? 0 : (a + b - 1) / b; }

std::tuple<unsigned int, unsigned int, unsigned int, unsigned int> compute_launch_dims(
    const ck::host::Solution& solution, unsigned int batch, unsigned int nhead,
    unsigned int M, unsigned int O) {
    auto bm0 = solution.GetTemplateParameter<std::size_t>("BM0");
    auto bn1 = solution.GetTemplateParameter<std::size_t>("BN1");

    auto rm0 = solution.GetTemplateParameter<std::size_t>("RM0");
    auto rn0 = solution.GetTemplateParameter<std::size_t>("RN0");
    auto rk0 = solution.GetTemplateParameter<std::size_t>("RK0");

    auto rm1 = solution.GetTemplateParameter<std::size_t>("RM1");
    auto rn1 = solution.GetTemplateParameter<std::size_t>("RN1");
    auto rk1 = solution.GetTemplateParameter<std::size_t>("RK1");

    const std::size_t num_warps = std::max(rm0 * rn0 * rk0, rm1 * rn1 * rk1);
    const unsigned int block    = static_cast<unsigned int>(num_warps * kWarpSize);

    const unsigned int grid_m     = ceil_div(M, static_cast<unsigned int>(bm0));
    const unsigned int grid_o     = ceil_div(O, static_cast<unsigned int>(bn1));
    const unsigned int grid_nhead = nhead;
    const unsigned int grid_mo    = grid_m * grid_o;
    const unsigned int grid_batch = batch;

    return {grid_nhead, grid_mo, grid_batch, block};
}

}  // namespace

RtcFmhaKernelInstance::RtcFmhaKernelInstance(ck::host::fmha_rtc::CompiledKernel compiled,
                                             ck::host::Solution solution,
                                             ck_tile::dispatcher::FmhaKernelKey key,
                                             std::string name, int batch, int nhead_q,
                                             int nhead_k, int seqlen_q, int seqlen_k)
    : compiled_(std::move(compiled)), solution_(std::move(solution)), key_(std::move(key)),
      name_(std::move(name)), compiled_batch_(batch), compiled_nhead_q_(nhead_q),
      compiled_nhead_k_(nhead_k), compiled_seqlen_q_(seqlen_q), compiled_seqlen_k_(seqlen_k) {}

bool RtcFmhaKernelInstance::supports(const ck_tile::dispatcher::FmhaProblem& problem) const {
    // Signature-level gate (family + dtype + flags + hdim): these must
    // match or the kernel physically cannot service the problem.
    if (key_.signature.family != problem.requested_family) return false;
    if (key_.signature.data_type != problem.data_type) return false;
    if (key_.signature.mask_type != problem.mask_type) return false;
    if (key_.signature.bias_type != problem.bias_type) return false;
    if (key_.signature.is_group_mode != problem.is_group_mode) return false;
    if (key_.signature.is_v_rowmajor != problem.is_v_rowmajor) return false;
    if (key_.signature.hdim_q != problem.hdim_q) return false;
    if (key_.signature.hdim_v != problem.hdim_v) return false;
    // Shape-level gate (batch / nhead / seqlen): the MGX template
    // hard-codes these as constexpr into the kernel descriptor, so a
    // kernel compiled for one shape cannot run another.
    if (compiled_batch_ != problem.batch) return false;
    if (compiled_nhead_q_ != problem.nhead_q) return false;
    if (compiled_nhead_k_ != problem.nhead_k) return false;
    if (compiled_seqlen_q_ != problem.seqlen_q) return false;
    if (compiled_seqlen_k_ != problem.seqlen_k) return false;
    return true;
}

void RtcFmhaKernelInstance::launch(const ck_tile::dispatcher::FmhaInvocation& invocation,
                                   const ck_tile::stream_config& stream_config) const {
    if (invocation.api_family != ck_tile::dispatcher::FmhaApiFamily::Fwd) {
        throw std::runtime_error("RtcFmhaKernelInstance: non-Fwd invocation not supported");
    }
    if (!std::holds_alternative<fmha_fwd_args>(invocation.args)) {
        throw std::runtime_error("RtcFmhaKernelInstance: invocation.args is not fmha_fwd_args");
    }
    const auto& args = std::get<fmha_fwd_args>(invocation.args);

    // Recompute launch dims each call so shape-aware adjustments (e.g.
    // runtime GQA) take effect. For batch-mode fp16 with identical
    // compile-time and launch-time shapes this is trivially the same
    // as what we cached at construction.
    auto [g_nhead, g_mo, g_batch, block_size] = compute_launch_dims(
        solution_, static_cast<unsigned int>(args.batch),
        static_cast<unsigned int>(args.nhead_q), static_cast<unsigned int>(args.seqlen_q),
        static_cast<unsigned int>(args.hdim_v));

    dim3 grid(g_nhead, g_mo, g_batch);
    dim3 block(block_size, 1, 1);

    if (std::getenv("CK_FMHA_RTC_TRACE") != nullptr) {
        // Opt-in runtime trace to confirm the hipRTC-compiled kernel
        // is being driven with the dispatcher's args. Useful when
        // validating new shapes / kernel families end-to-end.
        std::fprintf(stderr,
                     "[CK_FMHA_RTC] %s batch=%d nhead_q=%d Sq=%d Sk=%d Dq=%d Dv=%d"
                     " grid=(%u,%u,%u) block=(%u)\n",
                     name_.c_str(), args.batch, args.nhead_q, args.seqlen_q, args.seqlen_k,
                     args.hdim_q, args.hdim_v, grid.x, grid.y, grid.z, block.x);
        std::fflush(stderr);
    }

    // Kernel signature: f(q, k, v, bias, o).
    // `rtc::kernel::launch(stream, grid, block)` takes a vector of
    // pinned kernel_arguments. We pass raw const void* / void*.
    const void* q_ptr    = args.q_ptr;
    const void* k_ptr    = args.k_ptr;
    const void* v_ptr    = args.v_ptr;
    const void* bias_ptr = args.bias_ptr;
    void* o_ptr          = args.o_ptr;

    std::vector<rtc::kernel_argument> kargs;
    kargs.reserve(5);
    // kernel_argument stores void*; the rtc::kernel packer copies the
    // address into the arg buffer, so we must retain live l-values.
    kargs.emplace_back(q_ptr);
    kargs.emplace_back(k_ptr);
    kargs.emplace_back(v_ptr);
    kargs.emplace_back(bias_ptr);
    kargs.emplace_back(o_ptr);

    compiled_.kernel.launch(static_cast<hipStream_t>(stream_config.stream_id_), grid, block, kargs);
}

}  // namespace ck_fmha_plugin

#endif  // CK_FMHA_WITH_RTC
