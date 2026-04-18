// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Public facade over `rtc::compile_kernel` + `ck::host::GetTileHeaders`
// for hipRTC-based FMHA kernel compilation. A downstream consumer can
// now write:
//
//     ck::host::device_fmha_fwd::Problem prob{...};
//     auto solutions = prob.GetSolutions(arch);
//     auto compiled  = ck::host::fmha_rtc::compile_fwd(prob, solutions[0],
//                                                       arch,
//                                                       fwd_params);
//     compiled.kernel.launch(stream, grid, block)(q, k, v, bias, o);
//
// without needing to know about the kernel-source template, the header
// preprocessor, or the hipRTC wrapper machinery. Additional families
// (pagedkv, splitkv, appendkv, batch_prefill, bwd) are added as the
// migration phases land them.

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

#include "ck/host/device_fmha_fwd/problem.hpp"
#include "ck/host/device_fmha_fwd/operation.hpp"
#include "ck/host/device_fmha_fwd_pagedkv/problem.hpp"
#include "ck/host/device_fmha_fwd_splitkv/problem.hpp"
#include "ck/host/device_fmha_fwd_splitkv_combine/problem.hpp"
#include "ck/host/device_fmha_fwd_appendkv/problem.hpp"
#include "ck/host/device_fmha_batch_prefill/problem.hpp"
#include "ck/host/device_fmha_bwd_dot_do_o/problem.hpp"
#include "ck/host/device_fmha_bwd_dq_dk_dv/problem.hpp"
#include "ck/host/device_fmha_bwd_convert_dq/problem.hpp"
#include "ck/host/stringutils.hpp"
#include "ck/host/headers.hpp"
#include "ck/host/types.hpp"

#include <rtc/compile_kernel.hpp>
#include <rtc/kernel.hpp>
#include <utility>

namespace ck {
namespace host {
namespace fmha_rtc {

/// Lightweight container for a compiled FMHA kernel plus its metadata.
/// Exactly one `rtc::kernel` is held; multi-stage families (splitkv,
/// bwd) return a struct with multiple CompiledKernel members.
struct CompiledKernel
{
    rtc::kernel kernel{};
    std::string kernel_name{}; // entry point name passed to hipRTC (default "f")
    std::string solution_string{}; // for logging / cache keys
    double build_time_s = 0.0;
    /// True when the kernel was reconstituted from the on-disk HSACO
    /// cache (see Phase 14). False on a cold compile. Consumers use
    /// this to split compile telemetry into hot/cold buckets without
    /// having to infer from build_time_s.
    bool from_cache = false;
};

/// Parameters that the generated FMHA forward kernel template needs but
/// that do not come from the `Problem` / `Solution` pair. The caller
/// typically fills these from the actual tensor shapes it intends to
/// launch against.
///
/// NOTE: Phase 1 only covers the MGX signature (single-stage fwd,
/// fp16, causal-or-none, optional elementwise bias, row-major V,
/// batch mode, no LSE / dropout / soft-cap / sink / paged / group /
/// GQA). Phase 2 expands `FwdLaunchParams` with the fields required
/// to cover the full `fmha_fwd_traits` surface.
struct FwdLaunchParams
{
    // Reference shape used to materialise the descriptor template.
    // These are not the runtime launch shape (which is passed by the
    // kernel arguments); they only parameterise the constexpr
    // descriptor that feeds into `Kernel::make_descriptor(...)` inside
    // the generated kernel source. For MGX the same shapes must be
    // passed both at compile time (here) and launch time (via Q/K/V
    // tensor dimensions) because the kernel hard-codes them.
    std::size_t batch = 0;
    /// Number of query heads. Also used for Q / O tensor descriptors.
    std::size_t nhead = 0;
    /// Number of key/value heads. When 0, defaults to `nhead` (pure
    /// MHA). When set and < nhead, the kernel rotates Q heads onto K
    /// heads via `nhead_ratio_qk = nhead / nhead_k` (GQA / MQA).
    std::size_t nhead_k = 0;
    std::size_t M     = 0; // seqlen_q
    std::size_t N     = 0; // seqlen_k
    std::size_t K     = 0; // hdim_q
    std::size_t O     = 0; // hdim_v

    // Strides for Q, K, V, O, Bias. Passed to make_descriptor.
    std::size_t q_stride_batch = 0;
    std::size_t q_stride_nhead = 0;
    std::size_t q_stride_m     = 0;
    std::size_t k_stride_batch = 0;
    std::size_t k_stride_nhead = 0;
    std::size_t k_stride_n     = 0;
    std::size_t v_stride_batch = 0;
    std::size_t v_stride_nhead = 0;
    std::size_t v_stride_n     = 0;
    std::size_t o_stride_batch = 0;
    std::size_t o_stride_nhead = 0;
    std::size_t o_stride_m     = 0;
    std::size_t bias_stride_batch = 0;
    std::size_t bias_stride_nhead = 0;
    std::size_t bias_stride_m     = 0;

    float scale_s = 1.0f;

    /// Entry-point name for the generated `__global__`. Usually "f".
    std::string kernel_name = "f";

    /// Extra compile flags passed to hipRTC (in addition to the
    /// defaults: `-I. -O3 -std=c++20 -DCK_CODE_GEN_RTC --offload-arch=<device>`).
    std::string extra_flags{};
};

/// Build a vector<src_file> by taking every embedded CK Tile header
/// (already stripped of `CK_TILE_HOST` bodies by `GetTileHeaders`) and
/// prepending a leading whitespace character to each. hipRTC rejects
/// headers whose first byte is `#`, which is a known hipRTC quirk.
/// Returns a static cache populated on first call.
const std::vector<rtc::src_file>& tile_headers_for_rtc();

/// Compile a single-stage FMHA forward kernel from a (Problem,
/// Solution, arch, FwdLaunchParams) quad. Throws `std::runtime_error`
/// on codegen or hipRTC failure.
///
/// The returned `CompiledKernel` holds a `hipModule_t` + `hipFunction_t`
/// via `rtc::kernel`. Safe to copy (rtc::kernel is shared_ptr-backed).
CompiledKernel compile_fwd(const device_fmha_fwd::Problem& problem,
                           const Solution& solution,
                           const std::string& gfx_arch,
                           const FwdLaunchParams& params);

/// Convenience overload: auto-select the first Solution returned by
/// `problem.GetSolutions(arch)`. Throws if no solutions exist.
CompiledKernel compile_fwd(const device_fmha_fwd::Problem& problem,
                           const std::string& gfx_arch,
                           const FwdLaunchParams& params);

/// The host-side template used to produce kernel source for
/// `compile_fwd`. Exposed so that downstream consumers can inspect or
/// customise it (e.g. to add debug printfs). Matches the MGX test's
/// `kernel_template` in codegen/test/fmha_fwd.cpp.
std::string get_fwd_kernel_template();

// ===================================================================
// Phase 5: per-family facade entry points. These return CompiledKernel
// objects that Phase 9's CkFmhaJit::compile_rtc will register into the
// FmhaRegistry. Full Kargs plumbing for these families lands in the
// wrapper TODO(phase5-followup) sections; the facade signatures are
// stable now so downstream consumers can build against them.
// ===================================================================

/// PagedKV: single-stage forward on a paged KV cache.
CompiledKernel compile_pagedkv(const device_fmha_fwd_pagedkv::Problem& problem,
                               const Solution& solution,
                               const std::string& gfx_arch,
                               const FwdLaunchParams& params);

/// SplitKV: two-stage forward (split + combine). Returns both kernels;
/// the caller issues them on the same stream in order.
struct SplitKVCompiledKernels
{
    CompiledKernel split;
    CompiledKernel combine;
};
SplitKVCompiledKernels
compile_splitkv(const device_fmha_fwd_splitkv::Problem& split_problem,
                const device_fmha_fwd_splitkv_combine::Problem& combine_problem,
                const std::string& gfx_arch,
                const FwdLaunchParams& split_params);

/// AppendKV: memory-copy primitive that appends fresh K/V tokens into
/// the paged cache (optionally applying RoPE).
CompiledKernel compile_appendkv(const device_fmha_fwd_appendkv::Problem& problem,
                                const Solution& solution,
                                const std::string& gfx_arch,
                                const FwdLaunchParams& params);

/// BatchPrefill: fused prefill pass on a paged KV cache.
CompiledKernel compile_batch_prefill(const device_fmha_batch_prefill::Problem& problem,
                                     const Solution& solution,
                                     const std::string& gfx_arch,
                                     const FwdLaunchParams& params);

// ===================================================================
// Phase 6: backward three-stage pipeline. Dispatcher's plan() for Bwd
// produces a 2- or 3-stage plan (dot_do_o + dq_dk_dv + optional
// convert_dq). compile_bwd() returns all three compiled kernels; the
// caller registers them under distinct FmhaKernelFamily tags so
// FmhaDispatcher::run_plan() can chain them on one stream.
// ===================================================================

struct BwdCompiledKernels
{
    CompiledKernel dot_do_o;
    CompiledKernel dq_dk_dv;
    CompiledKernel convert_dq; // only populated when the problem needs it
    bool has_convert_dq = false;
};

BwdCompiledKernels
compile_bwd(const device_fmha_bwd_dot_do_o::Problem& dot_problem,
            const device_fmha_bwd_dq_dk_dv::Problem& dq_problem,
            const device_fmha_bwd_convert_dq::Problem& convert_problem,
            bool want_convert_dq,
            const std::string& gfx_arch,
            const FwdLaunchParams& shared_params);

} // namespace fmha_rtc
} // namespace host
} // namespace ck
