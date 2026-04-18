// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/host/fmha_rtc.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <string>

namespace ck {
namespace host {
namespace fmha_rtc {

namespace {

/// Stable 64-bit hash of a string (std::hash is allowed to vary across
/// runs; we need deterministic cache keys so we roll our own). FNV-1a.
std::uint64_t stable_hash(const std::string& s)
{
    constexpr std::uint64_t kOffset = 1469598103934665603ULL;
    constexpr std::uint64_t kPrime  = 1099511628211ULL;
    std::uint64_t h                 = kOffset;
    for(unsigned char c : s)
    {
        h ^= static_cast<std::uint64_t>(c);
        h *= kPrime;
    }
    return h;
}

std::string to_hex(std::uint64_t x)
{
    static const char kHex[] = "0123456789abcdef";
    std::string       out(16, '0');
    for(int i = 15; i >= 0; --i)
    {
        out[i] = kHex[x & 0xFu];
        x >>= 4;
    }
    return out;
}

/// Resolve the on-disk cache directory. Honour `CK_FMHA_RTC_CACHE_DIR`
/// if set; otherwise fall back to `$XDG_CACHE_HOME/ck_fmha_rtc` or
/// `$HOME/.cache/ck_fmha_rtc`. An empty value (explicit "") disables
/// caching and returns an empty path so callers can skip the blob I/O.
std::filesystem::path resolve_cache_dir()
{
    if(const char* env = std::getenv("CK_FMHA_RTC_CACHE_DIR"))
    {
        std::string v{env};
        if(v.empty())
            return {};
        return std::filesystem::path(v);
    }
    if(const char* xdg = std::getenv("XDG_CACHE_HOME"); xdg && *xdg)
        return std::filesystem::path(xdg) / "ck_fmha_rtc";
    if(const char* home = std::getenv("HOME"); home && *home)
        return std::filesystem::path(home) / ".cache" / "ck_fmha_rtc";
    return std::filesystem::temp_directory_path() / "ck_fmha_rtc";
}

/// Read a file into a vector<char>. Returns empty vector on any I/O
/// failure (cache miss). Files are treated as opaque byte blobs.
std::vector<char> read_file(const std::filesystem::path& p)
{
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if(!f) return {};
    auto size = f.tellg();
    if(size <= 0) return {};
    std::vector<char> buf(static_cast<std::size_t>(size));
    f.seekg(0);
    if(!f.read(buf.data(), size)) return {};
    return buf;
}

/// Atomically write bytes to `p`: write to a sibling tmp file, then
/// rename into place. Concurrent writers either produce identical
/// content (deterministic hash) or one overwrites the other -- either
/// way the cache is always valid.
void write_file_atomic(const std::filesystem::path& p, const std::vector<char>& bytes)
{
    std::error_code ec;
    std::filesystem::create_directories(p.parent_path(), ec);
    auto tmp = p;
    tmp += ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if(!f) return;
        f.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        if(!f) return;
    }
    std::filesystem::rename(tmp, p, ec);
    if(ec) std::filesystem::remove(tmp, ec); // best-effort cleanup
}

/// Sanitise an arch string for use in a filename ("gfx950:sramecc+:xnack-"
/// -> "gfx950_sramecc_xnack"). The naked gfx name is preserved when no
/// target features are present.
std::string arch_to_filename_tag(const std::string& arch)
{
    std::string out;
    out.reserve(arch.size());
    for(char c : arch)
    {
        if((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9'))
            out.push_back(c);
        else if(c == ':' || c == '+')
            out.push_back('_');
        // '-' is dropped: e.g. xnack- -> xnack
    }
    // Trim trailing underscores.
    while(!out.empty() && out.back() == '_') out.pop_back();
    return out;
}

// --- Minimal AMDGPU HSACO verifier ---------------------------------------
//
// Every ROCm code object produced by hiprtc for HIP kernels is an ELF64
// whose e_machine is EM_AMDGPU and whose e_flags encode the target
// mcpu in bits [0..9] using the EF_AMDGPU_MACH_AMDGCN_GFXxxx constants
// (see llvm/BinaryFormat/ELF.h). The magic-bytes check here catches:
//   - truncated / corrupt cache files
//   - caches copied from a different arch by mistake
//   - unrelated blobs shoved into the cache dir
//
// We deliberately do NOT parse the AMDGPU Metadata note: that's only
// needed for kernel-arg introspection, which hipModuleGetFunction
// handles downstream.
constexpr std::uint32_t kElfMagic      = 0x464c457fu; // "\x7fELF"
constexpr std::uint16_t kEmAMDGPU      = 224;
// EF_AMDGPU_MACH -- bottom 8 bits of e_flags identify the target mcpu
// (per LLVM's BinaryFormat/ELF.h AMDGPU section). Higher bits encode
// xnack/sramecc/ABI version; we ignore those when matching an arch.
constexpr std::uint32_t kEfAmdgpuMach  = 0x000000ffu;
// mapping from the AMDGPU machine flag to its gfx* short name. Only
// targets we care about are listed; an unknown value falls back to
// "unknown" which causes the sanity check to fail.
struct GfxMachine
{
    std::uint32_t flag;
    const char*   name;
};
constexpr GfxMachine kGfxTable[] = {
    {0x02c, "gfx900"},   {0x02e, "gfx906"},   {0x030, "gfx908"}, {0x033, "gfx90a"},
    {0x04c, "gfx942"},   {0x04f, "gfx950"},   {0x041, "gfx1030"}, {0x045, "gfx1100"},
    {0x046, "gfx1101"},  {0x047, "gfx1102"},  {0x036, "gfx1200"}, {0x037, "gfx1201"},
};

/// Extract the gfx* mcpu short name from an HSACO byte blob. Returns
/// empty string on any structural failure (not an ELF, not AMDGPU,
/// unknown mcpu flag).
std::string extract_gfx_mach(const std::vector<char>& blob)
{
    if(blob.size() < 0x40) return {};
    std::uint32_t magic;
    std::memcpy(&magic, blob.data(), 4);
    if(magic != kElfMagic) return {};
    if(blob[4] != 2 /*ELFCLASS64*/) return {};
    // ELF64 header: e_machine is at offset 0x12 (u16), e_flags at 0x30 (u32).
    std::uint16_t e_machine = 0;
    std::uint32_t e_flags   = 0;
    std::memcpy(&e_machine, blob.data() + 0x12, 2);
    std::memcpy(&e_flags,   blob.data() + 0x30, 4);
    if(e_machine != kEmAMDGPU) return {};
    const std::uint32_t mach = e_flags & kEfAmdgpuMach;
    for(const auto& entry : kGfxTable)
        if(entry.flag == mach) return entry.name;
    return {};
}

} // namespace

// Kept in sync with codegen/test/fmha_fwd.cpp's `kernel_template`. The
// only difference is the ${bias_ptr_type} placeholder, which lets the
// caller decide whether the bias argument is present in the signature
// (the MGX test always passes a dtype* even when has_bias=false).
static const std::string& fwd_kernel_template()
{
    static const std::string tpl = R"__ck__(
#include <${include}>

using KernelType = ${template};

extern "C" __launch_bounds__(KernelType::Kernel::kBlockSize, KernelType::Kernel::kBlockPerCu)
__global__ void ${kernel_name}(const ${dtype}* q, const ${dtype}* k, const ${dtype}* v, const ${dtype}* bias, ${dtype}* o) {

    constexpr float scale_s = ${scale_s};

    using Kernel = KernelType;

    // GQA: Q / O use nhead_q; K / V use nhead_k (may differ from nhead_q
    // when nhead_q > nhead_k the kernel rotates Q heads onto K heads via
    // nhead_ratio_qk = nhead_q / nhead_k inside fmha_fwd_wrapper.hpp).
    // Bias is sized against nhead_q (broadcast over KV heads).
    constexpr auto desc = Kernel::make_descriptor(
        // Q
        ck_tile::make_tuple(${batch}, ${nhead_q}, ${m}, ${k}),
        ck_tile::make_tuple(${q_stride_batch}, ${q_stride_nhead}, ${q_stride_m}),
        // K
        ck_tile::make_tuple(${batch}, ${nhead_k}, ${n}, ${k}),
        ck_tile::make_tuple(${k_stride_batch}, ${k_stride_nhead}, ${k_stride_n}),
        // V
        ck_tile::make_tuple(${batch}, ${nhead_k}, ${n}, ${o}),
        ck_tile::make_tuple(${v_stride_batch}, ${v_stride_nhead}, ${v_stride_n}),
        // O
        ck_tile::make_tuple(${batch}, ${nhead_q}, ${m}, ${o}),
        ck_tile::make_tuple(${o_stride_batch}, ${o_stride_nhead}, ${o_stride_m}),
        // Bias
        ck_tile::make_tuple(${batch}, ${nhead_q}, ${m}, ${n}),
        ck_tile::make_tuple(${bias_stride_batch}, ${bias_stride_nhead}, ${bias_stride_m}));

    static_assert(desc.IsValid(), "Invalid FMHA kernel configuration");

    Kernel::Run(desc, scale_s, q, k, v, bias, o);
}
)__ck__";
    return tpl;
}

std::string get_fwd_kernel_template() { return fwd_kernel_template(); }

// Matches create_tile_headers_for_test() in codegen/test/include/common.hpp.
// The leading whitespace before each header body is required for hipRTC
// to treat the string as a plain text header (hipRTC rejects headers
// whose first byte is `#`).
const std::vector<rtc::src_file>& tile_headers_for_rtc()
{
    static const std::vector<rtc::src_file> headers = [] {
        auto raw = GetTileHeaders();
        std::vector<rtc::src_file> result;
        result.reserve(raw.size());
        for(auto& [name, content] : raw)
        {
            std::string padded;
            padded.reserve(content.size() + 1);
            padded.push_back(' ');
            padded.append(content);
            result.emplace_back(name, std::move(padded));
        }
        return result;
    }();
    return headers;
}

static std::string make_fwd_source(const device_fmha_fwd::Problem& problem,
                                   const Solution& solution,
                                   const FwdLaunchParams& params)
{
    const auto& tpl = fwd_kernel_template();

    std::string dtype_cpp;
    switch(problem.dtype)
    {
    case DataType::Half: dtype_cpp = "ck_tile::fp16_t"; break;
    case DataType::Float: dtype_cpp = "float"; break;
    default:
        throw std::runtime_error(
            "fmha_rtc::compile_fwd currently only supports fp16/fp32; "
            "bf16 and fp8/bf8 arrive in Phases 2-3");
    }

    std::unordered_map<std::string, std::string> values = {
        {"include", problem.GetIncludeHeader()},
        {"template", solution.ToTemplateString()},
        {"dtype", dtype_cpp},
        {"kernel_name", params.kernel_name.empty() ? std::string("f") : params.kernel_name},
        {"batch", std::to_string(params.batch)},
        // `nhead` is kept for backward compatibility with callers that
        // don't set `nhead_k` explicitly. When nhead_k is 0 we default
        // it to `nhead` (pure MHA).
        {"nhead", std::to_string(params.nhead)},
        {"nhead_q", std::to_string(params.nhead)},
        {"nhead_k", std::to_string(params.nhead_k > 0 ? params.nhead_k : params.nhead)},
        {"m", std::to_string(params.M)},
        {"n", std::to_string(params.N)},
        {"k", std::to_string(params.K)},
        {"o", std::to_string(params.O)},
        {"q_stride_batch", std::to_string(params.q_stride_batch)},
        {"q_stride_nhead", std::to_string(params.q_stride_nhead)},
        {"q_stride_m", std::to_string(params.q_stride_m)},
        {"k_stride_batch", std::to_string(params.k_stride_batch)},
        {"k_stride_nhead", std::to_string(params.k_stride_nhead)},
        {"k_stride_n", std::to_string(params.k_stride_n)},
        {"v_stride_batch", std::to_string(params.v_stride_batch)},
        {"v_stride_nhead", std::to_string(params.v_stride_nhead)},
        {"v_stride_n", std::to_string(params.v_stride_n)},
        {"o_stride_batch", std::to_string(params.o_stride_batch)},
        {"o_stride_nhead", std::to_string(params.o_stride_nhead)},
        {"o_stride_m", std::to_string(params.o_stride_m)},
        {"bias_stride_batch", std::to_string(params.bias_stride_batch)},
        {"bias_stride_nhead", std::to_string(params.bias_stride_nhead)},
        {"bias_stride_m", std::to_string(params.bias_stride_m)},
        {"scale_s", std::to_string(params.scale_s) + "f"},
    };

    return InterpolateString(tpl, values);
}

CompiledKernel compile_fwd(const device_fmha_fwd::Problem& problem,
                           const Solution& solution,
                           const std::string& gfx_arch,
                           const FwdLaunchParams& params)
{
    const auto t0 = std::chrono::steady_clock::now();

    const std::string source      = make_fwd_source(problem, solution, params);
    const std::string kernel_name = params.kernel_name.empty() ? std::string("f") : params.kernel_name;

    // Cache key = stable_hash(arch + final-source). The source embeds
    // the interpolated batch / nhead / seqlen / strides / scale, so
    // distinct shapes naturally produce distinct cache entries. The
    // kernel_name is also folded in so caches built with non-default
    // entry points don't alias.
    const std::string key_input = gfx_arch + "\0" + kernel_name + "\0" + source;
    const std::string key_hex   = to_hex(stable_hash(key_input));

    const auto cache_dir = resolve_cache_dir();
    const bool cache_on  = !cache_dir.empty();

    // Cache entries are named `<arch>-<hash>.hsaco`. The arch prefix
    // makes a human-listable cache layout (`ls $CK_FMHA_RTC_CACHE_DIR`
    // immediately reveals which GPUs the cache was built for) and the
    // `.hsaco` extension aligns with the ROCm convention for
    // single-arch HSA code objects. A sibling `<stem>.json` sidecar
    // records the kernel name + solution string so rocm toolchains
    // (`llvm-readelf`, `llvm-objdump -d --arch=amdgcn --mcpu=gfxNNN`)
    // can still find them by filename and a human can grep the cache.
    const std::string arch_tag = arch_to_filename_tag(gfx_arch);
    const std::string stem     = arch_tag + "-" + key_hex;
    const auto        hsaco_path =
        cache_on ? (cache_dir / (stem + ".hsaco")) : std::filesystem::path{};
    const auto json_path =
        cache_on ? (cache_dir / (stem + ".json")) : std::filesystem::path{};
    const bool cache_hit_try = cache_on && std::filesystem::exists(hsaco_path);

    // --- Cache hit path: reconstitute kernel from on-disk code object ---
    if(cache_hit_try)
    {
        auto blob = read_file(hsaco_path);
        if(!blob.empty())
        {
            // Verify the on-disk blob is an AMDGPU HSACO for the
            // expected arch before handing it to the HIP runtime.
            // Catches truncated caches and caches copied from another
            // machine with a different GPU. A mismatch removes the
            // stale entry and falls through to recompile.
            const auto blob_gfx = extract_gfx_mach(blob);
            const bool arch_ok  = !blob_gfx.empty() &&
                                  gfx_arch.rfind(blob_gfx, 0) == 0; // prefix match ok
            if(arch_ok)
            {
                try
                {
                    auto kernel = rtc::kernel_from_code_object(blob, kernel_name);
                    const auto t1 = std::chrono::steady_clock::now();
                    const auto elapsed_s =
                        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() /
                        1e6;
                    return CompiledKernel{std::move(kernel),
                                          kernel_name,
                                          solution.ToTemplateString(),
                                          elapsed_s,
                                          /*from_cache=*/true};
                }
                catch(const std::exception&)
                {
                    // Bad cache entry -- fall through and recompile.
                }
            }
            std::error_code ec;
            std::filesystem::remove(hsaco_path, ec);
            std::filesystem::remove(json_path, ec);
        }
    }

    // --- Cache miss path: compile with hiprtc and write the blob ---
    auto srcs = tile_headers_for_rtc();
    srcs.emplace_back("main.cpp", source);

    rtc::compile_options opts;
    opts.kernel_name = kernel_name;
    opts.flags       = params.extra_flags;

    auto artifact = rtc::compile_kernel_with_code_object(srcs, opts);

    // Persist the fresh blob + metadata sidecar. Failures are silent
    // (best-effort caching); a non-existent cache dir or a read-only
    // FS falls back to compile-every-time without surfacing an error.
    // Skip when the backend did not produce a code-object blob (the
    // clang-based codegen test backend, not the hiprtc production
    // backend).
    if(cache_on && !artifact.code_object.empty())
    {
        write_file_atomic(hsaco_path, artifact.code_object);

        // Sidecar JSON with everything a human (or rocm-smi / MIOpen
        // cache tool) needs to understand the entry. Kept intentionally
        // small and escape-free: the only dynamic fields are kernel
        // name and solution string, both produced by ck_host and not
        // user-controlled.
        const std::string sidecar =
            std::string("{\n") +
            "  \"arch\": \"" + gfx_arch + "\",\n" +
            "  \"kernel_name\": \"" + kernel_name + "\",\n" +
            "  \"solution\": \"" + solution.ToTemplateString() + "\",\n" +
            "  \"family\": \"fwd\",\n" +
            "  \"hash\": \"" + key_hex + "\",\n" +
            "  \"hsaco_size\": " + std::to_string(artifact.code_object.size()) + "\n" +
            "}\n";
        std::vector<char> sidecar_bytes(sidecar.begin(), sidecar.end());
        write_file_atomic(json_path, sidecar_bytes);
    }

    const auto t1 = std::chrono::steady_clock::now();
    const auto elapsed_s =
        std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() / 1000.0;

    return CompiledKernel{std::move(artifact.k),
                          kernel_name,
                          solution.ToTemplateString(),
                          elapsed_s,
                          /*from_cache=*/false};
}

CompiledKernel compile_fwd(const device_fmha_fwd::Problem& problem,
                           const std::string& gfx_arch,
                           const FwdLaunchParams& params)
{
    auto solutions = problem.GetSolutions(gfx_arch);
    if(solutions.empty())
    {
        throw std::runtime_error(
            "fmha_rtc::compile_fwd: no solutions available for the given problem/arch");
    }
    return compile_fwd(problem, solutions.front(), gfx_arch, params);
}

// ===================================================================
// Phase 5 per-family facades.
// ===================================================================

// Generic compile helper shared by all forward-family facades. Each
// family has its own Problem::GetIncludeHeader() and wrapper template
// string, but the hipRTC path is identical: build the kernel source,
// prepend stripped tile headers, invoke rtc::compile_kernel.
template <class Problem>
static CompiledKernel compile_family_generic(const Problem& problem,
                                             const Solution& solution,
                                             const std::string& gfx_arch,
                                             const FwdLaunchParams& params)
{
    (void)gfx_arch;
    auto t0 = std::chrono::steady_clock::now();

    // Reuse the fwd kernel template's structure: the interpolation
    // dictionary treats `include` (the family header) and `template`
    // (the expanded wrapper string) the same way across families.
    const auto tpl = get_fwd_kernel_template();
    auto src = InterpolateString(tpl,
        std::unordered_map<std::string, std::string>{
            {"include", problem.GetIncludeHeader()},
            {"template", solution.ToTemplateString()},
            {"dtype", "ck_tile::fp16_t"}, // Phase 5 placeholder; family-specific
            {"kernel_name", params.kernel_name.empty() ? std::string("f") : params.kernel_name},
            {"batch", std::to_string(params.batch)},
            {"nhead", std::to_string(params.nhead)},
            {"m", std::to_string(params.M)}, {"n", std::to_string(params.N)},
            {"k", std::to_string(params.K)}, {"o", std::to_string(params.O)},
            {"q_stride_batch", std::to_string(params.q_stride_batch)},
            {"q_stride_nhead", std::to_string(params.q_stride_nhead)},
            {"q_stride_m", std::to_string(params.q_stride_m)},
            {"k_stride_batch", std::to_string(params.k_stride_batch)},
            {"k_stride_nhead", std::to_string(params.k_stride_nhead)},
            {"k_stride_n", std::to_string(params.k_stride_n)},
            {"v_stride_batch", std::to_string(params.v_stride_batch)},
            {"v_stride_nhead", std::to_string(params.v_stride_nhead)},
            {"v_stride_n", std::to_string(params.v_stride_n)},
            {"o_stride_batch", std::to_string(params.o_stride_batch)},
            {"o_stride_nhead", std::to_string(params.o_stride_nhead)},
            {"o_stride_m", std::to_string(params.o_stride_m)},
            {"bias_stride_batch", std::to_string(params.bias_stride_batch)},
            {"bias_stride_nhead", std::to_string(params.bias_stride_nhead)},
            {"bias_stride_m", std::to_string(params.bias_stride_m)},
            {"scale_s", std::to_string(params.scale_s) + "f"},
        });

    auto srcs = tile_headers_for_rtc();
    srcs.emplace_back("main.cpp", std::move(src));

    rtc::compile_options opts;
    opts.kernel_name = params.kernel_name.empty() ? std::string("f") : params.kernel_name;
    opts.flags       = params.extra_flags;

    auto kernel = rtc::compile_kernel(srcs, opts);

    const auto t1 = std::chrono::steady_clock::now();
    const auto elapsed_s =
        std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() / 1000.0;

    return CompiledKernel{std::move(kernel),
                          opts.kernel_name,
                          solution.ToTemplateString(),
                          elapsed_s};
}

CompiledKernel compile_pagedkv(const device_fmha_fwd_pagedkv::Problem& problem,
                               const Solution& solution,
                               const std::string& gfx_arch,
                               const FwdLaunchParams& params)
{
    return compile_family_generic(problem, solution, gfx_arch, params);
}

SplitKVCompiledKernels
compile_splitkv(const device_fmha_fwd_splitkv::Problem& split_problem,
                const device_fmha_fwd_splitkv_combine::Problem& combine_problem,
                const std::string& gfx_arch,
                const FwdLaunchParams& split_params)
{
    auto split_sols = split_problem.GetSolutions(gfx_arch);
    auto combine_sols = combine_problem.GetSolutions(gfx_arch);
    if(split_sols.empty() || combine_sols.empty())
    {
        throw std::runtime_error(
            "fmha_rtc::compile_splitkv: no solution for split or combine stage");
    }
    SplitKVCompiledKernels out;
    out.split   = compile_family_generic(split_problem, split_sols.front(),
                                         gfx_arch, split_params);
    out.combine = compile_family_generic(combine_problem, combine_sols.front(),
                                         gfx_arch, split_params);
    return out;
}

CompiledKernel compile_appendkv(const device_fmha_fwd_appendkv::Problem& problem,
                                const Solution& solution,
                                const std::string& gfx_arch,
                                const FwdLaunchParams& params)
{
    return compile_family_generic(problem, solution, gfx_arch, params);
}

CompiledKernel compile_batch_prefill(const device_fmha_batch_prefill::Problem& problem,
                                     const Solution& solution,
                                     const std::string& gfx_arch,
                                     const FwdLaunchParams& params)
{
    return compile_family_generic(problem, solution, gfx_arch, params);
}

// ===================================================================
// Phase 6 backward compile.
// ===================================================================

BwdCompiledKernels
compile_bwd(const device_fmha_bwd_dot_do_o::Problem& dot_problem,
            const device_fmha_bwd_dq_dk_dv::Problem& dq_problem,
            const device_fmha_bwd_convert_dq::Problem& convert_problem,
            bool want_convert_dq,
            const std::string& gfx_arch,
            const FwdLaunchParams& shared_params)
{
    auto dot_sols     = dot_problem.GetSolutions(gfx_arch);
    auto dq_sols      = dq_problem.GetSolutions(gfx_arch);
    auto convert_sols = convert_problem.GetSolutions(gfx_arch);
    if(dot_sols.empty() || dq_sols.empty())
    {
        throw std::runtime_error(
            "fmha_rtc::compile_bwd: missing solution(s) for dot_do_o or dq_dk_dv stage");
    }
    if(want_convert_dq && convert_sols.empty())
    {
        throw std::runtime_error(
            "fmha_rtc::compile_bwd: convert_dq requested but no solution available");
    }

    BwdCompiledKernels out;
    out.dot_do_o = compile_family_generic(dot_problem, dot_sols.front(),
                                          gfx_arch, shared_params);
    out.dq_dk_dv = compile_family_generic(dq_problem, dq_sols.front(),
                                          gfx_arch, shared_params);
    if(want_convert_dq)
    {
        out.convert_dq = compile_family_generic(convert_problem, convert_sols.front(),
                                                gfx_arch, shared_params);
        out.has_convert_dq = true;
    }
    return out;
}

} // namespace fmha_rtc
} // namespace host
} // namespace ck
