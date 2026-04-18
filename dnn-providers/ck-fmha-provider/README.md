# CK FMHA Provider Plugin for hipDNN

A hipDNN engine plugin that executes Scaled Dot-Product Attention (SDPA)
forward and backward operations through the Composable Kernel (CK) FMHA
dispatcher.

## Quickstart

Build hipDNN, build the plugin, run the demo -- all on a machine with ROCm and an AMD GPU:

```bash
# 1. Build hipDNN backend
cd projects/hipdnn
cmake -B build -G Ninja -DCMAKE_CXX_COMPILER=hipcc \
  -DHIPDNN_SKIP_TESTS=ON -DHIPDNN_GENERATE_SDK_HEADERS=OFF \
  -DENABLE_CLANG_TIDY=OFF -DENABLE_CLANG_FORMAT=OFF -DCMAKE_BUILD_TYPE=Release
ninja -C build hipdnn_backend

# 2. Build CK dispatcher FMHA library (precompiled GPU kernels)
cd ../composablekernel/dispatcher/build
cmake --build . --target dispatcher_fmha_lib -j$(nproc)

# 3. Build the CK FMHA plugin (see docs/EndToEndGuide.md for full commands)
#    Short version: compile plugin sources with hipcc, link with
#    libck_tile_dispatcher.a + generated kernel .o, install to plugin dir
mkdir -p projects/hipdnn/build/lib/hipdnn_plugins/engines
cp ck_fmha_provider_plugin.so projects/hipdnn/build/lib/hipdnn_plugins/engines/

# 4. Compile the demo (links only against libhipdnn_backend.so)
hipcc -std=c++17 -O2 -w <include flags> -lhipdnn_backend \
  dnn-providers/ck-fmha-provider/integration_tests/EndToEndSdpaDemo.cpp \
  -o ck_fmha_e2e_demo

# 5. Run
HIPDNN_PLUGIN_PATH=projects/hipdnn/build/lib/hipdnn_plugins/engines \
  ./ck_fmha_e2e_demo --warmup 2 --repeat 10
```

For the complete step-by-step with every flag and include path, see
**[docs/EndToEndGuide.md](docs/EndToEndGuide.md)** -- it documents exactly
what was run on an MI355X (gfx950) to produce the verified TFLOPS output below.

### JIT Mode (optional)

For shapes without precompiled kernels, the plugin can JIT-compile them
on demand. Two backends ship in the box:

- **`rtc`** (preferred) -- in-process hipRTC + `ck_tile_headers_preprocessor`.
  No Python, no `hipcc` subprocess, no `.so` on disk. First compile per
  (shape, arch) is ~7-15 s on MI355X, subsequent calls in the same
  process are a registry cache hit (~0.1 ms). See
  [**Using hipRTC (JIT backend)**](#using-hiprtc-jit-backend) below for
  the walk-through.
- **`hipcc`** (legacy) -- shells out to `python3 -> generate_fmha_fallback.py -> hipcc -> .so`,
  then dlopens. Kept as a fallback so deployments can still use the
  Phase 10 offline warm-up path (see `fmha_utils.setup_multiple_fmha_dispatchers`).

Quick-start for hipRTC:

```bash
CK_FMHA_ENABLE_JIT=1          \
CK_FMHA_JIT_BACKEND=rtc       \
HIPDNN_PLUGIN_PATH=<plugin_dir> \
  ./ck_fmha_e2e_rtc_demo --warmup 2 --repeat 5
```

Quick-start for legacy hipcc:

```bash
CK_FMHA_ENABLE_JIT=1          \
CK_FMHA_JIT_BACKEND=hipcc     \
CK_DISPATCHER_PYTHON_PATH=<ck>/dispatcher/python \
HIPDNN_PLUGIN_PATH=<plugin_dir> \
  ./ck_fmha_e2e_demo --warmup 2 --repeat 5
```

See [docs/EndToEndGuide.md](docs/EndToEndGuide.md#part-2-jit-path) for
the full walk-through of both paths, including kernel-cache semantics
and Phase 10's offline warm-up.

## Using hipRTC (JIT backend)

The hipRTC backend compiles FMHA kernels **in-process** via `libhiprtc`
directly against `CK_TILE_HOST`-stripped CK Tile headers. There is no
Python runtime, no `hipcc` subprocess, and no `.so` artefact on disk.

### Prerequisites

1. A ROCm install providing `libamdhip64.so`, `libhiprtc.so`, and
   `libcomgr.so` (tested against ROCm 7.0.1).
2. An AMD GPU. Primary support is `gfx950` (MI355X); `gfx942`, `gfx908`,
   `gfx1100`, `gfx1200` work as well after their per-arch tile tables
   and predicates land in upstream CK codegen.

### Build the plugin with the RTC backend

The CK FMHA plugin gates RTC behind a build-time flag. Default is
**on**, but you can flip it explicitly:

```bash
cd dnn-providers/ck-fmha-provider
cmake -B build -G Ninja                                                 \
  -DCMAKE_BUILD_TYPE=Release                                            \
  -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc                              \
  -DCK_ROOT=/workspace/rocm-lib-copy/projects/composablekernel          \
  -DHIPDNN_ROOT=/workspace/rocm-lib-copy/projects/hipdnn                \
  -DCK_FMHA_WITH_RTC=ON                                                 \
  -DCK_FMHA_PROVIDER_BUILD_TESTS=ON
ninja -C build ck_fmha_plugin ck_fmha_e2e_rtc_demo
```

This produces:

- `build/libck_fmha_provider_plugin.so` -- plugin with both backends
  linked in (`compile_rtc` and `compile_hipcc`).
- `build/integration_tests/ck_fmha_e2e_rtc_demo` -- the end-to-end
  demo that drives the full hipDNN Graph API -> CK plugin -> hipRTC
  -> GPU path and validates each shape against a CPU reference.

Verify the RTC backend is linked:

```bash
readelf -d build/libck_fmha_provider_plugin.so | grep NEEDED | grep hiprtc
#   0x0000000000000001 (NEEDED)  Shared library: [libhiprtc.so.7]

nm -D build/libck_fmha_provider_plugin.so | grep -E "compile_rtc|strip_host_bodies"
#   ... T ck_fmha_plugin::jit::compile_rtc(...)
#   ... T ck::host::strip_host_bodies(std::string_view)
```

Both symbols must be present for the RTC backend to be active.

### Runtime configuration

The plugin reads three environment variables:

| Variable | Purpose | Example |
|---|---|---|
| `CK_FMHA_ENABLE_JIT` | Must be `1` for any JIT path (RTC or hipcc). Without this the plugin only uses precompiled kernels. | `1` |
| `CK_FMHA_JIT_BACKEND` | Selects the JIT backend: `rtc`, `hipcc`, or `auto`. Default is `auto` (try RTC first, fall back to hipcc on failure). Set to `rtc` to pin RTC. | `rtc` |
| `CK_FMHA_RTC_TRACE` | Opt-in trace. Prints one line per kernel launch (name, grid, block, shape). Useful for validating that new shapes actually go through the RTC path. | `1` |

And hipDNN itself needs to know where your plugin `.so` lives:

| Variable | Purpose |
|---|---|
| `HIPDNN_PLUGIN_PATH` | Directory containing `ck_fmha_provider_plugin.so`. |

### Running the end-to-end demo

The included `ck_fmha_e2e_rtc_demo` target builds an SDPA forward graph
through the hipDNN Graph API (`Graph::sdpa` -> `graph.build` ->
`graph.execute`), forces the RTC backend, runs the kernel, and
compares the GPU output against a CPU reference computed as
`softmax(Q @ K^T / sqrt(d)) @ V`.

```bash
cd dnn-providers/ck-fmha-provider/build

HIPDNN_PLUGIN_PATH=$PWD                                                  \
LD_LIBRARY_PATH=/workspace/rocm-lib-copy/projects/hipdnn/build/release/lib:$LD_LIBRARY_PATH \
  ./integration_tests/ck_fmha_e2e_rtc_demo --warmup 2 --repeat 5
```

Expected output on MI355X (gfx950):

```
================================================================
  CK FMHA hipDNN plugin -- hipRTC end-to-end demo
  Device: AMD Instinct MI355X (gfx950:sramecc+:xnack-)
  Warmup: 2  Repeat: 5
  JIT backend: hipRTC (CK_FMHA_JIT_BACKEND=rtc)
================================================================

Shape                                 build(ms)   exec(ms)    TFLOPS    match?    max_abs     mean_abs    ref_rms
----------------------------------------------------------------------------------------------------------------------
B=2 H=4 Sq=128 Sk=128 Dq=128 Dv=128   7798.257    0.008       8.216     OK        0.000       0.000       0.002
B=2 H=8 Sq=256 Sk=256 Dq=128 Dv=128   7649.292    0.013       42.528    OK        0.000       0.000       0.001
B=1 H=8 Sq=512 Sk=512 Dq=128 Dv=128   7766.993    0.021       50.363    OK        0.000       0.000       0.001

================================================================
  Parity vs CPU reference: 3/3 shapes match.
================================================================
```

Three things to notice:

1. `build(ms) ~ 7-8 s per shape`. That's a full hipRTC compile of
   `ck_tile::TileFmhaFwdKernel<...>` for `gfx950`. The MGX template
   bakes `batch/nhead/seqlen_q/seqlen_k` into `make_descriptor` as
   `constexpr`, so each distinct shape triggers a fresh compile (see
   [_Per-shape compile is intrinsic to the MGX template_](#per-shape-compile-is-intrinsic-to-the-mgx-template)
   below).
2. `exec(ms) << 1 ms`. Once compiled, the kernel runs at standard CK
   Tile throughput. TFLOPS scales with workload as expected.
3. `match? = OK` under a strict, scale-aware tolerance
   (`atol = 1e-3 * ref_rms`, `rtol = 2e-2`). The demo fails loudly
   if the hipRTC path disagrees with the CPU reference.

### Telemetry

The plugin maintains a process-wide `RtcStats` counter so operators can
observe JIT churn in production. Enable the stderr summary with
`CK_FMHA_RTC_STATS=1`:

```bash
CK_FMHA_RTC_STATS=1 ./your_hipdnn_binary
# at process exit:
#   [CK FMHA hipRTC stats: attempts=5 (success=5, fail=0), cache=5/5 hits,
#    cold_avg=0.0ms, hot_avg=1.53ms, total_compile=0.00s,
#    total_cache_load=0.01s]
```

Fields:

- `attempts` -- total calls into `compile_rtc` (in-process registry hits
  don't count; those never reach the compile layer).
- `success / fail` -- split of `attempts` into successful compiles vs
  hipRTC errors. The last failure's message is kept in
  `RtcStats::last_error` for a richer post-mortem.
- `cache=<hits>/<attempts>` -- how many attempts were served from the
  on-disk HSACO cache. In a steady-state production workload this
  should approach 100 % of attempts after the first deploy.
- `cold_avg` -- mean `compile_rtc` wall-time for cache misses. Realistic
  numbers on gfx950 are ~7-9 s per kernel; `CK_FMHA_RTC_CACHE_DIR`
  amortises this across processes.
- `hot_avg` -- mean wall-time for cache hits. Should be ~1-5 ms on a
  warm page cache, dominated by `hipModuleLoadData`.
- `total_compile` / `total_cache_load` -- cumulative JIT wall-time, so
  you can account for the first-process tax when sizing warm-up
  budgets.

### Debugging a failing RTC compile

Set `CK_FMHA_RTC_SHOW_BUILD_STDERR=1` to un-silence the hipRTC
warnings that the demo suppresses by default. Real errors from
the RTC backend flow through `HIPDNN_PLUGIN_LOG_WARN` and appear
via `hipdnnSetCallback` once a logger is registered.

Set `CK_FMHA_RTC_TRACE=1` to get a per-launch diagnostic line,
useful for confirming that the right shape reached
`RtcFmhaKernelInstance::launch`:

```
[CK_FMHA_RTC] rtc_fwd_fp16_gfx950_B2H4Sq128Sk128 batch=2 nhead_q=4
              Sq=128 Sk=128 Dq=128 Dv=128 grid=(4,2,2) block=(256)
```

### On-disk RTC cache layout

The RTC backend persists every successful compile as a standard
AMDGPU HSA Code Object, so the cache is interoperable with the
rest of the ROCm toolchain:

```
$CK_FMHA_RTC_CACHE_DIR/
    gfx950-384b08261326a825.hsaco   # single-arch HSACO (ELF64 EM_AMDGPU)
    gfx950-384b08261326a825.json    # sidecar: arch / kernel / solution / hash
    gfx950-67ca4ae935f16c71.hsaco
    gfx950-67ca4ae935f16c71.json
    ...
```

Every `.hsaco` is a raw single-arch code object. No
`__CLANG_OFFLOAD_BUNDLE__` wrapper, so the file is directly
inspectable:

```bash
# Verify an entry is a valid gfx950 HSACO
file  $CK_FMHA_RTC_CACHE_DIR/gfx950-<hash>.hsaco
#   -> ELF 64-bit LSB shared object, AMD GPU architecture version 1

llvm-readelf -h $CK_FMHA_RTC_CACHE_DIR/gfx950-<hash>.hsaco | grep -E "OS/ABI|Machine|Flags"
#   OS/ABI:  AMDGPU - HSA
#   Machine: EM_AMDGPU
#   Flags:   0xE4F, gfx950, xnack-, sramecc+

# Disassemble the kernel's `.text` with standard ROCm tooling
llvm-objdump -d --arch-name=amdgcn --mcpu=gfx950 \
    $CK_FMHA_RTC_CACHE_DIR/gfx950-<hash>.hsaco

# Read the AMDGPU metadata note (kargs, group/private sizes, register usage)
llvm-readelf --notes $CK_FMHA_RTC_CACHE_DIR/gfx950-<hash>.hsaco

# Peek at the sidecar to see what CK-Tile template the hsaco came from
cat $CK_FMHA_RTC_CACHE_DIR/gfx950-<hash>.json
```

At load time the plugin re-verifies every `.hsaco` before handing
it to `hipModuleLoadData`:

- magic must be `\x7fELF` + ELF64
- `e_machine == EM_AMDGPU`
- `e_flags[0..9]` must decode to the expected gfx* machine for the
  current device

Stale or cross-arch entries are removed on sight and fall through
to recompile rather than risk a runtime memory fault.

Because the per-entry format is the standard HSACO, you can also:

- Package the cache into a fat binary with `clang-offload-bundler`
  when shipping multi-arch deployments
- Feed entries into existing rocprofv2 / MIOpen cache tooling
- Pre-build caches in CI (run the plugin once, archive the
  directory) and ship them with your container

### Per-shape compile is intrinsic to the MGX template

The current `ck::host::fmha_rtc::compile_fwd` renders a kernel source
derived from the MGX validation kernel in
`projects/composablekernel/codegen/test/fmha_fwd.cpp`. That template
passes runtime shapes as `constexpr` into `Kernel::make_descriptor`,
so the **compiled** kernel is valid only for the exact
`(batch, nhead, seqlen_q, seqlen_k)` tuple it was emitted for.

`RtcFmhaKernelInstance::supports()` enforces that invariant at
kernel-selection time -- a new shape cannot alias onto an existing
registry entry. Concretely:

- Each unique `(signature, tile, shape)` triple gets its own
  `FmhaKernelKey` (we fold a 32-bit hash of the runtime shape into
  `algorithm.selection_rank`) and its own hipRTC compile.
- The first call per shape costs ~7-15 s (fresh compile). Subsequent
  calls with the same shape in the same process cost ~0.1 ms
  (registry cache hit).
- Moving across processes invalidates the in-memory cache. See
  [Phase 10 on-disk RTC cache](#phase-10-on-disk-rtc-cache) below for
  how persistent caching drops the cold-cache cost to `hipModuleLoadData`
  (~50 ms).

Phase 2 follow-up work on `fmha_fwd_wrapper.hpp` will expose
`batch/nhead/seqlen_q/seqlen_k` as runtime kargs, at which point one
compiled kernel can service any shape in its hdim bucket, the way
production FMHA kernels already do. Until that lands, per-shape
compilation is the correct default.

### How RTC relates to precompiled kernels

The two are complementary. A production deployment typically uses:

```
Precompiled kernels (registered at handle construction) -----.
    .---> tried first by CkFmhaFwdPlanBuilder::isApplicable   |
    |                                                         |
RTC fallback on miss (CK_FMHA_ENABLE_JIT=1)   <---------------'
    .---> compile + register; subsequent calls hit the cache
```

Set `CK_FMHA_DEFAULT_BACKEND_RTC=ON` at build time to make `rtc` the
default backend (today the default is `auto`: try RTC first, fall
back to hipcc on failure).

## End-to-End Flow

### 1. Plugin Discovery

hipDNN loads `ck_fmha_provider_plugin.so` via `dlopen` at startup. The
`EnginePluginImpl.inl` macros expose the required C API surface
(`hipdnnEnginePluginGetAllEngineIds`, etc.).

```
hipDNN backend
  dlopen("ck_fmha_provider_plugin.so")
    hipdnnEnginePluginGetAllEngineIds()
      CkFmhaContainer::copyEngineIds()
        returns CK_FMHA_ENGINE_ID
```

### 2. Handle Creation

When `hipdnnEnginePluginCreate()` is called, a `CkFmhaHandle` is constructed:

```
CkFmhaHandle()
  detect_gfx_arch()           e.g. "gfx950"
  make_fmha_registry()        FmhaRegistry with all precompiled kernels
  filter_by_arch(arch)        Remove kernels for other architectures
  FmhaDispatcher(registry)    Kernel selection engine
  set_benchmarking(false)     One-shot execution, no timing overhead
  loadSupplementalKernels()   Scan CK_FMHA_KERNEL_LIB_PATH for extra .so
```

### 3. Graph Applicability Check

When the user constructs an SDPA graph and queries applicable engines:

```
graph.sdpa(Q, K, V, attrs)   SdpaAttributes graph

hipdnnEnginePluginGetApplicableEngineIds(handle, graph)
  CkFmhaEngine::isApplicable()
    CkFmhaFwdPlanBuilder::isApplicable()
      CkFmhaParamParser::isFwdSdpaGraph()   nodeCount==1, SdpaAttributes
      CkFmhaParamParser::parseFwdGraph()     Extract dims, dtype, mask, bias
      FmhaProblemBuilder::build()            Construct FmhaProblem
      dispatcher->select_kernel(problem)     non-null = applicable
```

### 4. Plan Building

When the user creates an execution context:

```
hipdnnEnginePluginCreateExecutionContext(handle, config, graph)
  CkFmhaEngine::initializeExecutionContext()
    CkFmhaFwdPlanBuilder::buildPlan()
      parseFwdGraph()                ParsedFwdParams (UIDs, dims, layout)
      buildFwdProblem()              FmhaProblem
      problem.canonical_key()        Cache key (25+ fields)
      handle.getCachedPlan(key)      Check cache (mutex-protected)
        (miss) dispatcher->plan()    FmhaExecutionPlan with kernel IDs
               handle.cachePlan()    Store for reuse
      CkFmhaFwdPlan(params, plan)    Ready for execution
```

For backward, the flow additionally queries `bwd_workspace_info()`:

```
CkFmhaBwdPlanBuilder::getMaxWorkspaceSize()
  bwd_workspace_info(problem)
    d_bytes      = B * Hq * Sq * 4          (fp32 scratch)
    dq_acc_bytes = B * Hq * Sq * Dq * 4     (fp32 accumulator)
    total_bytes  = align(d + dq_acc, 256)
```

### 5. Execution

When the user executes the graph:

```
hipdnnEnginePluginExecuteOpGraph(handle, ctx, workspace, buffers)
  CkFmhaFwdPlan::execute()
    findBuffer(q_uid)  void* Q device pointer
    findBuffer(k_uid)  void* K device pointer
    ...                Map all UIDs to device pointers
    Fill fmha_fwd_traits    dtype, mask, bias, dropout flags
    Fill fmha_fwd_args      pointers, dims, strides (BHSD/BSHD)
    FmhaInvocation::make(traits, args)
    dispatcher->run(invocation, stream)
      FmhaProblem::from_invocation()
      plan(problem)                  Select kernels
      run_plan(plan, invocation)     HIP kernel launch (one-shot)
```

For backward, workspace is suballocated before dispatch:

```
CkFmhaBwdPlan::execute(handle, buffers, workspace)
  d_ptr      = workspace + 0
  dq_acc_ptr = workspace + align(d_bytes, 256)
  Fill fmha_bwd_traits + fmha_bwd_args
  dispatcher->run(invocation, stream)
    BwdDotDoO stage    (compute d = rowsum(dO * O))
    BwdDqDkDv stage    (compute dQ, dK, dV)
    BwdConvertDq stage (optional: fp32 to output precision)
```

### 6. Teardown

```
hipdnnEnginePluginDestroyExecutionContext()  Drops IPlan
hipdnnEnginePluginDestroy()                  Drops CkFmhaHandle
                                              dlclose supplemental .so handles
                                              Registry + dispatcher destroyed
```

### 7. JIT Compilation Path

When the precompiled registry has no matching kernel, the plugin falls
back to a JIT compilation step. The backend is selected at runtime via
`CK_FMHA_JIT_BACKEND`:

```
CK_FMHA_JIT_BACKEND=rtc     in-process hipRTC (preferred)
CK_FMHA_JIT_BACKEND=hipcc   python3 + hipcc subprocess (legacy)
CK_FMHA_JIT_BACKEND=auto    try rtc, fall back to hipcc (default)
```

The compile-time flag `CK_FMHA_DEFAULT_BACKEND_RTC` (off by default)
changes the default from Auto to Rtc once the deployment has validated
RTC on its (arch, dtype, family, shape) matrix.

**RTC path** (preferred):

```
FmhaProblem (no kernel found in registry)
  jit::compile_rtc(problem, arch)
    ck::host::fmha_rtc::compile_fwd(...)
      GetTileHeaders()           strip_host_bodies on each ck_tile/**.hpp
      InterpolateString()        kernel source from wrapper template
      rtc::compile_kernel()      hipRTC -> hipModule_t + hipFunction_t
    RtcFmhaKernelInstance        registered directly into FmhaRegistry
  Retry: dispatcher->select_kernel()  now finds the RTC kernel
```

First-call latency: ~5-15 s (hipRTC skips the host-compile stage that
dominates hipcc). Cache hit: ~1 ms (`hipModule_t` held in-process).

**hipcc path** (legacy, still supported):

```
FmhaProblem (no kernel found in registry)
  jit::compile_hipcc(problem, arch)
    Shells out to: python3 -c "from fmha_utils import setup_fmha_dispatcher; ..."
    generate_fmha_fallback.py  hipcc  .so  (~20-40s first time)
    Cached to CK_FMHA_JIT_CACHE_DIR (instant on subsequent calls)
  load_jit_library(so_path, registry, arch)
    dlopen  dlsym("ck_fmha_register_kernels")  merge_from into registry
  Retry: dispatcher->select_kernel()  now finds the JIT'd kernel
```

The hipcc `.so` persists to disk, so it only compiles once per unique
kernel config per deployment. The offline warmup tool
(`fmha_utils.setup_multiple_fmha_dispatchers`) remains the recommended
way to pre-build kernels for production deployments with zero runtime
compilation cost, regardless of backend choice.

See [docs/ADRs/0001-hiprtc-as-jit-backend.md](docs/ADRs/0001-hiprtc-as-jit-backend.md)
for the architecture decision record and rollback procedure.

## End-to-End Usage (hipDNN Frontend API)

```cpp
#include <hipdnn_frontend.hpp>
using namespace hipdnn_frontend::graph;

// Create hipDNN handle (discovers + loads CK FMHA plugin)
hipdnnHandle_t handle;
hipdnnCreate(&handle);

// Build SDPA graph
Graph graph;
graph.set_io_data_type(DataType::HALF)
     .set_compute_data_type(DataType::FLOAT);

auto Q = Graph::tensor(TensorAttributes()
    .set_dim({B, Hq, Sq, Dq})
    .set_stride({Hq*Sq*Dq, Sq*Dq, Dq, 1})  // BHSD layout
    .set_uid(1));
auto K = Graph::tensor(TensorAttributes()
    .set_dim({B, Hk, Sk, Dq})
    .set_stride({Hk*Sk*Dq, Sk*Dq, Dq, 1})
    .set_uid(2));
auto V = Graph::tensor(TensorAttributes()
    .set_dim({B, Hk, Sk, Dv})
    .set_stride({Hk*Sk*Dv, Sk*Dv, Dv, 1})
    .set_uid(3));

SdpaAttributes attrs;
attrs.attn_scale_value = 1.0f / std::sqrt(float(Dq));
attrs.causal_mask = true;           // top-left causal
attrs.generate_stats = true;        // needed for backward

auto [O, stats] = graph.sdpa(Q, K, V, std::move(attrs));
O->set_output(true).set_uid(4);
stats->set_output(true).set_uid(5);

// Build (plugin applicability check + kernel selection + plan creation)
graph.build(handle);

// Query workspace
int64_t ws_size;
graph.get_workspace_size(ws_size);
void* workspace;
hipMalloc(&workspace, ws_size);

// Execute
std::unordered_map<std::shared_ptr<TensorAttributes>, void*> tensor_map = {
    {Q, d_q}, {K, d_k}, {V, d_v}, {O, d_o}, {stats, d_stats}
};
graph.execute(handle, tensor_map, workspace);

// Backward (requires stats from forward)
Graph bwd_graph;
bwd_graph.set_io_data_type(DataType::HALF)
         .set_compute_data_type(DataType::FLOAT);
// ... create backward tensors with same UIDs ...
SdpaBackwardAttributes bwd_attrs;
bwd_attrs.attn_scale_value = 1.0f / std::sqrt(float(Dq));
bwd_attrs.causal_mask = true;

auto [dQ, dK, dV] = bwd_graph.sdpa_backward(
    bQ, bK, bV, bO, bdO, bStats, std::move(bwd_attrs));
dQ->set_output(true).set_uid(7);
dK->set_output(true).set_uid(8);
dV->set_output(true).set_uid(9);

bwd_graph.build(handle);
```

## Verified End-to-End (MI355X gfx950)

The following shapes ran successfully through the full hipDNN Graph API path
(`Graph::sdpa()` -> `graph.build()` -> `graph.execute()` -> CK kernel on GPU):

```
Shape                                             Time(ms)    TFLOPS
--------------------------------------------------------------------------
B=2 Hq=4 Hk=4 Sq=128 Sk=128 Dq=128 Dv=128         0.01        6.61
B=1 Hq=32 Hk=8 Sq=2048 Sk=2048 Dq=128 Dv=128      0.18        375.58
B=4 Hq=32 Hk=8 Sq=2048 Sk=2048 Dq=128 Dv=128      0.70        390.89
B=4 Hq=64 Hk=4 Sq=2048 Sk=2048 Dq=128 Dv=128      1.36        404.38
B=1 Hq=14 Hk=2 Sq=1024 Sk=1024 Dq=64 Dv=64        0.04        88.47
B=2 Hq=8 Hk=4 Sq=1024 Sk=1024 Dq=128 Dv=128       0.04        193.80
```

These numbers are with a single precompiled kernel (h128 fp16 qr_async).
With the full receipt-curated kernel set (~50-100 kernels), seqtune-aware
selection picks optimal kernels per shape for higher peak TFLOPS.

## Testing Matrix Shapes

Benchmark shapes are aligned with `ck_fmha_testing_matrix.yaml` (smoke tier).
Run `ck_fmha_e2e_demo` to collect TFLOPS on your target hardware.

| Test Name | Key Shape | Model Archetype |
|-----------|-----------|-----------------|
| GQA_4to1_Prefill_Basic | B=1 Hq=32 Hk=8 Sq=2048 Sk=2048 D=128 | Llama-3-8B |
| GQA_4to1_Prefill_Basic | B=4 Hq=32 Hk=8 Sq=2048 Sk=2048 D=128 | Llama-3-8B batched |
| GQA_16to1_Large | B=1 Hq=64 Hk=4 Sq=2048 Sk=2048 D=128 | 70B-class |
| Small_GQA_7to1_SubWarp | B=1 Hq=14 Hk=2 Sq=1024 Sk=1024 D=64 | Sub-warp loads |
| CK_All_Hdim_Sweep | B=2 Hq=8 Hk=4 Sq=1024 Sk=1024 D=32..256 | hdim coverage |
| MQA_128to8_Decode | B=8..64 Hq=128 Hk=8 Sq=1 Sk=1024..4096 D=128 | 405B decode |
| MLA_Sparse_Decode | B=1..4 Hq=128 Hk=128 Sq=1 Sk=1024..4096 D=192/128 | R1-class MLA |
| Extreme_GQA_Ratios | B=2 Hq=5..48 Hk=1..8 Sq=1024 Sk=1024 D=128 | Exotic GQA |
| Prefill_Odd_Lengths | B=2 Hq=32 Hk=8 Sq=113..3131 D=128 | Padding stress |
| CK_Tiny_Sequences | B=1..2 Sq=1..33 Sk=10..99 D=128 | Edge cases |
| Vision_Transformer | B=1..4 Hq=16..40 D=88..128 | ViT hybrid |

## Supported Configurations

| Feature | Forward | Backward |
|---------|---------|----------|
| fp16 | Yes | Yes |
| bf16 | Yes | Yes |
| BHSD layout | Yes | Yes |
| BSHD layout | Yes | Yes |
| hdim 32/64/80/96/128/160/192/256 | Yes | Yes |
| GQA (nhead_q > nhead_k) | Yes | Yes |
| Causal mask (top-left) | Yes | Yes |
| Causal mask (bottom-right) | Yes | Yes |
| Window mask (generic) | Yes | Yes |
| Elementwise bias | Yes | Yes |
| ALiBi bias | Yes | Yes |
| Dropout | Yes | Yes |
| LSE / stats output | Yes | N/A (input) |
| dBias output | N/A | Yes |

## Building

```bash
# As part of hipDNN build (recommended)
cmake -DHIPDNN_BUILD_CK_FMHA_PLUGIN=ON ...

# Standalone
cd dnn-providers/ck-fmha-provider
cmake -B build -DCK_ROOT=/path/to/composablekernel
cmake --build build

# Run unit tests
./build/ck_fmha_provider_tests

# Run GPU integration tests
./build/integration_tests/ck_fmha_integration_tests

# Run end-to-end TFLOPS benchmark (full hipDNN graph path)
./build/integration_tests/ck_fmha_e2e_demo --warmup 5 --repeat 20 --bwd
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `HIPDNN_PLUGIN_PATH` | Directory hipDNN searches for `ck_fmha_provider_plugin.so`. Required by every demo and test that talks to the plugin. |
| `CK_FMHA_KERNEL_LIB_PATH` | Directory of supplemental precompiled kernel `.so` files, loaded at `CkFmhaHandle` construction. |
| `CK_FMHA_ENABLE_JIT` | Set to `1` to enable JIT compilation fallback for missing kernels. Without this the plugin only serves problems whose kernel is already in `FmhaRegistry`. |
| `CK_FMHA_JIT_BACKEND` | JIT backend selector. `rtc` = in-process hipRTC (requires `-DCK_FMHA_WITH_RTC=ON`); `hipcc` = python3 + hipcc subprocess (legacy); `auto` = try RTC, fall back to hipcc on failure (default). |
| `CK_FMHA_JIT_CACHE_DIR` | Cache directory for hipcc-backend JIT-compiled `.so`s (default: `/tmp/ck_fmha_jit`). Unused by the RTC backend. |
| `CK_FMHA_RTC_CACHE_DIR` | On-disk cache directory for RTC-compiled code objects (default: `$HOME/.cache/ck_fmha_rtc`). Set to `""` to disable persistent RTC caching. |
| `CK_FMHA_RTC_TRACE` | Set to `1` to print one diagnostic line per kernel launch from `RtcFmhaKernelInstance::launch`. |
| `CK_FMHA_RTC_SHOW_BUILD_STDERR` | Set to `1` to see the hipRTC compile warnings that the demo normally silences. Useful when debugging a compile failure. |
| `CK_FMHA_RTC_STATS` | Set to `1` to print a one-line summary of `RtcStats` (compile attempts, cache hits/misses, cold_avg, hot_avg, total_compile_time, total_cache_load_time) to stderr at process exit. |
| `CK_DISPATCHER_PYTHON_PATH` | Path to CK dispatcher Python modules. Required only when using the `hipcc` backend. |

## Architecture

```
CkFmhaPluginPublic.cpp          EnginePluginImpl.inl macros -> C API
  CkFmhaContainer               Engine factory (one CK_FMHA_ENGINE)
    CkFmhaEngine                 Routes to plan builders
      CkFmhaFwdPlanBuilder  ->   CkFmhaFwdPlan
      CkFmhaBwdPlanBuilder  ->   CkFmhaBwdPlan

CkFmhaHandle                    Per-device: registry, dispatcher, plan cache
CkFmhaParamParser               hipDNN SdpaAttributes <-> CK fmha_*_traits/args
CkFmhaJit                       hipcc subprocess JIT (slow mode)
```
