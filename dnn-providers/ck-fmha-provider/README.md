# CK FMHA Provider Plugin for hipDNN

A hipDNN engine plugin that executes Scaled Dot-Product Attention (SDPA)
forward and backward operations through the Composable Kernel (CK) FMHA
dispatcher.

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
  detect_gfx_arch()           "gfx942"
  make_fmha_registry()        FmhaRegistry with all precompiled kernels
  filter_by_arch("gfx942")    Remove kernels for other architectures
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

### 7. JIT Compilation Path (Slow Mode)

When the precompiled registry has no matching kernel:

```
FmhaProblem (no kernel found in registry)
  jit_compile_kernel(problem, arch)
    Shells out to: python3 -c "from fmha_utils import setup_fmha_dispatcher; ..."
    generate_fmha_fallback.py  hipcc  .so  (~20-40s first time)
    Cached to CK_FMHA_JIT_CACHE_DIR (instant on subsequent calls)
  load_jit_library(so_path, registry, arch)
    dlopen  dlsym("ck_fmha_register_kernels")  merge_from into registry
  Retry: dispatcher->select_kernel()  now finds the JIT'd kernel
```

The JIT `.so` persists to disk, so it only compiles once per unique kernel
config per deployment. The offline warmup tool
(`fmha_utils.setup_multiple_fmha_dispatchers`) can pre-build these for
production deployments with zero runtime compilation cost.

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

## Performance (Measured on MI300X gfx942, fp16)

Shapes aligned with `ck_fmha_testing_matrix.yaml`. TFLOPS from actual
benchmarks (`fmha_bench_all.csv`), best kernel per shape.

### Forward SDPA (No Mask)

| Shape | Model Archetype | Latency (ms) | TFLOPS |
|-------|-----------------|---------------|--------|
| B=4 Hq=32 Hk=8 Sq=2048 Sk=2048 D=128 | Llama-3-8B GQA prefill | 0.333 | 825.7 |
| B=1 Hq=64 Hk=4 Sq=2048 Sk=2048 D=128 | 70B-class GQA prefill | 0.169 | 811.3 |
| B=1 Hq=32 Hk=8 Sq=2048 Sk=2048 D=128 | Llama-3-8B single-batch | 0.087 | 794.5 |
| B=4 Hq=64 Hk=4 Sq=2048 Sk=2048 D=128 | 70B-class batched | 0.753 | 730.2 |
| B=2 Hq=8 Hk=4 Sq=1024 Sk=1024 D=192/128 | Asymmetric hdim (MLA) | 0.034 | 318.1 |
| B=2 Hq=8 Hk=4 Sq=1024 Sk=1024 D=256 | H256 high LDS pressure | 0.056 | 308.4 |
| B=2 Hq=8 Hk=4 Sq=1024 Sk=1024 D=128 | Standard GQA 2:1 | 0.031 | 277.5 |
| B=2 Hq=8 Hk=4 Sq=1024 Sk=1024 D=64 | Small hdim GQA | 0.024 | 178.6 |
| B=1 Hq=14 Hk=2 Sq=1024 Sk=1024 D=64 | Small GQA 7:1 | 0.024 | 155.6 |
| B=2 Hq=8 Hk=4 Sq=1024 Sk=1024 D=32 | Minimal hdim | 0.019 | 116.1 |

### Forward SDPA (Causal Mask, Top-Left)

| Shape | Latency (ms) | TFLOPS |
|-------|---------------|--------|
| B=4 Hq=32 Hk=8 Sq=2048 Sk=2048 D=128 | 0.338 | 814.4 |
| B=1 Hq=64 Hk=4 Sq=2048 Sk=2048 D=128 | 0.170 | 809.2 |
| B=1 Hq=32 Hk=8 Sq=2048 Sk=2048 D=128 | 0.087 | 792.4 |
| B=4 Hq=64 Hk=4 Sq=2048 Sk=2048 D=128 | 0.697 | 788.7 |
| B=2 Hq=8 Hk=4 Sq=1024 Sk=1024 D=128 | 0.031 | 281.1 |
| B=2 Hq=8 Hk=4 Sq=1024 Sk=1024 D=64 | 0.025 | 172.9 |

### Forward SDPA (bf16)

| Shape | Latency (ms) | TFLOPS |
|-------|---------------|--------|
| B=4 Hq=32 Hk=8 Sq=2048 Sk=2048 D=128 | 0.311 | 883.2 |
| B=1 Hq=64 Hk=4 Sq=2048 Sk=2048 D=128 | 0.157 | 873.3 |
| B=1 Hq=32 Hk=8 Sq=2048 Sk=2048 D=128 | 0.081 | 850.6 |
| B=4 Hq=64 Hk=4 Sq=2048 Sk=2048 D=128 | 0.720 | 763.3 |

### Decode Shapes (Single-Token Query)

| Shape | Latency (ms) | TFLOPS |
|-------|---------------|--------|
| B=64 Hq=128 Hk=8 Sq=1 Sk=1024 D=128 | 0.310 | 13.8 |
| B=64 Hq=128 Hk=8 Sq=1 Sk=4096 D=128 | 1.380 | 12.5 |
| B=8 Hq=128 Hk=8 Sq=1 Sk=4096 D=128 | 0.247 | 8.7 |
| B=8 Hq=128 Hk=8 Sq=1 Sk=1024 D=128 | 0.067 | 8.1 |
| B=1 Hq=128 Hk=8 Sq=1 Sk=4096 D=128 | 0.115 | 2.3 |
| B=1 Hq=128 Hk=8 Sq=1 Sk=1024 D=128 | 0.029 | 2.3 |

### Testing Matrix Shape Coverage

Shapes from `ck_fmha_testing_matrix.yaml` (smoke tier). Shapes marked
with a check have verified benchmark data:

| Test Name | Key Shape | TFLOPS | Status |
|-----------|-----------|--------|--------|
| GQA_4to1_Prefill_Basic | B=1 Hq=32 Hk=8 Sq=2048 Sk=2048 D=128 | 794.5 | Verified |
| GQA_4to1_Prefill_Basic (batch=4) | B=4 Hq=32 Hk=8 Sq=2048 Sk=2048 D=128 | 825.7 | Verified |
| Small_GQA_7to1_SubWarp | B=1 Hq=14 Hk=2 Sq=1024 Sk=1024 D=64 | 155.6 | Verified |
| CK_All_Hdim_Sweep (h32) | B=2 Hq=8 Hk=4 Sq=1024 Sk=1024 D=32 | 116.1 | Verified |
| CK_All_Hdim_Sweep (h64) | B=2 Hq=8 Hk=4 Sq=1024 Sk=1024 D=64 | 178.6 | Verified |
| CK_All_Hdim_Sweep (h128) | B=2 Hq=8 Hk=4 Sq=1024 Sk=1024 D=128 | 277.5 | Verified |
| CK_All_Hdim_Sweep (h160) | B=2 Hq=8 Hk=4 Sq=1024 Sk=1024 D=160 | 281.6 | Verified |
| CK_All_Hdim_Sweep (h192) | B=2 Hq=8 Hk=4 Sq=1024 Sk=1024 D=192 | 285.9 | Verified |
| CK_All_Hdim_Sweep (h256) | B=2 Hq=8 Hk=4 Sq=1024 Sk=1024 D=256 | 308.4 | Verified |
| CK_All_Hdim_Sweep (h192x128) | B=2 Hq=8 Hk=4 Sq=1024 Sk=1024 D=192/128 | 318.1 | Verified |
| CK_All_Hdim_Sweep (h80x96) | B=2 Hq=8 Hk=4 Sq=1024 Sk=1024 D=80/96 | 221.8 | Verified |
| CK_All_Hdim_Sweep (h96x128) | B=2 Hq=8 Hk=4 Sq=1024 Sk=1024 D=96/128 | 265.4 | Verified |
| GQA_16to1_Large | B=1 Hq=64 Hk=4 Sq=2048 Sk=2048 D=128 | 811.3 | Verified |
| GQA_16to1_Large (batch=4) | B=4 Hq=64 Hk=4 Sq=2048 Sk=2048 D=128 | 730.2 | Verified |
| MQA_128to8_Decode (B=8) | B=8 Hq=128 Hk=8 Sq=1 Sk=1024 D=128 | 8.1 | Verified |
| MQA_128to8_Decode (B=64) | B=64 Hq=128 Hk=8 Sq=1 Sk=1024 D=128 | 13.8 | Verified |
| Extreme_GQA_Ratios | B=2 Hq=5..48 Hk=1..8 Sq=1024 Sk=1024 D=128 | -- | Not yet benchmarked |
| Prefill_Odd_Lengths | B=2 Hq=32 Hk=8 Sq=113..3131 Sk=203..3131 D=128 | -- | Not yet benchmarked |
| CK_Tiny_Sequences | B=1..2 Sq=1..33 Sk=10..99 | -- | Not yet benchmarked |
| MLA_Sparse_Decode | B=1 Hq=128 Hk=128 Sq=1 Sk=1024 D=192/128 | -- | Not yet benchmarked |
| Vision_Transformer_Shapes | B=1..4 Hq=16..40 D=88..128 | -- | Not yet benchmarked |

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
| `CK_FMHA_KERNEL_LIB_PATH` | Directory of supplemental precompiled kernel `.so` files |
| `CK_FMHA_JIT_CACHE_DIR` | Cache directory for JIT-compiled kernels (default: `/tmp/ck_fmha_jit`) |
| `CK_DISPATCHER_PYTHON_PATH` | Path to CK dispatcher Python modules (for JIT) |

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
