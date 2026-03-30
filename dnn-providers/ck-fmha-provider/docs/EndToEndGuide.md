# End-to-End hipDNN SDPA Demo: Build & Run Guide

## What was demonstrated

A standalone C++ program (`EndToEndSdpaDemo.cpp`) that:

1. Calls the hipDNN frontend C++ API to construct an SDPA graph
2. hipDNN backend discovers and loads the CK FMHA plugin `.so`
3. Plugin translates the hipDNN graph to CK kernel args, selects a kernel, executes on GPU
4. HIP events measure wall-clock time, program reports TFLOPS

This is the same code path a framework (PyTorch, vLLM) would use.

## Files involved

```
dnn-providers/ck-fmha-provider/
├── integration_tests/EndToEndSdpaDemo.cpp   ← The demo program
├── src/CkFmhaPluginPublic.cpp               ← Plugin C API entry (EnginePluginImpl.inl)
├── src/CkFmhaHandle.cpp                     ← Registry init + kernel registration
├── src/engines/CkFmhaParamParser.cpp        ← hipDNN SdpaAttributes → CK traits/args
├── src/engines/plans/CkFmhaFwdPlan.cpp      ← Fills fmha_fwd_args, calls dispatcher

projects/hipdnn/
├── frontend/include/hipdnn_frontend.hpp     ← Graph API (Graph::sdpa(), graph.build())
├── backend/src/HipdnnBackend.cpp            ← Plugin discovery via dlopen
├── build/lib/libhipdnn_backend.so           ← Built backend shared library
├── build/lib/hipdnn_plugins/engines/
│   └── ck_fmha_provider_plugin.so           ← Our plugin, installed here

projects/composablekernel/dispatcher/
├── build/examples/fmha_python_fallback/
│   ├── fmha_python_dispatch.hpp             ← Generated: REGISTER_GENERATED_KERNELS macro
│   └── fmha_fwd_fp16_...01feefd2ef4b.o     ← Compiled GPU kernel object
├── build/libck_tile_dispatcher.a            ← Dispatcher static library
└── include/ck_tile/dispatcher_fmha.hpp      ← C++ API headers
```

## Step 1: Build hipDNN

```bash
cd /workspace/rocm-libraries/projects/hipdnn

cmake -B build -G Ninja \
  -DCMAKE_CXX_COMPILER=hipcc \
  -DHIPDNN_SKIP_TESTS=ON \
  -DHIPDNN_GENERATE_SDK_HEADERS=OFF \
  -DENABLE_CLANG_TIDY=OFF \
  -DENABLE_CLANG_FORMAT=OFF \
  -DCMAKE_BUILD_TYPE=Release

ninja -C build hipdnn_backend
```

Output: `build/lib/libhipdnn_backend.so`

## Step 2: Build CK dispatcher FMHA library (if not already built)

```bash
cd /workspace/rocm-libraries/projects/composablekernel/dispatcher/build
cmake --build . --target dispatcher_fmha_lib -j$(nproc)
```

Output: `build/examples/libdispatcher_fmha_lib.so` (contains precompiled kernels)

The generated dispatch header is at:
`build/examples/fmha_python_fallback/fmha_python_dispatch.hpp`

## Step 3: Compile the plugin .so

Each plugin source file is compiled separately, then linked into a shared library.

```bash
HIPDNN=/workspace/rocm-libraries/projects/hipdnn
CK_DISP=/workspace/rocm-libraries/projects/composablekernel/dispatcher
CK_ROOT=/workspace/rocm-libraries/projects/composablekernel
PLUGIN=/workspace/rocm-libraries/dnn-providers/ck-fmha-provider
GEN_DIR=${CK_DISP}/build/examples/fmha_python_fallback

INCS="-I${PLUGIN}/src \
  -I${HIPDNN}/data_sdk/include \
  -I${HIPDNN}/data_sdk/include/hipdnn_data_sdk/data_objects/v25_9_23 \
  -I${HIPDNN}/backend/include \
  -I${HIPDNN}/plugin_sdk/include \
  -I${HIPDNN}/frontend/include \
  -I${HIPDNN}/build/backend/src/backend/include \
  -I${HIPDNN}/build/backend/src/include \
  -I${HIPDNN}/build/frontend/include \
  -I${HIPDNN}/build/data_sdk/include \
  -I${HIPDNN}/build/plugin_sdk/include \
  -I${HIPDNN}/build/_deps/flatbuffers-src/include \
  -I${HIPDNN}/build/_deps/json-src/include \
  -I${CK_DISP}/include \
  -I${CK_ROOT}/include \
  -I${CK_ROOT} \
  -I${GEN_DIR} \
  -include ${GEN_DIR}/fmha_python_dispatch.hpp \
  -D__HIP_PLATFORM_AMD__"

FLAGS="-std=c++17 -O2 -fPIC -w"
mkdir -p /tmp/ck_plugin_build

# Compile each source
for src in CkFmhaPluginPublic CkFmhaContainer CkFmhaHandle; do
  hipcc ${FLAGS} ${INCS} -c ${PLUGIN}/src/${src}.cpp -o /tmp/ck_plugin_build/${src}.o
done
for src in CkFmhaEngine CkFmhaParamParser; do
  hipcc ${FLAGS} ${INCS} -c ${PLUGIN}/src/engines/${src}.cpp -o /tmp/ck_plugin_build/${src}.o
done
for src in CkFmhaFwdPlanBuilder CkFmhaFwdPlan CkFmhaBwdPlanBuilder CkFmhaBwdPlan; do
  hipcc ${FLAGS} ${INCS} -c ${PLUGIN}/src/engines/plans/${src}.cpp -o /tmp/ck_plugin_build/${src}.o
done

# Link plugin .so (include kernel .o and static dispatcher lib)
hipcc -shared -fPIC -w /tmp/ck_plugin_build/*.o \
  ${CK_DISP}/build/libck_tile_dispatcher.a \
  ${GEN_DIR}/fmha_fwd_fp16_batch_h128x128_qr_async_01feefd2ef4b.o \
  -o /tmp/ck_plugin_build/ck_fmha_provider_plugin.so
```

Output: `/tmp/ck_plugin_build/ck_fmha_provider_plugin.so`

### Why `-include fmha_python_dispatch.hpp`?

This generated header defines the `REGISTER_GENERATED_KERNELS` macro, which
`CkFmhaHandle.cpp` calls to register precompiled GPU kernels into the
plugin's `FmhaRegistry`. Without it, the registry is empty and no shapes
are applicable.

### Why link the kernel `.o` file?

The `.o` contains the actual compiled GPU kernel (HIP fat binary for gfx950).
The `REGISTER_GENERATED_KERNELS` macro references the symbol
`make_fmha_fwd_fp16_batch_h128x128_qr_async_...()` defined in that `.o`.

## Step 4: Install plugin where hipDNN can find it

```bash
PLUGIN_DIR=${HIPDNN}/build/lib/hipdnn_plugins/engines
mkdir -p ${PLUGIN_DIR}
cp /tmp/ck_plugin_build/ck_fmha_provider_plugin.so ${PLUGIN_DIR}/
```

hipDNN discovers plugins by scanning `hipdnn_plugins/engines/` relative to
`libhipdnn_backend.so`, or from the path set via `HIPDNN_PLUGIN_PATH` env var.

## Step 5: Compile the demo program

The demo only links against `libhipdnn_backend.so` -- it does NOT link
against the plugin or CK directly. The plugin is discovered at runtime
via dlopen.

```bash
hipcc -std=c++17 -O2 -w \
  -I${HIPDNN}/frontend/include \
  -I${HIPDNN}/data_sdk/include \
  -I${HIPDNN}/data_sdk/include/hipdnn_data_sdk/data_objects/v25_9_23 \
  -I${HIPDNN}/backend/include \
  -I${HIPDNN}/plugin_sdk/include \
  -I${HIPDNN}/build/backend/src/backend/include \
  -I${HIPDNN}/build/backend/src/include \
  -I${HIPDNN}/build/frontend/include \
  -I${HIPDNN}/build/data_sdk/include \
  -I${HIPDNN}/build/_deps/flatbuffers-src/include \
  -I${HIPDNN}/build/_deps/json-src/include \
  -L${HIPDNN}/build/lib -lhipdnn_backend \
  -Wl,-rpath,${HIPDNN}/build/lib \
  -D__HIP_PLATFORM_AMD__ \
  ${PLUGIN}/integration_tests/EndToEndSdpaDemo.cpp \
  -o /tmp/ck_fmha_e2e_demo
```

## Step 6: Run

```bash
HIPDNN_PLUGIN_PATH=${HIPDNN}/build/lib/hipdnn_plugins/engines \
LD_LIBRARY_PATH=${HIPDNN}/build/lib:${LD_LIBRARY_PATH} \
  /tmp/ck_fmha_e2e_demo --warmup 2 --repeat 10
```

### Actual output (MI355X gfx950)

```
============================================================
  CK FMHA hipDNN Plugin End-to-End Benchmark
  Device: AMD Instinct MI355X (gfx950:sramecc+:xnack-)
  Warmup: 2  Repeat: 5
============================================================

  Plugin path: /.../hipdnn_plugins/engines

=== Forward SDPA (fp16, BHSD, no mask) ===

Shape                                             Time(ms)    TFLOPS
--------------------------------------------------------------------------
B=2 Hq=4 Hk=4 Sq=128 Sk=128 Dq=128 Dv=128         0.01        6.61
B=1 Hq=32 Hk=8 Sq=2048 Sk=2048 Dq=128 Dv=128      0.18        375.58
B=4 Hq=32 Hk=8 Sq=2048 Sk=2048 Dq=128 Dv=128      0.70        390.89
B=4 Hq=64 Hk=4 Sq=2048 Sk=2048 Dq=128 Dv=128      1.36        404.38
B=1 Hq=14 Hk=2 Sq=1024 Sk=1024 Dq=64 Dv=64        0.04        88.47
B=2 Hq=8 Hk=4 Sq=1024 Sk=1024 Dq=128 Dv=128       0.04        193.80
```

## What happens at runtime (trace)

```
1. Demo calls hipdnnSetEnginePluginPaths_ext(HIPDNN_PLUGIN_PATH)
2. Demo calls hipdnnCreate(&handle)
   └─ Backend scans plugin dir, finds ck_fmha_provider_plugin.so
   └─ dlopen() → loads .so, static constructors register GPU kernels
   └─ hipdnnEnginePluginCreate() → CkFmhaHandle():
      ├─ detect_gfx_arch() → "gfx950"
      ├─ REGISTER_GENERATED_KERNELS(registry, "gfx950") → 1 kernel registered
      ├─ filter_by_arch("gfx950")
      └─ FmhaDispatcher(registry, benchmarking=false)

3. Demo constructs Graph with Q,K,V tensors + SdpaAttributes
4. Demo calls graph.build(handle)
   └─ Backend serializes graph to FlatBuffer
   └─ Queries plugin: hipdnnEnginePluginGetApplicableEngineIds()
      └─ CkFmhaEngine::isApplicable()
         └─ CkFmhaFwdPlanBuilder::isApplicable()
            ├─ isFwdSdpaGraph() → true (1 node, SdpaAttributes type)
            ├─ parseFwdGraph() → extracts dims, dtype, strides from tensors
            ├─ buildFwdProblem() → FmhaProblem via FmhaProblemBuilder
            └─ dispatcher->select_kernel(problem) → non-null → applicable!
   └─ Backend creates execution plan
      └─ hipdnnEnginePluginCreateExecutionContext()
         └─ CkFmhaFwdPlanBuilder::buildPlan()
            ├─ canonical_key() → "fwd\x1ffwd\x1ffp16\x1fgfx950\x1f128,128..."
            ├─ Cache miss → dispatcher->plan(problem) → FmhaExecutionPlan
            └─ CkFmhaFwdPlan created with parsed params + plan

5. Demo calls graph.execute(handle, tensor_map, workspace)
   └─ Backend maps tensor UIDs to device pointers
   └─ hipdnnEnginePluginExecuteOpGraph()
      └─ CkFmhaFwdPlan::execute()
         ├─ findBuffer(uid) for Q, K, V, O
         ├─ Fill fmha_fwd_traits (dtype, mask, bias, dropout flags)
         ├─ Fill fmha_fwd_args (pointers, batch, nhead, seqlen, hdim, strides)
         │   Stride logic (BHSD): stride=D, nhead_stride=S*D, batch_stride=H*S*D
         ├─ FmhaInvocation::make(traits, args)
         └─ dispatcher->run(invocation, stream)
            ├─ plan(problem) → seqtune-aware kernel selection
            └─ run_plan() → ck_tile::launch_kernel() → HIP kernel on GPU

6. HIP events measure elapsed time → TFLOPS = 2*B*H*Sq*Sk*(Dq+Dv) / ms / 1e9
```

## Key design decisions that made this work

1. **Plugin uses `REGISTER_GENERATED_KERNELS` directly** -- the same macro the
   CK ctypes library uses. This puts compiled GPU kernel instances into the
   plugin's own `FmhaRegistry`, avoiding the singleton-across-DSOs problem.

2. **Stride computation matches `fmha_ctypes_lib.cpp` exactly** -- the reference
   C API in the dispatcher fills `fmha_fwd_args` with specific stride formulas
   for BHSD/BSHD. The plugin replicates this logic line-for-line.

3. **Plugin links `libck_tile_dispatcher.a` + kernel `.o`** -- the static lib
   provides the `FmhaDispatcher`/`FmhaRegistry` C++ code, the `.o` provides
   the actual GPU fat binary. No dynamic dependency on `libdispatcher_fmha_lib.so`.

4. **`hipdnnSetEnginePluginPaths_ext` called BEFORE `hipdnnCreate`** -- the
   backend loads plugins during handle creation, not after. Setting the path
   after would have no effect.

## Part 2: JIT Path

When a shape has no precompiled kernel (e.g., h256 which is not in the 96
kernel set), the plugin transparently JIT-compiles one.

### What triggers JIT

```
CkFmhaFwdPlanBuilder::isApplicable()
  select_kernel(problem) -> nullptr (no precompiled kernel)
  handle.jitAndLoad(problem)
    check CK_FMHA_ENABLE_JIT=1 env var
    jit_compile_kernel(problem, gfx_arch)
      fork() -> python3 -c "from fmha_utils import setup_fmha_dispatcher; ..."
        generate_fmha_fallback.py -> hipcc -> .so  (~16-31s first time)
      capture stdout -> .so path
    load_jit_library(so_path, registry, gfx_arch)
      dlopen(.so)
      dlsym("ck_fmha_register_kernels")
      fn(registry, arch)  -> kernel merged into live registry
    select_kernel(problem) -> non-null -> return true
```

### Cache behavior

The JIT `.so` is cached on disk by `setup_fmha_dispatcher`. On subsequent
calls for the same kernel config, the `.so` is loaded from disk (~1ms).
The plugin's registry also caches in memory -- once a JIT kernel is loaded,
it stays in the registry for the lifetime of the handle.

### Running the JIT demo

```bash
CK_FMHA_ENABLE_JIT=1 \
CK_DISPATCHER_PYTHON_PATH=<ck>/dispatcher/python \
HIPDNN_PLUGIN_PATH=<hipDNN>/build/lib/hipdnn_plugins/engines \
  ./ck_fmha_e2e_demo --warmup 1 --repeat 3
```

### Verified output (MI355X gfx950)

```
=== Forward SDPA (fp16, BHSD, no mask) ===
Shape                                             Time(ms)    TFLOPS
B=2 Hq=8 Hk=4 Sq=1024 Sk=1024 Dq=128 Dv=128       0.05        184.43
B=2 Hq=8 Hk=4 Sq=1024 Sk=1024 Dq=256 Dv=256       0.06        274.09  <- JIT'd during isApplicable

=== Part 2: JIT Compilation (ENABLED) ===
Shape                                             JIT(s)      Time(ms)    TFLOPS
B=2 Hq=8 Hk=4 Sq=1024 Sk=1024 Dq=256 Dv=256       0.41        0.05        374.89  <- cached
B=1 Hq=8 Hk=8 Sq=2048 Sk=2048 Dq=256 Dv=256       0.60        0.08        457.80  <- cached
```

### Environment variables for JIT

| Variable | Required | Description |
|----------|----------|-------------|
| `CK_FMHA_ENABLE_JIT` | Yes | Set to `1` to enable JIT fallback |
| `CK_DISPATCHER_PYTHON_PATH` | Yes | Path to `<ck>/dispatcher/python` |
| `CK_FMHA_JIT_CACHE_DIR` | No | Cache directory (default: `/tmp/ck_fmha_jit`) |
