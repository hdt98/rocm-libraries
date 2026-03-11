# Design: Launching AITER ASM Attention Kernels from hipDNN (POC)

## Context

hipDNN's SDPA forward path is fully implemented in the frontend (`SdpaFpropNode`), but no plugin currently handles SDPA execution. AITER provides high-performance ASM Flash Attention v3 kernels for gfx942 (MI300X). This design describes a proof-of-concept plugin that copies the minimal necessary AITER code and a single pre-compiled `.co` kernel binary to launch BF16 Flash Attention v3 from hipDNN.

**Scope:** Single platform (gfx942 MI300X), single data type (BF16), single kernel variant (hd128, non-causal). No AITER dependency — all needed code is copied and adapted.

---

## What Gets Copied from AITER

### Files to copy and adapt

| # | What | AITER source | Adaptation needed |
|---|------|-------------|-------------------|
| 1 | **Kernel binary** | `hsa/gfx942/fmha_v3_fwd/MI300/fwd_hd128_bf16_rtne.co` | None — binary blob, ship as-is |
| 2 | **Kernel arg struct** | `fmha_fwd_v3_args` from `csrc/include/mha_fwd.h` | Extract struct + padding types (`p2`, `p3`). Remove CK tile includes. |
| 3 | **Arg init function** | `init_fmha_fwd_v3_args()` from `csrc/cpp_itfs/mha_fwd.cu` | Extract function. Replace `ck_tile::index_t` with `int32_t`. |
| 4 | **Grid dim calculation** | `get_grid_dim()` from `csrc/cpp_itfs/mha_fwd.cu` | Extract function. Trivial — `{ceil(seqlen_q/ts_qo), nhead_q, batch}`. |
| 5 | **HSA module loader** | `AiterAsmKernel` class from `csrc/include/aiter_hip_common.h` | Extract class. Remove CK tile includes, replace with direct HIP calls. |

### Files NOT needed

- All Python code (`codegen.py`, `setup.py`, pybind modules)
- PyTorch headers (`mha_common.h`, ATen)
- CK tile library (`ck_tile/core.hpp`, `fmha_fwd.hpp`, `mask.hpp`)
- All other `.co` variants (causal, hd192, fp8, group mode)
- The CSV metadata and codegen system (hardcode the single config)

### Dependencies eliminated

| AITER dependency | Replacement |
|-----------------|-------------|
| `ck_tile::index_t` | `int32_t` |
| `ck_tile::stream_config` | Not needed (direct `hipStream_t`) |
| `ck_tile::launch_kernel` | Direct `hipModuleLaunchKernel` call |
| `aiter_logger.h` | hipDNN plugin logging macros (`HIPDNN_PLUGIN_LOG_*`) |
| `$AITER_ASM_DIR` env var | Plugin-local path discovery |

---

## Plugin Architecture

### Directory Structure

```
dnn-providers/aiter-provider/
  CMakeLists.txt
  src/
    AiterPluginPublic.cpp             # 6-line entry point
    AiterContainer.hpp/.cpp           # Engine registration
    AiterHandle.hpp/.cpp              # Stream + container + detached buffers
    AiterContext.hpp                   # ExecutionContextBase
    AiterSettings.hpp                 # Empty for POC

    asm/
      AsmKernelLoader.hpp/.cpp        # Adapted from AiterAsmKernel
      FmhaFwdV3Args.hpp               # Adapted fmha_fwd_v3_args struct + init
      FmhaFwdV3Config.hpp             # Hardcoded kernel config (name, .co path, tile sizes)

    engines/
      AiterSdpaEngine.hpp/.cpp
      plans/
        AsmSdpaPlanBuilder.hpp/.cpp
        AsmSdpaPlan.hpp/.cpp

  asm_kernels/                        # Copied .co binaries
    gfx942/
      fwd_hd128_bf16_rtne.co

  tests/
    TestAsmSdpaPlanBuilder.cpp
  integration_tests/
    IntegrationGpuAiterSdpaFwd.cpp
```

### Type System (cloned from hip-kernel-provider)

Identical to hip-kernel-provider's types — `AiterHandle`, `AiterContext`, `AiterSettings`, `AiterContainer`. See `dnn-providers/hip-kernel-provider/src/` for the exact pattern to replicate.

### Engine Registration

```cpp
HIPDNN_REGISTER_ENGINE(AITER_SDPA_ENGINE, "AITER_SDPA_ENGINE")

const std::vector<AiterContainer::EngineDefinition>&
    AiterContainer::getEngineDefinitions()
{
    static const std::vector<EngineDefinition> s_engineDefinitions = {
        {AITER_SDPA_ENGINE_ID, []() {
            auto engine = std::make_unique<AiterSdpaEngine>(AITER_SDPA_ENGINE_ID);
            engine->addPlanBuilder(std::make_unique<AsmSdpaPlanBuilder>());
            return engine;
        }},
    };
    return s_engineDefinitions;
}
```

---

## Key Implementation Details

### AsmKernelLoader (adapted from AITER's AiterAsmKernel)

Stripped-down version of `csrc/include/aiter_hip_common.h`:

```cpp
class AsmKernelLoader {
public:
    AsmKernelLoader(const std::string& coPath, const std::string& kernelSymbol)
    {
        HIP_CHECK(hipModuleLoad(&_module, coPath.c_str()));
        HIP_CHECK(hipModuleGetFunction(&_function, _module, kernelSymbol.c_str()));
    }

    ~AsmKernelLoader()
    {
        if (_module) { hipModuleUnload(_module); }
    }

    void launch(void* args, size_t argSize, dim3 grid, dim3 block, hipStream_t stream)
    {
        void* config[] = {
            HIP_LAUNCH_PARAM_BUFFER_POINTER, args,
            HIP_LAUNCH_PARAM_BUFFER_SIZE, &argSize,
            HIP_LAUNCH_PARAM_END
        };
        HIP_CHECK(hipModuleLaunchKernel(
            _function, grid.x, grid.y, grid.z, block.x, block.y, block.z,
            0, stream, nullptr, config));
    }

private:
    hipModule_t _module = nullptr;
    hipFunction_t _function = nullptr;
};
```

### fmha_fwd_v3_args (adapted from AITER's mha_fwd.h)

The packed kernel arg struct with SGPR-aligned padding. This is the GPU kernel's ABI — layout must match exactly:

```cpp
struct __attribute__((packed)) p2 { char pad[8]; };   // 8-byte pad
struct __attribute__((packed)) p3 { char pad[12]; };  // 12-byte pad

struct __attribute__((packed)) fmha_fwd_v3_args {
    void* ptr_o;               p2 _p0;
    const void* ptr_q;         p2 _p1;
    const void* ptr_k;         p2 _p2;
    const void* ptr_v;         p2 _p3;
    void* ptr_lse;             p2 _p4;
    float scalar;              p3 _p5;   // 1/sqrt(d)
    unsigned int s_seq_len;    p3 _p6;   // query seq len
    unsigned int s_Seqs;       p3 _p7;   // Q stride[2] in bytes
    unsigned int s_Ts;         p3 _p8;
    unsigned int s_Hs;         p3 _p9;   // Q stride[1] in bytes
    unsigned int s_Bs;         p3 _p10;  // Q stride[0] in bytes
    unsigned int s_gqa;        p3 _p11;  // nhead_q / nhead_kv
    unsigned int s_k_Seqs;     p3 _p12;
    unsigned int s_k_Hs;       p3 _p13;
    unsigned int s_k_Bs;       p3 _p14;
    unsigned int s_opt;        p3 _p15;  // tuning option
    unsigned int s_lse;        p3 _p16;
    unsigned int s_kv_seq_len; p3 _p17;
    unsigned int s_qk_head_dim;p3 _p18;
    unsigned int s_v_head_dim; p3 _p19;
    unsigned int s_q_head_num; p3 _p20;
    unsigned int s_v_Seqs;     p3 _p21;
    unsigned int s_v_Hs;       p3 _p22;
    unsigned int s_v_Bs;       p3 _p23;
    unsigned int s_o_Seqs;     p3 _p24;
    unsigned int s_o_Hs;       p3 _p25;
    unsigned int s_o_Bs;       p3 _p26;
    const void* ptr_qseq;     p2 _p27;  // nullptr for non-group mode
    const void* ptr_kseq;     p2 _p28;  // nullptr for non-group mode
    unsigned int s_lse_Hs;     p3 _p29;
    const void* ptr_qseq_padding; p2 _p30;
    const void* ptr_kseq_padding; p2 _p31;
    const void* ptr_q_descale; p2 _p32;  // nullptr for BF16
    const void* ptr_k_descale; p2 _p33;  // nullptr for BF16
    const void* ptr_v_descale; p2 _p34;  // nullptr for BF16
    unsigned int s_descale_q_Bs; p3 _p35;
    unsigned int s_descale_q_Hs; p3 _p36;
    unsigned int s_descale_k_Bs; p3 _p37;
    unsigned int s_descale_k_Hs; p3 _p38;
    unsigned int s_descale_v_Bs; p3 _p39;
    unsigned int s_descale_v_Hs; p3 _p40;
};
```

All stride fields are in **bytes** (element strides × `sizeof(bfloat16)` = ×2).

### Hardcoded Kernel Config (FmhaFwdV3Config.hpp)

No codegen — just hardcode the single variant:

```cpp
struct FmhaFwdV3Config {
    static constexpr const char* kernelSymbol =
        "_ZN5aiter24fmha_fwd_hd128_bf16_rtneE";
    static constexpr const char* coFileName =
        "gfx942/fwd_hd128_bf16_rtne.co";
    static constexpr int tileSeqQ = 256;   // ts_qo from CSV
    static constexpr int tileSeqKV = 32;   // ts_kv from CSV
    static constexpr int blockSize = 512;  // bdx
    static constexpr int hdimQ = 128;
    static constexpr int hdimV = 128;
};
```

### AsmSdpaPlanBuilder::isApplicable()

Conservative matching — only handles the exact configuration this POC supports:

```cpp
bool AsmSdpaPlanBuilder::isApplicable(
    const AiterHandle& handle,
    const IGraph& opGraph) const
{
    try {
        if (opGraph.nodeCount() != 1) return false;

        const auto& node = opGraph.getNodeWrapper(0);
        if (node.attributesType() != NodeAttributes::SdpaAttributes)
            return false;

        const auto& sdpa = node.attributesAs<SdpaAttributes>();
        const auto& tensorMap = opGraph.getTensorMap();

        // Must be BF16
        auto qTensor = tensorMap.at(sdpa.q_tensor_uid());
        if (qTensor->data_type() != DataType::BFLOAT16)
            return false;

        // Must be hd128
        auto qDims = qTensor->dims();
        if (qDims->size() != 4 || qDims->Get(3) != 128)
            return false;

        // No causal masking (POC limitation)
        if (sdpa.causal_mask() || sdpa.causal_mask_bottom_right())
            return false;

        // No dropout
        if (sdpa.dropout_probability() != 0.0f)
            return false;

        // No paged attention
        if (sdpa.page_table_k_tensor_uid() != 0)
            return false;

        // Must be gfx942
        hipDeviceProp_t props;
        hipGetDeviceProperties(&props, 0);
        if (std::string(props.gcnArchName).find("gfx942") == std::string::npos)
            return false;

        return true;
    } catch (...) {
        return false;
    }
}
```

### AsmSdpaPlan::execute()

Maps hipDNN device buffers to kernel args and launches:

```cpp
void AsmSdpaPlan::execute(
    const AiterHandle& handle,
    const hipdnnPluginDeviceBuffer_t* deviceBuffers,
    uint32_t numDeviceBuffers,
    void* workspace) const
{
    // 1. Resolve device pointers from UIDs
    auto qBuf = findDeviceBuffer(_qUid, deviceBuffers, numDeviceBuffers);
    auto kBuf = findDeviceBuffer(_kUid, deviceBuffers, numDeviceBuffers);
    auto vBuf = findDeviceBuffer(_vUid, deviceBuffers, numDeviceBuffers);
    auto oBuf = findDeviceBuffer(_oUid, deviceBuffers, numDeviceBuffers);

    // 2. Fill kernel args (all strides in bytes, BF16 = 2 bytes)
    fmha_fwd_v3_args args = {};
    args.ptr_q = qBuf.ptr;
    args.ptr_k = kBuf.ptr;
    args.ptr_v = vBuf.ptr;
    args.ptr_o = oBuf.ptr;
    args.ptr_lse = workspace;  // or nullptr if not needed
    args.scalar = _scale;      // 1.0f / sqrt(hdim)
    args.s_seq_len = static_cast<unsigned>(_seqLenQ);
    args.s_kv_seq_len = static_cast<unsigned>(_seqLenKV);
    args.s_qk_head_dim = FmhaFwdV3Config::hdimQ;
    args.s_v_head_dim = FmhaFwdV3Config::hdimV;
    args.s_q_head_num = static_cast<unsigned>(_numHeads);
    args.s_gqa = static_cast<unsigned>(_numHeads / _numKVHeads);
    // ... fill byte strides from tensor strides × sizeof(bfloat16)

    // 3. Compute grid dimensions
    dim3 grid(
        (_seqLenQ + FmhaFwdV3Config::tileSeqQ - 1) / FmhaFwdV3Config::tileSeqQ,
        _numHeads,
        _batch);
    dim3 block(FmhaFwdV3Config::blockSize, 1, 1);

    // 4. Launch
    _kernelLoader.launch(&args, sizeof(args), grid, block, handle.getStream());
}
```

### .co File Discovery

For the POC, the `.co` file path is resolved relative to a configurable base directory:

```cpp
std::string resolveCoPath(const std::string& relPath) {
    // 1. Check env var
    if (auto* dir = std::getenv("HIPDNN_AITER_ASM_DIR"))
        return std::string(dir) + "/" + relPath;

    // 2. Use compiled-in default
    return std::string(AITER_ASM_DIR) + "/" + relPath;
}
```

The `AITER_ASM_DIR` compile definition points to where the `.co` files are installed (set in CMakeLists.txt).

---

## Build System

```cmake
cmake_minimum_required(VERSION 3.25.2)
project(aiter-provider VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(hip REQUIRED)
find_package(hipdnn_data_sdk CONFIG REQUIRED)
find_package(hipdnn_plugin_sdk CONFIG REQUIRED)

add_library(aiter_plugin SHARED
    src/AiterPluginPublic.cpp
    src/AiterContainer.cpp
    src/AiterHandle.cpp
    src/asm/AsmKernelLoader.cpp
    src/engines/AiterSdpaEngine.cpp
    src/engines/plans/AsmSdpaPlanBuilder.cpp
    src/engines/plans/AsmSdpaPlan.cpp
)

target_link_libraries(aiter_plugin
    PUBLIC hipdnn_data_sdk hipdnn_plugin_sdk hip::host)

# .co file install location
set(AITER_ASM_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/share/aiter/asm_kernels")
target_compile_definitions(aiter_plugin PRIVATE
    AITER_ASM_DIR="${AITER_ASM_INSTALL_DIR}")

set_target_properties(aiter_plugin PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${HIPDNN_BUILD_PLUGIN_ENGINE_DIR}")

install(TARGETS aiter_plugin
    LIBRARY DESTINATION ${HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR})
install(DIRECTORY asm_kernels/
    DESTINATION share/aiter/asm_kernels)
```

---

## Graph-to-Kernel Mapping (SdpaAttributes → fmha_fwd_v3_args)

| SdpaAttributes field | fmha_fwd_v3_args field | Notes |
|---------------------|----------------------|-------|
| `q_tensor_uid` → dims[0] | grid.z (batch) | Batch size |
| `q_tensor_uid` → dims[1] | `s_q_head_num`, grid.y | Number of Q heads |
| `q_tensor_uid` → dims[2] | `s_seq_len`, grid.x calc | Query sequence length |
| `q_tensor_uid` → dims[3] | `s_qk_head_dim` | Head dimension (128) |
| `k_tensor_uid` → dims[1] | `s_gqa` = nhead_q / nhead_kv | GQA ratio |
| `k_tensor_uid` → dims[2] | `s_kv_seq_len` | KV sequence length |
| `v_tensor_uid` → dims[3] | `s_v_head_dim` | V head dim (128) |
| `attn_scale_value` | `scalar` | 1/sqrt(d) if null |
| Tensor strides × 2 | `s_Seqs`, `s_Hs`, `s_Bs`, etc. | All strides in bytes |
| `q_tensor_uid` | `ptr_q` (via deviceBuffer UID) | Resolved at execute time |
| `k_tensor_uid` | `ptr_k` (via deviceBuffer UID) | Resolved at execute time |
| `v_tensor_uid` | `ptr_v` (via deviceBuffer UID) | Resolved at execute time |
| `o_tensor_uid` | `ptr_o` (via deviceBuffer UID) | Resolved at execute time |

Strides come from `TensorAttributes::strides()` in the graph's tensor map. They are element strides and must be multiplied by `sizeof(bfloat16)` = 2 to get byte strides for the kernel.

---

## Extending Beyond POC

To expand from POC to production:

| Feature | What to add |
|---------|-------------|
| Causal masking | Copy `fwd_hd128_bf16_causal_rtne.co`, add config entry, check `sdpa.causal_mask()` in `isApplicable()` |
| hd192 | Copy `fwd_hd192x128_bf16_*.co`, add config entry |
| FP8 | Copy FP8 `.co` variants, handle `descale_*` tensor UIDs, set FP8 arg fields |
| gfx950 | Copy `.co` files from `hsa/gfx950/`, add arch check |
| Variable-length batches | Copy `_group.co` variants, handle `seq_len_q/kv_tensor_uid` |
| Paged attention | Add `PagedAttentionPlanBuilder` using AITER's `cpp_itfs/pa/pa_ragged.h` |
| CK fallback | Add separate engine with direct CK dependency for unsupported configs |

---

## Risk Summary

| Risk | Severity | Mitigation |
|------|----------|------------|
| Kernel arg struct layout changes in future AITER versions | High | Document which AITER commit the struct was copied from; add `static_assert(sizeof(fmha_fwd_v3_args) == EXPECTED)` |
| `.co` binary incompatible with ROCm version | Medium | Test with target ROCm version; `.co` files are arch-specific but generally stable |
| Stride/padding errors corrupt results | High | Integration test comparing against reference implementation (CK or PyTorch) |
| `isApplicable()` too broad, matches unsupported configs | Medium | Start very restrictive (POC), widen only with test coverage |

---

## Reference Files

**hip-kernel-provider (template to clone):**
- `dnn-providers/hip-kernel-provider/src/HipKernelContainer.hpp/.cpp`
- `dnn-providers/hip-kernel-provider/src/HipKernelHandle.hpp/.cpp`
- `dnn-providers/hip-kernel-provider/src/HipKernelContext.hpp`
- `dnn-providers/hip-kernel-provider/src/HipKernelSettings.hpp`
- `dnn-providers/hip-kernel-provider/src/HipKernelPluginPublic.cpp`
- `dnn-providers/hip-kernel-provider/CMakeLists.txt`

**hipblaslt-provider (plan builder execution pattern):**
- `dnn-providers/hipblaslt-provider/engines/plans/HipblasltMatmulPlanBuilder.cpp` — `isApplicable()` / `buildPlan()` / attribute extraction
- `dnn-providers/hipblaslt-provider/engines/plans/HipblasltMatmulPlan.cpp` — `execute()` with device buffer UID resolution

**hipDNN Frontend (SDPA graph construction):**
- `hipdnn/frontend/include/hipdnn_frontend/node/SdpaFpropNode.hpp` — validates dims, packs FlatBuffer
- `hipdnn/frontend/include/hipdnn_frontend/attributes/SdpaAttributes.hpp` — attribute model
- `hipdnn/data_sdk/schemas/sdpa_attributes.fbs` — schema

**hipDNN Plugin SDK:**
- `hipdnn/plugin_sdk/include/hipdnn_plugin_sdk/EnginePluginImpl.inl`
- `hipdnn/plugin_sdk/include/hipdnn_plugin_sdk/EnginePluginApi.h`

**AITER (files to copy from):**
- `aiter/csrc/include/aiter_hip_common.h` — `AiterAsmKernel` class
- `aiter/csrc/include/mha_fwd.h` — `fmha_fwd_v3_args` struct
- `aiter/csrc/cpp_itfs/mha_fwd.cu` — `init_fmha_fwd_v3_args()`, `get_grid_dim()`
- `aiter/hsa/gfx942/fmha_v3_fwd/MI300/fwd_hd128_bf16_rtne.co` — kernel binary
