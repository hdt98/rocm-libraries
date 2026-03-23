# Kpack — CK Tile Kernel Distribution via Archive

## Motivation

[Kpack](https://github.com/ROCm/TheRock/blob/develop/docs/rfcs/rfc0008_kpack.md) (kernel pack) is a binary archive format that separates GPU device code from host code for efficient multi-architecture distribution. Instead of shipping fat binaries that embed kernels for every supported GPU, kpack packages per-architecture `.hsaco` code objects into compressed archives that are loaded at runtime based on the detected GPU.

This project demonstrates how kpack works with CK Tile kernels, progressing from a minimal hello-world to a production-ready pattern with multi-variant tuning and runtime kernel selection.

The core challenge is that CK Tile kernels are C++ template instantiations — host and device code share the same translation unit, and there is no standalone device-only kernel file. The **bridge pattern** (introduced in example 02) solves this with a thin `extern "C" __global__` wrapper that delegates to CK Tile on-device, cleanly separating compilation: device code compiles with CK Tile headers, the host binary only needs HIP runtime and kpack.

## Examples

Four progressive examples, each building on the last:

| Example | What it demonstrates |
|---------|---------------------|
| [01_hello_world](examples/01_hello_world/) | Minimal kpack pipeline — hand-written HIP kernel, one binary per arch |
| [02_ck_tile_vector_add](examples/02_ck_tile_vector_add/) | Bridge pattern — `extern "C"` wrapper around CK Tile's `ElementWiseKernel` |
| [03_rocm_ck_vector_add](examples/03_rocm_ck_vector_add/) | Full tuning surface — Signature/Algorithm split, 12 compiled variants, registry-based selection |
| [04_gemm](examples/04_gemm/) | GEMM with multi-type variants — operator-centric Signature, fp32/fp16/bf16, composed epilogue |

### Example 01: Hello World

Validates the end-to-end pipeline with a trivial hand-written vector-add kernel. One kernel source, compiled per architecture, packed into a single archive, loaded at runtime.

### Example 02: CK Tile Bridge

Proves that CK Tile kernels can be compiled into standalone `.hsaco` code objects via the bridge pattern. The device kernel (`ck_tile_add.hip`) wraps `ElementWiseKernel<Add>` behind an `extern "C"` entry point with flat arguments compatible with `hipModuleLaunchKernel`.

### Example 03: Full Tuning Surface

Production-ready pattern with:

- **Signature/Algorithm separation** — *what* (operator graph + data types) vs *how* (tile geometry, warp count, vector width, padding)
- **12 compiled variants** across FP32, FP16, BF16 with different block sizes, multi-warp, and mixed-precision
- **Constexpr validation** via `make_kernel` that catches invalid configurations at compile time
- **Variant registry** with `find_variant(DataType, problem_size)` for automatic kernel selection
- **Archive metadata** — tuning parameters stored in the kpack TOC for tooling

See the [example 03 README](examples/03_rocm_ck_vector_add/README.md) for full details.

### Example 04: GEMM (Multi-Type)

Extends the Signature pattern from elementwise to GEMM:

- **Operator-centric Signature** — `GemmOp`, `AddOp`, `ReluOp` compose the compute graph
- **Multi-type variants** — fp32, fp16, bf16 compiled from ~20-line `.hip` files via `runGemm<K>` template
- **Mixed-precision epilogue** — all variants accumulate in fp32; CShuffleEpilogue handles output type conversion
- **Composed epilogue** — `GemmOp + AddOp + ReluOp` for fused bias + activation
- **Typed host buffers** — `float_to_typed` / `typed_to_float` for type-agnostic verification

See the [example 04 README](examples/04_gemm/README.md) for full details.

## Pipeline

```text
*.hip kernel sources
    | (clang++ --cuda-device-only, per arch × per variant)
    v
*.hsaco code objects
    | (pack.py — variant-aware packer)
    v
kernels.kpack  (single archive: multiple architectures × multiple variants)
    | (loaded at runtime by kpack C API)
    v
host executable  (variant selection → hipModuleLoadData → hipModuleLaunchKernel)
```

## Directory Structure

```text
experimental/kpack/
├── CMakeLists.txt                  # Top-level (delegates to include/ and examples)
├── README.md
├── include/                        # Shared host-side headers (header-only)
│   ├── CMakeLists.txt              # INTERFACE library target "rocm_ck"
│   └── rocm_ck/
│       ├── hip_check.hpp           # HIP_CHECK error-checking macro
│       ├── gpu_arch.hpp            # get_gpu_arch() — GPU architecture detection
│       ├── datatype_utils.hpp      # DataType enum, type conversions, tolerances
│       ├── types.hpp               # Common types: index_t, warp_size
│       ├── typed_buffer.hpp        # TypedBuffer wrapper for host memory
│       ├── kpack_module.hpp        # RAII wrappers for kpack and HIP module handles
│       ├── ck_type_map.hpp         # DataType enum to CK Tile type mapping
│       ├── layout.hpp              # Layout enum (Row, Col, Contiguous, Auto)
│       ├── tensor_desc.hpp         # TensorDesc: name, dtype, rank, layout
│       ├── ops.hpp                 # Operator structs (GemmOp, AddOp, ReluOp, ...) + Op variant
│       ├── signature.hpp           # Tensor, Scalar, Signature (compute graph)
│       └── resolve.hpp             # consteval resolve(): dtype cascade, rank/layout propagation
├── rocm_kpack/                     # Vendored kpack C runtime library (from TheRock)
│   ├── CMakeLists.txt
│   ├── include/rocm_kpack/
│   │   ├── kpack.h
│   │   ├── kpack_types.h
│   │   └── kpack_export.h
│   └── src/
│       ├── kpack_internal.h
│       ├── isa_target_match.h
│       ├── kpack.cpp
│       ├── archive.cpp
│       ├── compression.cpp
│       ├── toc_parser.cpp
│       ├── loader.cpp
│       ├── path_resolution.cpp
│       └── isa_target_match.cpp
└── examples/
    ├── 01_hello_world/             # Minimal: hand-written HIP kernel
    │   ├── CMakeLists.txt
    │   ├── vector_add.hip
    │   ├── pack.py
    │   └── main.cpp
    ├── 02_ck_tile_vector_add/      # Bridge pattern: extern "C" + CK Tile
    │   ├── CMakeLists.txt
    │   ├── ck_tile_add.hip
    │   ├── pack.py
    │   └── main.cpp
    ├── 03_rocm_ck_vector_add/      # Full tuning surface: variants + registry
    │   ├── CMakeLists.txt
    │   ├── rocm_vector_add_api.hpp     # Signature/Algorithm types, make_kernel validation
    │   ├── rocm_vector_add_dev.hpp     # Device interface — maps config to CK Tile types
    │   ├── rocm_vector_add_registry.hpp # Variant table + find_variant selection
    │   ├── vector_add_*.hip            # 12 variant instantiations
    │   ├── pack.py                     # Variant-aware packer with metadata
    │   └── main.cpp                    # Variant selection demo + verify-all mode
    └── 04_gemm/                     # GEMM: multi-type via operator-centric Signature
        ├── CMakeLists.txt
        ├── gemm_api.hpp                # Signature-based make_kernel, GemmKernel, tile validation
        ├── gemm_dev.hpp                # CkTypeMap, CkLayoutMap, runGemm<K> template
        ├── gemm_args.hpp               # GemmArgs ABI struct, tile constants
        ├── gemm_fp32.hip               # fp32 variant instantiation
        ├── gemm_fp16.hip               # fp16 variant instantiation
        ├── gemm_fp16_w32.hip           # fp16 variant with WarpTile=32 (K=16 for fp16)
        ├── gemm_fp16_add.hip           # fp16 + bias addition (fused epilogue)
        ├── gemm_fp16_add_relu.hip      # fp16 + bias + ReLU (composed epilogue)
        ├── gemm_bf16.hip               # bf16 variant instantiation
        ├── cpu_ref.hpp                 # CPU reference GEMM implementation
        ├── cpu_ref.cpp                 # CPU reference implementation
        ├── pack.py                     # Variant-aware packer with dtype metadata
        └── main.cpp                    # Multi-variant loop with typed buffers
```

## Dependencies

- **ROCm** with HIP support (`/opt/rocm`)
- **System packages**: `libmsgpack-cxx-dev`, `libzstd-dev`
- **Python packages**: `msgpack` (`pip install msgpack`)

## Build

Each example is standalone — build from its directory:

```bash
cd experimental/kpack/examples/01_hello_world  # or 02_, 03_, 04_

cmake -B build -S . -G Ninja \
    -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
    -DCMAKE_PREFIX_PATH=/opt/rocm \
    -DGPU_TARGETS="gfx90a;gfx942"

ninja -C build
```

Examples 03 and 04 require C++20 for struct NTTPs and `consteval` validation.

## Run

On a machine with a supported GPU:

```bash
# Example 01
./build/kpack_hello_world build/kernels.kpack

# Example 02
./build/kpack_ck_tile_vector_add build/kernels.kpack

# Example 03
./build/kpack_rocm_ck_vector_add build/kernels.kpack

# Example 04
./build/kpack_gemm build/gemm.kpack
```

If the current GPU's architecture is not in the archive, the demo prints a clear error and exits.

## Inspect the Archive

```bash
python3 -c "
import msgpack, struct
f = open('build/kernels.kpack', 'rb')
magic, version, toc_offset = struct.unpack('<4sIQ', f.read(16))
print(f'Magic: {magic}, Version: {version}, TOC offset: {toc_offset}')
f.seek(toc_offset)
toc = msgpack.unpack(f, raw=False)
for binary, archs in toc['toc'].items():
    for arch, meta in archs.items():
        print(f'  {binary}/{arch}: {meta[\"original_size\"]} bytes')
"
```

## How It Works

1. **Compile**: `clang++ --cuda-device-only` compiles `.hip` sources into per-architecture `.hsaco` code objects
2. **Pack**: `pack.py` concatenates the `.hsaco` blobs after a 16-byte KPAK header, then appends a MessagePack table of contents (TOC) recording each blob's offset, size, and architecture
3. **Load**: The host binary opens the archive via the kpack C API, queries the detected GPU's architecture, extracts the matching code object, and loads it via `hipModuleLoadData`

## Kpack Archive Format

```text
[0x00]  "KPAK"              4 bytes   Magic
[0x04]  version             4 bytes   Little-endian uint32 (currently 1)
[0x08]  toc_offset          8 bytes   Little-endian uint64
[0x10]  blob_0              variable  Raw .hsaco for first arch
        blob_1              variable  Raw .hsaco for second arch
        ...
[toc_offset]  MessagePack TOC     variable  Compression scheme, arch list, blob metadata, nested TOC
```

The TOC can also carry optional `variant_metadata` sections (used by example 03) containing tuning parameters for each kernel variant.

## Vendored Runtime

The `rocm_kpack/` directory contains a local copy of the kpack C runtime library, taken from [TheRock](https://github.com/ROCm/TheRock) (`base/rocm-kpack/runtime/`) for this experimental demo. This avoids requiring a full TheRock build as a dependency — the runtime is small and self-contained. Once kpack ships as part of a ROCm release, this vendored copy should be replaced with a proper `find_package(rocm_kpack)`.

It provides:

- `kpack_open` / `kpack_close` — archive lifecycle
- `kpack_get_architecture_count` / `kpack_get_architecture` — enumerate architectures
- `kpack_get_binary_count` / `kpack_get_binary` — enumerate binaries
- `kpack_get_kernel` / `kpack_free_kernel` — extract a code object by binary name + architecture
- NoOp and Zstd per-kernel decompression
- Higher-level loader/cache API for runtime integration
