# Tile Engine

The **Tile Engine** is a GPU kernel implementation, benchmarking, and instance-generation framework built on top of the [`ck_tile`](../include/ck_tile/README.md) programming model. While `ck_tile` provides low-level tile-based abstractions (tensor descriptors, pipelines, MMA operations), Tile Engine provides the infrastructure to generate, compile, profile, and select production-ready GPU kernels from a parametric configuration space.

## Overview

```
tile_engine/
├── CMakeLists.txt          # Top-level build configuration
├── README.md               # This file
├── include/
│   └── utility/
│       └── validation.hpp  # Numerical validation utilities (rtol/atol)
└── ops/
    ├── gemm/               # GEMM variants (universal, multi-D, preshuffle)
    ├── gemm_streamk/       # StreamK GEMM (dynamic load balancing)
    └── reduce/             # Multi-dimensional reduction kernels
```

## Relationship to `ck_tile`

| Aspect | `ck_tile` (include/) | Tile Engine (tile_engine/) |
|--------|----------------------|---------------------------|
| **Role** | Core library of tile-based abstractions and APIs | Kernel instance generation, profiling, and benchmarking |
| **Contents** | Template headers for tensors, pipelines, MMA, data types | Python generators, JSON configs, profiler harnesses, benchmark executables |
| **Usage** | `#include "ck_tile/core.hpp"` in device kernels | Python/CMake scripts generate and compile concrete kernel binaries |
| **Abstraction level** | Low-level: `TileDescriptor`, `load_tile()`, `GemmPipeline` | High-level: `GemmProblem`, `GemmProfiler`, `benchmark()` |

Tile Engine consumes `ck_tile` APIs to instantiate kernels and does not have circular dependencies on legacy CK code.

## Supported GPU Targets

| Architecture | Target IDs |
|---|---|
| CDNA2 | `gfx90a` |
| CDNA3 | `gfx942` |
| CDNA3.5 | `gfx950` |
| RDNA4 | `gfx1201` |

## Operations

### GEMM (`ops/gemm/`)

Three GEMM variants are provided, each sharing a common architecture:

#### `gemm_universal/`

The primary GEMM kernel with the broadest configuration space. Supports:

- **Data types:** `fp16`, `fp8` (configurable via `GEMM_UNIVERSAL_DATATYPE`)
- **Layouts:** `rcr`, `rrr`, `crr`, `ccr` (Row/Column × Row/Column × Row) (configurable via `GEMM_UNIVERSAL_LAYOUT`)
- **Pipelines:** `compv3`, `compv4`, `mem`
- **Epilogues:** `cshuffle`, `default`
- **Schedulers:** `intrawave`, `interwave`
- **Padding:** per-dimension (`pad_m`, `pad_n`, `pad_k`)
- **Persistent kernels:** static or StreamK-style persistence
- **Split-K:** K-dimension splitting for tall-thin matrices

Kernel names follow the convention:

```
benchmark_gemm_universal_<dtype>_<layout>_<pipeline>_<epilogue>_<scheduler>_<padm>_<padn>_<padk>_<persistent>_<TileM>x<TileN>x<TileK>_<WarpM>x<WarpN>x<WarpK>_<WarpTileM>x<WarpTileN>x<WarpTileK>
```

Example:

```
benchmark_gemm_universal_fp16_rcr_compv4_cshuffle_intrawave_False_False_False_False_256x256x64_4x1x1_32x32x32
```

#### `gemm_multi_d/`

Extends `gemm_universal` with support for additional `D` input tensors and element-wise operations fused into the epilogue. Uses `ck_tile::element_wise` operators (e.g., `PassThrough`, `Relu`, custom activations).

#### `gemm_preshuffle/`

Implements a weight pre-shuffled GEMM, where the B matrix is reordered at load time to improve memory access efficiency. Adds a `PermuteN` option for N-dimension permutation and uses the `WeightPreshufflePipelineAGmemBGmemCRegV2` pipeline.

---

### GEMM StreamK (`ops/gemm_streamk/`)

A StreamK variant of GEMM that decomposes the K dimension across streaming multiprocessors for improved load balancing on non-ideal problem sizes. Replaces `GemmHostArgs` with `StreamKHostArgs` and adds:

- **Reduction strategy:** `reduction` (atomic accumulation across CUs)
- **Pipelines:** `mem`, `compv3`, `compv4`
- **Schedulers:** `intrawave`, `interwave`
- Supports the same layouts as `gemm_universal`

---

### Reduce (`ops/reduce/`)

Multi-dimensional block-wise and thread-wise reduction operations. Supports:

- Arbitrary input shapes (e.g., `[128, 64, 2]`, `[32, 8, 64, 16]`)
- Two kernel strategies: **threadwise** and **multi-block**
- Test generation via Google Test framework
- Supported targets: `gfx942`, `gfx950`

---

## Configuration System

Each operation reads its tile and trait parameters from JSON configuration files under `configs/`. CMake and Python instance builders use these files to generate the Cartesian product of valid kernel variants.

### Example: `gemm_universal/configs/default_config.json`

```json
{
    "tile_config": {
        "tile_m": { "min": 64, "max": 256, "step": 64 },
        "tile_n": { "min": 64, "max": 256, "step": 64 },
        "tile_k": { "min": 64, "max": 256, "step": 64 },
        "warp_m": { "values": [4, 2, 1] },
        "warp_n": { "values": [4, 2, 1] },
        "warp_k": { "values": [1] },
        "warp_tile_m": { "values": [4, 16, 32] },
        "warp_tile_n": { "values": [16, 32, 64] },
        "warp_tile_k": { "values": [8, 16, 32, 64, 128] }
    },
    "trait_config": {
        "pipeline":   { "values": ["compv3", "compv4", "mem"] },
        "scheduler":  { "values": ["intrawave", "interwave"] },
        "epilogue":   { "values": ["cshuffle", "default"] },
        "pad_m":      { "values": [false] },
        "pad_n":      { "values": [false] },
        "pad_k":      { "values": [false] },
        "persistent": { "values": [false, true] }
    },
    "k_block_per_cu": 1
}
```

Three built-in config presets are provided:

| File | Purpose |
|---|---|
| `default_config.json` | Full configuration space for development/tuning |
| `default_ci_config.json` | Reduced configuration space for CI speed |
| `user_provided_config.json` | Template for custom user configurations |

The active config can be selected via CMake variable or environment variable `GEMM_UNIVERSAL_CONFIG_FILE`.

---

## Build System

### CMake Build

Tile Engine integrates into the parent CK CMake build. To build with tile engine targets:

```bash
mkdir build && cd build

# Configure for one or more supported targets
cmake \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc \
  -DCMAKE_BUILD_TYPE=Release \
  -DGPU_TARGETS="gfx942" \
  ..
```

### Building Specific Kernel Targets

Each kernel variant is its own CMake target (`EXCLUDE_FROM_ALL`). Collection targets allow building groups:

```bash
# Build all gemm_universal kernels
make -j benchmark_gemm_universal_all

# Build all fp16 kernels
make -j benchmark_gemm_universal_fp16

# Build all kernels for a specific layout
make -j benchmark_gemm_universal_rcr

# Build all kernels using a specific pipeline
make -j benchmark_gemm_universal_compv4_pipeline

# Build a single specific kernel
make -j benchmark_gemm_universal_fp16_rcr_compv4_cshuffle_intrawave_False_False_False_False_256x256x64_4x1x1_32x32x32
```

### CMake Configuration Variables

| Variable | Default | Description |
|---|---|---|
| `GEMM_UNIVERSAL_DATATYPE` | `fp8;fp16` | Data types to build |
| `GEMM_UNIVERSAL_LAYOUT` | `rcr;rrr;crr;ccr` | Matrix layouts to build |
| `GEMM_UNIVERSAL_CONFIG_FILE` | (default_config.json) | Custom config file name (must be in `configs/`) |
| `ENABLE_CCACHE_GEMM_UNIVERSAL` | `OFF` | Enable ccache for faster recompilation |

You can also override the config file per-build via environment variable:

```bash
GEMM_UNIVERSAL_CONFIG_FILE=user_provided_config.json cmake ..
```

### Kernel Instance Generation

At CMake configure time, Python scripts (`gemm_universal_instance_builder.py`) enumerate all valid kernel configurations from the JSON config and write a kernel list file. At build time, each target generates its own C++ header (via `add_custom_command`) that instantiates the exact kernel variant, then compiles `gemm_benchmark_single.cpp` with that header force-included via `-include`.

This approach keeps compilation modular: changing one kernel does not recompile others.

---

## Running Benchmarks

Each compiled benchmark executable accepts the following arguments:

| Argument | Default | Description |
|---|---|---|
| `--m` | 3840 | M dimension |
| `--n` | 4096 | N dimension |
| `--k` | 2048 | K dimension |
| `--stride_a` | 0 | Leading dimension of A (0 = contiguous) |
| `--stride_b` | 0 | Leading dimension of B |
| `--stride_c` | 0 | Leading dimension of C |
| `--split_k` | 1 | K-dimension split factor |
| `--warmup` | 50 | Warmup iterations |
| `--repeat` | 100 | Benchmark iterations |
| `--timer` | true | Use GPU timer (`true`) or CPU timer (`false`) |
| `--verify` | 2 | Validation: `0` = off, `1` = CPU, `2` = GPU |
| `--init` | 0 | Tensor init: `0` = random, `1` = linear, `2` = constant(1) |
| `--flush_cache` | true | Flush cache between iterations |
| `--rotating_count` | 1000 | Cache rotation buffer count |
| `--metric` | 0 | Report metric: `0` = latency, `1` = TFLOPS, `2` = bandwidth |
| `--csv_filename` | (none) | Save results to CSV file |
| `--json_output` | false | Output results in JSON format |
| `--structured_sparsity` | false | Use sparse kernel variant |
| `--log` | false | Print kernel instance information |

### Example Usage

```bash
# Run with default problem size, validate on GPU
./benchmark_gemm_universal_fp16_rcr_compv4_cshuffle_intrawave_False_False_False_False_256x256x64_4x1x1_32x32x32 \
    --verify 2

# Custom problem size, report TFLOPS, save to CSV
./benchmark_gemm_universal_fp16_rcr_compv4_cshuffle_intrawave_False_False_False_False_256x256x64_4x1x1_32x32x32 \
    --m 8192 --n 8192 --k 4096 \
    --warmup 20 --repeat 200 \
    --metric 1 \
    --csv_filename results.csv

# StreamK kernel with split-k disabled (StreamK handles K parallelism internally)
./benchmark_gemm_streamk_fp16_rcr_compv4_cshuffle_intrawave_... \
    --m 3840 --n 4096 --k 2048 \
    --verify 2 --repeat 50
```

---

## Validation

`include/utility/validation.hpp` provides `calculate_rtol_atol<>()` and `compare()` template utilities for numerical correctness checking. Error thresholds are computed based on:

- **Input data types** (FP16, BF16, FP8, FP32, INT8, etc.)
- **Accumulation strategy** (single-pass vs. split-K)
- **K dimension** (larger K accumulates more rounding error)

These are used internally by `GemmProfiler` when `--verify` is non-zero.

---

## Adding a New Operation

1. Create a new subdirectory under `ops/`, e.g., `ops/my_op/`.
2. Write a JSON config file in `ops/my_op/configs/`.
3. Write a Python instance builder (`my_op_instance_builder.py`) that reads the config and generates C++ kernel headers.
4. Write a C++ benchmark entry point (`my_op_benchmark_single.cpp`) that uses `ck_tile::ArgParser` and a profiler.
5. Write a `CMakeLists.txt` that calls the Python builder at configure time and creates per-kernel `add_executable` targets.
6. Add `add_subdirectory(ops/my_op)` to `tile_engine/CMakeLists.txt`.

---

## Related Resources

- [ck_tile programming model](../include/ck_tile/README.md) — core abstractions used by all Tile Engine kernels
- [ck_tile examples](../example/ck_tile/) — standalone examples showing how to write `ck_tile` kernels
- [ck_tile tutorials](../tutorial/ck_tile/) — step-by-step walkthroughs of copy kernels, GEMM, and tile distribution
- [Composable Kernel README](../README.md) — top-level project documentation and build instructions
