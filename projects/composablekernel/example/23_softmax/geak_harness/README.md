# GEAK Test Harness - Softmax PoC

A standalone test harness for LLM-driven CK kernel optimization. The baseline
and optimized kernels are compiled as shared libraries (`.so`) and loaded into a
Python test script via ctypes. The baseline kernel output is used as ground
truth — the optimized kernel is verified against it. No CK build system
dependency — only CK headers + HIP runtime.

## Quick Start

```bash
# 1. Set up Python environment (one-time)
uv venv .venv && .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/rocm7.1

# 2. Build kernel .so files (auto-detects GPU architecture)
./compile.py

# 3. Verify correctness
python test_harness.py libbaseline.so liboptimized.so --correctness

# 4. Benchmark
python test_harness.py libbaseline.so liboptimized.so --benchmark
```

## Iteration Loop

The workflow for LLM-driven kernel optimization:

1. **Edit** tuning parameters in `optimized.cpp` (the `TUNING PARAMETERS` section)
2. **Build** the optimized kernel: `./compile.py optimized` (~2.5s with `--arch`, ~7.5s with auto-detect)
3. **Verify**: `python test_harness.py libbaseline.so liboptimized.so --correctness`
4. **Benchmark**: `python test_harness.py libbaseline.so liboptimized.so --benchmark`
5. **Read** the output: verification status, bandwidth, speedup vs baseline
6. Repeat from step 1

The baseline `.so` is built once and never changes. Only `optimized.cpp` is
modified and recompiled during iteration. The Python test script never needs
recompilation.

## Files

| File | Purpose |
|------|---------|
| `baseline.cpp` | Kernel .so with original tile configuration (compile once) |
| `optimized.cpp` | Kernel .so with tunable parameters (LLM edits this) |
| `test_harness.py` | Python test script: load, verify, benchmark, compare |
| `compile.py` | Build wrapper: auto-detects GPU arch from PyTorch, calls make |
| `Makefile` | hipcc build rules (called by compile.py) |

## Usage

### compile.py

```bash
./compile.py                # build all .so files (auto-detect GPU arch)
./compile.py optimized      # rebuild only the optimized kernel
./compile.py clean           # remove .so files
./compile.py --arch gfx942  # override GPU architecture
```

### test_harness.py

The harness has four mutually exclusive modes. Exactly one must be specified.

```bash
# Verify optimized kernel against baseline (ground truth)
python test_harness.py libbaseline.so liboptimized.so --correctness

# Run optimized kernel for profiling (minimal launches, clean trace)
python test_harness.py libbaseline.so liboptimized.so --profile

# Benchmark on HARNESS_SHAPES (20-25 shapes)
python test_harness.py libbaseline.so liboptimized.so --benchmark

# Benchmark on ALL_SHAPES (full coverage)
python test_harness.py libbaseline.so liboptimized.so --full-benchmark

# Control iteration count (default: 20, or env GEAK_BENCHMARK_ITERATIONS)
python test_harness.py libbaseline.so liboptimized.so --benchmark --iterations 50

# Change reduction dimension (default: last dim)
python test_harness.py libbaseline.so liboptimized.so --benchmark --reduce-dim 1
```

### Modes

| Flag | Shapes | Description |
|------|--------|-------------|
| `--correctness` | HARNESS_SHAPES | Deterministic seed, run both kernels on same input, compare with `torch.testing.assert_close`, exit 1 on failure |
| `--profile` | PROFILE_SHAPES (5) | Run optimized kernel once per shape for rocprofv3 capture. Tensors generated on CPU then moved to GPU for clean profiler trace. |
| `--benchmark` | HARNESS_SHAPES | Benchmark both kernels, report per-shape latency/speedup and summary with median |
| `--full-benchmark` | ALL_SHAPES | Same as `--benchmark` but over all shapes |

### Shape lists

Three shape tiers are defined at the top of `test_harness.py`:

- **ALL_SHAPES** (25 shapes): 3D softmax tensors `(B, S, H)` sorted by element count, from `[1, 8, 256]` to `[128, 128, 4096]`.
- **HARNESS_SHAPES**: same as ALL_SHAPES (since there are exactly 25).
- **PROFILE_SHAPES** (5): evenly-spaced subset of ALL_SHAPES for fast profiling.

### Example output

#### `--correctness`

```
Correctness check: liboptimized.so vs libbaseline.so (ground truth)
Shapes: 25

  PASS  [1, 8, 256]  max_err=0.00e+00
  PASS  [1, 8, 1024]  max_err=0.00e+00
  ...
  PASS  [128, 128, 4096]  max_err=0.00e+00

RESULT: PASS
```

#### `--benchmark`

```
Benchmark: liboptimized.so vs libbaseline.so
Shapes: 25, iterations: 20

                         Shape  Baseline ms  Optimized ms   BW GB/s  Speedup
--------------------------------------------------------------------------------
                   [1, 8, 256]       0.0041        0.0041       2.0    1.00x
                  [8, 128, 2048]     0.0173        0.0173     486.2    1.00x
  ...
              [128, 128, 4096]       0.1540        0.1540    1742.9    1.00x

Shapes benchmarked: 25
median_wall_time_us: 17.28
median_baseline_us:  17.29
median_speedup:      1.0009
mean_speedup:        1.0000
```

## Tuning Parameters (optimized.cpp)

The `TUNING PARAMETERS` section in `optimized.cpp` controls the kernel
configuration:

```cpp
using KernelInstance = ck::tensor_operation::device::DeviceSoftmaxImpl<
    ck::half_t,       // InDataType
    float,            // AccDataType
    ck::half_t,       // OutDataType
    PassThrough,      // InElementwiseOperation
    PassThrough,      // AccElementwiseOperation
    3,                // Rank
    1,                // NumReduceDim
    256,              // BlockSize
    8,                // ClusterM
    32,               // ClusterK
    1,                // SliceM
    8,                // SliceK
    1,                // SrcVecDim (0=M, 1=K)
    8,                // SrcScalarPerVector
    8                 // OutScalarPerVector
>;
```

**Constraints:**
- `ClusterM * ClusterK` must equal `BlockSize`
- `SrcScalarPerVector` and `OutScalarPerVector` must be powers of 2 (1, 2, 4, 8)
- `SrcVecDim`: 0 = vectorize along M dimension, 1 = vectorize along K dimension
- Not all combinations are valid — `IsSupportedArgument` returns false for
  invalid configs (reported as UNSUPPORTED in the test output)

## Adapting to Other Kernel Families

This harness is a template. To adapt it for a different CK kernel (e.g., GEMM,
convolution), follow this pattern:

### 1. Kernel .so (`optimized.cpp`)

Replace the `DeviceSoftmaxImpl` template with your target kernel and adjust the
`run_kernel` C ABI signature to match the kernel's argument requirements:

```cpp
// GEMM example
extern "C" float run_kernel(
    const void* a_dev, const void* b_dev, void* c_dev,
    int64_t M, int64_t N, int64_t K,
    int64_t lda, int64_t ldb, int64_t ldc,
    bool time_kernel, int warmup, int nrepeat)
{
    GemmInstance op;
    auto arg = op.MakeArgumentPointer(/* ... */);
    // ... same pattern: construct, validate, invoke, sync
}
```

### 2. Python test harness (`test_harness.py`)

Update three things:
- **`load_kernel()`**: change `argtypes` to match new C ABI
- **`call_kernel()`**: change arguments passed to `run_kernel`
- **Shape lists**: replace `ALL_SHAPES` / `HARNESS_SHAPES` / `PROFILE_SHAPES`
  with shapes appropriate for the new kernel family

The verification model stays the same: the baseline kernel output is the ground
truth, and the optimized kernel is compared against it.

### 3. What stays the same

- `compile.py` and `Makefile` work unchanged (same hipcc flags)
- The overall pattern: `.cpp` with tuning params + `extern "C"` wrapper +
  Python test script with baseline-as-ground-truth verification
- The four GEAK modes (`--correctness`, `--profile`, `--benchmark`,
  `--full-benchmark`) and `--iterations N`
- ctypes loading, GPU synchronization, error reporting

## Testing Limitations

### Baseline-as-ground-truth is sufficient for tuning, not for kernel development

This harness validates the optimized kernel's output against the baseline
kernel's output. This works well when the LLM is only modifying **tuning
parameters** (tile sizes, vector widths, cluster dimensions) — the kernel logic
is unchanged, so comparing outputs between configurations is sufficient.

However, when the LLM modifies **kernel source code** (e.g., CK grid/block
kernels, pipeline logic, memory access patterns), baseline-only validation has
limitations:

- **Assumes baseline correctness**: the baseline is treated as ground truth. If
  the baseline itself has a bug, the harness won't catch it.
- **No isolation**: if the output diverges, the comparison only tells you
  *that* it's wrong, not *where*. The LLM cannot easily test individual
  components (e.g., just the reduction, just the memory layout transform).
- **No edge case coverage**: the harness tests one random input per shape.
  CK's test infrastructure covers boundary conditions, alignment edge cases,
  and specific parameter combinations that trigger different code paths.
- **No unit test granularity**: old CK has limited unit tests, but CK Tile has
  better coverage. An LLM modifying kernel internals should run the relevant
  CK/CK Tile unit tests, not just the end-to-end comparison.

## Prerequisites

- ROCm with hipcc (`/opt/rocm/bin/hipcc`)
- CK source tree (for headers at `../../../include` relative to this directory)
- Python 3.10+ with PyTorch (ROCm build)
