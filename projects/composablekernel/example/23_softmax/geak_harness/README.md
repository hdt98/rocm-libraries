# GEAK Test Harness - Softmax PoC

A standalone test harness for LLM-driven CK kernel optimization. The kernel is
compiled as a shared library (`.so`) and loaded into a Python/PyTorch test
script via ctypes. No CK build system dependency — only CK headers + HIP
runtime.

## Quick Start

```bash
# 1. Set up Python environment (one-time)
uv venv .venv && .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/rocm7.1

# 2. Build kernel .so files (auto-detects GPU architecture)
./compile.py

# 3. Run tests
python test_harness.py libbaseline.so liboptimized.so
```

## Iteration Loop

The workflow for LLM-driven kernel optimization:

1. **Edit** tuning parameters in `optimized.cpp` (the `TUNING PARAMETERS` section)
2. **Build** the optimized kernel: `./compile.py optimized` (~2.5s with `--arch`, ~7.5s with auto-detect)
3. **Test** against baseline: `python test_harness.py libbaseline.so liboptimized.so`
4. **Read** the output: verification status, bandwidth, speedup vs baseline
5. Repeat from step 1

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

```bash
# Verify and benchmark a single kernel
python test_harness.py liboptimized.so

# Compare two kernels (first kernel is the baseline for speedup)
python test_harness.py libbaseline.so liboptimized.so

# Custom shapes (semicolon-separated)
python test_harness.py liboptimized.so --shapes "8,128,2048;32,64,4096"

# Control timing: 10 warmup iterations, 200 timed iterations
python test_harness.py liboptimized.so --warmup 10 --nrepeat 200

# Change reduction dimension (default: last dim)
python test_harness.py liboptimized.so --reduce-dim 1

# Skip verification (timing only)
python test_harness.py liboptimized.so --no-verify
```

### Example output

```
Shape: [8, 128, 2048], reduce_dim=2
------------------------------------------------------------
  libbaseline.so                     0.016 ms     517.1 GB/s  [PASS, max_err=1.91e-06]
  liboptimized.so                    0.014 ms     590.2 GB/s  [PASS, max_err=3.81e-06]

  liboptimized.so vs libbaseline.so: 1.14x
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
- **Reference implementation**: replace `torch.softmax()` with the appropriate
  PyTorch operation (e.g., `torch.matmul()`, `torch.nn.functional.conv2d()`)

### 3. What stays the same

- `compile.py` and `Makefile` work unchanged (same hipcc flags)
- The overall pattern: `.cpp` with tuning params + `extern "C"` wrapper +
  Python test script with PyTorch reference
- ctypes loading, GPU synchronization, error reporting

## Prerequisites

- ROCm with hipcc (`/opt/rocm/bin/hipcc`)
- CK source tree (for headers at `../../../include` relative to this directory)
- Python 3.10+ with PyTorch (ROCm build)
