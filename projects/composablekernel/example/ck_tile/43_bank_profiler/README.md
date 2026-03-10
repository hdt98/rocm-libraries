# CK Tile GEMM Bank Profiler

Tools to analyze AMD GPU LDS (Local Data Share) memory bank conflict behavior using CK tile GEMM kernels and `rocprofv3` hardware counters.

## What It Does

**Part 1 - Hardware Probing:** Determines the GPU's LDS bank count and thread phase structure using a minimal probe kernel with controlled LDS access patterns.

**Part 2 - GEMM Profiling:** Measures `SQ_LDS_BANK_CONFLICT` counters across 6 different tile configurations to quantify how well CK's LDS layout (3D + padding) avoids bank conflicts.

## Tile Configurations

| Config | M_Tile | N_Tile | K_Tile | Warps | BlockSize |
|--------|--------|--------|--------|-------|-----------|
| 128x128x32 | 128 | 128 | 32 | 4 | 256 |
| 256x128x32 | 256 | 128 | 32 | 8 | 512 |
| 128x256x32 | 128 | 256 | 32 | 8 | 512 |
| 256x256x32 | 256 | 256 | 32 | 16 | 1024 |
| 64x64x32 | 64 | 64 | 32 | 1 | 64 |
| 128x128x64 | 128 | 128 | 64 | 4 | 256 |

## Quick Start

```bash
# Build all binaries
cd /path/to/composable_kernel/build
make tile_bank_profiler -j

# Run everything
cd /path/to/composable_kernel/example/ck_tile/16_bank_profiler
CK_BUILD_DIR=/path/to/build bash script/run_all.sh
```

## Step-by-Step Usage

### 1. Build

```bash
cd /path/to/composable_kernel/build
make tile_bank_profiler -j$(nproc)
```

### 2. Probe hardware (one-time, ~35 minutes for phase solver)

```bash
cd example/ck_tile/16_bank_profiler

# Bank count (~5-10 min)
python3 script/bank_solver.py --build-dir=/path/to/build --mode=2

# Phase structure (~30 min)
python3 script/phase_solver.py --build-dir=/path/to/build --mode=2
```

Instruction modes: `0`=ds_read_b64, `1`=ds_read_b96, `2`=ds_read_b128, `3`=ds_write_b64

### 3. Profile GEMM kernels

```bash
# All configs, all problem sizes
python3 script/gemm_profiler.py --build-dir=/path/to/build

# Specific configs
python3 script/gemm_profiler.py --build-dir=/path/to/build \
    --configs 128x128x32 256x128x32

# Specific problem sizes
python3 script/gemm_profiler.py --build-dir=/path/to/build \
    --sizes 1024x1024x1024 4096x4096x4096
```

### 4. Analyze results

```bash
python3 script/analyze_results.py
```

## Output Files

| File | Description |
|------|-------------|
| `bank_results.txt` | Detected LDS bank count |
| `phase_results.txt` | Thread phase assignments + conflict matrix |
| `out/gemm_bank_profile_results.csv` | Raw profiling data for all GEMM configs |
| `out/analysis_report.txt` | Comparative analysis report |

## Dependencies

- ROCm with `rocprofv3`
- Python 3 with `pandas`
- CK build with HIP compiler

## How It Works

### Bank Probing

The probe kernel (`lds_probe`) launches 1 wavefront (64 threads) and selectively enables 2 threads to access specific LDS addresses. By sweeping offsets under `rocprofv3`, we detect:

- **Bank count**: The first offset that causes a wraparound conflict reveals `NUM_BANKS`
- **Phase structure**: Thread pairs that conflict when accessing the same bank are in the same "phase"

### GEMM Profiling

CK's GEMM pipeline stages matrix tiles through LDS using a 3D + padding layout:

```
A in LDS: shape [K/8, M, 8], stride [(M+1)*8, 8, 1]
B in LDS: shape [K/8, N, 8], stride [(N+1)*8, 8, 1]
```

The `+1` padding on the M/N stride dimension is designed to avoid bank conflicts. Profiling quantifies its effectiveness across different tile sizes and problem dimensions.
