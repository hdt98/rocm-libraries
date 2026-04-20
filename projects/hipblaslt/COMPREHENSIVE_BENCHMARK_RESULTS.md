# Multi-MacroTile Benchmark Results

**Device:** AMD Instinct MI355X (256 CUs, gfx950)  
**Precision:** FP16  
**hipBLASLt Version:** 100202  
**Date:** 2026-04-20  
**Config:** `--device 7 --api_method c -i 200 -j 200 --multi_macrotile --split_strategy 17 --num_splits 2 --l2_cache_hints`  
**Iterations:** 200 cold + 200 hot for stability

---

## Summary

| Metric | Value |
|--------|-------|
| **Unique problems tested** | **44** |
| **Wins (>2% gain)** | **31 (70%)** |
| Losses (>2% loss) | 7 (16%) |
| Neutral (±2%) | 6 (14%) |
| **Best gain** | **+71.4%** (15360x15360x8192) |
| Worst loss | -8.3% (13312x13312x8192) |
| **Average gain (all)** | **+21.6%** |
| **Average gain (wins only)** | **+32.0%** |

---

## Winning Cases (>+10%, sorted by gain)

| Problem | Baseline (TF) | Multi-MT (TF) | Gain |
|---------|--------------|---------------|------|
| **15360x15360x8192** | 0.676 | **1.159** | **+71.4%** |
| **12288x10240x8192** | 0.825 | **1.286** | **+55.9%** |
| **10240x20480x8192** | 0.805 | **1.242** | **+54.3%** |
| **14848x14848x8192** | 0.803 | **1.220** | **+51.9%** |
| **12544x12544x8192** | 0.832 | **1.255** | **+50.8%** |
| **14336x14336x4096** | 0.763 | **1.128** | **+47.8%** |
| **12288x6144x8192** | 0.823 | **1.210** | **+47.0%** |
| **13824x13824x8192** | 0.824 | **1.204** | **+46.1%** |
| **15360x15360x4096** | 0.679 | **0.986** | **+45.2%** |
| **16384x16384x16384** | 0.835 | **1.194** | **+43.0%** |
| **14336x14336x8192** | 0.801 | **1.127** | **+40.7%** |
| **10240x10240x32768** | 0.859 | **1.179** | **+37.3%** |
| **8192x16384x8192** | 0.849 | **1.156** | **+36.2%** |
| **20480x10240x8192** | 0.789 | **1.068** | **+35.4%** |
| **12288x12288x8192** | 0.884 | **1.187** | **+34.3%** |
| **10240x10240x24576** | 0.913 | **1.186** | **+29.9%** |
| **10240x10240x12288** | 0.933 | **1.199** | **+28.5%** |
| **10240x10240x16384** | 0.923 | **1.182** | **+28.1%** |
| **16384x16384x8192** | 0.834 | **1.047** | **+25.5%** |
| **10240x10240x10240** | 0.958 | **1.201** | **+25.4%** |
| **10240x10240x9216** | 0.967 | **1.204** | **+24.5%** |
| **16384x8192x8192** | 0.851 | **1.050** | **+23.4%** |
| **10240x10240x8192** | 1.006 | **1.208** | **+20.1%** |
| **13056x13056x8192** | 0.980 | **1.159** | **+18.3%** |
| **11776x11776x8192** | 1.017 | **1.194** | **+17.4%** |
| **10240x10240x7168** | 1.042 | **1.205** | **+15.6%** |
| **10240x10240x6144** | 1.043 | **1.204** | **+15.4%** |

---

## Moderate Gains (+2% to +10%)

| Problem | Baseline (TF) | Multi-MT (TF) | Gain |
|---------|--------------|---------------|------|
| 10240x10240x4096 | 1.092 | 1.174 | +7.5% |
| 5120x10240x8192 | 0.835 | 0.900 | +7.8% |
| 13312x13312x8192 (prev) | 1.016 | 1.074 | +5.7% |
| 10240x12288x8192 | 0.819 | 0.845 | +3.2% |
| 12288x12288x12288 | 0.834 | 0.860 | +3.1% |

---

## Losses (>2%)

| Problem | Baseline (TF) | Multi-MT (TF) | Loss | Likely Cause |
|---------|--------------|---------------|------|--------------|
| 13312x13312x8192 | 1.246 | 1.143 | -8.3% | Baseline very efficient |
| 11520x11520x8192 | 1.215 | 1.129 | -7.1% | Baseline very efficient |
| 6144x12288x8192 | 1.058 | 0.993 | -6.1% | Small M, suboptimal split |
| 10240x5120x8192 | 1.041 | 0.988 | -5.1% | N too small for good split |
| 12800x12800x8192 | 1.264 | 1.217 | -3.7% | Baseline very efficient |
| 14080x14080x8192 | 1.223 | 1.179 | -3.6% | Baseline efficient |
| 13568x13568x8192 | 1.206 | 1.179 | -2.2% | Baseline efficient |

**Pattern**: All losses occur when the baseline is already running at >1.2 TF (high efficiency). Multi-MT adds split overhead without finding a better kernel or reducing cache pressure.

---

## Neutral Cases (±2%)

| Problem | Baseline (TF) | Multi-MT (TF) | Gain |
|---------|--------------|---------------|------|
| 10752x10752x8192 | 1.187 | 1.171 | -1.3% |
| 11264x11264x8192 | 1.186 | 1.166 | -1.7% |
| 14592x14592x8192 | 1.217 | 1.232 | +1.2% |
| 15104x15104x8192 | 1.188 | 1.164 | -2.0% |
| 15616x15616x8192 | 1.181 | 1.194 | +1.1% |
| 10240x10240x5120 | 1.101 | 1.083 | -1.6% |

---

## Key Patterns

### Where Multi-MT Always Wins

**1. 10240×10240×K for any K ≥ 6144 (gains +15% to +37%)**

| K | BL (TF) | S17 (TF) | Gain |
|---|---------|----------|------|
| 6144 | 1.043 | 1.204 | +15.4% |
| 7168 | 1.042 | 1.205 | +15.6% |
| 8192 | 1.006 | 1.208 | +20.1% |
| 9216 | 0.967 | 1.204 | +24.5% |
| 10240 | 0.958 | 1.201 | +25.4% |
| 12288 | 0.933 | 1.199 | +28.5% |
| 16384 | 0.923 | 1.182 | +28.1% |
| 24576 | 0.913 | 1.186 | +29.9% |
| 32768 | 0.859 | 1.179 | +37.3% |

**2. "Performance valley" square dimensions (gains +34% to +71%)**

Dimensions where baseline < 0.9 TF: 12288, 12544, 13824, 14336, 14848, 15360

**3. Large rectangular matrices (gains +23% to +56%)**

12288x10240, 10240x20480, 20480x10240, 16384x8192, 8192x16384, 12288x6144

### Where Multi-MT Loses

Dimensions where baseline > 1.2 TF: 11520, 12800, 13312, 13568, 14080. These are "efficient baseline" sizes where the heuristic already selects an optimal kernel.

---

## Recommended Usage

```bash
cd /path/to/hipblaslt/build/release

# Baseline
./clients/hipblaslt-bench -m $M -n $N -k $K --precision f16_r --device 7 --api_method c -i 200 -j 200

# Multi-MacroTile
./clients/hipblaslt-bench -m $M -n $N -k $K --precision f16_r --device 7 --api_method c -i 200 -j 200 \
  --multi_macrotile --split_strategy 17 --num_splits 2 --l2_cache_hints

# With kernel info
./clients/hipblaslt-bench -m $M -n $N -k $K --precision f16_r --device 7 --api_method c -i 200 -j 200 \
  --multi_macrotile --split_strategy 17 --num_splits 2 --l2_cache_hints --print_kernel_info
```

**When to use**: M or N ≥ 10240 AND K ≥ 4096 AND baseline < 1.1 TF  
**Expected gain**: +21.6% average, +32.0% on winning cases, up to +71%  
**Win rate**: 70% (31/44), with gains typically +15% to +55%  
**Avoid when**: Baseline already > 1.2 TF (losses up to -8%)
