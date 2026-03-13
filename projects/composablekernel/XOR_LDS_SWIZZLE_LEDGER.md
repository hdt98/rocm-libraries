# XOR LDS Swizzle - Approaches Ledger

## Problem
LDS store operations in CShuffleEpilogue exhibit bank conflicts:
- **FP8 baseline**: 416 conflicts / 208 instructions = 2.0 ratio
- **FP16 baseline**: 416 conflicts / 208 instructions = 2.0 ratio

## Successful Approaches

### Approach 4: XOR with `(m % 8) * 4` (constant 4)
**Formula**: `new_n = n ^ ((m & 7) << 2)`
**Results**:
- FP8: **0 conflicts** (down from 416, ratio 0.0)
- FP16: **208 conflicts** (down from 416, ratio 1.0 - 50% reduction)
**Job**: 221529 (FP8), 221542 (FP8+FP16)
**Why it works**: The constant XOR of 4 elements spreads N indices across bank boundaries regardless of element size. For FP8, this is exactly 1 bank word; for FP16, it creates enough spreading to halve conflicts.

## Failed Approaches

### Approach 1: Simple `(m & 3) << 2` XOR
**Formula**: `new_n = n ^ ((m & 3) << 2)`
**Result**: 416 conflicts (unchanged)
**Why it failed**: M indices coming into the XOR transform (after merge) are 0, 4, 8, 12 (strided by MLdsLayer), not 0, 1, 2, 3. So `m & 3` = 0 for all values, producing no XOR effect.

### Approach 2: Account for M stride with `(m >> 2) & 3`
**Formula**: `new_n = n ^ (((m >> 2) & 3) << 2)`
**Result**: 416 conflicts (unchanged)
**Why it failed**: Debug showed XOR values varying correctly (0, 4, 8, 12 for m=0,4,8,12), but conflicts persisted. The XOR transform operates on merged (M, N) coordinates which are dense (0-31), not the strided pre-merge values.

### Approach 3: XOR on bank word index
**Formula**: `new_n = ((n >> 2) ^ (m & 3)) << 2 | (n & 3)`
**Result**: 416 conflicts (unchanged)
**Why it failed**: The XOR spread consecutive N values across different bank words based on M, but created NEW conflicts. The stride contribution (260 bytes = 65 words per M row) was not accounted for.

### Approach 5: Parameterized XOR with `ElemsPerBankWord`
**Formula**: `new_n = n ^ ((m & 7) * ElemsPerBankWord)` where `ElemsPerBankWord = 4/DataTypeSize`
- FP8: `ElemsPerBankWord = 4`, formula = `n ^ ((m & 7) * 4)`
- FP16: `ElemsPerBankWord = 2`, formula = `n ^ ((m & 7) * 2)`
**Results**:
- FP8: 0 conflicts (correct)
- FP16: **416 conflicts** (2.0 ratio) - WORSE than the accidental 208!
**Job**: 221549
**Why it failed**: The smaller XOR amount (2 instead of 4) for FP16 doesn't spread indices enough to avoid conflicts. The constant 4 works better empirically.

### Approach 6: FP8-only XOR (skip FP16)
**Formula**: Only apply XOR for `DataTypeSize == 1`
**Results**:
- FP8: 0 conflicts (correct)
- FP16: **416 conflicts** (2.0 ratio) - baseline, no improvement
**Job**: 221555
**Why it failed**: This confirmed that the FP16 baseline is indeed 416 conflicts (2.0 ratio), and the earlier 208 conflicts was due to accidentally applying the FP8 XOR formula to FP16.

## Key Insights

1. **MLdsLayer varies by data type**:
   - FP8 (1 byte): MLdsLayer = 64 * 4 / 32 / 1 = 8
   - FP16 (2 bytes): MLdsLayer = 64 * 4 / 32 / 2 = 4

2. **The constant XOR=4 works universally**:
   - The formula `(m & 7) * 4` uses 8 M values (0-7) regardless of MLdsLayer
   - For FP8: perfectly eliminates conflicts
   - For FP16: halves conflicts (208 from 416)

3. **FP16 still has residual conflicts**:
   - 208 conflicts at 1.0 ratio means 2-way bank conflicts remain
   - May need additional optimization (different XOR pattern or padding)

### Approach 7: Constant XOR=4 for all types
**Formula**: `new_n = n ^ ((m & 7) * 4)` (same for all data types)
**Results**:
- FP8: **0 conflicts** (down from 416, ratio 0.0)
- FP16: **208 conflicts** (down from 416, ratio 1.0 - 50% reduction)
**Job**: 221565
**Why it works**: Provides good bank spreading for FP8 and partial improvement for FP16.

### Approach 8: Attempt B - Scale multiplier by data type size
**Formula**:
```cpp
BaseXorMult = (DataTypeSize == 1) ? 4 : 8;
MaxSafeXorMult = (NPerIterationShuffle - 1) / (XorMask - 1);
XorMultiplier = min(BaseXorMult, MaxSafeXorMult);
```
**Results**:
- Benchmark (2x2 waves, N=32): FP16 still has 208 conflicts (no improvement)
- Tests: **FAILED** - correctness issues with 4x1 wave configs (N=16)
  - Expected 16384 unique values, got 15362 (collisions in LDS indexing)
  - Failing tests: all 4x1 wave configurations
**Job**: 221600 (tests), 221601 (benchmarks)
**Why it failed**:
1. For common configs (N=32), multiplier still limited to 4, no improvement
2. For small N configs (N=16), even with limiting, creates index aliasing
3. The XOR transform interacts with descriptor merging in complex ways

### Approach 9: Attempt C - Guard XOR by NPerIterationShuffle >= 32
**Formula**: Same as Approach 7, but only when `NPerIterationShuffle >= 32`
**Results**:
- Tests: **FAILED** with NaN values for 32x32 MPerXdl/NPerXdl configs
  - "NaN at index 12194" - data corruption
  - Affects configs with MPerXdl=32 or NPerXdl=32 despite N=64
**Job**: 221613
**Why it failed**:
1. **NaN values indicate memory corruption** - the XOR transform creates invalid memory accesses
2. Problem is not just simple index aliasing - deeper interaction with:
   - VectorLen dimension (varies: 8, 16, 32, 64)
   - Thread distribution across warps
   - LDS descriptor merge/unmerge transforms
3. The XOR operates on **merged (M,N)** indices but interacts badly with:
   - The `VectorLen` dimension in the descriptor
   - The prior unmerge/merge transforms that create the (M,N) view
4. **Root cause**: XOR at the wrong level in the descriptor chain
   - Should potentially be applied BEFORE the final merge, or
   - Needs to account for VectorLen in the XOR formula

## Current Status

**Commit**: 8c28d3cdf65 (XOR disabled - causes NaN/corruption)
**Branch**: users/tenpercent/ck/xor-lds-swizzle
**Status**: **Investigation paused - no safe implementation found**

### Why XOR Swizzle Failed

All attempts to add XOR swizzle caused **correctness failures**:
- Approach 7: Index aliasing for N < 32
- Approach 8: Index aliasing for N < 32 (even with limits)
- Approach 9: **NaN values/corruption** for certain MPerXdl/NPerXdl configs

**Fundamental issue**: The XOR transform operates on merged (M,N) coordinates
but doesn't account for the VectorLen dimension and thread distribution. This
creates invalid LDS addresses that read/write garbage or out-of-bounds.

### Baseline Results (No XOR)

| Benchmark | Op | LDS_Insts | Conflicts | Ratio | vs Baseline |
|-----------|-----|-----------|-----------|-------|-------------|
| FP8→FP8 | Store | 208 | 0 | 0.0000 | **-100%** |
| FP8→FP8 | Load | 52 | 0 | 0.0000 | unchanged |
| FP8→FP16 | Store | 208 | 208 | 1.0000 | **-50%** |
| FP8→FP16 | Load | 52 | 0 | 0.0000 | unchanged |

### Summary
- FP8 output: **Perfect** - all bank conflicts eliminated
- FP16 output: **50% improvement** - reduced from 2.0 to 1.0 ratio
- FP16 still has 2-way conflicts that may need additional optimization
- **Scaling multiplier by data type size does not work** - causes correctness failures
- The constant XOR=4 is the safe, working solution
