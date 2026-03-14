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

### Approach 10: XOR Before VectorLen Merge (4D Transform)
**Hypothesis**: Previous NaN issues were caused by XOR corrupting the VectorLen dimension. Apply XOR to the 4D descriptor BEFORE the final merge to preserve VectorLen thread mapping.

**Implementation**:
- Created new `xor_lds_bank_4d_t` transform operating on `[M_coarse, MLdsLayer, N_vec, VectorLen]`
- XOR only dimension 2 (N_vector): `new_n_vec = n_vec ^ ((merged_m & 7) * 4)`
- Dimensions 0, 1, 3 pass through unchanged
- Applied between `lds_block_desc_1` (after unmerge) and `lds_block_desc_2` (before merge)
- Safety constraint: `N_vectors >= 32` to ensure sufficient spread

**Results**:
- Tests: **ALL PASSED** ✅ - No NaN values, no memory corruption
  - test_ck_tile_cshuffle_epilogue_fp16 - PASS
  - test_ck_tile_cshuffle_epilogue_fp8 - PASS
  - test_ck_tile_cshuffle_epilogue_fp8_gfx950 - PASS
  - test_ck_tile_cshuffle_epilogue_scale - PASS
- Benchmarks: **416 conflicts** ❌ - Same as baseline (no improvement)
  - FP8→FP8 Store: 416 conflicts (regression from 0)
  - FP8→FP16 Store: 416 conflicts (regression from 208)

**Job**: 221651 (tests), 221649 (benchmarks)
**Commit**: 52a79f26c1c

**Why it failed**:
1. **Safety constraint too strict**: Test config has only 4 N_vectors (not 32)
   - 2x2 waves, 16x16 MPerXdl/NPerXdl → NPerIterationShuffle = 32
   - FP16: VectorLen=8 → N_vectors = 32/8 = **4** (< 32, XOR skipped)
   - FP8: VectorLen=16 → N_vectors = 32/16 = **2** (< 32, XOR skipped)
2. **XOR was never applied** - fell back to baseline path
3. The 416 conflicts match baseline (no XOR), not a regression from XOR
4. **Correctness success is significant** - proves 4D transform preserves VectorLen correctly

**Conclusion**: Implementation is correct but needs relaxed safety constraint to actually run.

### Approach 11: Relaxed Constraint with Scaled Parameters
**Hypothesis**: Approach 10's XOR implementation is correct but constraint too strict. Relax to N_vectors >= 4 and scale XorMult based on configuration size.

**Implementation**:
- Relaxed constraint: N_vectors >= 4 (from 32)
- Scaled XorMask: 4 for N < 8, else 8
- Scaled XorMult: 1 for N < 32, else 4
- For N_vectors=4: XorMask=4, XorMult=1 → XOR offset {0,1,2,3}

**Results**:
- Tests: **ALL PASSED** ✅
  - test_ck_tile_cshuffle_epilogue_fp16 - PASS
  - test_ck_tile_cshuffle_epilogue_fp8 - PASS
  - test_ck_tile_cshuffle_epilogue_fp8_gfx950 - PASS
  - test_ck_tile_cshuffle_epilogue_scale - PASS
- Benchmarks: **416 conflicts** ❌ - No improvement
  - FP8→FP8 Store: 416 conflicts (no change from baseline)
  - FP8→FP16 Store: 416 conflicts (no change from baseline)

**Job**: 221661 (tests), 221662 (benchmarks)
**Commit**: c3523b5838e

**Why it failed**:
1. **XorMult=1 too weak**: XOR offset of only 0-3 N_vectors insufficient to change bank assignment
2. **Limited M coverage**: Mask `(m & 3)` only uses 4 out of 32 M values, providing 25% coverage per offset
3. **Bank formula insensitive**: Small XOR offsets don't alter `(M_contribution + N/2) % 64` enough
4. **FP8 baseline mystery**: FP8 showing 416 conflicts (Approach 7 had 0) - config may have changed

**Conclusion**: Correctness proven again, but XorMult=1 provides insufficient bank spreading. Need larger XOR offsets.

### Approach 12: Adaptive XOR with Modulo Safety
**Hypothesis**: Add modulo operation to prevent aliasing and enable more aggressive XOR multipliers. Use adaptive parameters that scale with N_vectors size.

**Implementation**:
- Adaptive XorMask: `(N_vectors >= 8) ? 8 : N_vectors`
- Adaptive XorMult: `(N_vectors >= 32) ? 4 : (N_vectors >= 8) ? 2 : 1`
- Relaxed constraint: `N_vectors >= 2` (enables even FP8)
- **Modulo safety**: `xor_val = ((merged_m & (XorMask-1)) * XorMult) % N_vectors`
  - Prevents aliasing even with aggressive multipliers
  - Runtime extraction of N_vectors from descriptor

**Expected for N_vectors=4 (FP16 test)**:
- XorMask=4, XorMult=1 → XOR range {0,1,2,3} (full 4-way spreading)
- Bank shift: 4, 8, 12 banks after VectorLen merge

**Results**:
- Tests: **ALL PASSED** ✅
  - test_ck_tile_cshuffle_epilogue_fp16 - PASS
  - test_ck_tile_cshuffle_epilogue_fp8 - PASS
  - test_ck_tile_cshuffle_epilogue_fp8_gfx950 - PASS
  - test_ck_tile_cshuffle_epilogue_scale - PASS
- Benchmarks: **416 conflicts** ❌ - No improvement
  - FP8→FP8 Store: 416 conflicts (identical to Approach 11)
  - FP8→FP16 Store: 416 conflicts (identical to Approach 11)

**Job**: 221675 (tests), 221674 (benchmarks)
**Commit**: 100819b2f0e

**Why it failed**:
1. **XOR pattern correct but ineffective**: Full 4-way XOR spreading applied, but doesn't break bank conflict pattern
2. **Bank formula mismatch**: XOR shifts in N dimension (4, 8, 12 banks) don't redistribute threads to avoid conflicts caused by M dimension
3. **M dimension dominates**: Bank assignment formula `((M/4)*65 + (M%4)*16 + N/2) % 32` creates conflicts primarily from M distribution
4. **XOR in N can't fix M conflicts**: The problem is fundamentally a **2D conflict pattern** (M+N interaction), but XOR only affects N
5. **Same result as Approach 11**: For N_vectors=4, both use XorMask=4, XorMult=1 → identical XOR pattern → identical conflicts

**Key Insight**: The bank conflict structure is **entangled between M and N dimensions**. XOR swizzling N_vectors alone (even with full utilization and modulo safety) cannot break conflicts that arise from the wave layout and M-layer structure's contribution to bank assignment.

**Conclusion**: N-dimension-only XOR approaches exhausted. Need to either:
- Attack M dimension (2D XOR, change MLdsLayer, different wave mapping)
- Change bank alignment (padding)
- Accept partial improvement and move on

### Approach 13: Safe 2D XOR with Modulo
**Hypothesis**: Return to 2D XOR but add modulo safety to prevent index aliasing and corruption that plagued Approach 7-9.

**Implementation**:
- Apply XOR to merged 2D (M, N) descriptor with modulo: `xor_val = ((m & 7) * 4) % N`
- Operates on `lds_block_desc_2` (final merged descriptor)
- Modulo ensures XOR value stays within N range for any N size

**Results**:
- Tests: **FAILED** - NaN values, same corruption as Approach 9
- The modulo doesn't prevent the underlying memory corruption

**Why it failed**:
1. **2D XOR after merge breaks Vec structure**: Even with modulo, XOR at merged level corrupts thread ownership
2. **The problem is WHERE XOR is applied, not the formula**: Once Vec is merged into N, any N modification breaks thread-to-element mapping
3. **Vec structure is not algebraic** - it's a physical mapping constraint

**Conclusion**: 2D XOR approaches (7, 8, 9, 13) all fail because they operate after Vec merge.

---

### Approach 14: 3D XOR at Level 0 (Current)
**Hypothesis**: Apply XOR to `lds_block_desc_0` which has 3D structure `[M_coarse, N_interleaved, Vec]` where Vec is a **separate dimension**, not merged into N.

**Key Insight**: At `lds_block_desc_0`:
- `Vec` is dimension 2 (separate, not merged)
- `N_interleaved = MLdsLayer × N_vectors` (e.g., 4×4=16 values for FP16)
- XORing N_interleaved (dim 1) based on M_coarse (dim 0) preserves Vec completely

**Implementation**:
- New `xor_lds_bank_3d_t` transform operating on 3D descriptor
- XOR formula: `new_n_interleaved = n_interleaved ^ ((m_coarse & 7) * 2) % n_interleaved_len`
- Vec (dim 2) passed through unchanged
- Applied BEFORE unmerge transform, not after merge

**Expected Coverage**:
- 4D XOR (Approaches 10-12): Only affected N_vectors (4 values) = 12.5% of N
- 3D XOR: Affects N_interleaved (16 values) = ~50% of N

**XOR Parameters**:
- XorMask = 8 (use M_coarse & 7 for 8 distinct patterns)
- XorMult = 2 (spread by 2 positions)

**Status**: Implemented, pending verification
**Commit**: TBD

**Expected Results**:
- Correctness: ✅ PASS (Vec preserved as separate dimension)
- Bank Conflicts: Improved from 416 (TBD via profiling)

---

### Approach 15: 4D Dual-XOR (M_layer + N_vec)
**Hypothesis**: Return to 4D XOR (which passed all correctness tests) but double coverage by XORing BOTH M_layer and N_vec dimensions.

**Key Insight**: XOR is SAFE at 4D level (after unmerge, before merge) because:
- Merge is ADDITIVE: `N = N_vec × VectorLen + Vec`
- Vec appears ONLY as addition, never through XOR
- Therefore: `(π(N_vec) × VectorLen + Vec) % VectorLen = Vec` ✓
- Thread assignment of `vec ∈ [0, VectorLen)` is ALWAYS preserved

**Why Different Levels Fail/Succeed**:
| Level | Operation After | XOR Safe? | Why |
|-------|-----------------|-----------|-----|
| **3D** (before unmerge) | div/mod | ❌ NO | `(a⊕v) ÷ k ≠ (a÷k) ⊕ something` |
| **4D** (after unmerge) | multiply/add | ✅ YES | `(a⊕v) × k + c` preserves Vec |
| **2D** (after merge) | thread access | ❌ NO | Vec merged into N, corruption |

**Implementation**:
- Modified `xor_lds_bank_4d_t` to XOR both dimensions:
  - `new_m_layer = m_layer ^ (n_vec & (m_layer_len - 1))`
  - `new_n_vec = n_vec ^ ((merged_m & (XorMask-1)) * XorMult % n_vectors)`
- Vec (dim 3) passed through unchanged (CRITICAL!)
- Applied AFTER unmerge, BEFORE merge in `cshuffle_epilogue.hpp`

**Bijectivity Proof**:
```
T(m, n) = (m ⊕ f(n), n ⊕ g(m))
Inverse: T⁻¹(m', n') = (m' ⊕ f(n' ⊕ g(m')), n' ⊕ g(m'))
XOR is self-inverse, so this always works.
```

**Coverage Improvement**:
| Approach | XOR Dimensions | Patterns | Coverage |
|----------|---------------|----------|----------|
| N_vec only (10-12) | 1 | 4 | 12.5% |
| **M_layer + N_vec** | 2 | 4×4=16 | ~50% |

**XOR Parameters**:
- XorMask = 8 (use merged_m & 7 for 8 distinct N patterns)
- XorMult = 2 (spread by 2 positions)
- M_layer XOR: `n_vec & (m_layer_len - 1)` (4 patterns for M_layer_len=4)

**Results**:
- Tests: **ALL PASSED** ✅
  - test_ck_tile_cshuffle_epilogue_fp16 - PASS (5 tests)
  - test_ck_tile_cshuffle_epilogue_fp8 - PASS (7 tests)
  - test_ck_tile_cshuffle_epilogue_fp8_gfx950 - PASS (6 tests)
  - test_ck_tile_cshuffle_epilogue_scale - PASS (2 tests)
- Benchmarks: **416 conflicts** ❌ - No improvement
  - FP8→FP8 Store: 416 conflicts (2.0 ratio)
  - FP8→FP16 Store: 416 conflicts (2.0 ratio)

**Job**: 222784 (tests), 222789 (benchmarks)
**Commit**: 69b9e36526e

**Why it failed to reduce conflicts**:
1. **XOR at 4D level shuffles vector buckets, not bank addresses**: The XOR permutes which N_vec slot data goes to, but after merge the final bank assignment is unchanged
2. **Bank conflicts are determined by final merged addresses**: `bank = ((M/4)*65 + (M%4)*16 + N/2) % 32` - this formula doesn't change just because we permuted N_vec before merge
3. **Coverage doesn't equal effectiveness**: Even with 50% coverage (16 patterns), the XOR values don't redistribute threads to different banks
4. **The problem is structural**: Bank conflicts arise from wave layout × LDS stride interaction, not from which vector bucket data lands in

**Conclusion**: 4D XOR (Approaches 10-12, 15) is the ONLY safe level for XOR transforms, but it cannot break the bank conflict pattern because it operates on logical indices that don't affect bank assignment after merge.

---

### Approach 16: Column-Major Wave Ordering

**Hypothesis**: Change wave index mapping from row-major to column-major to alter which threads access which (M,N) coordinates, potentially breaking the bank conflict pattern.

**Implementation**:
- Changed `tile_distribution_encoding` from `sequence<1, 2>` (row-major) to `sequence<2, 1>` (column-major)
- Row-major: `wave_id = m_wave * NWave + n_wave`
- Column-major: `wave_id = n_wave * MWave + m_wave`

```
Row-major (baseline):            Column-major:
Wave0(0,0)  Wave1(0,1)           Wave0(0,0)  Wave2(0,1)
Wave2(1,0)  Wave3(1,1)           Wave1(1,0)  Wave3(1,1)
```

**Results**:

| Configuration | Correctness | Bank Conflicts |
|---------------|-------------|----------------|
| Column-major + 4D XOR | ✅ PASS | 416 (no improvement) |
| Column-major only (no XOR) | ❌ **FAIL** (1 test) | 416 (no improvement) |

**Job**: 222868/222869 (with XOR), 222870/222871 (without XOR)

**Why it failed**:
1. **Column-major alone breaks correctness**: The wave ordering change requires compensating transforms (like 4D XOR) to maintain correct thread-to-element mapping
2. **No effect on bank conflicts**: Even when correct (with XOR), conflicts remain at 416
3. **Bank assignment is stride-dependent, not wave-order-dependent**: The bank formula depends on final LDS addresses, not which wave accesses them

**Conclusion**: Wave ordering alone cannot reduce bank conflicts. The conflict pattern is determined by the LDS stride and MLdsLayer interleaving, not by wave assignment.

---

### Wave Layout Comparison (16x16x128 MFMA Baseline)

Benchmarked all wave layouts WITHOUT any XOR transforms to find if any layout naturally avoids conflicts.

**Results** (Job 222884):

| Wave Layout | FP8→FP8 Store | FP8→FP16 Store | Loads |
|-------------|---------------|----------------|-------|
| **4x1** | **0** ✅ | 416 | 0 |
| **2x2** | 416 | 416 | 0 |
| **1x4** | 416 | 416 | 0 |

**Key Finding**: The **4x1 wave layout with FP8→FP8 output** achieves **ZERO bank conflicts** without any XOR swizzle!

**Why 4x1 + FP8 works**:
- FP8 output: VectorLen=16, MLdsLayer=8
- 4x1 layout: 4 waves stacked in M direction, 1 wave in N direction
- This specific combination aligns thread access patterns to avoid bank conflicts

**Why FP8→FP16 still has conflicts**:
- FP16 output: VectorLen=8, MLdsLayer=4
- Different interleaving factor changes the stride pattern
- All wave layouts produce 416 conflicts for FP16 output

**Conclusion**: For 16x16x128 MFMA:
- **FP8→FP8**: Use 4x1 wave layout for zero conflicts (no code changes needed, just config selection)
- **FP8→FP16**: No wave layout eliminates conflicts; still needs investigation

---

### Summary
- **FP8→FP8 with 4x1 wave layout**: **ZERO conflicts** ✅ (baseline, no XOR needed)
- **FP8→FP16**: 416 conflicts across all wave layouts (unsolved)
- **XOR at 2D level (after merge)**: Eliminates conflicts but causes memory corruption (Approaches 7-9, 13)
- **XOR at 4D level (before merge)**: Preserves correctness but has NO effect on conflicts (Approaches 10-12, 15)
- **XOR at 3D level (before unmerge)**: Breaks correctness due to division/modulo operations (Approach 14)
- **Column-major wave ordering**: No effect on conflicts; breaks correctness without XOR (Approach 16)
- **Fundamental insight**: There is NO level where XOR can both preserve correctness AND reduce bank conflicts
  - Safe levels (4D) don't affect bank addresses
  - Effective levels (2D) corrupt thread-to-element mapping
- **Recommendation**: Use 4x1 wave layout for FP8→FP8 workloads; FP8→FP16 requires further investigation (padding, different MFMA tiles)
