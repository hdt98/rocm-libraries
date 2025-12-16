# Memory Alignment Optimization for hipBLASLt

## Overview

This document describes the memory alignment optimization features added to hipBLASLt/Tensile to address performance issues caused by misaligned matrix strides.

### Problem Statement

A 20-25% performance drop was observed when matrix stride is not aligned to cacheline boundaries. 

**Example Problem:**
- GEMM: 2048x3072x1880, TN (A transposed, B non-transformed)
- Data type: BFP16 (2 bytes per element)
- Stride: K × 2 = 1880 × 2 = 3760 bytes
- Problem: 3760 is NOT a multiple of 256-byte cacheline
- Observation: Padding K to 1920 (stride=3840 bytes, aligned) → 20-25% improvement

### Root Cause

For K-major matrices (transposed A or non-transposed B):
- Consecutive elements in a row are K elements apart
- Memory address stride = K × bytes_per_element
- When stride is misaligned to cacheline (typically 256 bytes):
  - Poor cache utilization
  - Increased memory bank conflicts
  - Suboptimal memory coalescing

## Implemented Solutions

Three complementary techniques have been implemented:

### 1. Address Interleaving

**Concept:** Remap row indices to spread accesses across cachelines.

Instead of linear: `J = Rn + Kn × Sn`  
Use interleaved: `J = Rn × tileSize + Kn`

**Benefits:**
- Simple to implement
- Low overhead
- Works well for moderately misaligned strides
- No LDS overhead

**Configuration:**
```yaml
EnableAddressInterleave: True
AddressInterleaveDirection: "N"       # or "M"
AddressInterleaveAlignment: 256       # target alignment in bytes
AddressInterleaveTileSize: 16         # number of tiles to interleave
```

**When to use:**
- Stride misalignment < 128 bytes
- Limited LDS available
- Want minimal code complexity

### 2. LDS Re-alignment with Multi-buffering

**Concept:** Use Local Data Share (LDS) to realign data with multi-buffering.

**How it works:**
1. Calculate per-row aligned read pointers: `read_ptr[row] = (stride × row) & ~(K-1)`
2. Calculate LDS offsets: `lds_offset[row] = stride × row - read_ptr[row]`
3. Use 2 or 3 LDS buffers for overlap
4. Wrap reads/writes within LDS buffer space

**Benefits:**
- Handles any stride misalignment
- Can combine with address interleaving
- Flexible (2 or 3 buffers)

**Costs:**
- 2x-3x LDS usage
- Additional address calculations
- May limit occupancy

**Configuration:**
```yaml
EnableLDSAlignment: True
LDSAlignmentBuffers: 3                # 2 for double-buffer, 3 for triple
LDSAlignmentTarget: 128               # alignment target (128 or 256)
LDSAlignmentKernelK: 64               # use MxNx64 instead of MxNx128
```

**When to use:**
- Severe stride misalignment
- Sufficient LDS available
- Need guaranteed alignment fix
- Performance is critical

### 3. Row Re-alignment

**Concept:** Group rows by alignment pattern to reduce VALU operations.

**How it works:**
1. Group rows by `(row × stride) % alignment_unit`
2. For odd strides: separate "even" and "odd" aligned rows
3. Apply alignment shift once per group instead of per row
4. Reduces VALU operations from 10-15 to 5-6

**Configuration:**
```yaml
EnableRowReAlignment: True
RowReAlignGroupSize: 16               # number of rows per group
```

**When to use:**
- Odd strides (stride % 2 == 1)
- Want to reduce VALU overhead
- Can combine with other techniques

## Files Modified/Added

### New Files:
1. `Tensile/Components/AlignmentUtils.py` - Core alignment utility classes
2. `test-configs/test-alignment-interleave.yaml` - Test config for address interleaving
3. `test-configs/test-alignment-lds.yaml` - Test config for LDS alignment
4. `ALIGNMENT_OPTIMIZATION_PLAN.md` - Detailed implementation plan
5. `ALIGNMENT_README.md` - This file

### Modified Files:
1. `Tensile/Common/GlobalParameters.py` - Added configuration parameters

### Files to be Modified (Next Steps):
1. `Tensile/KernelWriterAssembly.py` - Integrate address calculation
2. `Tensile/Components/LocalWrite.py` - LDS write with alignment
3. `Tensile/Components/LocalRead.py` - LDS read with alignment
4. `Tensile/Components/CustomSchedule.py` - Instruction scheduling
5. `Tensile/SolutionStructs/Solution.py` - Parameter validation

## Configuration Parameters

All parameters added to `defaultBenchmarkCommonParameters`:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `EnableAddressInterleave` | bool | False | Enable address interleaving |
| `AddressInterleaveDirection` | string | "N" | "M" or "N" - which dimension |
| `AddressInterleaveAlignment` | int | 256 | Target alignment (128/256 bytes) |
| `AddressInterleaveTileSize` | int | 16 | Tile size for interleaving |
| `EnableLDSAlignment` | bool | False | Enable LDS re-alignment |
| `LDSAlignmentBuffers` | int | 2 | Number of LDS buffers (2 or 3) |
| `LDSAlignmentTarget` | int | 128 | Alignment target (128/256 bytes) |
| `LDSAlignmentKernelK` | int | 64 | Kernel K dimension for alignment |
| `EnableRowReAlignment` | bool | False | Enable row grouping |
| `RowReAlignGroupSize` | int | 16 | Rows per alignment group |

## Usage Examples

### Example 1: Address Interleaving Only

```yaml
BenchmarkCommonParameters:
  - EnableAddressInterleave: [True]
  - AddressInterleaveDirection: ["N"]
  - AddressInterleaveAlignment: [256]
  - AddressInterleaveTileSize: [16]
  - EnableLDSAlignment: [False]
  - EnableRowReAlignment: [False]
```

### Example 2: LDS Alignment with Triple Buffering

```yaml
BenchmarkCommonParameters:
  - EnableLDSAlignment: [True]
  - LDSAlignmentBuffers: [3]
  - LDSAlignmentTarget: [128]
  - LDSAlignmentKernelK: [64]
  - EnableAddressInterleave: [False]
  - EnableRowReAlignment: [False]
```

### Example 3: Combined Approach

```yaml
BenchmarkCommonParameters:
  # Use both address interleaving and LDS alignment
  - EnableAddressInterleave: [True]
  - AddressInterleaveDirection: ["N"]
  - AddressInterleaveAlignment: [256]
  - AddressInterleaveTileSize: [16]
  
  - EnableLDSAlignment: [True]
  - LDSAlignmentBuffers: [2]
  - LDSAlignmentTarget: [128]
  - LDSAlignmentKernelK: [64]
  
  - EnableRowReAlignment: [True]
  - RowReAlignGroupSize: [16]
```

## Testing

### Quick Test

Test the original problem case:
```bash
cd tensilelite
python3 Tensile/Tensile.py test-configs/test-alignment-interleave.yaml
```

### Comprehensive Test

Test all techniques:
```bash
# Address interleaving
python3 Tensile/Tensile.py test-configs/test-alignment-interleave.yaml

# LDS alignment (2 and 3 buffers)
python3 Tensile/Tensile.py test-configs/test-alignment-lds.yaml
```

### Expected Results

For the problem case (K=1880, stride=3760 bytes):

**Baseline (no optimization):**
- K=1880: X TFLOPS
- K=1920: X × 1.25 TFLOPS (20-25% better)

**With optimization:**
- K=1880: ~X × 1.20-1.25 TFLOPS (close to aligned performance)
- Performance gap should be reduced to < 5%

### Performance Metrics to Monitor

1. **TFLOPS**: Main performance metric
2. **LDS Usage**: Should not exceed hardware limits
3. **VGPR Usage**: Watch for register pressure
4. **Occupancy**: May decrease with LDS alignment
5. **Memory Bandwidth**: Should improve with alignment

## Implementation Status

### Phase 1: Infrastructure ✅ COMPLETE
- [x] Add kernel parameters to GlobalParameters.py
- [x] Create AlignmentUtils.py with utility classes
- [x] Create test configuration files
- [x] Document implementation plan

### Phase 2: Address Interleaving ⏳ IN PROGRESS
- [ ] Integrate `AddressInterleaveCalculator` into `KernelWriterAssembly.py`
- [ ] Modify `graUnrollAssignment()` or similar for interleaved addressing
- [ ] Test with simple GEMM cases
- [ ] Validate performance improvement

### Phase 3: LDS Re-alignment ⏳ PENDING
- [ ] Integrate `LDSAlignmentManager` into local write/read
- [ ] Modify LDS size calculations
- [ ] Implement multi-buffer rotation
- [ ] Test 2-buffer vs 3-buffer performance

### Phase 4: Row Re-alignment ⏳ PENDING
- [ ] Integrate `RowReAlignmentHelper`
- [ ] Implement grouped shift operations
- [ ] Test on odd-stride cases

### Phase 5: Integration & Testing ⏳ PENDING
- [ ] Combine multiple techniques
- [ ] Performance benchmarking across various cases
- [ ] Edge case handling
- [ ] Documentation updates

## Design Decisions

### Why Multiple Techniques?

Different problems require different solutions:
- **Mild misalignment** (< 64 bytes): Address interleaving sufficient
- **Moderate misalignment** (64-128 bytes): Combination of techniques
- **Severe misalignment** (> 128 bytes): LDS re-alignment necessary
- **Odd strides**: Row re-alignment helps reduce overhead

### LDS Buffer Count: 2 vs 3

**Double Buffering (2 buffers):**
- Pros: Lower LDS usage, higher potential occupancy
- Cons: Less overlap opportunity

**Triple Buffering (3 buffers):**
- Pros: Better read/compute/write overlap
- Cons: 50% more LDS usage, may reduce occupancy

**Recommendation:** Try both and profile.  Generally:
- 2 buffers for smaller kernels or LDS-limited configurations
- 3 buffers for larger problems with good LDS availability

### Target Alignment: 128 vs 256

**128 bytes:**
- Pros: Easier to achieve, works for most GPUs
- Cons: May not be optimal for all architectures

**256 bytes:**
- Pros: Full cacheline alignment, better performance
- Cons: Harder to achieve, may require padding

**Recommendation:** Start with 128, upgrade to 256 if LDS permits.

## Debugging

### Enable Debug Output

Set in test YAML:
```yaml
GlobalParameters:
  LibraryPrintDebug: True
  PrintLevel: 2
```

### Check Generated Assembly

Look for alignment-related code in generated `.s` files:
```bash
grep -n "ComputeInterleaved\|AlignedKStart\|LDSOffset" generated_kernel.s
```

### Common Issues

1. **LDS Overflow**
   - Symptom: Kernel launch failure or incorrect results
   - Solution: Reduce `LDSAlignmentBuffers` or `LDSAlignmentKernelK`

2. **Register Pressure**
   - Symptom: Lower occupancy, reduced performance
   - Solution: Reduce temporary VGPR usage in alignment calculations

3. **Wrong Direction**
   - Symptom: No performance improvement
   - Solution: Check `AddressInterleaveDirection` matches K-major matrix

## Future Enhancements

1. **Auto-detection**: Automatically enable alignment based on stride analysis
2. **Hybrid Buffering**: Mix 2 and 3 buffers for different parts of kernel
3. **Architecture-specific Tuning**: Different parameters for different GPU architectures
4. **Runtime Padding**: Automatic padding at library level for known bad cases

## References

- Original issue description (user-provided problem statement)
- AMD GPU Architecture documentation
- ROCm/hipBLAS performance optimization guides
- Tensile code generator documentation

## Contact

For questions or issues:
- Check `ALIGNMENT_OPTIMIZATION_PLAN.md` for detailed implementation
- Review generated assembly for debugging
- Consult Tensile documentation for kernel writer details

## License

Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

See individual source files for full license text (MIT License).

