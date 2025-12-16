# Memory Alignment Optimization Implementation Plan

## Problem Statement
20-25% performance drop when matrix stride is not aligned to cacheline boundaries.
- Example: GEMM 2048x3072x1880 TN with BFP16, stride=3760 bytes
- Padding K from 1880 to 1920 fixes the issue

## Proposed Solutions

### Solution 1: Address Interleaving (Single-Direction Alignment)
**Target**: Align memory accesses for one dimension using address remapping

**Implementation Steps**:

1. **Add New Kernel Parameters** (`Solution.py`)
   ```python
   # Add to defaultSolution or defaultInternalSupportParams
   "EnableAddressInterleave": False,          # Enable address interleaving optimization
   "AddressInterleaveDirection": "N",          # 'M' or 'N' - which direction to interleave
   "AddressInterleaveAlignment": 256,          # Target alignment in bytes (128 or 256)
   "AddressInterleaveTileSize": 16,            # Number of kernels for interleaving
   ```

2. **Modify Global Read Address Calculation** (`KernelWriterAssembly.py`)
   - Add method `computeInterleavedAddress()` to remap row indices
   - Modify `graUnrollAssignment()` or `graTileAssignment()` to use interleaved addressing
   
   ```python
   def computeInterleavedAddress(self, kernel, tP):
       """
       Compute interleaved address for better alignment.
       Instead of J = Rn + Kn * Sn, use J = Rn * 16 + Kn
       """
       if not kernel["EnableAddressInterleave"]:
           return None
       
       tc = tP["tensorChar"]
       tileSize = kernel["AddressInterleaveTileSize"]
       
       # Generate code to compute interleaved indices
       module = Module("AddressInterleave%s" % tc)
       # ... implementation
       return module
   ```

3. **Calculate Aligned Starting K** 
   ```python
   def computeAlignedKStart(self, kernel, tP, rowVgpr):
       """
       Calculate K start position: K = ceil((J * stride) / alignment) * alignment
       """
       alignment = kernel["AddressInterleaveAlignment"]
       # Generate code to compute aligned K start
       # Use vgpr math to calculate aligned offset
   ```

4. **Implement Wrapped K Iteration**
   - Modify loop unrolling code to handle K wrapping
   - Start from aligned K, iterate to end, then wrap to beginning

### Solution 2: LDS Re-alignment with Multi-buffering
**Target**: Fix alignment issues using Local Data Share (LDS) with buffer management

**Implementation Steps**:

1. **Add New Kernel Parameters** (`Solution.py`)
   ```python
   "EnableLDSAlignment": False,                # Enable LDS re-alignment
   "LDSAlignmentBuffers": 3,                   # 2 or 3 buffers
   "LDSAlignmentTarget": 128,                  # Target alignment: 128 or 256 bytes
   "LDSAlignmentKernelK": 64,                  # Use MxNx64 instead of MxNx128
   ```

2. **Modify LDS Size Calculation** (`Solution.py` in `assignDerivedParameters()`)
   ```python
   def calculateLDSWithAlignment(self, kernel):
       """
       Calculate LDS size considering alignment buffers.
       """
       numBuffers = kernel["LDSAlignmentBuffers"] if kernel["EnableLDSAlignment"] else kernel["ExpandPointerSwap"]
       kernelK = kernel["LDSAlignmentKernelK"] if kernel["EnableLDSAlignment"] else kernel["DepthU"]
       
       # Calculate per-row LDS requirements
       ldsPerRow = kernelK * numBuffers
       # ... rest of calculation
   ```

3. **Add Per-Row Offset Tracking** (`KernelWriterAssembly.py`)
   ```python
   def initializeLDSOffsets(self, kernel, tP):
       """
       Initialize per-row LDS offsets for alignment.
       lds_offset[row] = stride * row - read_pointer[row]
       """
       module = Module("InitLDSOffsets%s" % tP["tensorChar"])
       
       # Calculate read_pointer[row] = (stride * row) & ~(KERNEL_K - 1)
       # Calculate lds_offset[row] = stride * row - read_pointer[row]
       
       return module
   ```

4. **Modify Local Write Code** (`LocalWrite.py`)
   ```python
   def localWriteWithAlignment(self, writer, kernel, tP):
       """
       Write to LDS with proper offset for alignment.
       lds_addr(row) = lds_base + KERNEL_K * NUM_BUFFERS * row + 
                       (lds_offset[row] + lane_index + KERNEL_K * iter) % (KERNEL_K * NUM_BUFFERS)
       """
       # Implement wrapped LDS addressing
       # Handle buffer rotation
   ```

5. **Modify Local Read Code** (`LocalRead.py`)
   ```python
   def localReadWithAlignment(self, writer, kernel, bufferIdx, iui, tP):
       """
       Read from LDS with alignment-aware addressing.
       """
       if not kernel["EnableLDSAlignment"]:
           return self.originalLocalRead(writer, kernel, bufferIdx, iui, tP)
       
       # Calculate wrapped LDS address for current buffer
       # Apply per-row offset
   ```

6. **Implement Triple Buffer Management** (`CustomSchedule.py`)
   ```python
   def scheduleAlignedLDSAccess(self, kernel):
       """
       Schedule LDS access with triple buffering.
       Ensure entire buffer is consumed before issuing new reads.
       """
       # Modify instruction scheduling
       # Add appropriate s_waitcnt for LDS completion
   ```

### Solution 3: Row Re-alignment within Kernel
**Target**: Group rows by alignment pattern to optimize VALU operations

**Implementation Steps**:

1. **Add New Kernel Parameters** (`Solution.py`)
   ```python
   "EnableRowReAlignment": False,              # Enable row re-alignment
   "RowReAlignGroupSize": 16,                  # Number of rows to group
   ```

2. **Add Row Sorting Logic** (`KernelWriterAssembly.py`)
   ```python
   def sortRowsByAlignment(self, kernel, tP, numRows):
       """
       Sort rows into groups based on alignment pattern.
       For stride % 2 == 1, group even and odd aligned rows.
       """
       module = Module("SortRows%s" % tP["tensorChar"])
       
       # Group rows by (stride * row) % alignment_unit
       # Generate permutation indices
       
       return module
   ```

3. **Apply Alignment Shift During Load** (`GlobalRead.py`)
   ```python
   def applyAlignmentShift(self, kernel, tP, dataVgpr):
       """
       Shift data by alignment offset for grouped rows.
       Reduces VALU operations from 10-15 to 5-6.
       """
       if not kernel["EnableRowReAlignment"]:
           return Module()
       
       module = Module("AlignmentShift%s" % tP["tensorChar"])
       
       # Apply v_alignbit_b32 or similar to shift data
       # Only needed once per group instead of per row
       
       return module
   ```

## Implementation Priority

### Phase 1: Infrastructure (Week 1-2)
1. Add kernel parameters to `Solution.py`
2. Create utility functions for alignment calculations
3. Add configuration options to YAML files

### Phase 2: Address Interleaving (Week 2-3)
1. Implement `computeInterleavedAddress()`
2. Modify global read address calculation
3. Test with simple GEMM cases

### Phase 3: LDS Re-alignment (Week 3-5)
1. Implement LDS offset calculation
2. Modify local write/read with alignment
3. Implement multi-buffer management
4. Test and tune buffer count (2 vs 3)

### Phase 4: Row Re-alignment (Week 5-6)
1. Implement row sorting logic
2. Add grouped shift operations
3. Integrate with existing global read

### Phase 5: Integration & Testing (Week 6-8)
1. Combine multiple techniques
2. Performance benchmarking
3. Edge case handling
4. Documentation

## Testing Strategy

### Test Cases
1. **Aligned Strides**: Verify no regression
   - Stride = 1024, 2048, 4096 (multiples of cacheline)

2. **Misaligned Strides**: Verify improvement
   - Stride = 3760 (original problem case)
   - Stride = 1880, 2560, 3760

3. **Different Geometries**:
   - Square matrices: 2048x2048xK
   - Rectangular: 2048x3072xK
   - Various K values: 1880, 1920, 2000

4. **Different Data Types**:
   - BFP16, FP16, FP32
   - Mixed precision

### Performance Metrics
- Measure TFLOPS before/after
- Measure cache hit rates (if available)
- Measure memory bandwidth utilization
- Verify 20-25% improvement for misaligned cases

## Configuration Files

Create test configuration YAML files:

### `test-alignment-interleave.yaml`
```yaml
GlobalParameters:
  MinimumRequiredVersion: 4.4.0

BenchmarkProblems:
  - # TN GEMM with alignment optimization
    - ProblemType:
        OperationType: GEMM
        DataType: b
        TransposeA: True
        TransposeB: False
    Solutions:
      - Solution:
          EnableAddressInterleave: True
          AddressInterleaveDirection: N
          AddressInterleaveAlignment: 256
          AddressInterleaveTileSize: 16
```

### `test-alignment-lds.yaml`
```yaml
BenchmarkProblems:
  - Solutions:
      - Solution:
          EnableLDSAlignment: True
          LDSAlignmentBuffers: 3
          LDSAlignmentTarget: 128
          LDSAlignmentKernelK: 64
```

## Code Locations

### Key Files to Modify:
1. `tensilelite/Tensile/SolutionStructs/Solution.py` - Add parameters
2. `tensilelite/Tensile/KernelWriterAssembly.py` - Address calculation
3. `tensilelite/Tensile/Components/LocalWrite.py` - LDS writes
4. `tensilelite/Tensile/Components/LocalRead.py` - LDS reads
5. `tensilelite/Tensile/Components/CustomSchedule.py` - Instruction scheduling
6. `tensilelite/Tensile/Components/GlobalRead.py` (if exists) - Global reads

## Next Steps

1. Review this plan with team
2. Prioritize which solution to implement first
3. Create feature branch for development
4. Start with Phase 1 infrastructure
5. Incremental testing after each phase

## Notes
- Consider backward compatibility - make all features optional
- Add comprehensive error checking for invalid configurations
- Document performance characteristics in comments
- Consider interaction with existing features (DirectToLDS, etc.)

