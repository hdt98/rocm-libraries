# Layout Pruning - PR Analysis and Implementation Plan

## Executive Summary

This document analyzes two related PRs for splitting the Composable Kernel convolution library by data layout and provides a comprehensive implementation plan for the rocm-libraries repository.

- **Total Convolution Files**: 727 .cpp files
- **Target**: Split monolithic libraries into layout-specific libraries
- **Primary Layouts**: GNHWC, NHWGC, NGCHW, GNDHWC, NDHWGC, NGCDHW (for 2D/3D), GNWC, NWGC (for 1D)

---

## PR #3010 Analysis (composable_kernel - CLOSED)

**Title**: Split convolution library by data layout
**Status**: Closed (December 17, 2025) - Superseded by ck_builder approach
**Purpose**: Modularize convolution operations for independent building and linking

### Key Changes

The PR decomposed the monolithic `device_conv_operations` library into **nine specialized libraries**:

1. **`device_conv_gnwc_operations`** - 1D grouped convolutions with GNWC layout
2. **`device_conv_nwgc_operations`** - 1D grouped convolutions with NWGC layout
3. **`device_conv_nhwgc_operations`** - 2D grouped convolutions with NHWGC layout
4. **`device_conv_ngchw_operations`** - 2D grouped convolutions with NGCHW layout
5. **`device_conv_gnhwc_operations`** - 2D grouped convolutions with GNHWC layout
6. **`device_conv_gndhwc_operations`** - 3D grouped convolutions with GNDHWC layout
7. **`device_conv_ndhwgc_operations`** - 3D grouped convolutions with NDHWGC layout
8. **`device_conv_ngcdhw_operations`** - 3D grouped convolutions with NGCDHW layout
9. **`device_conv_old_operations`** - Non-grouped convolutions and backward-weight instances

### Pattern Applied

- **Directory reorganization**: Source files moved into layout-specific subdirectories
- **CMake modularization**: Separate library targets per layout
- **Backward compatibility**: Umbrella `device_conv_operations` includes sub-libraries per GPU arch

### Why It Was Closed

The PR author stated that the approach became unnecessary due to planned migration toward `ck_builder` integration with MIOpen instead of static library consumption.

---

## PR #2099 Analysis (rocm-libraries - CLOSED)

**Title**: [MIOpen] Use specific CK libs
**Status**: Closed (January 10, 2026) - Blocked by upstream PR rejection
**Purpose**: Optimize MIOpen build times by using subset of CK libraries

### MIOpen's Required Libraries

The PR identified that MIOpen only needs **three specialized libraries**:

1. **`device_conv_ndhwgc_operations`** - 3D convolutions, NDHWGC layout
2. **`device_conv_nhwgc_operations`** - 2D convolutions, NHWGC layout
3. **`device_conv_old_operations`** - Legacy non-grouped convolutions

**Replacement Pattern**: Changed from linking monolithic `device_conv_operations` to the three specific libraries above.

### Commits

1. `6dddbba` - "[MIOpen] Use just the CK libraries that MIOpen needs"
2. `4a7dbc4` - "TMP: update CK to match Illia's PR" (temporary)
3. `3443b5e` - Resurrection attempt (March 2026) by brockhargreaves-amd

### Why It Was Closed

Blocked because the prerequisite composable_kernel PR #3010 was rejected upstream.

---

## Current Repository State

### File Distribution by Operation Type

```
Operation Category                      | File Count
----------------------------------------|-----------
grouped_conv2d_fwd                      | 400
grouped_conv3d_fwd (all variants)       | 452
grouped_conv2d_bwd_weight               | 132
grouped_conv2d_bwd_data                 | 78
grouped_conv3d_bwd_weight (all variants)| 154
grouped_conv3d_bwd_data (all variants)  | 104
grouped_conv1d_bwd_weight               | 18
grouped_conv1d_fwd                      | 8
conv2d_* (non-grouped)                  | 13
conv3d_bwd_data                         | 4
conv1d_bwd_data                         | 4
quantization/conv2d_fwd                 | 8
----------------------------------------|-----------
TOTAL                                   | 727
```

### Layout Patterns in Filenames

The analysis shows layouts are encoded in filenames following these patterns:

**For 2D Convolutions (H=height, W=width, C=channel, G=group):**
- `nhwgc` - NHWGC layout (N, Height, Width, Group, Channel)
- `gnhwc` - GNHWC layout (Group, N, Height, Width, Channel)
- `ngchw` - NGCHW layout (N, Group, Channel, Height, Width)

**For 3D Convolutions (D=depth added):**
- `ndhwgc` - NDHWGC layout
- `gndhwc` - GNDHWC layout
- `ngcdhw` - NGCDHW layout

**For 1D Convolutions (W=width):**
- `nwgc` - NWGC layout
- `gnwc` - GNWC layout

### Existing Directory Structure

Some operations already have layout-based subdirectories:

```
grouped_conv2d_fwd/
├── dl/          # Direct link method instances
├── gnhwc/       # GNHWC layout instances (NEW - untracked)
├── ngchw/       # NGCHW layout instances (NEW - untracked)
├── nhwgc/       # NHWGC layout instances (NEW - untracked)
├── wmma/        # Wave matrix multiply accumulate
└── xdl/         # XDL (matrix core) method instances
    ├── comp/
    ├── large_tensor/
    ├── mem/
    └── merged_groups/
```

**Note**: The git status shows `gnhwc/`, `ngchw/`, and `nhwgc/` as untracked - suggesting layout split work has begun.

---

## Implementation Plan

### Phase 1: File Categorization and Inventory ✅ COMPLETE

**Status**: Complete - 727 files cataloged in `all_convolution_files.txt`

### Phase 2: Layout Extraction and Mapping

**Objective**: Create comprehensive mapping of each file to its target layout library

**Tasks**:
1. Parse all 727 filenames to extract layout information
2. Create per-layout file manifests
3. Identify files requiring special handling (multi-layout, old/legacy)
4. Generate coverage report

**Deliverables**:
- `layout_mapping.json` - Machine-readable layout assignments
- `LAYOUT_FILE_MANIFEST.md` - Human-readable breakdown per layout
- `coverage_report.txt` - Verification that all 727 files are assigned

### Phase 3: Directory Reorganization

**Objective**: Move source files into layout-specific subdirectories

**Pattern** (following PR #3010 approach):
```
grouped_conv{1,2,3}d_{fwd,bwd_data,bwd_weight}/
├── gnhwc/     # or gnwc for 1D, gndhwc for 3D
├── nhwgc/     # or nwgc for 1D, ndhwgc for 3D
├── ngchw/     # or ngcdhw for 3D
├── old/       # Legacy non-grouped, backward compatibility
└── [existing subdirs like wmma/, xdl/, dl/]
```

**Execution Strategy**:
- Use git mv to preserve history
- Process operation types in order (conv1d → conv2d → conv3d)
- Within each type: fwd → bwd_data → bwd_weight
- Verify builds after each major category

### Phase 4: CMake Library Splitting

**Objective**: Create separate CMake library targets per layout

**New Library Targets** (following PR #3010 naming):

**1D Convolutions:**
- `device_conv1d_gnwc_operations`
- `device_conv1d_nwgc_operations`

**2D Convolutions:**
- `device_conv2d_gnhwc_operations`
- `device_conv2d_nhwgc_operations`
- `device_conv2d_ngchw_operations`

**3D Convolutions:**
- `device_conv3d_gndhwc_operations`
- `device_conv3d_ndhwgc_operations`
- `device_conv3d_ngcdhw_operations`

**Legacy:**
- `device_conv_old_operations` (non-grouped convs, special cases)

**Backward Compatibility:**
- Keep umbrella `device_conv_operations` target that includes appropriate sub-libraries per GPU arch

### Phase 5: Build System Integration

**CMakeLists.txt changes**:

1. Define layout-specific source lists via GLOB or explicit listing
2. Create conditional library targets based on GPU architecture
3. Link sub-libraries into umbrella target
4. Update installation rules

**Example Pattern**:
```cmake
# Layout-specific libraries
add_library(device_conv2d_nhwgc_operations
    grouped_conv2d_fwd/nhwgc/*.cpp
    grouped_conv2d_bwd_data/nhwgc/*.cpp
    grouped_conv2d_bwd_weight/nhwgc/*.cpp
)

# Umbrella library
add_library(device_conv_operations)
target_link_libraries(device_conv_operations
    device_conv2d_nhwgc_operations
    device_conv2d_gnhwc_operations
    device_conv2d_ngchw_operations
    # ... etc
)
```

### Phase 6: MIOpen Integration (PR #2099 resurrection)

**Objective**: Update MIOpen to link only required CK libraries

**Changes in MIOpen's CMakeLists.txt**:

```cmake
# OLD:
target_link_libraries(MIOpen ... device_conv_operations ...)

# NEW:
target_link_libraries(MIOpen
    device_conv2d_nhwgc_operations
    device_conv3d_ndhwgc_operations
    device_conv_old_operations
)
```

**Benefits**:
- Reduced link time
- Smaller binary size
- Faster incremental builds

### Phase 7: Verification and Testing

**Build Verification**:
1. Full clean build with all layouts
2. Selective builds (single layout)
3. MIOpen integration build
4. Cross-GPU architecture validation (gfx908, gfx90a, gfx940, gfx941, gfx942, etc.)

**Test Coverage**:
1. Unit tests for each layout library
2. Integration tests with MIOpen
3. Performance regression tests
4. Binary size measurements

---

## Coverage Verification

### All 727 Files Must Be Assigned

**Breakdown by Target Library** (to be completed in Phase 2):

| Library Target | Expected File Count | Status |
|----------------|---------------------|--------|
| device_conv1d_gnwc_operations | TBD | Pending |
| device_conv1d_nwgc_operations | TBD | Pending |
| device_conv2d_gnhwc_operations | TBD | Pending |
| device_conv2d_nhwgc_operations | TBD | Pending |
| device_conv2d_ngchw_operations | TBD | Pending |
| device_conv3d_gndhwc_operations | TBD | Pending |
| device_conv3d_ndhwgc_operations | TBD | Pending |
| device_conv3d_ngcdhw_operations | TBD | Pending |
| device_conv_old_operations | TBD | Pending |
| **TOTAL** | **727** | **Must Match** |

---

## Risk Analysis and Mitigation

### Risk 1: Incomplete Layout Coverage

**Risk**: Some files may contain multiple layouts or unclear layout patterns
**Mitigation**: Manual review of ambiguous files, create explicit mapping manifest

### Risk 2: Build Time Regression

**Risk**: Additional library targets may slow build despite modularization
**Mitigation**: Benchmark build times before/after, optimize CMake parallelization

### Risk 3: Breaking Downstream Consumers

**Risk**: Other projects besides MIOpen may depend on monolithic library
**Mitigation**: Keep umbrella `device_conv_operations` for backward compatibility

### Risk 4: Git History Loss

**Risk**: Moving 727 files could fragment git history
**Mitigation**: Use `git mv` for all moves, document mapping in this file

---

## Success Criteria

1. ✅ All 727 convolution files cataloged
2. ⏳ All files assigned to layout-specific libraries (coverage = 100%)
3. ⏳ Clean build succeeds with split libraries
4. ⏳ MIOpen builds with only 3 required libraries
5. ⏳ Build time reduced by >20% for MIOpen
6. ⏳ All tests pass (unit, integration, performance)
7. ⏳ Documentation updated

---

## Next Steps

1. **Phase 2**: Run layout extraction script to populate `layout_mapping.json`
2. **Review**: Validate layout assignments for all 727 files
3. **Approval**: Get stakeholder sign-off on reorganization plan
4. **Execute**: Begin Phase 3 directory reorganization
5. **Iterate**: Phase 4-7 implementation with continuous validation

---

## References

- Upstream PR composable_kernel#3010: https://github.com/ROCm/composable_kernel/pull/3010
- Downstream PR rocm-libraries#2099: https://github.com/ROCm/rocm-libraries/pull/2099
- File inventory: `all_convolution_files.txt` (727 files)
- Working directory: `/home/AMD/bhargrea/github/rocm-libraries/projects/composablekernel/layout_pruning/`

---

**Document Version**: 1.0
**Last Updated**: 2026-03-25
**Author**: Analysis based on PR #3010 and PR #2099
