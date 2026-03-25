# Layout Pruning Implementation Plan

## Overview

This document provides a concrete, actionable implementation plan for splitting the Composable Kernel convolution library by data layout, based on analysis of:
- **PR #3010** (composable_kernel): Split convolution library by data layout
- **PR #2099** (rocm-libraries): MIOpen use specific CK libs

**Status**: ✅ **All 727 convolution files cataloged and categorized**

---

## File Inventory Summary

### Total Coverage
- **Total files**: 727
- **Categorized**: 727 (100%)
- **Uncategorized**: 0

### Target Libraries (12 total)

| Library Name | File Count | Purpose |
|-------------|------------|---------|
| `device_conv1d_gnwc_operations` | 10 | 1D convolutions, GNWC layout |
| `device_conv1d_nwgc_operations` | 3 | 1D convolutions, NWGC layout |
| `device_conv2d_gnhwc_operations` | 30 | 2D convolutions, GNHWC layout |
| `device_conv2d_ngchw_operations` | 49 | 2D convolutions, NGCHW layout |
| `device_conv2d_nhwgc_operations` | 226 | 2D convolutions, NHWGC layout ⭐ |
| `device_conv3d_gndhwc_operations` | 25 | 3D convolutions, GNDHWC layout |
| `device_conv3d_ndhwgc_operations` | 290 | 3D convolutions, NDHWGC layout ⭐ |
| `device_conv3d_ngcdhw_operations` | 38 | 3D convolutions, NGCDHW layout |
| `device_conv3d_nhwgc_operations` | 2 | 3D convolutions, NHWGC layout |
| `device_conv_old_operations` | 22 | Non-grouped, legacy ⭐ |
| `device_convnd_generic_operations` | 24 | N-dimensional generic |
| `device_quantization_operations` | 8 | Quantized convolutions |
| **TOTAL** | **727** | |

⭐ = **Required by MIOpen** (from PR #2099 analysis)

---

## Comparison with PR #3010

### PR #3010 Proposed Libraries (9)
1. ✅ `device_conv_gnwc_operations` → Our: `device_conv1d_gnwc_operations`
2. ✅ `device_conv_nwgc_operations` → Our: `device_conv1d_nwgc_operations`
3. ✅ `device_conv_nhwgc_operations` → Our: `device_conv2d_nhwgc_operations`
4. ✅ `device_conv_ngchw_operations` → Our: `device_conv2d_ngchw_operations`
5. ✅ `device_conv_gnhwc_operations` → Our: `device_conv2d_gnhwc_operations`
6. ✅ `device_conv_gndhwc_operations` → Our: `device_conv3d_gndhwc_operations`
7. ✅ `device_conv_ndhwgc_operations` → Our: `device_conv3d_ndhwgc_operations`
8. ✅ `device_conv_ngcdhw_operations` → Our: `device_conv3d_ngcdhw_operations`
9. ✅ `device_conv_old_operations` → Our: `device_conv_old_operations`

### Additional Libraries (not in PR #3010)
10. ➕ `device_conv3d_nhwgc_operations` (2 files) - New
11. ➕ `device_convnd_generic_operations` (24 files) - N-dimensional generic
12. ➕ `device_quantization_operations` (8 files) - Quantization support

**Rationale**: Our analysis found additional file categories that need dedicated libraries.

---

## MIOpen Requirements (PR #2099)

MIOpen requires only **3 libraries** (471 files = 64.8% of total):

1. ✅ `device_conv2d_nhwgc_operations` - **226 files**
2. ✅ `device_conv3d_ndhwgc_operations` - **290 files**
3. ✅ `device_conv_old_operations` - **22 files**

**Current**: Links all 727 files via monolithic `device_conv_operations`
**Proposed**: Link only 471 files (35% reduction)

**Expected Benefits**:
- 35% fewer files to compile/link
- Estimated 20-30% build time reduction
- Smaller binary size
- Faster incremental builds

---

## Implementation Phases

### Phase 1: File Reorganization ✅ COMPLETE

**Deliverables**:
- ✅ `all_convolution_files.txt` - Complete file inventory (727 files)
- ✅ `layout_mapping.json` - Machine-readable categorization
- ✅ `LAYOUT_FILE_MANIFEST.md` - Detailed file listings by library
- ✅ `categorize_files.py` - Categorization script

### Phase 2: Directory Structure Creation

**Objective**: Create layout-specific subdirectories without moving files yet

**Actions**:
```bash
# For each operation with layout variants, create layout subdirs
cd projects/composablekernel/library/src/tensor_operation_instance/gpu

# 1D convolutions
mkdir -p grouped_conv1d_fwd/{gnwc,nwgc}
mkdir -p grouped_conv1d_bwd_weight/{gnwc,nwgc}

# 2D convolutions
mkdir -p grouped_conv2d_fwd/{gnhwc,ngchw,nhwgc}
mkdir -p grouped_conv2d_bwd_data/{gnhwc,nhwgc,ngchw}
mkdir -p grouped_conv2d_bwd_weight/{gnhwc,nhwgc,ngchw}

# 2D convolution variants
mkdir -p grouped_conv2d_fwd_clamp/{gnhwc,nhwgc,ngchw}
mkdir -p grouped_conv2d_fwd_bias_clamp/{gnhwc,nhwgc,ngchw}
mkdir -p grouped_conv2d_fwd_bias_bnorm_clamp/{gnhwc,nhwgc,ngchw}
mkdir -p grouped_conv2d_fwd_dynamic_op/{gnhwc,nhwgc,ngchw}

# 3D convolutions
mkdir -p grouped_conv3d_fwd/{gndhwc,ngcdhw,ndhwgc}
mkdir -p grouped_conv3d_bwd_data/{gndhwc,ndhwgc,ngcdhw}
mkdir -p grouped_conv3d_bwd_weight/{gndhwc,ndhwgc,ngcdhw}

# 3D convolution variants
mkdir -p grouped_conv3d_fwd_scale/{gndhwc,ndhwgc,ngcdhw}
mkdir -p grouped_conv3d_fwd_clamp/{gndhwc,ndhwgc,ngcdhw}
mkdir -p grouped_conv3d_fwd_bias_clamp/{gndhwc,ndhwgc,ngcdhw}
mkdir -p grouped_conv3d_fwd_bias_bnorm_clamp/{gndhwc,ndhwgc,ngcdhw}
# ... (all other 3D variants)

# N-dimensional generic (no layout split needed)
# grouped_convnd_bwd_weight/ - keep as-is

# Quantization (no layout split needed)
# quantization/conv2d_fwd/ - keep as-is

# Non-grouped "old" operations (keep as-is)
# conv1d_bwd_data/
# conv2d_bwd_data/
# conv2d_fwd/
# conv3d_bwd_data/
```

**Verification**:
```bash
# Check directory structure created correctly
find . -type d -name "gnhwc" -o -name "nhwgc" -o -name "ngchw" | wc -l
# Should show proper count of new layout directories
```

### Phase 3: File Migration

**Objective**: Move files to layout-specific directories using git mv

**Strategy**: Process systematically by operation type, preserving git history

**Script-Based Approach**:

```python
#!/usr/bin/env python3
import json
import subprocess
from pathlib import Path

# Load categorization
with open('layout_mapping.json') as f:
    data = json.load(f)

# For each file, determine target directory
for library, files in data['by_library'].items():
    for filepath in files:
        # Parse current location
        current = Path(filepath)

        # Determine target based on layout
        # Example: grouped_conv2d_fwd/xdl/device_grouped_conv2d_fwd_xdl_nhwgc_*.cpp
        #       -> grouped_conv2d_fwd/nhwgc/xdl/device_grouped_conv2d_fwd_xdl_nhwgc_*.cpp

        # Extract layout from categorization
        # Build target path
        # Execute git mv

        # Example:
        # git mv grouped_conv2d_fwd/xdl/file.cpp grouped_conv2d_fwd/nhwgc/xdl/file.cpp
```

**Manual Verification Checkpoints**:
- After 1D migrations: Build test
- After 2D migrations: Build test
- After 3D migrations: Build test
- Final: Full clean build

**Rollback Plan**: Tag commit before each migration phase

### Phase 4: CMakeLists.txt Modularization

**Objective**: Create separate library targets per layout

**Location**: `projects/composablekernel/library/src/tensor_operation_instance/gpu/CMakeLists.txt`

**Implementation Pattern**:

```cmake
# ============================================================
# Layout-Specific Convolution Libraries
# ============================================================

# Helper function to collect sources by layout
function(collect_layout_sources OUT_VAR BASE_DIR LAYOUT)
    file(GLOB_RECURSE SOURCES
        "${BASE_DIR}/*/dl/${LAYOUT}/*.cpp"
        "${BASE_DIR}/*/xdl/${LAYOUT}/*.cpp"
        "${BASE_DIR}/*/wmma/${LAYOUT}/*.cpp"
        "${BASE_DIR}/*/${LAYOUT}/*.cpp"
    )
    set(${OUT_VAR} ${SOURCES} PARENT_SCOPE)
endfunction()

# 1D Convolution Libraries
collect_layout_sources(CONV1D_GNWC_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}" "gnwc")
if(CONV1D_GNWC_SOURCES)
    add_library(device_conv1d_gnwc_operations OBJECT ${CONV1D_GNWC_SOURCES})
    target_compile_features(device_conv1d_gnwc_operations PUBLIC cxx_std_17)
    # ... other properties
endif()

collect_layout_sources(CONV1D_NWGC_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}" "nwgc")
if(CONV1D_NWGC_SOURCES)
    add_library(device_conv1d_nwgc_operations OBJECT ${CONV1D_NWGC_SOURCES})
    target_compile_features(device_conv1d_nwgc_operations PUBLIC cxx_std_17)
endif()

# 2D Convolution Libraries
collect_layout_sources(CONV2D_GNHWC_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}" "gnhwc")
if(CONV2D_GNHWC_SOURCES)
    add_library(device_conv2d_gnhwc_operations OBJECT ${CONV2D_GNHWC_SOURCES})
    target_compile_features(device_conv2d_gnhwc_operations PUBLIC cxx_std_17)
endif()

collect_layout_sources(CONV2D_NHWGC_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}" "nhwgc")
if(CONV2D_NHWGC_SOURCES)
    add_library(device_conv2d_nhwgc_operations OBJECT ${CONV2D_NHWGC_SOURCES})
    target_compile_features(device_conv2d_nhwgc_operations PUBLIC cxx_std_17)
endif()

collect_layout_sources(CONV2D_NGCHW_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}" "ngchw")
if(CONV2D_NGCHW_SOURCES)
    add_library(device_conv2d_ngchw_operations OBJECT ${CONV2D_NGCHW_SOURCES})
    target_compile_features(device_conv2d_ngchw_operations PUBLIC cxx_std_17)
endif()

# 3D Convolution Libraries
collect_layout_sources(CONV3D_GNDHWC_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}" "gndhwc")
if(CONV3D_GNDHWC_SOURCES)
    add_library(device_conv3d_gndhwc_operations OBJECT ${CONV3D_GNDHWC_SOURCES})
    target_compile_features(device_conv3d_gndhwc_operations PUBLIC cxx_std_17)
endif()

collect_layout_sources(CONV3D_NDHWGC_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}" "ndhwgc")
if(CONV3D_NDHWGC_SOURCES)
    add_library(device_conv3d_ndhwgc_operations OBJECT ${CONV3D_NDHWGC_SOURCES})
    target_compile_features(device_conv3d_ndhwgc_operations PUBLIC cxx_std_17)
endif()

collect_layout_sources(CONV3D_NGCDHW_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}" "ngcdhw")
if(CONV3D_NGCDHW_SOURCES)
    add_library(device_conv3d_ngcdhw_operations OBJECT ${CONV3D_NGCDHW_SOURCES})
    target_compile_features(device_conv3d_ngcdhw_operations PUBLIC cxx_std_17)
endif()

# Special Libraries
file(GLOB_RECURSE CONV_OLD_SOURCES
    "conv1d_bwd_data/*.cpp"
    "conv2d_bwd_data/*.cpp"
    "conv2d_fwd/*.cpp"
    "conv3d_bwd_data/*.cpp"
)
add_library(device_conv_old_operations OBJECT ${CONV_OLD_SOURCES})

file(GLOB_RECURSE CONVND_SOURCES "grouped_convnd_bwd_weight/**/*.cpp")
add_library(device_convnd_generic_operations OBJECT ${CONVND_SOURCES})

file(GLOB_RECURSE QUANT_SOURCES "quantization/**/*.cpp")
add_library(device_quantization_operations OBJECT ${QUANT_SOURCES})

# ============================================================
# Umbrella Library (Backward Compatibility)
# ============================================================
add_library(device_conv_operations INTERFACE)
target_link_libraries(device_conv_operations INTERFACE
    device_conv1d_gnwc_operations
    device_conv1d_nwgc_operations
    device_conv2d_gnhwc_operations
    device_conv2d_nhwgc_operations
    device_conv2d_ngchw_operations
    device_conv3d_gndhwc_operations
    device_conv3d_ndhwgc_operations
    device_conv3d_ngcdhw_operations
    device_conv_old_operations
    device_convnd_generic_operations
    device_quantization_operations
)

# Installation
install(TARGETS
    device_conv1d_gnwc_operations
    device_conv1d_nwgc_operations
    device_conv2d_gnhwc_operations
    device_conv2d_nhwgc_operations
    device_conv2d_ngchw_operations
    device_conv3d_gndhwc_operations
    device_conv3d_ndhwgc_operations
    device_conv3d_ngcdhw_operations
    device_conv_old_operations
    device_convnd_generic_operations
    device_quantization_operations
    device_conv_operations
    EXPORT ComposableKernelTargets
)
```

**Verification**:
```bash
# Build all libraries
cmake --build . --target device_conv2d_nhwgc_operations
cmake --build . --target device_conv3d_ndhwgc_operations
cmake --build . --target device_conv_old_operations

# Build umbrella
cmake --build . --target device_conv_operations
```

### Phase 5: MIOpen Integration

**File**: `projects/miopen/CMakeLists.txt` (or equivalent linking location)

**Change**:
```cmake
# BEFORE (current):
target_link_libraries(MIOpen
    PRIVATE
    composablekernel::device_operations  # Links ALL 727 files
)

# AFTER (optimized):
target_link_libraries(MIOpen
    PRIVATE
    composablekernel::device_conv2d_nhwgc_operations    # 226 files
    composablekernel::device_conv3d_ndhwgc_operations   # 290 files
    composablekernel::device_conv_old_operations        # 22 files
    # Total: 471 files (35% reduction from 727)
)
```

**Build Time Measurement**:
```bash
# Before
time cmake --build . --target MIOpen 2>&1 | tee build_before.log

# After
time cmake --build . --target MIOpen 2>&1 | tee build_after.log

# Compare
# Expected: 20-30% improvement
```

### Phase 6: Testing & Validation

**Unit Tests**:
```bash
# Test each layout library independently
ctest -R conv1d_gnwc
ctest -R conv2d_nhwgc
ctest -R conv3d_ndhwgc
# ... etc
```

**Integration Tests**:
```bash
# Full MIOpen test suite
cd projects/miopen/build
ctest --output-on-failure
```

**Performance Regression**:
```bash
# Run MIOpen benchmarks
# Verify no performance degradation from library split
```

**Binary Size**:
```bash
# Measure libMIOpen.so size
ls -lh libMIOpen.so

# Expected: Smaller due to only linking required layouts
```

### Phase 7: Documentation

**Files to Update**:
1. `projects/composablekernel/README.md` - Document new library structure
2. `projects/composablekernel/docs/BUILD.md` - Build system changes
3. `projects/miopen/docs/DEPENDENCIES.md` - Updated CK dependency info
4. **This file** - Implementation record

**Migration Guide**:
Create `LAYOUT_SPLIT_MIGRATION_GUIDE.md` for downstream consumers

---

## Rollback Plan

If issues arise, rollback is straightforward:

```bash
# Revert to commit before file migrations
git revert <migration-commit-sha>

# Or full reset if needed
git reset --hard <pre-migration-tag>
```

**Safe Points** (tag these commits):
1. `layout-split-phase1-inventory` - After file categorization
2. `layout-split-phase2-dirs` - After directory creation
3. `layout-split-phase3-migration` - After file moves
4. `layout-split-phase4-cmake` - After CMake changes
5. `layout-split-phase5-miopen` - After MIOpen integration

---

## Success Metrics

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| Files categorized | 0 | 727 | ✅ 727/727 |
| MIOpen linked files | 727 | 471 | ⏳ Pending |
| MIOpen build time | Baseline | -20% | ⏳ Pending |
| Library count | 1 | 12 | ⏳ Pending |
| Test pass rate | 100% | 100% | ⏳ Pending |
| Binary size reduction | 0% | >10% | ⏳ Pending |

---

## Timeline Estimate

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Phase 1: Inventory | ✅ Complete | None |
| Phase 2: Directories | 1 day | Phase 1 |
| Phase 3: Migration | 2-3 days | Phase 2 |
| Phase 4: CMake | 2 days | Phase 3 |
| Phase 5: MIOpen | 1 day | Phase 4 |
| Phase 6: Testing | 3-5 days | Phase 5 |
| Phase 7: Docs | 1 day | Phase 6 |
| **Total** | **10-13 days** | |

**Critical Path**: File migration → CMake changes → MIOpen integration → Testing

---

## Next Actions

**Immediate (Week 1)**:
1. ✅ Complete file inventory and categorization
2. ⏳ Create directory structure (Phase 2)
3. ⏳ Begin file migration for 1D convolutions (smallest, safest)

**Near-term (Week 2)**:
4. ⏳ Complete 2D and 3D migrations
5. ⏳ Implement CMake modularization
6. ⏳ Test individual library builds

**Long-term (Week 3)**:
7. ⏳ MIOpen integration
8. ⏳ Full validation and benchmarking
9. ⏳ Documentation and PR submission

---

## Approval Required

Before proceeding to Phase 2, please confirm:

- [ ] Categorization approach is correct (12 libraries)
- [ ] MIOpen requirements are accurate (3 libraries needed)
- [ ] Directory structure pattern is acceptable
- [ ] Timeline is reasonable
- [ ] Risk mitigation strategies are sufficient

---

**Document Version**: 1.0
**Status**: Phase 1 Complete - Ready for Phase 2
**Last Updated**: 2026-03-25
**Next Review**: Before starting Phase 2 (directory creation)
