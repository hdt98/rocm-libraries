# Phase 4: CMake Modularization - Status

**Date**: 2026-03-25
**Status**: ✅ **Structure Ready - Manual Integration Needed**

---

## ✅ Completed

### 1. Layout-Specific CMakeLists.txt Generated

**42 layout CMakeLists.txt files** created in layout subdirectories:

```
grouped_conv{1,2,3}d_*/
├── {layout}/
│   └── CMakeLists.txt  ← Defines layout-specific source list
```

Each file:
- Defines a variable with all source files for that layout
- Calls `add_instance_library()` to create the library target
- Example: `device_grouped_conv2d_fwd_nhwgc_instance`

### 2. Operation-Level CMakeLists.txt Generated

**28 operation CMakeLists.txt files** that include layout subdirectories:

```
grouped_conv{1,2,3}d_*/CMakeLists.txt  ← Includes all layout subdirs
```

Each file calls `add_subdirectory()` for each layout it contains.

### 3. Top-Level Library Targets Defined

Created `toplevel_layout_libraries.cmake` with **12 layout-specific libraries**:

**1D Libraries:**
- `device_conv1d_gnwc_operations`
- `device_conv1d_nwgc_operations`

**2D Libraries:**
- `device_conv2d_gnhwc_operations`
- `device_conv2d_nhwgc_operations` ⭐ (MIOpen required)
- `device_conv2d_ngchw_operations`

**3D Libraries:**
- `device_conv3d_gndhwc_operations`
- `device_conv3d_ndhwgc_operations` ⭐ (MIOpen required)
- `device_conv3d_ngcdhw_operations`
- `device_conv3d_nhwgc_operations`

**Special Libraries:**
- `device_conv_old_operations` ⭐ (MIOpen required - non-grouped)
- `device_convnd_generic_operations` (N-dimensional generic)
- `device_quantization_operations` (quantization)

**Umbrella Library:**
- `device_conv_operations` (backward compatibility - includes all)

---

## 📝 Manual Integration Required

The CMake structure is ready but needs to be integrated into the main build system:

### Step 1: Backup Original CMakeLists.txt

The original was listing all files in one place. Before modifying:

```bash
cp projects/composablekernel/library/src/tensor_operation_instance/gpu/CMakeLists.txt \
   projects/composablekernel/library/src/tensor_operation_instance/gpu/CMakeLists.txt.bak
```

### Step 2: Modify Main CMakeLists.txt

Find where convolution operations are currently added (around line 300-400) and:

**REMOVE** the old approach that scans all subdirectories:
```cmake
# OLD: Auto-scan approach
FOREACH(subdir_path ${dir_list})
    IF(IS_DIRECTORY "${subdir_path}")
        set(cmake_instance)
        file(READ "${subdir_path}/CMakeLists.txt" cmake_instance)
        # ... lots of conditional logic ...
        if(add_inst EQUAL 1)
            add_subdirectory("${subdir_path}")
        endif()
    ENDIF()
ENDFOREACH()
```

**REPLACE** with explicit includes for operations with layout splits:
```cmake
# NEW: Explicit includes for layout-split operations
add_subdirectory(grouped_conv1d_fwd)
add_subdirectory(grouped_conv1d_bwd_weight)
add_subdirectory(grouped_conv2d_fwd)
add_subdirectory(grouped_conv2d_bwd_data)
add_subdirectory(grouped_conv2d_bwd_weight)
add_subdirectory(grouped_conv2d_fwd_clamp)
add_subdirectory(grouped_conv2d_fwd_bias_clamp)
add_subdirectory(grouped_conv2d_fwd_bias_bnorm_clamp)
add_subdirectory(grouped_conv2d_fwd_dynamic_op)
add_subdirectory(grouped_conv3d_fwd)
add_subdirectory(grouped_conv3d_bwd_data)
add_subdirectory(grouped_conv3d_bwd_weight)
add_subdirectory(grouped_conv3d_fwd_scale)
add_subdirectory(grouped_conv3d_fwd_clamp)
add_subdirectory(grouped_conv3d_fwd_bias_clamp)
add_subdirectory(grouped_conv3d_fwd_bias_bnorm_clamp)
add_subdirectory(grouped_conv3d_fwd_bilinear)
add_subdirectory(grouped_conv3d_fwd_convscale)
add_subdirectory(grouped_conv3d_fwd_convscale_add)
add_subdirectory(grouped_conv3d_fwd_convscale_relu)
add_subdirectory(grouped_conv3d_fwd_convinvscale)
add_subdirectory(grouped_conv3d_fwd_dynamic_op)
add_subdirectory(grouped_conv3d_fwd_scaleadd_ab)
add_subdirectory(grouped_conv3d_fwd_scaleadd_scaleadd_relu)
add_subdirectory(grouped_conv3d_bwd_data_bilinear)
add_subdirectory(grouped_conv3d_bwd_data_scale)
add_subdirectory(grouped_conv3d_bwd_weight_bilinear)
add_subdirectory(grouped_conv3d_bwd_weight_scale)

# Non-grouped and special operations (keep existing approach)
add_subdirectory(conv1d_bwd_data)
add_subdirectory(conv2d_bwd_data)
add_subdirectory(conv2d_fwd)
add_subdirectory(conv2d_fwd_bias_relu)
add_subdirectory(conv2d_fwd_bias_relu_add)
add_subdirectory(conv3d_bwd_data)
add_subdirectory(grouped_convnd_bwd_weight)
add_subdirectory(quantization)
```

### Step 3: Add Layout Library Definitions

After the subdirectory includes, add the content from `toplevel_layout_libraries.cmake`:

```cmake
# Include layout-specific library definitions
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../projects/composablekernel/layout_pruning/toplevel_layout_libraries.cmake)
```

OR copy the contents directly into the CMakeLists.txt file.

### Step 4: Remove/Update Old device_conv_operations Target

Find and **REMOVE** the old monolithic library definition:
```cmake
# OLD - Remove this:
add_library(device_conv_operations ${CK_DEVICE_CONV_INSTANCES})
add_library(composablekernels::device_conv_operations ALIAS device_conv_operations)
```

The new `device_conv_operations` is defined in `toplevel_layout_libraries.cmake` as an INTERFACE library.

---

## 🧪 Testing the Build

After integration, test the build:

### Test 1: Build Individual Layout Libraries

```bash
cd build
cmake --build . --target device_conv2d_nhwgc_operations
cmake --build . --target device_conv3d_ndhwgc_operations
cmake --build . --target device_conv_old_operations
```

### Test 2: Build Umbrella Library

```bash
cmake --build . --target device_conv_operations
```

### Test 3: Build MIOpen with Selective Linking

In MIOpen's CMakeLists.txt, change from:
```cmake
# OLD
target_link_libraries(MIOpen PRIVATE composablekernels::device_conv_operations)
```

To:
```cmake
# NEW - Only link required layouts
target_link_libraries(MIOpen PRIVATE
    composablekernels::device_conv2d_nhwgc_operations
    composablekernels::device_conv3d_ndhwgc_operations
    composablekernels::device_conv_old_operations
)
```

### Test 4: Verify Build Time Improvement

```bash
# Clean build
rm -rf build && mkdir build && cd build

# Time the build
time cmake --build . --target MIOpen 2>&1 | tee build_log.txt

# Expected: 20-30% faster than before
```

---

## 📊 Expected Results

### File Count Verification

Run this to verify all files are included:

```bash
# Count cpp files in each layout library's CMakeLists.txt
for cmake in $(find . -path "*/*/CMakeLists.txt" -name "CMakeLists.txt"); do
    echo "$cmake: $(grep '\.cpp' "$cmake" | wc -l) files"
done | grep -E "(gnhwc|nhwgc|ngchw|gndhwc|ndhwgc|ngcdhw)" | \
awk '{sum += $2} END {print "Total: " sum " files"}'

# Should show: Total: 673 files
```

### Library Target Verification

```bash
# After successful cmake configuration
cmake --build . --target help | grep "device_conv.*operations"

# Should show:
# device_conv1d_gnwc_operations
# device_conv1d_nwgc_operations
# device_conv2d_gnhwc_operations
# device_conv2d_nhwgc_operations
# device_conv2d_ngchw_operations
# device_conv3d_gndhwc_operations
# device_conv3d_ndhwgc_operations
# device_conv3d_ngcdhw_operations
# device_conv3d_nhwgc_operations
# device_conv_old_operations
# device_convnd_generic_operations
# device_quantization_operations
# device_conv_operations (umbrella)
```

---

## ⚠️ Known Issues / Considerations

### 1. Conditional Targets

Some layout libraries are only created if targets exist (e.g., `device_conv1d_gnwc_operations`). The `if(TARGET ...)` checks handle this gracefully.

### 2. GPU Architecture Filtering

The `add_instance_library()` function already filters files based on:
- GPU target (gfx9, gfx10, gfx11, etc.)
- Data types (fp16, fp32, int8, etc.)
- Kernel types (XDL, WMMA, DL)

This filtering still applies to layout-specific libraries.

### 3. Backward Compatibility

The umbrella `device_conv_operations` target maintains backward compatibility for projects that:
- Don't know about layout-specific libraries
- Want to link against all convolution operations

### 4. Installation

Update installation rules to install all layout-specific libraries:

```cmake
install(TARGETS
    device_conv1d_gnwc_operations
    device_conv1d_nwgc_operations
    device_conv2d_gnhwc_operations
    device_conv2d_nhwgc_operations
    device_conv2d_ngchw_operations
    device_conv3d_gndhwc_operations
    device_conv3d_ndhwgc_operations
    device_conv3d_ngcdhw_operations
    device_conv3d_nhwgc_operations
    device_conv_old_operations
    device_convnd_generic_operations
    device_quantization_operations
    device_conv_operations
    EXPORT ComposableKernelTargets
)
```

---

## 📁 Generated Files

All generated files are in `/projects/composablekernel/layout_pruning/`:

- `generate_layout_cmake_v2.py` - Script that generated CMakeLists.txt files
- `toplevel_layout_libraries.cmake` - Top-level library definitions
- This file: `PHASE4_STATUS.md`

---

## ✅ Phase 4 Checklist

- [x] Generate layout-specific CMakeLists.txt (42 files)
- [x] Generate operation-level CMakeLists.txt (28 files)
- [x] Define top-level layout libraries (12 libraries)
- [x] Define umbrella library (backward compatibility)
- [ ] **Manual**: Integrate into main CMakeLists.txt
- [ ] **Manual**: Test build
- [ ] **Manual**: Update installation rules
- [ ] **Verify**: All 673 files included
- [ ] **Verify**: Layout libraries build independently

---

## Next Steps

1. **Backup** the original CMakeLists.txt
2. **Integrate** the changes into the main build system
3. **Test** individual layout library builds
4. **Proceed** to Phase 5: MIOpen Integration

---

**Status**: CMake structure complete, ready for integration and testing
