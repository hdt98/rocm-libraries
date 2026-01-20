# AOCL OpenMP Integration on Windows - Technical Report

## Executive Summary

Successfully enabled OpenMP multithreading support for AOCL (AMD Optimizing CPU Libraries) on Windows when building with ROCm Clang. This required solving three key technical challenges related to compiler compatibility, header generation, and OpenMP version mismatches.

**Result:** AOCL 5.2 now builds successfully with OpenMP support using Visual Studio's OpenMP 2.0 library and ROCm Clang 20.0.0.

**🔷 Platform Scope:** All changes are strictly **Windows-only**. Zero impact on Linux builds. See [WINDOWS_ONLY_CONFIRMATION.md](./WINDOWS_ONLY_CONFIRMATION.md) for detailed analysis.

## Problem Statement

AOCL was building in single-threaded mode (`ENABLE_MULTITHREADING=OFF`) to avoid OpenMP dependency issues. However, for production use in rocBLAS testing, OpenMP multithreading is required for performance validation.

### Environment
- **OS:** Windows 10/11
- **Compiler:** ROCm Clang 20.0.0 (from ROCm 6.4)
- **Build System:** CMake + Ninja
- **OpenMP Library:** Visual Studio MSVC 14.50 (OpenMP 2.0)
- **AOCL Version:** 5.2 (AOCL-BLAS built on BLIS)

## Technical Challenges & Solutions

### Challenge 1: Compiler Flag Incompatibility

**Problem:** BLIS CMakeLists.txt uses MSVC-specific OpenMP SIMD flags when detecting Windows:
```cmake
if(WIN32)
    set(COMPSIMDFLAGS /openmp:experimental)  # MSVC syntax
```

This causes `clang: fatal error: no such file or directory: '/openmp:experimental'` because ROCm Clang expects GNU-style flags.

**Root Cause:** BLIS's build system assumes Windows = MSVC compiler, but we're using Clang with MSVC compatibility mode.

**Solution:** Patch BLIS CMakeLists.txt to use Clang-compatible flags:
```cmake
if(WIN32)
    set(COMPSIMDFLAGS -fopenmp-simd)  # Clang syntax
```

**Implementation:** `rdeps.py` clones BLIS separately and patches it before AOCL's build process.

---

### Challenge 2: BLIS Header Flattening Skips System Headers

**Problem:** BLIS uses `flatten-headers.py` to create a monolithic `blis.h` header file. This script marks system headers like `<omp.h>` as "skipped":
```c
#include <omp.h> // skipped
```

This results in OpenMP functions being undeclared during compilation, causing errors like:
```
error: call to undeclared function 'omp_get_active_level'
```

**Root Cause:** BLIS's header flattening script intentionally skips system headers that aren't in the BLIS source tree to avoid conflicts. The generated `blis.h` contains the comment but not the actual include.

**Solution:** Force-include a custom header before all source files using the `-include` compiler flag:
```cmake
add_compile_options(-include "${CMAKE_CURRENT_SOURCE_DIR}/omp_stub.h")
```

This bypasses the skipped `#include <omp.h>` in the generated header.

---

### Challenge 3: OpenMP Version Mismatch (Critical Issue)

**Problem:** Visual Studio's OpenMP implementation only supports **OpenMP 2.0**, but BLIS requires **OpenMP 3.0+** functions:
- `omp_get_active_level()` - Added in OpenMP 3.0
- `omp_get_max_active_levels()` - Added in OpenMP 3.0
- `omp_set_max_active_levels()` - Added in OpenMP 3.0

**Root Cause:** Microsoft's OpenMP runtime (vcomp*.dll) has not been updated beyond OpenMP 2.0 specification, despite supporting OpenMP 5.0+ in their LLVM-based compiler.

**Solution:** Create a stub header (`omp_stub.h`) that provides minimal implementations of the missing OpenMP 3.0 functions:

```c
/* omp_stub.h - OpenMP 3.0 compatibility shim for Visual Studio OpenMP 2.0 */
#ifndef OMP_STUB_H
#define OMP_STUB_H

#include <omp.h>

static inline int omp_get_active_level(void) {
    /* Visual Studio OpenMP doesn't support nested parallelism.
       Return 0 for top level, 1 if inside a parallel region. */
    return (omp_in_parallel() ? 1 : 0);
}

static inline int omp_get_max_active_levels(void) {
    /* Return 1 since Visual Studio OpenMP doesn't support nested parallelism */
    return 1;
}

static inline void omp_set_max_active_levels(int max_levels) {
    /* No-op - Visual Studio OpenMP doesn't support nested parallelism */
    (void)max_levels;
}

#endif /* OMP_STUB_H */
```

**Rationale:** These implementations are correct for BLIS's use case because:
1. BLIS uses these functions to detect nested parallel regions
2. Visual Studio OpenMP doesn't support nested parallelism anyway
3. The stub correctly reports "no nesting support" (max_levels = 1)
4. BLIS will run single-threaded when called from within a parallel region, which is the safe/correct behavior

---

## Implementation Details

### File Modified: `projects/rocblas/rdeps.py`

**Changes to AOCL build process:**

1. **Use Visual Studio's OpenMP library** (lines 146-164):
   ```python
   msvc_base = r'C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC'
   msvc_dir = os.path.join(msvc_base, msvc_versions[0])
   openmp_lib = os.path.join(msvc_dir, 'lib', 'x64', 'libomp.lib')
   openmp_include = os.path.join(msvc_dir, 'include')
   ```

2. **Clone and patch BLIS** (lines 166-258):
   - Clone BLIS from GitHub
   - Patch CMakeLists.txt to fix OpenMP flags
   - Create `omp_stub.h` with OpenMP 3.0 compatibility
   - Add preprocessor defines and include paths

3. **Configure AOCL with OpenMP** (lines 269-290):
   ```python
   '-DENABLE_MULTITHREADING=ON',
   f'-DOpenMP_libomp_LIBRARY={openmp_lib}',
   f'-DBLAS_PATH={blis_src_dir.parent}',  # Use patched BLIS
   ```

### Key CMake Path Handling

**Important:** Windows backslashes must be converted to forward slashes for CMake:
```python
openmp_include_cmake = openmp_include.replace('\\', '/')
```

CMake handles forward slashes correctly on all platforms, but backslashes cause parsing errors in `include_directories()` commands.

---

## Potential Cleanup/Streamlining

### Redundant Environment Variables (Minor)

**Current code** (lines 265-267):
```python
import_env = os.environ.copy()
import_env['CFLAGS'] = f'-I"{openmp_include}"'
import_env['CXXFLAGS'] = f'-I"{openmp_include}"'
```

**Also passes** (lines 277-278):
```python
f'-DCMAKE_C_FLAGS=-I"{openmp_include}" -fopenmp=libomp',
f'-DCMAKE_CXX_FLAGS=-I"{openmp_include}" -fopenmp=libomp',
```

**Assessment:** The environment variables are likely redundant since CMake flags take precedence. However, this is a "belt and suspenders" approach and doesn't hurt. Could be removed for cleanliness if testing confirms it's unnecessary.

**Recommendation:** Keep as-is for robustness, or test removal during integration phase.

---

## False Trails Encountered (Already Removed)

1. ❌ **Attempted to use ROCm's OpenMP library** - ROCm 6.4 on Windows doesn't ship with `libomp.lib`
2. ❌ **Hardcoded RC_COMPILER path** - Was due to running in wrong terminal (not Visual Studio Developer Command Prompt)
3. ❌ **Attempted to disable PRAGMA_OMP_SIMD via CMake** - Flag was set by try_compile test, not configuration option

All false trails have been removed from the final implementation.

---

## Testing & Verification

### Build Success Criteria
- ✅ AOCL builds without errors
- ✅ OpenMP is enabled (`ENABLE_MULTITHREADING=ON`)
- ✅ BLIS compiles with OpenMP support
- ✅ Static library generated: `libaocl.lib`

### Build Output Confirmation
```
Using OpenMP from Visual Studio: C:\Program Files\Microsoft Visual Studio\...\libomp.lib
✓ Fixed OpenMP SIMD flags
✓ Created OpenMP 3.0 stub header at ...\blis\omp_stub.h
✓ Added OpenMP include directory, preprocessor define, and force-include
✓ Successfully patched BLIS CMakeLists.txt
✓ AOCL 5.2 successfully built with ILP64 support (static)
```

---

## Recommendations for AOCL Team

### 1. BLIS Windows/Clang Compatibility

**Issue:** BLIS CMakeLists.txt assumes Windows = MSVC compiler

**Suggested Fix:** Detect compiler family instead of OS:
```cmake
if(CMAKE_C_COMPILER_ID MATCHES "MSVC")
    set(COMPSIMDFLAGS /openmp:experimental)
elseif(CMAKE_C_COMPILER_ID MATCHES "Clang|GNU")
    set(COMPSIMDFLAGS -fopenmp-simd)
endif()
```

### 2. OpenMP 3.0 Requirement Documentation

**Issue:** BLIS requires OpenMP 3.0+ but doesn't document this clearly

**Suggested Fix:** Update documentation to specify:
- Minimum OpenMP version: 3.0
- Functions required: `omp_get_active_level()`, `omp_get_max_active_levels()`
- Note that OpenMP 2.0 implementations (Visual Studio) are incompatible without shims

### 3. Header Flattening and System Headers

**Issue:** `flatten-headers.py` skips system headers, causing issues when using force-includes

**Suggested Fix:** Consider one of:
- Option A: Include `#pragma message` in flattened header to warn about skipped includes
- Option B: Provide a CMake option to disable header flattening for platforms with module/PCH support
- Option C: Document that users must ensure OpenMP headers are available in all translation units

### 4. Windows OpenMP Support

**Issue:** No clear guidance on OpenMP library selection for Windows/Clang builds

**Suggested Fix:** Document the Windows OpenMP landscape:
- Visual Studio OpenMP: 2.0 only (requires shims for BLIS)
- Intel OpenMP (libomp): Full support, but requires separate install
- LLVM libomp: Not included with ROCm on Windows
- Recommendation: Point users to Intel's OpenMP runtime or provide bundled solution

---

## Next Steps

### Phase 2: rocBLAS Client Integration (Upcoming)

1. Build rocBLAS with AOCL reference BLAS
2. Ensure OpenMP is properly linked in rocBLAS clients
3. Verify OpenMP threading works correctly in test suite
4. Performance validation with multithreaded AOCL

### Potential Upstream Contributions

1. **BLIS Pull Request:** Windows/Clang compatibility fix (compiler detection)
2. **AOCL Documentation:** OpenMP version requirements and Windows build guide
3. **ROCm Issue:** Request bundled OpenMP runtime for Windows builds

---

## References

### Code Files Modified
- `projects/rocblas/rdeps.py` - Main build script for AOCL integration

### Generated Files (Build Artifacts)
- `build/deps/blis/omp_stub.h` - OpenMP 3.0 compatibility shim
- `build/deps/blis/CMakeLists.txt` - Patched BLIS build configuration
- `build/deps/aocl/install_package/lib/libaocl.lib` - Final library

### External Dependencies
- AOCL 5.2: https://github.com/amd/aocl (branch AOCL-5.2)
- BLIS: https://github.com/amd/blis (branch master)
- Visual Studio MSVC: OpenMP 2.0 runtime (vcomp140.dll)

### Key Discoveries
- Visual Studio OpenMP is OpenMP 2.0 only (as of MSVC 14.50 / VS 2025)
- ROCm 6.4 on Windows does not bundle OpenMP runtime
- BLIS header flattening can interfere with force-included headers
- CMake requires forward slashes in paths on Windows for `include_directories()`

---

## Authors & Timeline

**Date:** January 20, 2026
**Contributor:** Tony Davis
**Reviewer/Collaborator:** AI Assistant (Claude)
**Context:** rocBLAS Windows testing infrastructure improvement

---

## Appendix: Complete omp_stub.h

```c
/* Stub implementations for OpenMP 3.0+ functions missing in Visual Studio OpenMP 2.0 */
#ifndef OMP_STUB_H
#define OMP_STUB_H

#include <omp.h>

#ifdef __cplusplus
extern "C" {
#endif

/* These functions were added in OpenMP 3.0 */
static inline int omp_get_active_level(void) {
    /* Return 0 for top level (not inside a parallel region), 
       Visual Studio OpenMP doesn't support nested parallelism anyway */
    return (omp_in_parallel() ? 1 : 0);
}

static inline int omp_get_max_active_levels(void) {
    /* Return 1 since Visual Studio OpenMP doesn't support nested parallelism */
    return 1;
}

static inline void omp_set_max_active_levels(int max_levels) {
    /* No-op since Visual Studio OpenMP doesn't support nested parallelism */
    (void)max_levels;
}

#ifdef __cplusplus
}
#endif

#endif /* OMP_STUB_H */
```

---

*End of Technical Report*
