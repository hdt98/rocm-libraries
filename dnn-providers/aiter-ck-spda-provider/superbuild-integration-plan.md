# Plan: Add aiter-ck-spda-provider to the Superbuild

## Context

The AITER CK SDPA provider is a new POC plugin that launches pre-compiled ASM Flash Attention v3 kernels on gfx942 (MI300X). The provider directory already exists at `dnn-providers/aiter-ck-spda-provider/` with only a design document. It needs to be wired into the rocm-libraries superbuild so it can be built alongside hipDNN using `cmake --preset aiter-ck-spda-provider` or via `ROCM_LIBS_ENABLE_COMPONENTS`.

## Files to Modify

### 1. `CMakeLists.txt` (repo root)

Add `aiter-ck-spda-provider` to `AVAILABLE_COMPONENTS` list (line 55, after `hipblaslt-provider`):

```cmake
set(AVAILABLE_COMPONENTS
    ${DEFAULT_COMPONENTS}
    # dnn
    hipdnn
    miopen-provider
    hipblaslt-provider
    aiter-ck-spda-provider          # <-- add
)
```

Add `add_subdirectory_with_message` block after the hipblaslt-provider block (after line 188):

```cmake
if("aiter-ck-spda-provider" IN_LIST ROCM_LIBS_ENABLE_COMPONENTS)
    add_subdirectory_with_message(
        COMPONENT aiter-ck-spda-provider
        PREFIX_PATH dnn-providers
        EXPECT_TARGET aiter_plugin
    )
endif()
```

### 2. `CMakePresets.json` (repo root)

Add a new configure preset (after the hipblaslt-provider preset, before the closing `]` of `configurePresets`):

```json
{
    "name": "aiter-ck-spda-provider",
    "description": "Build hipdnn and aiter-ck-spda-provider",
    "inherits": ["default:release"],
    "cacheVariables": {
        "ROCM_LIBS_ENABLE_COMPONENTS": "hipdnn;aiter-ck-spda-provider"
    }
}
```

## Files to Create

### 3. `dnn-providers/aiter-ck-spda-provider/CMakeLists.txt`

Follow the hipblaslt-provider pattern (`dnn-providers/hipblaslt-provider/CMakeLists.txt`). Key structure:

- `cmake_minimum_required(VERSION 3.25.2)`
- Toolchain / project setup (project name: `aiter-ck-spda-provider`)
- `CMAKE_CXX_STANDARD 17`
- Find `hip REQUIRED`
- SDK dual-mode discovery (`if(NOT TARGET hipdnn_data_sdk) find_package(...)`)
- `include(Dependencies)`, `include(Tests)`, `include(ClangCheck)`
- Plugin engine directory setup (`HIPDNN_BUILD_PLUGIN_ENGINE_DIR`)
- `add_library(aiter_plugin_private ...)` for implementation sources (listed in the design doc)
- `add_library(aiter_plugin SHARED AiterPluginPublic.cpp)` linking to `aiter_plugin_private`
- Symbol hiding: `-Xlinker --exclude-libs=ALL`, `CXX_VISIBILITY_PRESET hidden`
- Output directories point to `HIPDNN_BUILD_PLUGIN_ENGINE_DIR`
- `.co` file handling: `AITER_ASM_DIR` compile definition pointing to install location
- `install(DIRECTORY asm_kernels/ DESTINATION share/aiter/asm_kernels)` for the `.co` binary
- Test subdirectories + `finalize_test_targets("${PROJECT_NAME}")`
- `install(TARGETS aiter_plugin ...)` to `HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR`

**No external library dependency** beyond HIP (unlike hipblaslt-provider which needs `find_package(hipblaslt)`). This makes the CMakeLists.txt simpler.

### 4. `dnn-providers/aiter-ck-spda-provider/cmake/ClangToolChain.cmake`

Copy from `dnn-providers/hipblaslt-provider/cmake/ClangToolChain.cmake` (identical across all providers).

### 5. `dnn-providers/aiter-ck-spda-provider/cmake/Dependencies.cmake`

Copy from `dnn-providers/hipblaslt-provider/cmake/Dependencies.cmake` (GTest fetch logic, identical across providers).

### 6. `dnn-providers/aiter-ck-spda-provider/cmake/Tests.cmake`

Copy from `dnn-providers/hipblaslt-provider/cmake/Tests.cmake`, adapting:
- Property name: `HIPBLASLT_TEST_TARGETS` -> `AITER_TEST_TARGETS`
- Install function name: `install_hipblaslt_plugin_ctest_files` -> `install_aiter_plugin_ctest_files`
- Install path: `hipblaslt_plugin` -> `aiter_plugin`

### 7. `dnn-providers/aiter-ck-spda-provider/cmake/ClangCheck.cmake`

Copy from `dnn-providers/hipblaslt-provider/cmake/ClangCheck.cmake` (identical across providers).

### 8. Placeholder source and test directories

Create minimal placeholder files so the superbuild can configure and build:

- `dnn-providers/aiter-ck-spda-provider/src/AiterPluginPublic.cpp` - empty plugin entry point (just enough to compile a shared library)
- `dnn-providers/aiter-ck-spda-provider/tests/CMakeLists.txt` - empty (no tests yet)
- `dnn-providers/aiter-ck-spda-provider/integration_tests/CMakeLists.txt` - empty (no tests yet)
- `dnn-providers/aiter-ck-spda-provider/asm_kernels/gfx942/.gitkeep` - placeholder for `.co` binaries

## Verification

1. Configure with the new preset:
   ```bash
   cmake --preset aiter-ck-spda-provider
   ```
   Verify cmake configures successfully, finds hipdnn SDKs, and reports the aiter-ck-spda-provider component.

2. Build:
   ```bash
   cmake --build build --target aiter_plugin
   ```
   Verify the `aiter_plugin.so` shared library is produced in the plugin engine directory.

3. Verify the existing presets still work (no regressions):
   ```bash
   cmake --preset hipdnn
   ```
