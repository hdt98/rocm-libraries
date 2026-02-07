# Rebuild Failed Include Tests

## Current Status

- **Total include tests**: 74
- **Passed on first build**: 53
- **Failed on first build**: 21

## Fixes Applied

The following header files have been fixed with missing includes:

1. **`include/ck_tile/ops/gemm/pipeline/gemm_pipeline_problem.hpp`**
   - Added: `<string>`, `<type_traits>`, `<tuple>`
   - Reason: Uses `std::conditional_t`, `std::tuple_element_t`, `std::string`

2. **`include/ck_tile/ops/gemm/kernel/grouped_gemm_kernel.hpp`**
   - Added: `<array>`
   - Reason: Uses `std::array`

3. **`include/ck_tile/ops/gemm/warp/warp_gemm_attribute_wmma_impl_base_traits.hpp`**
   - Added: `"ck_tile/core.hpp"`
   - Reason: Uses core types like `gfx11_t`, `ext_vector_t`, `sequence`, `index_t`

4. **`include/ck_tile/ops/gemm/warp/warp_gemm_attribute_wmma_impl_16bit_traits.hpp`**
   - Added: `"ck_tile/core.hpp"`
   - Reason: Uses `WmmaTraits` template and core types

5. **`include/ck_tile/ops/gemm/warp/warp_gemm_attribute_wmma_impl_8bit_traits.hpp`**
   - Added: `"ck_tile/core.hpp"`
   - Reason: Uses `WmmaTraits` template and core types

## Failed Targets (21)

```
test_include_kernel_grouped_gemm_kernel
test_include_pipeline_gemm_pipeline_ag_bg_cr_comp_async
test_include_pipeline_gemm_pipeline_ag_bg_cr_comp_async_default_policy
test_include_pipeline_gemm_pipeline_ag_bg_cr_comp_v3
test_include_pipeline_gemm_pipeline_ag_bg_cr_comp_v4
test_include_pipeline_gemm_pipeline_ag_bg_cr_comp_v4_default_policy
test_include_pipeline_gemm_pipeline_ag_bg_cr_comp_v5
test_include_pipeline_gemm_pipeline_ag_bg_cr_comp_v5_default_policy
test_include_pipeline_gemm_pipeline_ag_bg_cr_comp_v6
test_include_pipeline_gemm_pipeline_ag_bg_cr_comp_v6_default_policy
test_include_pipeline_gemm_pipeline_ag_bg_cr_mem
test_include_pipeline_gemm_pipeline_agmem_bgmem_creg_v1
test_include_pipeline_gemm_pipeline_agmem_bgmem_creg_v1_default_policy
test_include_pipeline_gemm_pipeline_agmem_bgmem_creg_v2
test_include_pipeline_gemm_pipeline_agmem_bgmem_creg_v2_default_policy
test_include_pipeline_gemm_pipeline_problem
test_include_pipeline_wp_pipeline_agmem_bgmem_creg_base_policy
test_include_pipeline_wp_pipeline_agmem_bgmem_creg_v2
test_include_warp_warp_gemm_attribute_wmma_impl_16bit_traits
test_include_warp_warp_gemm_attribute_wmma_impl_8bit_traits
test_include_warp_warp_gemm_attribute_wmma_impl_base_traits
```

## How to Rebuild

### Option 1: Using Docker Script (Recommended)

```bash
cd /home/AMD/congma13/home_build/fork/rocm-libraries/projects/composablekernel

# Run the Docker build script
./run_docker_build.sh
```

This will:
1. Start Docker container with GPU access
2. Configure CMake
3. Build all 21 failed targets
4. Move passed targets to `passed_include_test_targets.txt`
5. Move still-failed targets to `failed_include_test_targets.txt`
6. Clear `include_test_targets.txt`

### Option 2: Manual Docker Build

```bash
cd /home/AMD/congma13/home_build/fork/rocm-libraries/projects/composablekernel

docker run --rm -it \
    --network=host \
    --device=/dev/kfd \
    --device=/dev/dri \
    --security-opt seccomp=unconfined \
    --group-add video \
    -v $(pwd):$(pwd) \
    -w $(pwd) \
    61474c2a0070 \
    bash

# Inside container:
./build_include_tests.sh
exit
```

### Option 3: Build Single Target for Testing

```bash
cd /home/AMD/congma13/home_build/fork/rocm-libraries/projects/composablekernel

docker run --rm -it \
    --network=host \
    --device=/dev/kfd \
    --device=/dev/dri \
    --security-opt seccomp=unconfined \
    --group-add video \
    -v $(pwd):$(pwd) \
    -w $(pwd)/build \
    61474c2a0070 \
    ninja test_include_pipeline_gemm_pipeline_problem
```

## After Rebuild

Check the results:

```bash
# View passed targets
cat passed_include_test_targets.txt

# View failed targets
cat failed_include_test_targets.txt

# If there are still failures, analyze the errors
cat build_errors/*.txt
```

## Next Steps

If targets still fail after this rebuild:
1. Check error logs in `build_errors/` directory
2. Identify missing includes from compilation errors
3. Add missing includes to the header files
4. Copy failed targets back: `cp failed_include_test_targets.txt include_test_targets.txt`
5. Run rebuild again

## Files in This Repository

- `include_test_targets.txt` - Currently contains 21 targets to rebuild
- `passed_include_test_targets.txt` - 53 targets that passed
- `failed_include_test_targets.txt` - Backup of 21 failed targets
- `build_include_tests.sh` - Build script (runs inside Docker)
- `run_docker_build.sh` - Docker launcher script
- `analyze_build_errors.sh` - Script to analyze build errors (if needed)
