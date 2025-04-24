add_custom_target(example_gemm_dl)

add_example_executable(example_gemm_dl_fp32 gemm_dl_fp32.cpp)
add_example_dependencies(example_gemm_dl example_gemm_dl_fp32)

add_example_executable(example_gemm_dl_fp16 gemm_dl_fp16.cpp)
add_example_dependencies(example_gemm_dl example_gemm_dl_fp16)

add_example_executable(example_gemm_dpp_fp16 gemm_dpp_fp16.cpp)

add_example_executable(example_gemm_dl_int8 gemm_dl_int8.cpp)
add_example_dependencies(example_gemm_dl example_gemm_dl_int8)
if(USE_BITINT_EXTENSION_INT4)
    add_example_executable(example_gemm_dl_int4 gemm_dl_int4.cpp)
    add_example_dependencies(example_gemm_dl example_gemm_dl_int4)
endif(USE_BITINT_EXTENSION_INT4)

add_custom_target(example_gemm_xdl)
add_example_executable(example_gemm_xdl_fp16 gemm_xdl_fp16.cpp)
add_example_dependencies(example_gemm_xdl example_gemm_xdl_fp16)

add_example_executable(example_gemm_xdl_fp16_v2 gemm_xdl_fp16_v2.cpp)
add_example_dependencies(example_gemm_xdl example_gemm_xdl_fp16_v2)

add_example_executable(example_gemm_xdl_fp16_streamk_v3 gemm_xdl_fp16_streamk_v3.cpp)
add_example_dependencies(example_gemm_xdl example_gemm_xdl_fp16_streamk_v3)
add_example_executable(example_gemm_xdl_fp16_v3 gemm_xdl_fp16_v3.cpp)
add_example_dependencies(example_gemm_xdl example_gemm_xdl_fp16_v3)
add_example_executable(example_gemm_xdl_fp8_v3 gemm_xdl_fp8_v3.cpp)
add_example_dependencies(example_gemm_xdl example_gemm_xdl_fp8_v3)
add_example_executable(example_gemm_xdl_fp16_fp8_v3 gemm_xdl_fp16_fp8_v3.cpp)
add_example_executable(example_gemm_xdl_fp16_pk_i4_v3 gemm_xdl_fp16_pk_i4_v3.cpp)
add_example_executable(example_gemm_xdl_fp16_pk_i4_v3_b_scale gemm_xdl_fp16_pk_i4_v3_b_scale.cpp)
add_example_executable(example_gemm_xdl_bf16_pk_i4_v3 gemm_xdl_bf16_pk_i4_v3.cpp)
add_example_dependencies(example_gemm_xdl example_gemm_xdl_fp16_fp8_v3)
add_example_executable(example_gemm_xdl_bf16_v3 gemm_xdl_bf16_v3.cpp)
add_example_dependencies(example_gemm_xdl example_gemm_xdl_bf16_v3)

list(APPEND gpu_list gfx942 gfx950)
set(target 0)
foreach(gpu IN LISTS GPU_TARGETS)
    if(gpu IN_LIST gpu_list AND target EQUAL 0)
        add_example_executable(example_gemm_xdl_fp8_pk_i4_bpreshuffle_v3 gemm_xdl_fp8_pk_i4_bpreshuffle_v3.cpp)
        add_example_executable(example_gemm_xdl_fp8_pk_i4_v3 gemm_xdl_fp8_pk_i4_v3.cpp)
        set(target 1)
    endif()
endforeach()

add_example_executable(example_gemm_xdl_wavelet_fp16 gemm_xdl_wavelet_fp16.cpp)
add_example_dependencies(example_gemm_xdl example_gemm_xdl_wavelet_fp16)

add_example_executable(example_gemm_xdl_skip_b_lds_fp16 gemm_xdl_skip_b_lds_fp16.cpp)
add_example_dependencies(example_gemm_xdl example_gemm_xdl_skip_b_lds_fp16)

add_example_executable(example_gemm_xdl_bf16 gemm_xdl_bf16.cpp)
add_example_dependencies(example_gemm_xdl example_gemm_xdl_bf16)

add_example_executable(example_gemm_xdl_int8 gemm_xdl_int8.cpp)
add_example_dependencies(example_gemm_xdl example_gemm_xdl_int8)

if(USE_BITINT_EXTENSION_INT4)
    add_example_executable(example_gemm_xdl_int4 gemm_xdl_int4.cpp)
    add_example_dependencies(example_gemm_xdl example_gemm_xdl_int4)
endif(USE_BITINT_EXTENSION_INT4)

add_example_executable(example_gemm_xdl_fp64 gemm_xdl_fp64.cpp)
add_example_dependencies(example_gemm_xdl example_gemm_xdl_fp64)

add_example_executable(example_gemm_xdl_streamk gemm_xdl_streamk.cpp)

list(APPEND gpu_list gfx90a gfx942 gfx950)
set(target 0)
foreach(gpu IN LISTS GPU_TARGETS)
    if(gpu IN_LIST gpu_list AND target EQUAL 0)
        add_example_executable(example_gemm_xdl_lds_direct_load_fp32 gemm_xdl_lds_direct_load_fp32.cpp)
        add_example_dependencies(example_gemm_xdl example_gemm_xdl_lds_direct_load_fp32)

        add_example_executable(example_gemm_xdl_lds_direct_load_fp16 gemm_xdl_lds_direct_load_fp16.cpp)
        add_example_dependencies(example_gemm_xdl example_gemm_xdl_lds_direct_load_fp16)

        add_example_executable(example_gemm_xdl_bf16_streamk_v3 gemm_xdl_bf16_streamk_v3.cpp)
        add_example_dependencies(example_gemm_xdl example_gemm_xdl_bf16_streamk_v3)

        add_example_executable(example_gemm_xdl_fp8_streamk_v3 gemm_xdl_fp8_streamk_v3.cpp)
        add_example_dependencies(example_gemm_xdl example_gemm_xdl_fp8_streamk_v3)
        set(target 1)
    endif()
endforeach()

add_example_executable(example_gemm_xdl_fp8 gemm_xdl_fp8.cpp)
add_example_dependencies(example_gemm_xdl example_gemm_xdl_fp8)

add_example_executable(example_gemm_xdl_fp8_bf8 gemm_xdl_fp8_bf8.cpp)
add_example_dependencies(example_gemm_xdl example_gemm_xdl_fp8_bf8)

add_example_executable(example_gemm_xdl_fp16_fp8 gemm_xdl_fp16_fp8.cpp)
add_example_dependencies(example_gemm_xdl example_gemm_xdl_fp16_fp8)

add_custom_target(example_gemm_wmma)
add_example_executable(example_gemm_wmma_fp16 gemm_wmma_fp16.cpp)
add_example_dependencies(example_gemm_wmma example_gemm_wmma_fp16)
add_example_executable(example_gemm_wmma_bf16 gemm_wmma_bf16.cpp)
add_example_dependencies(example_gemm_wmma example_gemm_wmma_bf16)
add_example_executable(example_gemm_wmma_int8 gemm_wmma_int8.cpp)
add_example_dependencies(example_gemm_wmma example_gemm_wmma_int8)

add_example_executable(example_gemm_wmma_fp16_nolds gemm_wmma_fp16_nolds.cpp)
add_example_executable(example_gemm_wmma_int8_nolds gemm_wmma_int8_nolds.cpp)
add_example_dependencies(example_gemm_wmma example_gemm_wmma_fp16_nolds)
add_example_dependencies(example_gemm_wmma example_gemm_wmma_int8_nolds)

if(SUPPORTED_GPU_TARGETS MATCHES "gfx13")
    add_example_executable(example_gemm_wmma_int8_kmultiplier gemm_wmma_int8_kmultiplier.cpp)
    add_example_executable(example_gemm_wmma_tr_load_fp16 gemm_wmma_tr_load_fp16.cpp)
    add_example_executable(example_gemm_wmma_async_load_fp16 gemm_wmma_async_load_fp16.cpp)
    add_example_dependencies(example_gemm_wmma example_gemm_wmma_int8_kmultiplier)
    add_example_dependencies(example_gemm_wmma example_gemm_wmma_tr_load_fp16)
    add_example_dependencies(example_gemm_wmma example_gemm_wmma_async_load_fp16)
    add_example_dependencies(example_gemm_wmma example_gemm_wmma_wavegroup_fp16)
    if (ENABLE_MX_TYPE)
        add_example_executable(example_gemm_wmma_mtx_fp8 gemm_wmma_mtx_fp8.cpp)
        add_example_dependencies(example_gemm_wmma example_gemm_wmma_mtx_fp8)

        add_example_executable(example_gemm_wmma_mtx_fp8_nolds gemm_wmma_mtx_fp8.cpp)
        target_compile_definitions(example_gemm_wmma_mtx_fp8_nolds PRIVATE GEMM_A_DISABLE_LDS=1)
        target_compile_definitions(example_gemm_wmma_mtx_fp8_nolds PRIVATE GEMM_B_DISABLE_LDS=1)
        add_example_dependencies(example_gemm_wmma example_gemm_wmma_mtx_fp8_nolds)
    endif()

    # wavegroup tests
    add_example_executable(example_gemm_wmma_wavegroup_fp16 gemm_wmma_wavegroup_fp16.cpp)
    target_compile_options(example_gemm_wmma_wavegroup_fp16 PRIVATE -save-temps=obj -Wno-gnu-line-marker)
    add_example_executable(example_gemm_wmma_wavegroup_fp16_nolds gemm_wmma_wavegroup_fp16.cpp)
    target_compile_options(example_gemm_wmma_wavegroup_fp16_nolds PRIVATE -save-temps=obj -Wno-gnu-line-marker)
    target_compile_definitions(example_gemm_wmma_wavegroup_fp16_nolds PRIVATE GEMM_A_DISABLE_LDS=1)
    target_compile_definitions(example_gemm_wmma_wavegroup_fp16_nolds PRIVATE GEMM_B_DISABLE_LDS=1)

    add_example_executable(example_gemm_wmma_async_load_wavegroup_fp16 gemm_wmma_async_load_wavegroup_fp16.cpp)
    target_compile_options(example_gemm_wmma_async_load_wavegroup_fp16 PRIVATE -save-temps=obj -Wno-gnu-line-marker)
    add_example_dependencies(example_gemm_wmma example_gemm_wmma_wavegroup_fp16)
    add_example_dependencies(example_gemm_wmma example_gemm_wmma_wavegroup_fp16_nolds)
    add_example_dependencies(example_gemm_wmma example_gemm_wmma_async_load_wavegroup_fp16)

    if (ENABLE_MX_TYPE)
        add_example_executable(example_gemm_wmma_mtx_wavegroup_fp8 gemm_wmma_mtx_wavegroup_fp8.cpp)
        target_compile_options(example_gemm_wmma_mtx_wavegroup_fp8 PRIVATE -save-temps=obj -Wno-gnu-line-marker)
        add_example_executable(example_gemm_wmma_mtx_wavegroup_fp8_nolds gemm_wmma_mtx_wavegroup_fp8.cpp)
        target_compile_options(example_gemm_wmma_mtx_wavegroup_fp8_nolds PRIVATE -save-temps=obj -Wno-gnu-line-marker)
        target_compile_definitions(example_gemm_wmma_mtx_wavegroup_fp8_nolds PRIVATE GEMM_A_DISABLE_LDS=1)
        target_compile_definitions(example_gemm_wmma_mtx_wavegroup_fp8_nolds PRIVATE GEMM_B_DISABLE_LDS=1)

        add_example_dependencies(example_gemm_wmma example_gemm_wmma_mtx_wavegroup_fp8)
        add_example_dependencies(example_gemm_wmma example_gemm_wmma_mtx_wavegroup_fp8_nolds)
    endif()
endif()
