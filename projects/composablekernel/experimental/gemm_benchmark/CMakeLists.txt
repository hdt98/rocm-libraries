
set(GEMM_XDL_V3_BENCHMARK gemm_xdl_v3_benchmark.cpp)
set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
generate_sharded_instantiations(
  INSTANCES_NAME example_gemm_xdl_v3_benchmark_instances
  TEMPLATE_FILE example_gemm_xdl_v3_benchmark_instances.in
  NUM_SHARDS 160
  SRC_LIST GEMM_XDL_V3_BENCHMARK
  OUTPUT_DIR ${GENERATED_DIR}
)
add_custom_target(example_gemm_xdl_v3_benchmark)

set(EXAMPLE_GEMM_COMPILE_OPTIONS)
#list(APPEND EXAMPLE_GEMM_COMPILE_OPTIONS "SHELL: -mllvm -greedy-reverse-local-assignment=1 -mllvm -enable-noalias-to-md-conversion=0")
#list(APPEND EXAMPLE_GEMM_COMPILE_OPTIONS -Wno-unused-local-typedef)
list(APPEND EXAMPLE_GEMM_COMPILE_OPTIONS --save-temps=obj -Wno-gnu-line-marker)
list(APPEND EXAMPLE_GEMM_COMPILE_OPTIONS -mllvm -amdgpu-hard-clause-length-limit=1)
#list(APPEND EXAMPLE_GEMM_COMPILE_OPTIONS "SHELL: -Rpass-analysis=kernel-resource-usage ")

function(add_example_gemm_xdl_v3_benchmark PrecDataType ALayout BLayout)
    set(EXE_NAME example_gemm_xdl_v3_benchmark_${PrecDataType}_${ALayout}_${BLayout})
    add_example_executable(${EXE_NAME} ${GEMM_XDL_V3_BENCHMARK})
    add_example_dependencies(example_gemm_xdl_v3_benchmark ${EXE_NAME})
    example_compile_options(${EXE_NAME} PRIVATE -Wno-global-constructors)
    example_compile_options(${EXE_NAME} PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
    if (TARGET  ${EXE_NAME})
       target_compile_definitions(${EXE_NAME} PRIVATE PREC_DATATYPE=${PrecDataType})
       target_compile_definitions(${EXE_NAME} PRIVATE A_LAYOUT=${ALayout})
       target_compile_definitions(${EXE_NAME} PRIVATE B_LAYOUT=${BLayout})
    endif()
endfunction(add_example_gemm_xdl_v3_benchmark)

add_example_gemm_xdl_v3_benchmark(fp16 Row Col)
add_example_gemm_xdl_v3_benchmark(fp16 Row Row)
add_example_gemm_xdl_v3_benchmark(fp16 Col Col)
add_example_gemm_xdl_v3_benchmark(fp16 Col Row)
add_example_gemm_xdl_v3_benchmark(i8 Row Col)
add_example_gemm_xdl_v3_benchmark(fp8 Row Col)
add_example_gemm_xdl_v3_benchmark(fp8 Row Row)
add_example_gemm_xdl_v3_benchmark(pk_i4 Row Col)
#add_example_gemm_xdl_v3_benchmark(pk_fp4 Row Col)

add_example_executable(example_gemm_xdl_ck_tile_wrap_fp16 gemm_xdl_ck_tile_wrap_fp16.cpp)
example_compile_options(example_gemm_xdl_ck_tile_wrap_fp16 PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
add_example_executable(example_gemm_xdl_ck_tile_wrap_fp16_async gemm_xdl_ck_tile_wrap_fp16_async.cpp)
example_compile_options(example_gemm_xdl_ck_tile_wrap_fp16_async PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
add_example_executable(example_gemm_xdl_ck_tile_wrap_fp16_tdm gemm_xdl_ck_tile_wrap_fp16_tdm.cpp)
example_compile_options(example_gemm_xdl_ck_tile_wrap_fp16_tdm PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})