# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

find_package(GTest REQUIRED)

add_executable(tensilelite-tests)
add_subdirectory(configs)

set(TEST_DATA_DIR "${CMAKE_CURRENT_BINARY_DIR}/data")
file(MAKE_DIRECTORY "${TEST_DATA_DIR}")

foreach(FILE ${SOLUTION_LIBRARY_FILES})
    if(EXISTS "${FILE}.gz")
        execute_process(COMMAND gzip --keep --decompress --force "${FILE}.gz")
    endif()
    file(COPY ${FILE} DESTINATION "${TEST_DATA_DIR}")
endforeach()

target_sources(tensilelite-tests
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/AlmostEqual_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/ContractionSelectionLibrary_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/DataTypes_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/Predicates_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/TestData.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/GEMMKernelTest.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/GEMMKernelTest_impl.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/RunGEMMKernelTest.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/RunGEMMKernelTest_impl.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/TestData.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/TestUtils.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/RangeLibrary_test.cpp"
)

target_include_directories(tensilelite-tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(tensilelite-tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../client/include")

target_link_libraries(tensilelite-tests
    PUBLIC
        tensilelite::tensilelite-host
    PRIVATE
        hip::device
        ${CMAKE_DL_LIBS}
)

gtest_discover_tests(tensilelite-tests WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} TIMEOUT 60)

target_link_libraries(tensilelite-tests PUBLIC GTest::gtest)

if(HIPBLASLT_ENABLE_OPENMP)
    target_link_libraries(tensilelite-tests PRIVATE OpenMP::OpenMP_CXX)
endif()


enable_testing()

#################################################
#        Tests for cpu-gemm-driver              #
#################################################

function(add_cpu_gemm_test TEST_NAME)
    cmake_parse_arguments(ARG "" "" "ARGS" ${ARGN})

    add_test(
        NAME "CPUGemm.${TEST_NAME}"
        COMMAND cpu-gemm-driver ${ARG_ARGS}
    )
endfunction()

# 2. Basic Sanity Check (F32)
add_cpu_gemm_test(F32_Sanity
    ARGS -M 31 -N 41 -K 51 --type f32 --validate
)

# 3. Transpose Combinations (T/N, N/T, T/T)
add_cpu_gemm_test(F32_TransA
    ARGS -M 31 -N 41 -K 51 --transA --type f32 --validate
)
add_cpu_gemm_test(F32_TransB
    ARGS -M 31 -N 41 -K 51 --transB --type f32 --validate
)
add_cpu_gemm_test(F32_TransAB
    ARGS -M 31 -N 41 -K 51 --transA --transB --type f32 --validate
)

# 4. Small sizes
add_cpu_gemm_test(F32_Odd_Sizes
    ARGS -M 2 -N 3 -K 5 --type f32 --validate
)

# 5. BFloat16 Tests
# These verify your "cast to f32 -> compute -> cast back" logic
add_cpu_gemm_test(BF16_Standard
    ARGS -M 128 -N 128 -K 128 --type bf16 --validate
)

# 6. FP16 Tests
add_cpu_gemm_test(F16_Standard
    ARGS -M 128 -N 128 -K 128 --type f16 --validate
)

# 7. Verify fast flag on / off works
add_cpu_gemm_test(FastPath_On
    ARGS -M 256 -N 256 -K 256 --tryFastPath --validate --type f32
)
add_cpu_gemm_test(FastPath_Off
    ARGS -M 256 -N 256 -K 256 --validate --type f32
)

# 8. Cross-validate fast path against golden reference
#    Covers all transpose combos, types, features, and edge cases
#    supported by solveCPUFastInF32.

# 8a. Transpose combinations per type
add_cpu_gemm_test(CrossValidate_F32_NN
    ARGS -M 33 -N 47 -K 65 --type f32 --validate
)
add_cpu_gemm_test(CrossValidate_F32_TN
    ARGS -M 33 -N 47 -K 65 --transA --type f32 --validate
)
add_cpu_gemm_test(CrossValidate_F32_NT
    ARGS -M 33 -N 47 -K 65 --transB --type f32 --validate
)
add_cpu_gemm_test(CrossValidate_F32_TT
    ARGS -M 33 -N 47 -K 65 --transA --transB --type f32 --validate
)
add_cpu_gemm_test(CrossValidate_F16_NN
    ARGS -M 31 -N 41 -K 51 --type f16 --validate
)
add_cpu_gemm_test(CrossValidate_F16_TN
    ARGS -M 31 -N 41 -K 51 --transA --type f16 --validate
)
add_cpu_gemm_test(CrossValidate_F16_NT
    ARGS -M 31 -N 41 -K 51 --transB --type f16 --validate
)
add_cpu_gemm_test(CrossValidate_F16_TT
    ARGS -M 31 -N 41 -K 51 --transA --transB --type f16 --validate
)
add_cpu_gemm_test(CrossValidate_BF16_NN
    ARGS -M 31 -N 41 -K 51 --type bf16 --validate
)
add_cpu_gemm_test(CrossValidate_BF16_TN
    ARGS -M 31 -N 41 -K 51 --transA --type bf16 --validate
)
add_cpu_gemm_test(CrossValidate_BF16_NT
    ARGS -M 31 -N 41 -K 51 --transB --type bf16 --validate
)
add_cpu_gemm_test(CrossValidate_BF16_TT
    ARGS -M 31 -N 41 -K 51 --transA --transB --type bf16 --validate
)

# 8b. Beta != 0
add_cpu_gemm_test(CrossValidate_F32_Beta
    ARGS -M 33 -N 47 -K 65 --type f32 --beta 1.0 --validate
)
add_cpu_gemm_test(CrossValidate_F16_Beta
    ARGS -M 31 -N 41 -K 51 --type f16 --beta 1.0 --validate
)

# 8c. Features (f32)
add_cpu_gemm_test(CrossValidate_F32_Bias
    ARGS -M 33 -N 47 -K 65 --type f32 --validate --bias
)
add_cpu_gemm_test(CrossValidate_F32_Relu
    ARGS -M 33 -N 47 -K 65 --type f32 --validate --activation relu
)
add_cpu_gemm_test(CrossValidate_F32_ScaleAlphaVec
    ARGS -M 33 -N 47 -K 65 --type f32 --validate --scaleAlphaVec
)
add_cpu_gemm_test(CrossValidate_F32_AllFeatures
    ARGS -M 33 -N 47 -K 65 --type f32 --validate --bias --activation relu --scaleAlphaVec
)

# 8d. Features with non-f32 types
add_cpu_gemm_test(CrossValidate_F16_Bias
    ARGS -M 31 -N 41 -K 51 --type f16 --validate --bias
)
add_cpu_gemm_test(CrossValidate_F16_AllFeatures
    ARGS -M 31 -N 41 -K 51 --type f16 --validate --bias --activation relu --scaleAlphaVec
)
add_cpu_gemm_test(CrossValidate_BF16_Bias
    ARGS -M 31 -N 41 -K 51 --type bf16 --validate --bias
)
add_cpu_gemm_test(CrossValidate_BF16_AllFeatures
    ARGS -M 31 -N 41 -K 51 --type bf16 --validate --bias --activation relu --scaleAlphaVec
)

# 8e. Features with transpose
add_cpu_gemm_test(CrossValidate_F32_TN_AllFeatures
    ARGS -M 33 -N 47 -K 65 --transA --type f32 --validate --bias --activation relu --scaleAlphaVec
)
add_cpu_gemm_test(CrossValidate_F16_TN_AllFeatures
    ARGS -M 31 -N 41 -K 51 --transA --type f16 --validate --bias --activation relu --scaleAlphaVec
)

# 8f. Edge cases: dimensions smaller than tile size (BLOCK=32), M=1, N=1
add_cpu_gemm_test(CrossValidate_F32_M1
    ARGS -M 1 -N 47 -K 65 --type f32 --validate
)
add_cpu_gemm_test(CrossValidate_F32_N1
    ARGS -M 33 -N 1 -K 65 --type f32 --validate
)
add_cpu_gemm_test(CrossValidate_F32_K1
    ARGS -M 33 -N 47 -K 1 --type f32 --validate
)
add_cpu_gemm_test(CrossValidate_F32_1x1
    ARGS -M 1 -N 1 -K 1 --type f32 --validate
)
add_cpu_gemm_test(CrossValidate_F32_ExactBlock
    ARGS -M 32 -N 32 -K 8 --type f32 --validate
)
add_cpu_gemm_test(CrossValidate_F32_Beta_AllFeatures
    ARGS -M 33 -N 47 -K 65 --type f32 --beta 0.5 --validate --bias --activation relu --scaleAlphaVec
)
