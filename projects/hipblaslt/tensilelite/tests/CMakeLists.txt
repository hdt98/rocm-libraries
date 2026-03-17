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

# Helper: get default test sizes per type. f32 uses non-tile-aligned sizes
# that differ from f16/bf16 to increase coverage diversity.
macro(get_test_sizes TYPE OUT_SIZES)
    if("${TYPE}" STREQUAL "f32")
        set(${OUT_SIZES} -M 33 -N 47 -K 65)
    else()
        set(${OUT_SIZES} -M 31 -N 41 -K 51)
    endif()
endmacro()

# Helper: convert transpose tag (NN/TN/NT/TT) to flags
macro(get_trans_flags TRANS OUT_FLAGS)
    set(${OUT_FLAGS} "")
    if("${TRANS}" MATCHES "^T")
        list(APPEND ${OUT_FLAGS} --transA)
    endif()
    if("${TRANS}" MATCHES "T$")
        list(APPEND ${OUT_FLAGS} --transB)
    endif()
endmacro()

# 2-6. Basic sanity, transpose combos, odd sizes, type coverage
add_cpu_gemm_test(F32_Sanity
    ARGS -M 31 -N 41 -K 51 --type f32 --validate
)
add_cpu_gemm_test(F32_Odd_Sizes
    ARGS -M 2 -N 3 -K 5 --type f32 --validate
)
foreach(TYPE f32 f16 bf16)
    get_test_sizes(${TYPE} SIZES)
    foreach(TRANS NN TN NT TT)
        get_trans_flags(${TRANS} TFLAGS)
        add_cpu_gemm_test(${TYPE}_${TRANS}
            ARGS ${SIZES} ${TFLAGS} --type ${TYPE} --validate
        )
    endforeach()
endforeach()

# 7. Fast path on/off
add_cpu_gemm_test(FastPath_On
    ARGS -M 256 -N 256 -K 256 --tryFastPath --validate --type f32
)
add_cpu_gemm_test(FastPath_Off
    ARGS -M 256 -N 256 -K 256 --validate --type f32
)

# 8. Cross-validate: features per type
foreach(TYPE f32 f16 bf16)
    get_test_sizes(${TYPE} SIZES)
    add_cpu_gemm_test(${TYPE}_Beta
        ARGS ${SIZES} --type ${TYPE} --beta 1.0 --validate
    )
    add_cpu_gemm_test(${TYPE}_Bias
        ARGS ${SIZES} --type ${TYPE} --validate --bias
    )
    add_cpu_gemm_test(${TYPE}_AllFeatures
        ARGS ${SIZES} --type ${TYPE} --validate --bias --activation relu --scaleAlphaVec
    )
    add_cpu_gemm_test(${TYPE}_TN_AllFeatures
        ARGS ${SIZES} --transA --type ${TYPE} --validate --bias --activation relu --scaleAlphaVec
    )
endforeach()

# 9. F32-only individual feature tests
add_cpu_gemm_test(f32_Relu
    ARGS -M 33 -N 47 -K 65 --type f32 --validate --activation relu
)
add_cpu_gemm_test(f32_ScaleAlphaVec
    ARGS -M 33 -N 47 -K 65 --type f32 --validate --scaleAlphaVec
)
add_cpu_gemm_test(f32_Beta_AllFeatures
    ARGS -M 33 -N 47 -K 65 --type f32 --beta 0.5 --validate --bias --activation relu --scaleAlphaVec
)

# 10. Edge cases: dimensions at or below tile size (BLOCK_M=32, BLOCK_N=32, BLOCK_K=8)
foreach(CASE "M1;1;47;65" "N1;33;1;65" "K1;33;47;1" "1x1;1;1;1" "ExactBlock;32;32;8")
    list(GET CASE 0 NAME)
    list(GET CASE 1 EM)
    list(GET CASE 2 EN)
    list(GET CASE 3 EK)
    add_cpu_gemm_test(f32_${NAME}
        ARGS -M ${EM} -N ${EN} -K ${EK} --type f32 --validate
    )
endforeach()
