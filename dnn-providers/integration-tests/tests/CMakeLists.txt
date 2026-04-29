# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

add_executable(hipdnn_integration_tests_unit_tests
    main.cpp
    TestPlatformUtils.cpp
    TestTestConfig.cpp
    TestTestSettings.cpp
    TestGpuReferenceGraphExecutor.cpp
    TestGpuConvolutionFwdSignatureKey.cpp
    TestGpuConvolutionFwdPlan.cpp
)

target_include_directories(hipdnn_integration_tests_unit_tests
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_include_directories(hipdnn_integration_tests_unit_tests SYSTEM PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/third_party/tomlplusplus)

target_link_libraries(hipdnn_integration_tests_unit_tests
    PRIVATE
        hip::host
        hipdnn_gpu_ref
        hipdnn_test_sdk
        hipdnn_plugin_sdk
        GTest::gtest
        Threads::Threads
        $<$<PLATFORM_ID:Windows>:shlwapi>
)

target_compile_options(hipdnn_integration_tests_unit_tests
    PRIVATE ${INTEGRATION_TESTS_WARNING_COMPILE_OPTIONS})

clang_tidy_check(hipdnn_integration_tests_unit_tests)

add_unit_test_target(hipdnn_integration_tests_unit_tests ${CMAKE_CURRENT_BINARY_DIR})

add_executable(hipdnn_gpu_ref_tests
    gpu_ref/main.cpp
    gpu_ref/TestGpuFpReferenceConvolution.cpp
    gpu_ref/TestGpuFpReferenceDgrad.cpp
    gpu_ref/TestGpuFpReferenceWgrad.cpp
    gpu_ref/TestGpuFpReferenceValidation.cpp
    gpu_ref/TestGpuRefHipError.cpp
    gpu_ref/TestGpuRefKernelCompiler.cpp
    gpu_ref/TestShallowDeviceOnlyMigratableMemory.cpp
    gpu_ref/TestShallowGpuTensor.cpp
)

target_link_libraries(hipdnn_gpu_ref_tests
    PRIVATE
        hipdnn_gpu_ref
        hipdnn_test_sdk
        hipdnn_plugin_sdk
        GTest::gtest
        Threads::Threads
        hip::host
)

target_compile_options(hipdnn_gpu_ref_tests
    PRIVATE ${INTEGRATION_TESTS_WARNING_COMPILE_OPTIONS})

clang_tidy_check(hipdnn_gpu_ref_tests)

# -- Infra setup (RPATH, install, test-name validation) --
# We don't use add_unit_test_target() here because it registers an unfiltered
# ctest entry that would overlap with the YAML-generated category suites,
# causing every test to run twice under ninja check / ninja unit-check.
set(CHECK_DEPENDS_GLOBAL ${CHECK_DEPENDS_GLOBAL} hipdnn_gpu_ref_tests
    CACHE INTERNAL "Accumulated global dependencies for test name validation" FORCE)
set(_gpu_ref_exe hipdnn_gpu_ref_tests${CMAKE_EXECUTABLE_SUFFIX})
set(CHECK_EXECUTABLE_PATHS_GLOBAL ${CHECK_EXECUTABLE_PATHS_GLOBAL}
    "${CMAKE_INSTALL_BINDIR}/${_gpu_ref_exe}"
    CACHE INTERNAL "Accumulated global check executable paths" FORCE)
set_target_properties(hipdnn_gpu_ref_tests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}"
    INSTALL_RPATH
        "\$ORIGIN/../${CMAKE_INSTALL_LIBDIR};\$ORIGIN/../${CMAKE_INSTALL_LIBDIR}/hipdnn_plugins/engines"
    INSTALL_RPATH_USE_LINK_PATH TRUE
    BUILD_RPATH_USE_ORIGIN TRUE)
set_property(GLOBAL APPEND PROPERTY INTEGRATION_TESTS_TEST_TARGETS hipdnn_gpu_ref_tests)
install(TARGETS hipdnn_gpu_ref_tests RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# -- YAML-based test categories (generates ctest entries with gtest filters) --
# quick and nightly are disjoint and together cover all tests. Only quick carries
# unit_test so ninja unit-check is fast; ninja check (bare ctest) runs both.
# See shared/ctest/README.md for the full architecture.
include(${ROCM_LIBRARIES_ROOT}/shared/ctest/TestCategories.cmake)
apply_test_category_labels(
    hipdnn_gpu_ref_tests
    "${CMAKE_CURRENT_SOURCE_DIR}/gpu_ref/test_categories.yaml"
    "${CMAKE_CURRENT_BINARY_DIR}")
