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
# ctest entry that would overlap with the filtered quick/comprehensive suites,
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
install(TARGETS hipdnn_gpu_ref_tests RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# -- ctest entries with gtest filters (disjoint, label-based grouping) --
# Each entry runs one prefix. Labels control which tiers include it.
# ctest -L standard  →  runs quick + standard entries (both carry "standard" label).
# No YAML needed — YAML can be adopted later for per-GPU exclusions or multiple binaries.
add_test(NAME hipdnn_gpu_ref_tests_quick
    COMMAND hipdnn_gpu_ref_tests --gtest_filter=Small*)
set_tests_properties(hipdnn_gpu_ref_tests_quick PROPERTIES
    LABELS "quick;standard;comprehensive;full;unit_test" TIMEOUT 300)

add_test(NAME hipdnn_gpu_ref_tests_standard
    COMMAND hipdnn_gpu_ref_tests --gtest_filter=Test*)
set_tests_properties(hipdnn_gpu_ref_tests_standard PROPERTIES
    LABELS "standard;comprehensive;full;unit_test" TIMEOUT 600)

add_test(NAME hipdnn_gpu_ref_tests_comprehensive
    COMMAND hipdnn_gpu_ref_tests --gtest_filter=Medium*)
set_tests_properties(hipdnn_gpu_ref_tests_comprehensive PROPERTIES
    LABELS "comprehensive;full" TIMEOUT 1800)

add_test(NAME hipdnn_gpu_ref_tests_full
    COMMAND hipdnn_gpu_ref_tests --gtest_filter=Large*)
set_tests_properties(hipdnn_gpu_ref_tests_full PROPERTIES
    LABELS "full" TIMEOUT 3600)

# -- Install staging: quick only --
# TheRock CI runs bare ctest on the installed CTestTestfile (no -L filter).
# Only quick entries are installed so comprehensive tests don't time out in CI.
# Other tiers run locally via: ctest -L standard / comprehensive / full
set(GPU_REF_INSTALL_CTEST_STAGING "${CMAKE_BINARY_DIR}/gpu_ref_install_ctest.cmake.staging"
    CACHE INTERNAL "Staging file for gpu_ref install-time ctest entries")
file(WRITE "${GPU_REF_INSTALL_CTEST_STAGING}"
    "add_test(hipdnn_gpu_ref_tests_quick \"../hipdnn_gpu_ref_tests\" --gtest_filter=Small*)\n"
    "set_tests_properties(hipdnn_gpu_ref_tests_quick PROPERTIES LABELS \"quick\" TIMEOUT 300)\n")
