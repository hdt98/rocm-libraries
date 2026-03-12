# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

find_package(rocprofiler-sdk REQUIRED)
find_package(rocprofiler-sdk-roctx REQUIRED)

list(APPEND CMAKE_MODULE_PATH "${rocprofiler-sdk_DIR}/Modules")
find_package(libdw REQUIRED)

target_link_libraries(rocroller-tests-rocprofiler
    PRIVATE
        Catch2::Catch2WithMain
        rocprofiler-sdk::rocprofiler-sdk
        rocprofiler-sdk-roctx::rocprofiler-sdk-roctx
        amd_comgr
        libdw::libdw
        ${BLAS_LIBRARIES}
        rocroller::common-test-utilities
)

target_sources(rocroller-tests-rocprofiler
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/Agent.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/BankConflictTest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/DSObserverTest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/LDSTestUtils.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/MultipleKernelTest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/Utils.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/WaitcntTest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/../catch/CustomAssertions.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/../catch/TestContext.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/../catch/TestKernels.cpp"
)
