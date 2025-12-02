# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

add_library(common-test-utilities OBJECT)
add_library(rocroller::common-test-utilities ALIAS common-test-utilities)

target_link_libraries(common-test-utilities
    PUBLIC
        roc::rocroller-interface
        rocroller-private-interface # For testing internal functionality
        roc::mxDataGenerator
        hip::host
        OpenMP::OpenMP_CXX
        ${CBLAS_LIBRARIES}
)

if(ROCROLLER_ENABLE_ASAN)
    rocroller_target_configure_sanitizers(common-test-utilities PRIVATE)
endif()

if(ROCROLLER_ENABLE_TIMERS)
    target_compile_definitions(common-test-utilities PUBLIC ROCROLLER_ENABLE_TIMERS)
endif()

add_subdirectory(include)
add_subdirectory(src)
