# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

target_sources(rocroller-tests-api
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/LazySingletonAPITest.cpp"
)

target_link_libraries(rocroller-tests-api
    PRIVATE
        roc::rocroller-interface
        Catch2::Catch2WithMain
)
