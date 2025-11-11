# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

target_sources(rocsolver-test-common PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/lapack_host_reference.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/rocsolver_test.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/clients_utility.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/program_options.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/client_environment_helpers.cpp"
)
