# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

find_package(CLI11 QUIET)
if(NOT CLI11_FOUND)
    if(ROCROLLER_ENABLE_FETCH)
        FetchContent_Declare(
            CLI11
            GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
            GIT_TAG main
        )
        FetchContent_MakeAvailable(CLI11)
    else()
        message(FATAL_ERROR "Failed to find CLI11")
    endif()
endif()

add_subdirectory(gemm)
add_subdirectory(codegen_stress)

add_custom_target(all_clients)
add_dependencies(all_clients rocroller-gemm rocroller-codegen-stress)
