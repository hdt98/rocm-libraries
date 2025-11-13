# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(NOT ROCPRIM_USE_HIPCXX)
    if(NOT DEFINED HIP_COMPILER)
        message(FATAL_ERROR "HIP_COMPILER is not defined. Please ensure find_package(hip) has been called.")
    endif()
    
    if(HIP_COMPILER STREQUAL "clang")
        if(NOT (HIP_CXX_COMPILER MATCHES ".*hipcc" OR HIP_CXX_COMPILER MATCHES ".*clang\\+\\+"))
            message(FATAL_ERROR "On ROCm platform 'hipcc' or HIP-aware Clang must be used as C++ compiler.")
        endif()
    else()
        message(FATAL_ERROR "HIP_COMPILER must be 'clang' (AMD ROCm platform)")
    endif()
endif()
