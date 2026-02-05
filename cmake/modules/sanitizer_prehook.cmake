# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Sanitizer prehook to propagate THEROCK_SANITIZER settings to BUILD_ADDRESS_SANITIZER

if(THEROCK_SANITIZER STREQUAL "ASAN")
    set(BUILD_ADDRESS_SANITIZER ON CACHE BOOL "Enable Address Sanitizer" FORCE)
    message(STATUS "sanitizer_prehook: Setting BUILD_ADDRESS_SANITIZER=ON due to THEROCK_SANITIZER=ASAN")
elseif(THEROCK_SANITIZER STREQUAL "HOST_ASAN")
    set(BUILD_ADDRESS_SANITIZER ON CACHE BOOL "Enable Address Sanitizer (host-only)" FORCE)
    message(STATUS "sanitizer_prehook: Setting BUILD_ADDRESS_SANITIZER=ON due to THEROCK_SANITIZER=HOST_ASAN")
else()
    set(BUILD_ADDRESS_SANITIZER OFF CACHE BOOL "Disable Address Sanitizer" FORCE)
    message(STATUS "sanitizer_prehook: Setting BUILD_ADDRESS_SANITIZER=OFF (THEROCK_SANITIZER is not ASAN or HOST_ASAN)")
endif()
