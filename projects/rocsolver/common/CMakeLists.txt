# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_library(rocsolver-common INTERFACE)

target_include_directories(rocsolver-common INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_compile_definitions(rocsolver-common INTERFACE 
    __HIP_HCC_COMPAT_MODE__=1
)

if(WIN32)
    target_compile_definitions(rocsolver-common INTERFACE
        WIN32_LEAN_AND_MEAN
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
        __HIP_ROCclr__=1
        __HIP_PLATFORM_AMD__=1
    )
    target_compile_options(rocsolver-common INTERFACE
        -fms-extensions
        -fms-compatibility
        -Wno-ignored-attributes
        -Wno-unused-command-line-argument
    )
endif()

if(ROCSOLVER_ENABLE_WERROR)
    target_compile_options(rocsolver-common INTERFACE
        -Werror=vla
        -Werror=reorder
        -Werror=return-type
    )
endif()

if(ROCSOLVER_ENABLE_ASAN)
    target_compile_options(rocsolver-common INTERFACE
        -fsanitize=address
        -shared-libasan
    )
    target_link_options(rocsolver-common INTERFACE
        -fsanitize=address
        -shared-libasan
        -fuse-ld=lld
    )
endif()

if(ROCSOLVER_ENABLE_COVERAGE)
    target_compile_options(rocsolver-common INTERFACE
        -g
        -O0
        -fprofile-instr-generate
        -fcoverage-mapping
    )
    target_link_options(rocsolver-common INTERFACE
        -fprofile-instr-generate
    )
endif()

if(ROCSOLVER_ENABLE_COMPRESSED_DBG)
    target_compile_options(rocsolver-common INTERFACE
        "$<$<CONFIG:Debug>:-gz>"
        "$<$<CONFIG:RelWithDebInfo>:-gz>"
    )
endif()
