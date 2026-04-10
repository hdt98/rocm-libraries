enable_language(HIP)

# ---- Map requested GPUs to logical ISAs ----
set(_hipconv_gpu_isa_pairs
    gfx942  cdna3
    gfx950  cdna4
)

function(hipconv_gpu_to_logical _gpu _out_logical)
    string(REGEX REPLACE ":.*" "" _gpu_base "${_gpu}")
    list(LENGTH _hipconv_gpu_isa_pairs _len)
    math(EXPR _last "${_len} - 1")
    foreach(_i RANGE 0 ${_last} 2)
        list(GET _hipconv_gpu_isa_pairs ${_i} _g)
        if(_gpu_base STREQUAL "${_g}")
            math(EXPR _j "${_i} + 1")
            list(GET _hipconv_gpu_isa_pairs ${_j} _l)
            set("${_out_logical}" "${_l}" PARENT_SCOPE)
            return()
        endif()
    endforeach()
    set("${_out_logical}" "" PARENT_SCOPE)
endfunction()

set(_hipconv_logical "")
set(HIPCONV_HAS_ARCH_CDNA3 0)
set(HIPCONV_HAS_ARCH_CDNA4 0)
foreach(_gpu IN LISTS CMAKE_HIP_ARCHITECTURES)
    string(STRIP "${_gpu}" _gpu)
    if(_gpu STREQUAL "")
        continue()
    endif()
    hipconv_gpu_to_logical("${_gpu}" _logical)
    if(NOT _logical STREQUAL "")
        list(APPEND _hipconv_logical "${_logical}")
    endif()
endforeach()
list(REMOVE_DUPLICATES _hipconv_logical)

# Per-ISA object libraries.
set(_arch_root "${CMAKE_CURRENT_SOURCE_DIR}/src/arch")

function(hipconv_add_arch_lib _logical _gpu)
    string(TOUPPER "${_logical}" _LOGICAL)
    set(HIPCONV_HAS_ARCH_${_LOGICAL} 1 PARENT_SCOPE)

    add_library(hipconv_arch_${_logical} OBJECT
        src/arch/${_logical}/grouped_backend.cpp
    )
    set_source_files_properties(
        src/arch/${_logical}/grouped_backend.cpp
        PROPERTIES LANGUAGE HIP
    )
    target_include_directories(hipconv_arch_${_logical} PRIVATE
        src include "${_arch_root}/${_logical}")
    target_link_libraries(hipconv_arch_${_logical} PRIVATE hip::device)
    set_target_properties(hipconv_arch_${_logical} PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        HIP_ARCHITECTURES ${_gpu}
    )
endfunction()

function(hipconv_logical_to_gpu _logical _out_gpu)
    list(LENGTH _hipconv_gpu_isa_pairs _len)
    math(EXPR _last "${_len} - 1")
    foreach(_i RANGE 0 ${_last} 2)
        math(EXPR _j "${_i} + 1")
        list(GET _hipconv_gpu_isa_pairs ${_j} _l)
        if(_logical STREQUAL "${_l}")
            list(GET _hipconv_gpu_isa_pairs ${_i} _g)
            set("${_out_gpu}" "${_g}" PARENT_SCOPE)
            return()
        endif()
    endforeach()
    message(FATAL_ERROR "hipconv: unknown logical ISA '${_logical}'")
endfunction()

# Only create arch object libs for ISAs that have source files.
set(_hipconv_active_logical "")
foreach(_logical IN LISTS _hipconv_logical)
    if(EXISTS "${_arch_root}/${_logical}")
        hipconv_logical_to_gpu("${_logical}" _gpu)
        hipconv_add_arch_lib("${_logical}" "${_gpu}")
        list(APPEND _hipconv_active_logical "${_logical}")
    endif()
endforeach()

# ---- hipconv static library ----
add_library(hipconv STATIC
    src/registry.cpp
    src/tolerance.cpp
    src/grouped/grouped_conv.cpp
)
set_target_properties(hipconv PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    HIP_ARCHITECTURES gfx950
)
set_source_files_properties(
    src/registry.cpp
    src/tolerance.cpp
    src/grouped/grouped_conv.cpp
    PROPERTIES LANGUAGE HIP
)
foreach(_logical IN LISTS _hipconv_active_logical)
    target_sources(hipconv PRIVATE $<TARGET_OBJECTS:hipconv_arch_${_logical}>)
endforeach()

target_include_directories(hipconv
    PUBLIC  include
    PRIVATE src
)
target_link_libraries(hipconv PUBLIC hip::device)
target_compile_options(hipconv PRIVATE -fvisibility=hidden -fvisibility-inlines-hidden)
target_compile_definitions(hipconv PUBLIC HIPCONV_BUILDING_LIBRARY)
target_compile_definitions(hipconv PRIVATE
    HIPCONV_HAS_ARCH_CDNA3=${HIPCONV_HAS_ARCH_CDNA3}
    HIPCONV_HAS_ARCH_CDNA4=${HIPCONV_HAS_ARCH_CDNA4}
)
