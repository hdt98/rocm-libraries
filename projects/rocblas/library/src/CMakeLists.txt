# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

target_sources(rocblas-helper
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/buildinfo.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/check_numerics_matrix.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/check_numerics_vector.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/handle.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/logging.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_auxiliary.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblas_ostream.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/utility.cpp"
)

if(ROCBLAS_ENABLE_TENSILE OR ROCBLAS_ENABLE_HIPBLASLT)
    add_library(rocblas-gemm-host OBJECT)
    add_library(rocblas::tensile-host ALIAS rocblas-gemm-host)

    target_link_libraries(rocblas-gemm-host
        PRIVATE
            hip::device
            hip::host
            rocblas::warnings
    )

    target_include_directories(rocblas-gemm-host
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}"
            "${CMAKE_CURRENT_SOURCE_DIR}/include"
            "${PROJECT_BINARY_DIR}/include"
            "${PROJECT_BINARY_DIR}/include/rocblas"
            "${PROJECT_BINARY_DIR}/include/rocblas/internal"
            "${PROJECT_SOURCE_DIR}/library/include"
            "${PROJECT_SOURCE_DIR}/library/include/internal"
    )

    target_compile_definitions(rocblas-gemm-host
        PRIVATE
            ROCBLAS_BETA_FEATURES_API
            ROCBLAS_INTERNAL_API
            ROCM_USE_FLOAT16
    )
# === MERGE: develop additions, resolve later ===
# set( rocblas64_blas1_source
#   src64/blas1/rocblas_iamax_iamin_kernels_64.cpp
#   src64/blas1/rocblas_iamin_64.cpp
#   src64/blas1/rocblas_iamin_batched_64.cpp
#   src64/blas1/rocblas_iamin_strided_batched_64.cpp
#   src64/blas1/rocblas_iamax_64.cpp
#   src64/blas1/rocblas_iamax_batched_64.cpp
#   src64/blas1/rocblas_iamax_strided_batched_64.cpp
#   src64/blas1/rocblas_axpy_64.cpp
#   src64/blas1/rocblas_axpy_kernels_64.cpp
#   src64/blas1/rocblas_axpy_batched_64.cpp
#   src64/blas1/rocblas_axpy_strided_batched_64.cpp
#   src64/blas1/rocblas_copy_64.cpp
#   src64/blas1/rocblas_copy_kernels_64.cpp
#   src64/blas1/rocblas_copy_batched_64.cpp
#   src64/blas1/rocblas_copy_strided_batched_64.cpp
#   src64/blas1/rocblas_asum_64.cpp
#   src64/blas1/rocblas_asum_nrm2_kernels_64.cpp
#   src64/blas1/rocblas_asum_batched_64.cpp
#   src64/blas1/rocblas_asum_strided_batched_64.cpp
#   src64/blas1/rocblas_dot_64.cpp
#   src64/blas1/rocblas_dot_kernels_64.cpp
#   src64/blas1/rocblas_dot_strided_batched_64.cpp
#   src64/blas1/rocblas_dot_batched_64.cpp
#   src64/blas1/rocblas_nrm2_64.cpp
#   src64/blas1/rocblas_nrm2_batched_64.cpp
#   src64/blas1/rocblas_nrm2_strided_batched_64.cpp
#   src64/blas1/rocblas_rot_64.cpp
#   src64/blas1/rocblas_rot_kernels_64.cpp
#   src64/blas1/rocblas_rot_batched_64.cpp
#   src64/blas1/rocblas_rot_strided_batched_64.cpp
#   src64/blas1/rocblas_rotg_64.cpp
#   src64/blas1/rocblas_rotg_kernels_64.cpp
#   src64/blas1/rocblas_rotg_batched_64.cpp
#   src64/blas1/rocblas_rotg_strided_batched_64.cpp
#   src64/blas1/rocblas_rotm_64.cpp
#   src64/blas1/rocblas_rotm_kernels_64.cpp
#   src64/blas1/rocblas_rotm_batched_64.cpp
#   src64/blas1/rocblas_rotm_strided_batched_64.cpp
#   src64/blas1/rocblas_rotmg_64.cpp
#   src64/blas1/rocblas_rotmg_kernels_64.cpp
#   src64/blas1/rocblas_rotmg_batched_64.cpp
#   src64/blas1/rocblas_rotmg_strided_batched_64.cpp
#   src64/blas1/rocblas_scal_64.cpp
#   src64/blas1/rocblas_scal_kernels_64.cpp
#   src64/blas1/rocblas_scal_batched_64.cpp
#   src64/blas1/rocblas_scal_strided_batched_64.cpp
#   src64/blas1/rocblas_swap_64.cpp
#   src64/blas1/rocblas_swap_kernels_64.cpp
#   src64/blas1/rocblas_swap_batched_64.cpp
#   src64/blas1/rocblas_swap_strided_batched_64.cpp
# )
#
# set( subdir_src64_list
#   ${rocblas64_ex_source}
#   ${rocblas64_ex_source_no_tensile}
#   ${rocblas64_blas3_source}
#   ${rocblas64_blas3_source_no_tensile}
#   ${rocblas64_blas2_source}
#   ${rocblas64_blas1_source}
#   ${rocblas64_auxiliary_source}
# )
#
# prepend_path( ".." rocblas_headers_public relative_rocblas_headers_public )
#
# add_library( rocblas
#   # ordered from generally slowest to compile to faster to increase parallelism
#   ${rocblas_blas3_source}
#   ${rocblas_blas3_source_no_tensile}
#   ${rocblas_blas2_source}
#   ${rocblas_ex_source}
#   ${rocblas_ex_source_no_tensile}
#   ${rocblas_blas1_source}
#   ${relative_rocblas_headers_public}
#   ${rocblas_auxiliary_source}
#   ${subdir_src64_list}
# )
#
# # Add Windows version resource to library
# if(WIN32 AND BUILD_SHARED_LIBS)
#   target_sources(rocblas PRIVATE "${PROJECT_BINARY_DIR}/version.rc")
# endif()
#
# foreach( file ${subdir_src64_list} )
#   SET_SOURCE_FILES_PROPERTIES( ${file} PROPERTIES COMPILE_DEFINITIONS ROCBLAS_INTERNAL_ILP64 )
# endforeach()
#
# message(STATUS "** NOTE: blas2/rocblas_ger_kernels.cpp is compiled with the verbose flag -v for QC purposes.")
# SET_SOURCE_FILES_PROPERTIES( blas2/rocblas_ger_kernels.cpp PROPERTIES COMPILE_FLAGS "-v" )
#
# #if( WIN32 )
# #  set_target_properties(rocblas_fortran PROPERTIES LINKER_LANGUAGE CXX)
# #  target_link_directories(rocblas_fortran PRIVATE "C:\\cygwin64\\lib\\gcc\\x86_64-pc-cygwin\\9.3.0" "C:\\cygwin64\\lib" "C:\\cygwin64\\lib\\w32api")
# #endif( )
#
# add_library( roc::rocblas ALIAS rocblas )
#
# rocblas_library_settings( rocblas )
#
# target_link_libraries( rocblas PRIVATE "-Xlinker --exclude-libs=ALL" ) # HIDE symbols
#
#
# if( BUILD_WITH_TENSILE )
#
#   if( BUILD_SHARED_LIBS )
#     target_link_libraries( rocblas PRIVATE TensileHost )
#   endif()
# === END MERGE ===

    if(ROCBLAS_ENABLE_TENSILE)
        target_link_libraries(rocblas-gemm-host PRIVATE roc::tensile-host)

        target_compile_definitions(rocblas-gemm-host
            PRIVATE
                BUILD_WITH_TENSILE
                ROCBLAS_TENSILE_LAZY_LOAD=1
                ROCBLAS_TENSILE_SEPARATE_ARCH=1
        )
    endif()

    if(ROCBLAS_ENABLE_HIPBLASLT)
        target_link_libraries(rocblas-gemm-host PRIVATE roc::hipblaslt)
        target_compile_definitions(rocblas-gemm-host PRIVATE BUILD_WITH_HIPBLASLT)
    endif()

    # NOTE: C++ standard inherited from global CMAKE_CXX_STANDARD=17 setting
    # NOTE: CXX_EXTENSIONS inherited from global CMAKE_CXX_EXTENSIONS=OFF setting

    # Compiler flags for rocblas-gemm-host
    target_compile_options(rocblas-gemm-host PRIVATE --offload-compress)
    target_compile_definitions(rocblas-gemm-host PRIVATE rocblas_EXPORTS)

    # POSITION_INDEPENDENT_CODE required for OBJECT library linking into shared library
    set_target_properties(rocblas-gemm-host
        PROPERTIES
            CXX_VISIBILITY_PRESET hidden
            POSITION_INDEPENDENT_CODE ON
            VISIBILITY_INLINES_HIDDEN ON
    )

    target_link_libraries(rocblas PRIVATE rocblas-gemm-host)

    if(ROCBLAS_ENABLE_TENSILE)
        target_sources(rocblas-gemm-host PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/tensile_host.cpp")
        if(ROCBLAS_ENABLE_ASAN)
            rocblas_target_configure_sanitizers(rocblas-gemm-host PRIVATE)
        endif()
    endif()

    if(ROCBLAS_ENABLE_HIPBLASLT)
        target_sources(rocblas PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt_host.cpp")
        # Link hipBLASLt to main rocblas target (not just rocblas-gemm-host)
        target_link_libraries(rocblas PRIVATE roc::hipblaslt)
    endif()
endif()

add_subdirectory(blas1)
add_subdirectory(blas2)
add_subdirectory(blas3)
add_subdirectory(blas_ex)
add_subdirectory(src64)

# generate rocblas_device_malloc.hpp header for device memory allocation
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/include/rocblas_device_malloc.hpp" "${PROJECT_BINARY_DIR}/include/rocblas/rocblas_device_malloc.hpp")

set(ROCBLAS_PROTO_TEMPLATES "${PROJECT_BINARY_DIR}/include/rocblas/internal/rocblas-exported-proto.hpp")

add_custom_command(OUTPUT ${ROCBLAS_PROTO_TEMPLATES}
  COMMAND ${Python3_EXECUTABLE} 
    "${CMAKE_CURRENT_SOURCE_DIR}/template-proto.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/"
    "${CMAKE_CURRENT_SOURCE_DIR}/blas3/Tensile/"
    "${CMAKE_CURRENT_SOURCE_DIR}/blas3/"
    "${CMAKE_CURRENT_SOURCE_DIR}/blas2/"
    "${CMAKE_CURRENT_SOURCE_DIR}/blas1/"
    "${CMAKE_CURRENT_SOURCE_DIR}/src64/blas3/"
    "${CMAKE_CURRENT_SOURCE_DIR}/src64/blas2/"
    "${CMAKE_CURRENT_SOURCE_DIR}/src64/blas1/" > ${ROCBLAS_PROTO_TEMPLATES}
  COMMENT "Generating prototypes from ${CMAKE_CURRENT_SOURCE_DIR}."
)
add_custom_target(rocblas_proto_templates ALL DEPENDS ${ROCBLAS_PROTO_TEMPLATES})
# === MERGE: develop additions, resolve later ===
#
# if( BUILD_WITH_TENSILE )
#   if (WIN32)
#     set( ROCBLAS_TENSILE_LIBRARY_DIR "\${CPACK_PACKAGING_INSTALL_PREFIX}/bin/rocblas" CACHE PATH "path to tensile library" )
#   else()
#     set( ROCBLAS_TENSILE_LIBRARY_DIR "\${CPACK_PACKAGING_INSTALL_PREFIX}${CMAKE_INSTALL_LIBDIR}/rocblas" CACHE PATH "path to tensile library" )
#   endif()
#   # For ASAN package, Tensile library files(which are not shared libraries) are not required
#   if( NOT ENABLE_ASAN_PACKAGING )
#     if( BUILD_SHARED_LIBS )
#       set( TENSILE_DATA_COMPONENT_NAME ${CMAKE_INSTALL_DEFAULT_COMPONENT_NAME} )
#     else()
#       set( TENSILE_DATA_COMPONENT_NAME devel )
#     endif()
#     # TensileManifest.txt is a build-time verification artifact whose contents
#     # are absolute paths into the build tree (e.g. /work/build-.../Tensile/library/...).
#     # Nothing at runtime reads it (rocBLAS opens TensileLibrary_lazy_<arch>.dat
#     # directly), and shipping it would also collide across kpack per-arch shards
#     # since the filename has no arch suffix.
#     rocm_install(
#       DIRECTORY ${CMAKE_BINARY_DIR}/Tensile/library
#       DESTINATION ${ROCBLAS_TENSILE_LIBRARY_DIR}
#       COMPONENT ${TENSILE_DATA_COMPONENT_NAME} # Use this cmake variable to be compatible with rocm-cmake 0.6 and 0.7
#       PATTERN "TensileManifest.txt" EXCLUDE)
#   endif()
# endif()
#
# if(NOT WIN32)
#   if(RUN_HEADER_TESTING)
#   # Compilation tests to ensure that header files work independently,
#   # and that public header files work across several languages
#   add_custom_command(
#     TARGET rocblas
#     POST_BUILD
#     COMMAND ${CMAKE_HOME_DIRECTORY}/header_compilation_tests.sh
#     WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
#   )
#   endif()
# endif()
# === END MERGE ===
