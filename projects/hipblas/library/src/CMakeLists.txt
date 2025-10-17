# ########################################################################
# Copyright (C) 2016-2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ########################################################################

# ########################################################################
# A helper function to prefix a source list of files with a common path into a new list (non-destructive)
# ########################################################################
# function( prepend_path prefix source_list_of_files return_list_of_files )
#   foreach( file ${${source_list_of_files}} )
#     if(IS_ABSOLUTE ${file} )
#       list( APPEND new_list ${file} )
#     else( )
#       list( APPEND new_list ${prefix}/${file} )
#     endif( )
#   endforeach( )
#   set( ${return_list_of_files} ${new_list} PARENT_SCOPE )
# endfunction( )

# ########################################################################
# Main
# ########################################################################
# prepend_path( ".." hipblas_headers_public relative_hipblas_headers_public )
include(GenerateExportHeader)
find_package(hipblas-common REQUIRED)

if(HIPBLAS_BUILD_SHARED_LIBS OR BUILD_SHARED_LIBS)
  add_library(hipblas SHARED)
  #${relative_hipblas_headers_public}
else()
  add_library(hipblas STATIC)
  # set_target_properties(hipblas PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

add_library(roc::hipblas ALIAS hipblas)

# Add sources based on backend
target_sources(hipblas PRIVATE
  "hipblas_auxiliary.cpp"
)
if(HIPBLAS_ENABLE_HIP)
  target_sources(hipblas PRIVATE "amd_detail/hipblas.cpp")
else() # HIPBLAS_ENABLE_CUDA
  target_sources(hipblas PRIVATE "nvidia_detail/hipblas.cpp")
endif()

if(HIPBLAS_ENABLE_FORTRAN)
  add_library(hipblas_fortran
    "hipblas_module.f90"
  )
endif()

# target_compile_features(hipblas PUBLIC cxx_std_${HIPBLAS_CXX_STANDARD}) # placeholder
set_target_properties(hipblas PROPERTIES
  CXX_EXTENSIONS OFF
)

if(HIPBLAS_ENABLE_ASAN)
  # add_link_options(-fuse-ld=lld)
  target_link_options(hipblas PRIVATE -fuse-ld=lld)
  target_compile_options(hipblas PRIVATE -fsanitize=address)
  target_link_options(hipblas PRIVATE -fsanitize=address)
endif()
if(HIPBLAS_BUILD_COVERAGE OR BUILD_COVERAGE)
  target_compile_options( hipblas PRIVATE -fprofile-arcs -ftest-coverage )
  target_link_libraries( hipblas PRIVATE --coverage -lgcov)
endif()

target_link_libraries( hipblas PUBLIC roc::hipblas-common)

if(HIPBLAS_ENABLE_HIP)
  # list(APPEND static_depends PACKAGE rocblas)
  target_link_libraries( hipblas PRIVATE roc::rocblas )
  target_link_libraries( hipblas PUBLIC hip::host )

  if( HIPBLAS_BUILD_ROCSOLVER )
    # list(APPEND static_depends PACKAGE rocsolver)
    target_link_libraries( hipblas PRIVATE roc::rocsolver )
    target_compile_definitions(hipblas PUBLIC __HIP_PLATFORM_SOLVER__)
  endif( )
else()
  target_compile_definitions(hipblas PRIVATE ${HIPBLAS_HIP_PLATFORM_COMPILER_DEFINES} __HIP_PLATFORM_NVCC__ __HIP_PLATFORM_NVIDIA__)

  target_link_libraries(hipblas PRIVATE CUDA::cublas CUDA::cusolver) # check if hipblas builds with cuda

  # External header includes included as system files
  target_include_directories(hipblas
    SYSTEM PRIVATE
      $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>
  )
endif( )

# install needs to be in root cmake
target_include_directories( hipblas
  SYSTEM PRIVATE
    $<BUILD_INTERFACE:${ROCBLAS_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${ROCSOLVER_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
)

# Internal header includes
target_include_directories( hipblas
  PUBLIC  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/library/include>
          $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include/hipblas>
          $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
          $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
          $<BUILD_INTERFACE:${HIPBLAS-COMMON_INCLUDE_DIRS}>
  PRIVATE
          ${CMAKE_CURRENT_SOURCE_DIR}/include
          ${CMAKE_CURRENT_SOURCE_DIR}
)

rocm_set_soversion( hipblas ${hipblas_SOVERSION} )
# set_target_properties( hipblas PROPERTIES CXX_EXTENSIONS NO )
set_target_properties( hipblas PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )

if (WIN32)
  add_custom_command( TARGET hipblas POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_BINARY_DIR}/staging/$<TARGET_FILE_NAME:hipblas> ${PROJECT_BINARY_DIR}/clients/staging/$<TARGET_FILE_NAME:hipblas> )
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_custom_command( TARGET hipblas POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_BINARY_DIR}/staging/hipblas.pdb ${PROJECT_BINARY_DIR}/clients/staging/hipblas.pdb )
  endif()
endif()

<<<<<<< HEAD
=======
if(BUILD_CODE_COVERAGE)
  target_compile_options(hipblas PRIVATE -g -O0 -fprofile-instr-generate -fcoverage-mapping)
  target_link_options(hipblas PUBLIC -fprofile-instr-generate)
endif()

>>>>>>> upstream/develop
# Package that helps me set visibility for function names exported from shared library
set_target_properties( hipblas PROPERTIES CXX_VISIBILITY_PRESET "hidden" VISIBILITY_INLINES_HIDDEN ON )
generate_export_header( hipblas EXPORT_FILE_NAME ${PROJECT_BINARY_DIR}/include/hipblas/hipblas-export.h )

# Following Boost conventions of prefixing 'lib' on static built libraries, across all platforms
if(NOT HIPBLAS_BUILD_SHARED_LIBS)
  set_target_properties( hipblas PROPERTIES PREFIX "lib" )
<<<<<<< HEAD
endif()

=======
endif( )

############################################################
# Installation

rocm_install_targets(
  TARGETS hipblas
  INCLUDE
    ${CMAKE_BINARY_DIR}/include
)
#         PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_EXECUTE GROUP_READ WORLD_EXECUTE WORLD_READ

if(HIP_PLATFORM STREQUAL amd )
  rocm_export_targets(
    TARGETS roc::hipblas
    DEPENDS PACKAGE hip
    DEPENDS PACKAGE hipblas-common
    STATIC_DEPENDS ${static_depends}
    NAMESPACE roc::
  )
else( )
  rocm_export_targets(
    TARGETS roc::hipblas
    DEPENDS PACKAGE HIP
    DEPENDS PACKAGE hipblas-common
    NAMESPACE roc::
  )
endif( )

# Force installation of .f90 module files
rocm_install(FILES "hipblas_module.f90"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/hipblas")
>>>>>>> upstream/develop
