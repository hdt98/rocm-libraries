# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# ckProfiler
set(CK_PROFILER_OP_FILTER "" CACHE STRING "Filter for the operators to be profiled. Default is to include all")
set(CK_PROFILER_INSTANCE_FILTER "" CACHE STRING "Filter for the kernels instances to be profiled. Default is to be the same as the operator filter")
if (CK_PROFILER_OP_FILTER STREQUAL "")
  set(CK_PROFILER_OP_FILTER ".+")
endif()
if (CK_PROFILER_INSTANCE_FILTER STREQUAL "")
  set(CK_PROFILER_INSTANCE_FILTER ${CK_PROFILER_OP_FILTER})
endif()
message(STATUS "CK_PROFILER_OP_FILTER: ${CK_PROFILER_OP_FILTER}")
message(STATUS "CK_PROFILER_INSTANCE_FILTER: ${CK_PROFILER_INSTANCE_FILTER}")

# SCAFFOLD-421: Only build 3D conv profiler ops
set(PROFILER_OPS
    profile_grouped_conv_fwd.cpp
    # profile_grouped_conv_bwd_data.cpp
    # profile_grouped_conv_bwd_weight.cpp
)

set(PROFILER_SOURCES profiler.cpp)
foreach(SOURCE ${PROFILER_OPS})
  string(REGEX REPLACE "profile_(.+)\.cpp" "\\1" OP_NAME ${SOURCE})
  if (OP_NAME STREQUAL "") 
    message(FATAL_ERROR "Unexpected source file name: ${SOURCE}")
  endif()
  if("${OP_NAME}" MATCHES "${CK_PROFILER_OP_FILTER}")
    list(APPEND PROFILER_SOURCES ${SOURCE})
  endif()
endforeach()
message(VERBOSE "ckProfiler sources: ${PROFILER_SOURCES}")

set(PROFILER_EXECUTABLE ckProfiler)

add_executable(${PROFILER_EXECUTABLE} ${PROFILER_SOURCES})
target_compile_options(${PROFILER_EXECUTABLE} PRIVATE -Wno-global-constructors)
# flags to compress the library
if(NOT WIN32 AND ${hip_VERSION_FLAT} GREATER 600241132)
  message(DEBUG "Adding --offload-compress flag for ${PROFILER_EXECUTABLE}")
  target_compile_options(${PROFILER_EXECUTABLE} PRIVATE --offload-compress)
endif()


# SCAFFOLD-421: Only 3D conv instance libraries
set(DEVICE_INSTANCES "")
list(APPEND DEVICE_INSTANCES device_grouped_conv1d_fwd_instance)
list(APPEND DEVICE_INSTANCES device_grouped_conv2d_fwd_instance)
list(APPEND DEVICE_INSTANCES device_grouped_conv3d_fwd_instance)
# list(APPEND DEVICE_INSTANCES device_grouped_conv3d_fwd_convinvscale_instance)
# list(APPEND DEVICE_INSTANCES device_grouped_conv3d_fwd_convscale_add_instance)
# list(APPEND DEVICE_INSTANCES device_grouped_conv3d_fwd_convscale_relu_instance)
# list(APPEND DEVICE_INSTANCES device_grouped_conv3d_fwd_convscale_instance)
# list(APPEND DEVICE_INSTANCES device_grouped_conv3d_fwd_clamp_instance)
# list(APPEND DEVICE_INSTANCES device_grouped_conv3d_fwd_scale_instance)
# list(APPEND DEVICE_INSTANCES device_grouped_conv3d_fwd_scaleadd_scaleadd_relu_instance)
# list(APPEND DEVICE_INSTANCES device_grouped_conv3d_fwd_bias_clamp_instance)
# list(APPEND DEVICE_INSTANCES device_grouped_conv3d_fwd_bias_bnorm_clamp_instance)
# list(APPEND DEVICE_INSTANCES device_grouped_conv3d_fwd_bilinear_instance)
# list(APPEND DEVICE_INSTANCES device_grouped_conv3d_fwd_dynamic_op_instance)
# list(APPEND DEVICE_INSTANCES device_grouped_conv3d_bwd_data_instance)
# list(APPEND DEVICE_INSTANCES device_grouped_conv3d_bwd_weight_instance)
# list(APPEND DEVICE_INSTANCES device_grouped_convnd_bwd_weight_instance)
# list(APPEND DEVICE_INSTANCES device_conv3d_bwd_data_instance)

set(PROFILER_LIBS utility getopt::getopt)
foreach(LIB ${DEVICE_INSTANCES})
  string(REGEX REPLACE "device_(.+)_instance" "\\1" INSTANCE_NAME ${LIB})
  if (INSTANCE_NAME STREQUAL "")
    message(FATAL_ERROR "Unexpected kernel instance name: ${LIB}")
  endif()
  if("${INSTANCE_NAME}" MATCHES "${CK_PROFILER_INSTANCE_FILTER}")
    # Only link if the target was actually created
    if(TARGET ${LIB})
      list(APPEND PROFILER_LIBS ${LIB})
    else()
      message(VERBOSE "Skipping ${LIB} - no instances built for current GPU targets")
    endif()
  endif()
endforeach()
message(VERBOSE "ckProfiler libs: ${PROFILER_LIBS}")
target_link_libraries(${PROFILER_EXECUTABLE} PRIVATE ${PROFILER_LIBS})

rocm_install(TARGETS ${PROFILER_EXECUTABLE} COMPONENT profiler)