# MIT License
#
# Copyright (c) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Benchmarks

# Add a rocRAND benchmark executable and install it.
function(add_rocrand_benchmark benchmark_src legacy)
  get_filename_component(BENCHMARK_TARGET ${benchmark_src} NAME_WE)
  add_executable(${BENCHMARK_TARGET} ${benchmark_src})

  if(HIP_COMPILER STREQUAL "nvcc")
    set_property(TARGET ${BENCHMARK_TARGET} PROPERTY CUDA_STANDARD 14)
    set_source_files_properties(${benchmark_src} PROPERTIES LANGUAGE CUDA)
    target_compile_options(${BENCHMARK_TARGET}
      PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
    )
    target_link_libraries(${BENCHMARK_TARGET} ${CUDA_curand_LIBRARY})
  else()
    target_link_libraries(${BENCHMARK_TARGET} hip::device)
  endif()
  target_link_libraries(${BENCHMARK_TARGET} rocrand)
  if(NOT legacy)
    target_link_libraries(${BENCHMARK_TARGET} benchmark::benchmark)
  endif()

  set_target_properties(${BENCHMARK_TARGET}
      PROPERTIES
          RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/benchmark"
  )
  rocm_install(TARGETS ${BENCHMARK_TARGET} COMPONENT benchmarks)
endfunction()

# Copy DLLs and test support files at configure time on Windows.
# Uses file(COPY) at file scope rather than per-target POST_BUILD commands
# to avoid build-time races when parallel targets copy the same files.
# The file list is determined by glob at configure time in either approach;
# if HIP or other dependencies are rebuilt, a reconfigure is needed to pick
# up changes (consistent with the find_package(hip) contract).
if(WIN32)
  file(GLOB _test_support_files
    ${HIP_DIR}/bin/*.dll
    ${CMAKE_SOURCE_DIR}/rtest.*
  )
  if(_test_support_files)
    file(COPY ${_test_support_files} DESTINATION ${PROJECT_BINARY_DIR}/benchmark)
  endif()
endif()

add_rocrand_benchmark("benchmark_rocrand_host_api.cpp" OFF)
add_rocrand_benchmark("benchmark_rocrand_device_api.cpp" OFF)
if(HIP_COMPILER STREQUAL "nvcc")
  add_rocrand_benchmark("benchmark_curand_host_api.cpp" OFF)
  add_rocrand_benchmark("benchmark_curand_device_api.cpp" OFF)
endif()

if(BUILD_LEGACY_BENCHMARK)
  add_rocrand_benchmark("benchmark_rocrand_generate.cpp" ON)
  add_rocrand_benchmark("benchmark_rocrand_kernel.cpp"   ON)
  if(HIP_COMPILER STREQUAL "nvcc")
    add_rocrand_benchmark("benchmark_curand_generate.cpp" ON)
    add_rocrand_benchmark("benchmark_curand_kernel.cpp"   ON)
  endif()
endif()

if(BUILD_BENCHMARK_TUNING)
  add_subdirectory(tuning)
endif()
