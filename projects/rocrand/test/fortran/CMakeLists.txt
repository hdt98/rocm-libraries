# Fortran Wrapper Tests

set(test_rocrand "test_rocrand_fortran_wrapper")

# Build unit test (rocrand)
add_executable(${test_rocrand})
target_sources(${test_rocrand} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/fruit/fruit.f90"
    "${CMAKE_CURRENT_SOURCE_DIR}/fruit/fruit_util.f90"
    "${CMAKE_CURRENT_SOURCE_DIR}/fruit_helpers.f90"
    "${CMAKE_CURRENT_SOURCE_DIR}/fruit_test_driver_rocrand.f90"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_rocrand.f90"
)
target_include_directories(${test_rocrand}
    PRIVATE
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/library/fortran>"
)
target_link_libraries(${test_rocrand} PRIVATE rocrand_fortran hip::host)
if(HIP_COMPILER STREQUAL "nvcc")
    if(CMAKE_VERSION VERSION_LESS 3.17)
        find_package(CUDA REQUIRED)
    else()
        find_package(CUDAToolkit)
        set(CUDA_LIBRARIES CUDA::cudart)
    endif()
    target_link_libraries(${test_rocrand} PRIVATE ${CUDA_LIBRARIES})
endif()
set_target_properties(${test_rocrand}
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test"
)

add_relative_test(${test_rocrand} ${test_rocrand})
rocm_install(TARGETS ${test_rocrand} COMPONENT tests)
