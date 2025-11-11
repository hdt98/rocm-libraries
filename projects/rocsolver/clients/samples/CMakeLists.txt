# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# C samples
add_executable(example-c-basic)
target_sources(example-c-basic PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/example_basic.c"
)
target_compile_features(example-c-basic PRIVATE c_std_99)
target_link_libraries(example-c-basic PRIVATE roc::rocsolver)
rocm_install(TARGETS example-c-basic COMPONENT samples)

add_executable(example-c-graph)
target_sources(example-c-graph PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/example_graph.c"
)
target_compile_features(example-c-graph PRIVATE c_std_99)
target_link_libraries(example-c-graph PRIVATE roc::rocsolver)
rocm_install(TARGETS example-c-graph COMPONENT samples)

add_executable(example-c-hmm)
target_sources(example-c-hmm PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/example_hmm.c"
)
target_compile_features(example-c-hmm PRIVATE c_std_99)
target_link_libraries(example-c-hmm PRIVATE roc::rocsolver)
rocm_install(TARGETS example-c-hmm COMPONENT samples)

add_executable(example-c-batched)
target_sources(example-c-batched PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/example_batched.c"
)
target_compile_features(example-c-batched PRIVATE c_std_99)
target_link_libraries(example-c-batched PRIVATE roc::rocsolver)
rocm_install(TARGETS example-c-batched COMPONENT samples)

add_executable(example-c-strided-batched)
target_sources(example-c-strided-batched PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/example_strided_batched.c"
)
target_compile_features(example-c-strided-batched PRIVATE c_std_99)
target_link_libraries(example-c-strided-batched PRIVATE roc::rocsolver)
rocm_install(TARGETS example-c-strided-batched COMPONENT samples)

# C++ samples
add_executable(example-cpp-basic)
target_sources(example-cpp-basic PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/example_basic.cpp"
)
target_compile_features(example-cpp-basic PRIVATE cxx_std_11)
target_link_libraries(example-cpp-basic PRIVATE roc::rocsolver)
rocm_install(TARGETS example-cpp-basic COMPONENT samples)

add_executable(example-cpp-logging)
target_sources(example-cpp-logging PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/example_logging.cpp"
)
target_compile_features(example-cpp-logging PRIVATE cxx_std_11)
target_link_libraries(example-cpp-logging PRIVATE roc::rocsolver)
rocm_install(TARGETS example-cpp-logging COMPONENT samples)
