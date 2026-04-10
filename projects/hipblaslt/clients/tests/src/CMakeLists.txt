# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

target_sources(hipblaslt-test
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt_gtest_main.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt_test.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/matmul_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/auxiliary_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/matrix_transform_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt_gtest_ext_op.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_gtest.cpp
        # New coverage improvement tests
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_core_gemm_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_epilogue_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_scaling_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_layout_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_extended_ops_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_advanced_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_error_handling_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_performance_gtest.cpp
        # New comprehensive multi-GPU tests
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_complete_implementations_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_fp8_int8_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_grouped_gemm_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_streams_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_communication_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_algorithm_selection_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_combined_features_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_large_scale_gtest.cpp
        # Parameter variation tests (systematic testing of all hipBLASLt parameters)
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_matrix_dimensions_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_data_types_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_transpose_ops_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_scalar_params_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_epilogue_ops_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_batching_params_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_scaling_params_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_matrix_layout_variations_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_algorithm_params_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_auxiliary_outputs_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_stream_sync_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_device_params_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_edge_cases_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_leading_dim_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_problem_sizes_gtest.cpp
        # Real multi-GPU computation tests (distributed GEMM with CPU verification)
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_real_compute_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_real_params_gtest.cpp
        # Comprehensive real multi-GPU tests covering all parameters
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_real_data_types_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_real_transpose_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_real_scalars_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_real_epilogue_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_real_combinations_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_real_k_partition_gtest.cpp
        # Complete coverage: dimensions, batching, leading dims/strides
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_real_dimensions_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_real_batching_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_real_leading_stride_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_real_matrix_coverage_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_real_fp8_int8_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_gradient_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_transformer_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_scaling_efficiency_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_fault_tolerance_gtest.cpp
        # Gap-filling tests (topology, heterogeneous, dynamic/adaptive patterns)
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_topology_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_heterogeneous_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_dynamic_adaptive_gtest.cpp
        # Critical P0 gap tests (IMPLEMENTED - addresses critical production needs)
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_memory_management_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_synchronization_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_ml_workloads_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_checkpoint_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_collectives_gtest.cpp
)
