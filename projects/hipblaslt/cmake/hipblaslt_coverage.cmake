# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

target_compile_definitions(hipblaslt-clients-common PUBLIC HIPBLASLT_ENABLE_COVERAGE)
target_compile_options(hipblaslt PRIVATE -g -O0 -fprofile-instr-generate -fcoverage-mapping)
target_link_options(hipblaslt PRIVATE -fprofile-instr-generate)
target_compile_options(hipblaslt-clients-common PRIVATE -g -O0 -fprofile-instr-generate -fcoverage-mapping)
target_link_options(hipblaslt-clients-common PRIVATE -fprofile-instr-generate)
target_compile_options(hipblaslt-bench PRIVATE -g -O0 -fprofile-instr-generate -fcoverage-mapping)
target_link_options(hipblaslt-bench PRIVATE -fprofile-instr-generate)
target_compile_options(hipblaslt-test PRIVATE -g -O0 -fprofile-instr-generate -fcoverage-mapping -fvisibility=default)
target_link_options(hipblaslt-test PRIVATE -fprofile-instr-generate)

rocm_get_git_commit_tag(hipblaslt_COMMIT)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/code_coverage_tuning_file.txt"
            "Git Version: ${hipblaslt_COMMIT}\n"
            "    transA,transB,grouped_gemm,batch_count,m,n,k,alpha,lda,stride_a,beta,ldb,stride_b,ldc,stride_c,ldd,stride_d,"
            "a_type,b_type,c_type,d_type,compute_type,scaleA,scaleB,scaleC,scaleD,amaxD,activation_type,bias_vector,bias_type,"
            "aux_type,rotating_buffer,hipblaslt-Gflops,hipblaslt-GB/s,us,solution_index,gcnArchName,CUs\n"
            "    T,N,0,1,512,1024,13,1,13,6656,0,13,13312,512,524288,512,524288,f32_r,f32_r,f32_r,f32_r,xf32_r,0,0,0,0,0,none,"
            "1,f32_r,f32_r,512,464.651,69.1759,29.3371,532705,gfx942:sramecc+:xnack-,304"
)

set(coverage_dir "${CMAKE_CURRENT_BINARY_DIR}/coverage-report")
add_custom_command(
    OUTPUT "${coverage_dir}/coverage-test.stamp"
    DEPENDS sample_hipblaslt_basic_matmul_for_cov hipblaslt-test tensilelite-device-libraries
    COMMAND ${CMAKE_COMMAND} -E rm -rf "${coverage_dir}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${coverage_dir}/profraw"
    COMMAND ${CMAKE_COMMAND} -E env HIPBLASLT_BENCH_PERF=1
                                    HIPBLASLT_ENABLE_MARKER=2
                                    HIPBLASLT_LOG_MASK=64
                                    LLVM_PROFILE_VERBOSE=1
                                    GTEST_LISTENER=NO_PASS_LINE_IN_LOG
                                    LLVM_PROFILE_FILE="${coverage_dir}/profraw/hipblaslt-coverage_test_perf_64_%m.profraw"
                                    $<TARGET_FILE:hipblaslt-test>
                                    --gtest_filter="-*matmul_test*:*aux_test.conversion/pre_checkin_aux_rocblaslt_rocroller_host_func_f16_r*"
                                    --precompile=hipblaslt-test-precompile.db
                                    --output-on-failure
    COMMAND ${CMAKE_COMMAND} -E env HIPBLASLT_ENABLE_MARKER=2
                                    HIPBLASLT_LOG_MASK=128
                                    LLVM_PROFILE_VERBOSE=1
                                    LLVM_PROFILE_FILE="${coverage_dir}/profraw/hipblaslt-coverage_sample_hipblaslt_gemm_128_%m.profraw"
                                    $<TARGET_FILE:sample_hipblaslt_gemm>
                                    --output-on-failure
    COMMAND ${CMAKE_COMMAND} -E env HIPBLASLT_ENABLE_MARKER=2
                                    HIPBLASLT_LOG_MASK=128
                                    LLVM_PROFILE_VERBOSE=1
                                    LLVM_PROFILE_FILE="${coverage_dir}/profraw/hipblaslt-coverage_sample_hipblaslt_gemm_ext_128_%m.profraw"
                                    $<TARGET_FILE:sample_hipblaslt_gemm_ext>
                                    --output-on-failure
    COMMAND ${CMAKE_COMMAND} -E env HIPBLASLT_ENABLE_MARKER=2
                                    HIPBLASLT_LOG_MASK=128
                                    LLVM_PROFILE_VERBOSE=1
                                    LLVM_PROFILE_FILE="${coverage_dir}/profraw/hipblaslt-coverage_sample_hipblaslt_gemm_batched_128_%m.profraw"
                                    $<TARGET_FILE:sample_hipblaslt_gemm_batched>
                                    --output-on-failure
    COMMAND ${CMAKE_COMMAND} -E env HIPBLASLT_ENABLE_MARKER=2
                                    HIPBLASLT_LOG_MASK=128
                                    LLVM_PROFILE_VERBOSE=1
                                    LLVM_PROFILE_FILE="${coverage_dir}/profraw/hipblaslt-coverage_sample_hipblaslt_gemm_batched_ext_128_%m.profraw"
                                    $<TARGET_FILE:sample_hipblaslt_gemm_batched_ext>
                                    --output-on-failure
    COMMAND ${CMAKE_COMMAND} -E env HIPBLASLT_ENABLE_MARKER=2
                                    HIPBLASLT_LOG_MASK=128
                                    LLVM_PROFILE_VERBOSE=1
                                    LLVM_PROFILE_FILE="${coverage_dir}/profraw/hipblaslt-coverage_sample_hipblaslt_gemm_tuning_splitk_ext_128_%m.profraw"
                                    $<TARGET_FILE:sample_hipblaslt_gemm_tuning_splitk_ext>
                                    --output-on-failure
    COMMAND ${CMAKE_COMMAND} -E env HIPBLASLT_ENABLE_MARKER=2
                                    HIPBLASLT_LOG_MASK=128
                                    LLVM_PROFILE_VERBOSE=1
                                    LLVM_PROFILE_FILE="${coverage_dir}/profraw/hipblaslt-coverage_sample_hipblaslt_gemm_bias_128_%m.profraw"
                                    $<TARGET_FILE:sample_hipblaslt_gemm_bias>
                                    --output-on-failure
    COMMAND ${CMAKE_COMMAND} -E env HIPBLASLT_ENABLE_MARKER=2
                                    HIPBLASLT_LOG_MASK=128
                                    LLVM_PROFILE_VERBOSE=1
                                    LLVM_PROFILE_FILE="${coverage_dir}/profraw/hipblaslt-coverage_sample_hipblaslt_gemm_bias_ext_128_%m.profraw"
                                    $<TARGET_FILE:sample_hipblaslt_gemm_bias_ext>
                                    --output-on-failure
    COMMAND ${CMAKE_COMMAND} -E env HIPBLASLT_ENABLE_MARKER=2
                                    HIPBLASLT_LOG_MASK=128
                                    LLVM_PROFILE_VERBOSE=1
                                    LLVM_PROFILE_FILE="${coverage_dir}/profraw/hipblaslt-coverage_sample_hipblaslt_gemm_get_all_algos_128_%m.profraw"
                                    $<TARGET_FILE:sample_hipblaslt_gemm_get_all_algos>
                                    --output-on-failure
    COMMAND ${CMAKE_COMMAND} -E env HIPBLASLT_ENABLE_MARKER=2
                                    HIPBLASLT_LOG_MASK=128
                                    LLVM_PROFILE_VERBOSE=1
                                    LLVM_PROFILE_FILE="${coverage_dir}/profraw/hipblaslt-coverage_sample_hipblaslt_gemm_get_all_algos_ext_128_%m.profraw"
                                    $<TARGET_FILE:sample_hipblaslt_gemm_get_all_algos_ext>
                                    --output-on-failure
    COMMAND ${CMAKE_COMMAND} -E env HIPBLASLT_ENABLE_MARKER=2
                                    HIPBLASLT_LOG_MASK=128
                                    LLVM_PROFILE_VERBOSE=1
                                    LLVM_PROFILE_FILE="${coverage_dir}/profraw/hipblaslt-coverage_sample_hipblaslt_gemm_get_algo_by_index_ext_128_%m.profraw"
                                    $<TARGET_FILE:sample_hipblaslt_gemm_get_algo_by_index_ext>
                                    --output-on-failure
    COMMAND ${CMAKE_COMMAND} -E env HIPBLASLT_ENABLE_MARKER=2
                                    HIPBLASLT_LOG_MASK=128
                                    LLVM_PROFILE_VERBOSE=1
                                    LLVM_PROFILE_FILE="${coverage_dir}/profraw/hipblaslt-coverage_sample_hipblaslt_gemm_alphavec_ext_128_%m.profraw"
                                    $<TARGET_FILE:sample_hipblaslt_gemm_alphavec_ext>
                                    --output-on-failure
    COMMAND ${CMAKE_COMMAND} -E env HIPBLASLT_ENABLE_MARKER=2
                                    HIPBLASLT_LOG_MASK=128
                                    LLVM_PROFILE_VERBOSE=1
                                    LLVM_PROFILE_FILE="${coverage_dir}/profraw/hipblaslt-coverage_sample_hipblaslt_gemm_gelu_aux_bias_128_%m.profraw"
                                    $<TARGET_FILE:sample_hipblaslt_gemm_gelu_aux_bias>
                                    --output-on-failure
    COMMAND ${CMAKE_COMMAND} -E env HIPBLASLT_ENABLE_MARKER=2
                                    HIPBLASLT_LOG_MASK=128
                                    LLVM_PROFILE_VERBOSE=1
                                    LLVM_PROFILE_FILE="${coverage_dir}/profraw/hipblaslt-coverage_sample_hipblaslt_gemm_gelu_aux_bias_ext_128_%m.profraw"
                                    $<TARGET_FILE:sample_hipblaslt_gemm_gelu_aux_bias_ext>
                                    --output-on-failure
    COMMAND ${CMAKE_COMMAND} -E env HIPBLASLT_ENABLE_MARKER=2
                                    HIPBLASLT_LOG_MASK=128
                                    LLVM_PROFILE_VERBOSE=1
                                    LLVM_PROFILE_FILE="${coverage_dir}/profraw/hipblaslt-coverage_sample_hipblaslt_gemm_amax_128_%m.profraw"
                                    $<TARGET_FILE:sample_hipblaslt_gemm_amax>
                                    --output-on-failure
    COMMAND ${CMAKE_COMMAND} -E env HIPBLASLT_ENABLE_MARKER=2
                                    HIPBLASLT_LOG_MASK=128
                                    LLVM_PROFILE_VERBOSE=1
                                    LLVM_PROFILE_FILE="${coverage_dir}/profraw/hipblaslt-coverage_sample_hipblaslt_gemm_amax_ext_128_%m.profraw"
                                    $<TARGET_FILE:sample_hipblaslt_gemm_amax_ext>
                                    --output-on-failure
    COMMAND ${CMAKE_COMMAND} -E env HIPBLASLT_ENABLE_MARKER=2
                                    HIPBLASLT_LOG_MASK=128
                                    LLVM_PROFILE_VERBOSE=1
                                    LLVM_PROFILE_FILE="${coverage_dir}/profraw/hipblaslt-coverage_sample_hipblaslt_gemm_amax_with_scale_128_%m.profraw"
                                    $<TARGET_FILE:sample_hipblaslt_gemm_amax_with_scale>
                                    --output-on-failure
    COMMAND ${CMAKE_COMMAND} -E env HIPBLASLT_ENABLE_MARKER=2
                                    HIPBLASLT_LOG_MASK=128
                                    LLVM_PROFILE_VERBOSE=1
                                    LLVM_PROFILE_FILE="${coverage_dir}/profraw/hipblaslt-coverage_sample_hipblaslt_gemm_amax_with_scale_ext_128_%m.profraw"
                                    $<TARGET_FILE:sample_hipblaslt_gemm_amax_with_scale_ext>
                                    --output-on-failure
    COMMAND ${CMAKE_COMMAND} -E env HIPBLASLT_ENABLE_MARKER=2
                                    HIPBLASLT_LOG_MASK=128
                                    LLVM_PROFILE_VERBOSE=1
                                    LLVM_PROFILE_FILE="${coverage_dir}/profraw/hipblaslt-coverage_sample_hipblaslt_gemm_bgradb_128_%m.profraw"
                                    $<TARGET_FILE:sample_hipblaslt_gemm_bgradb>
                                    --output-on-failure
    COMMAND ${CMAKE_COMMAND} -E env HIPBLASLT_ENABLE_MARKER=2
                                    HIPBLASLT_LOG_MASK=128
                                    LLVM_PROFILE_VERBOSE=1
                                    LLVM_PROFILE_FILE="${coverage_dir}/profraw/hipblaslt-coverage_sample_hipblaslt_gemm_ext_bgradb_128_%m.profraw"
                                    $<TARGET_FILE:sample_hipblaslt_gemm_ext_bgradb>
                                    --output-on-failure
    COMMAND ${CMAKE_COMMAND} -E env HIPBLASLT_ENABLE_MARKER=2
                                    HIPBLASLT_LOG_MASK=128
                                    LLVM_PROFILE_VERBOSE=1
                                    LLVM_PROFILE_FILE="${coverage_dir}/profraw/hipblaslt-coverage_sample_hipblaslt_gemm_dgelu_bgrad_128_%m.profraw"
                                    $<TARGET_FILE:sample_hipblaslt_gemm_dgelu_bgrad>
                                    --output-on-failure
    COMMAND ${CMAKE_COMMAND} -E env HIPBLASLT_ENABLE_MARKER=2
                                    HIPBLASLT_LOG_MASK=128
                                    LLVM_PROFILE_VERBOSE=1
                                    LLVM_PROFILE_FILE="${coverage_dir}/profraw/hipblaslt-coverage_sample_hipblaslt_gemm_dgelu_bgrad_ext_128_%m.profraw"
                                    $<TARGET_FILE:sample_hipblaslt_gemm_dgelu_bgrad_ext>
                                    --output-on-failure
    COMMAND ${CMAKE_COMMAND} -E env HIPBLASLT_ENABLE_MARKER=2
                                    HIPBLASLT_LOG_MASK=128
                                    LLVM_PROFILE_VERBOSE=1
                                    LLVM_PROFILE_FILE="${coverage_dir}/profraw/hipblaslt-coverage_sample_hipblaslt_gemm_is_tuned_ext_128_%m.profraw"
                                    $<TARGET_FILE:sample_hipblaslt_gemm_is_tuned_ext>
                                    --output-on-failure
    COMMAND ${CMAKE_COMMAND} -E env HIPBLASLT_ENABLE_MARKER=2
                                    HIPBLASLT_LOG_MASK=128
                                    LLVM_PROFILE_VERBOSE=1
                                    LLVM_PROFILE_FILE="${coverage_dir}/profraw/hipblaslt-coverage_sample_hipblaslt_gemm_tuning_wgm_ext_128_%m.profraw"
                                    $<TARGET_FILE:sample_hipblaslt_gemm_tuning_wgm_ext>
                                    --output-on-failure
    COMMAND ${CMAKE_COMMAND} -E env HIPBLASLT_ENABLE_MARKER=2
                                    HIPBLASLT_LOG_MASK=128
                                    LLVM_PROFILE_VERBOSE=1
                                    LLVM_PROFILE_FILE="${coverage_dir}/profraw/hipblaslt-coverage_sample_hipblaslt_gemm_with_scale_a_b_128_%m.profraw"
                                    $<TARGET_FILE:sample_hipblaslt_gemm_with_scale_a_b>
                                    --output-on-failure
    COMMAND ${CMAKE_COMMAND} -E env HIPBLASLT_ENABLE_MARKER=2
                                    HIPBLASLT_LOG_MASK=128
                                    LLVM_PROFILE_VERBOSE=1
                                    LLVM_PROFILE_FILE="${coverage_dir}/profraw/hipblaslt-coverage_sample_hipblaslt_gemm_with_scale_a_b_ext_128_%m.profraw"
                                    $<TARGET_FILE:sample_hipblaslt_gemm_with_scale_a_b_ext>
                                    --output-on-failure
    COMMAND ${CMAKE_COMMAND} -E env HIPBLASLT_ENABLE_MARKER=2
                                    HIPBLASLT_LOG_MASK=128
                                    LLVM_PROFILE_VERBOSE=1
                                    LLVM_PROFILE_FILE="${coverage_dir}/profraw/hipblaslt-coverage_sample_hipblaslt_groupedgemm_ext_128_%m.profraw"
                                    $<TARGET_FILE:sample_hipblaslt_groupedgemm_ext>
                                    --output-on-failure
    COMMAND ${CMAKE_COMMAND} -E env HIPBLASLT_ENABLE_MARKER=2
                                    HIPBLASLT_LOG_MASK=128
                                    LLVM_PROFILE_VERBOSE=1
                                    LLVM_PROFILE_FILE="${coverage_dir}/profraw/hipblaslt-coverage_sample_hipblaslt_groupedgemm_fixed_mk_ext_128_%m.profraw"
                                    $<TARGET_FILE:sample_hipblaslt_groupedgemm_fixed_mk_ext>
                                    --output-on-failure
    COMMAND ${CMAKE_COMMAND} -E env HIPBLASLT_ENABLE_MARKER=2
                                    HIPBLASLT_LOG_MASK=128
                                    LLVM_PROFILE_VERBOSE=1
                                    LLVM_PROFILE_FILE="${coverage_dir}/profraw/hipblaslt-coverage_sample_hipblaslt_groupedgemm_get_all_algos_ext_128_%m.profraw"
                                    $<TARGET_FILE:sample_hipblaslt_groupedgemm_get_all_algos_ext>
                                    --output-on-failure
    COMMAND ${CMAKE_COMMAND} -E env HIPBLASLT_ENABLE_MARKER=2
                                    HIPBLASLT_LOG_MASK=128
                                    LLVM_PROFILE_VERBOSE=1
                                    LLVM_PROFILE_FILE="${coverage_dir}/profraw/hipblaslt-coverage_sample_hipblaslt_gemm_mix_precision_128_%m.profraw"
                                    $<TARGET_FILE:sample_hipblaslt_gemm_mix_precision>
                                    --output-on-failure
    COMMAND ${CMAKE_COMMAND} -E env HIPBLASLT_ENABLE_MARKER=2
                                    HIPBLASLT_LOG_MASK=128
                                    LLVM_PROFILE_VERBOSE=1
                                    LLVM_PROFILE_FILE="${coverage_dir}/profraw/hipblaslt-coverage_sample_hipblaslt_gemm_mix_precision_ext_128_%m.profraw"
                                    $<TARGET_FILE:sample_hipblaslt_gemm_mix_precision_ext>
                                    --output-on-failure
    COMMAND ${CMAKE_COMMAND} -E env HIPBLASLT_ENABLE_MARKER=2
                                    HIPBLASLT_LOG_MASK=128
                                    LLVM_PROFILE_VERBOSE=1
                                    LLVM_PROFILE_FILE="${coverage_dir}/profraw/hipblaslt-coverage_sample_hipblaslt_gemm_mix_precision_with_amax_ext_128_%m.profraw"
                                    $<TARGET_FILE:sample_hipblaslt_gemm_mix_precision_with_amax_ext>
                                    --output-on-failure
    COMMAND ${CMAKE_COMMAND} -E env HIPBLASLT_ENABLE_MARKER=2
                                    HIPBLASLT_LOG_MASK=128
                                    LLVM_PROFILE_VERBOSE=1
                                    LLVM_PROFILE_FILE="${coverage_dir}/profraw/hipblaslt-coverage_sample_hipblaslt_gemm_attr_tciA_tciB_128_%m.profraw"
                                    $<TARGET_FILE:sample_hipblaslt_gemm_attr_tciA_tciB>
                                    --output-on-failure
    COMMAND ${CMAKE_COMMAND} -E env HIPBLASLT_ENABLE_MARKER=2
                                    HIPBLASLT_LOG_MASK=128
                                    LLVM_PROFILE_VERBOSE=1
                                    LLVM_PROFILE_FILE="${coverage_dir}/profraw/hipblaslt-coverage_sample_hipblaslt_ext_op_layernorm_128_%m.profraw"
                                    $<TARGET_FILE:sample_hipblaslt_ext_op_layernorm>
                                    --output-on-failure
    COMMAND ${CMAKE_COMMAND} -E env HIPBLASLT_ENABLE_MARKER=2
                                    HIPBLASLT_LOG_MASK=128
                                    LLVM_PROFILE_VERBOSE=1
                                    LLVM_PROFILE_FILE="${coverage_dir}/profraw/hipblaslt-coverage_sample_hipblaslt_ext_op_amax_128_%m.profraw"
                                    $<TARGET_FILE:sample_hipblaslt_ext_op_amax>
                                    --output-on-failure
    COMMAND ${CMAKE_COMMAND} -E env HIPBLASLT_ENABLE_MARKER=2
                                    HIPBLASLT_LOG_MASK=128
                                    LLVM_PROFILE_VERBOSE=1
                                    LLVM_PROFILE_FILE="${coverage_dir}/profraw/hipblaslt-coverage_sample_hipblaslt_gemm_with_TF32_128_%m.profraw"
                                    $<TARGET_FILE:sample_hipblaslt_gemm_with_TF32>
                                    --output-on-failure
    COMMAND ${CMAKE_COMMAND} -E env HIPBLASLT_ENABLE_MARKER=2
                                    HIPBLASLT_LOG_MASK=128
                                    LLVM_PROFILE_VERBOSE=1
                                    LLVM_PROFILE_FILE="${coverage_dir}/profraw/hipblaslt-coverage_sample_hipblaslt_gemm_swizzle_a_128_%m.profraw"
                                    $<TARGET_FILE:sample_hipblaslt_gemm_swizzle_a>
                                    --output-on-failure
    COMMAND ${CMAKE_COMMAND} -E env HIPBLASLT_ENABLE_MARKER=2
                                    HIPBLASLT_LOG_MASK=128
                                    LLVM_PROFILE_VERBOSE=1
                                    LLVM_PROFILE_FILE="${coverage_dir}/profraw/hipblaslt-coverage_sample_hipblaslt_gemm_bias_swizzle_a_ext_128_%m.profraw"
                                    $<TARGET_FILE:sample_hipblaslt_gemm_bias_swizzle_a_ext>
                                    --output-on-failure
    COMMAND ${CMAKE_COMMAND} -E env HIPBLASLT_ENABLE_MARKER=2
                                    HIPBLASLT_LOG_MASK=128
                                    LLVM_PROFILE_VERBOSE=1
                                    LLVM_PROFILE_FILE="${coverage_dir}/profraw/hipblaslt-coverage_sample_hipblaslt_weight_swizzle_padding_128_%m.profraw"
                                    $<TARGET_FILE:sample_hipblaslt_weight_swizzle_padding>
                                    --output-on-failure
    COMMAND ${CMAKE_COMMAND} -E env HIPBLASLT_ENABLE_MARKER=2
                                    HIPBLASLT_LOG_MASK=128
                                    LLVM_PROFILE_VERBOSE=1
                                    LLVM_PROFILE_FILE="${coverage_dir}/profraw/hipblaslt-coverage_sample_hipblaslt_gemm_swish_bias_128_%m.profraw"
                                    $<TARGET_FILE:sample_hipblaslt_gemm_swish_bias>
                                    --output-on-failure
    COMMAND ${CMAKE_COMMAND} -E env HIPBLASLT_ENABLE_MARKER=2
                                    HIPBLASLT_LOG_MASK=128
                                    LLVM_PROFILE_VERBOSE=1
                                    LLVM_PROFILE_FILE="${coverage_dir}/profraw/hipblaslt-coverage_sample_hipblaslt_basic_matmul_for_cov_128_%m.profraw"
                                    $<TARGET_FILE:sample_hipblaslt_basic_matmul_for_cov>
                                    --output-on-failure
    COMMAND ${CMAKE_COMMAND} -E touch "${coverage_dir}/coverage-test.stamp"
)
add_custom_target(
    coverage-test
    DEPENDS "${coverage_dir}/coverage-test.stamp"
)
find_program(
  LLVM_PROFDATA
  llvm-profdata
  HINTS llvm/bin
  REQUIRED
)
find_program(
  LLVM_COV
  llvm-cov
  REQUIRED
  HINTS llvm/bin
)
add_custom_command(
  OUTPUT "${coverage_dir}/coverage-merge.stamp"
  DEPENDS "${coverage_dir}/coverage-test.stamp"
  COMMAND ${LLVM_PROFDATA} merge -sparse "${coverage_dir}/profraw/hipblaslt-coverage_*.profraw" -o "${coverage_dir}/hipblaslt.profdata"
  COMMAND ${CMAKE_COMMAND} -E touch "${coverage_dir}/coverage-merge.stamp"
)
add_custom_target(
  coverage-merge
  DEPENDS "${coverage_dir}/coverage-merge.stamp"
)
set(hipblaslt_coverage_ignore_regex "'.*/clients/.*|.*/build/.*|.*/Tensile/Source/.*|.*/llvm/include/.*|.*/hipblaslt/hipblaslt_xfloat32.h|.*/include/rocblaslt/.*|.*tensilelite/.*|.*origami/.*|.*rocroller/.*|.*mxdatagenerator/.*'")
add_custom_command(
  OUTPUT "${coverage_dir}/coverage-report.stamp"
  DEPENDS "${coverage_dir}/coverage-merge.stamp"
  COMMAND ${LLVM_COV} report -object $<TARGET_FILE:hipblaslt> -instr-profile="${coverage_dir}/hipblaslt.profdata" --ignore-filename-regex=${hipblaslt_coverage_ignore_regex}
  COMMAND ${LLVM_COV} show -object $<TARGET_FILE:hipblaslt>
                           -object $<TARGET_FILE:hipblaslt-test>
                           -object $<TARGET_FILE:sample_hipblaslt_gemm>
                           -object $<TARGET_FILE:sample_hipblaslt_gemm_ext>
                           -object $<TARGET_FILE:sample_hipblaslt_gemm_batched>
                           -object $<TARGET_FILE:sample_hipblaslt_gemm_batched_ext>
                           -object $<TARGET_FILE:sample_hipblaslt_gemm_tuning_splitk_ext>
                           -object $<TARGET_FILE:sample_hipblaslt_gemm_bias>
                           -object $<TARGET_FILE:sample_hipblaslt_gemm_bias_ext>
                           -object $<TARGET_FILE:sample_hipblaslt_gemm_get_all_algos>
                           -object $<TARGET_FILE:sample_hipblaslt_gemm_get_all_algos_ext>
                           -object $<TARGET_FILE:sample_hipblaslt_gemm_get_algo_by_index_ext>
                           -object $<TARGET_FILE:sample_hipblaslt_gemm_alphavec_ext>
                           -object $<TARGET_FILE:sample_hipblaslt_gemm_gelu_aux_bias>
                           -object $<TARGET_FILE:sample_hipblaslt_gemm_gelu_aux_bias_ext>
                           -object $<TARGET_FILE:sample_hipblaslt_gemm_amax>
                           -object $<TARGET_FILE:sample_hipblaslt_gemm_amax_ext>
                           -object $<TARGET_FILE:sample_hipblaslt_gemm_amax_with_scale>
                           -object $<TARGET_FILE:sample_hipblaslt_gemm_amax_with_scale_ext>
                           -object $<TARGET_FILE:sample_hipblaslt_gemm_bgradb>
                           -object $<TARGET_FILE:sample_hipblaslt_gemm_ext_bgradb>
                           -object $<TARGET_FILE:sample_hipblaslt_gemm_dgelu_bgrad>
                           -object $<TARGET_FILE:sample_hipblaslt_gemm_dgelu_bgrad_ext>
                           -object $<TARGET_FILE:sample_hipblaslt_gemm_is_tuned_ext>
                           -object $<TARGET_FILE:sample_hipblaslt_gemm_tuning_wgm_ext>
                           -object $<TARGET_FILE:sample_hipblaslt_gemm_with_scale_a_b>
                           -object $<TARGET_FILE:sample_hipblaslt_gemm_with_scale_a_b_ext>
                           -object $<TARGET_FILE:sample_hipblaslt_groupedgemm_ext>
                           -object $<TARGET_FILE:sample_hipblaslt_groupedgemm_fixed_mk_ext>
                           -object $<TARGET_FILE:sample_hipblaslt_groupedgemm_get_all_algos_ext>
                           -object $<TARGET_FILE:sample_hipblaslt_gemm_mix_precision>
                           -object $<TARGET_FILE:sample_hipblaslt_gemm_mix_precision_ext>
                           -object $<TARGET_FILE:sample_hipblaslt_gemm_mix_precision_with_amax_ext>
                           -object $<TARGET_FILE:sample_hipblaslt_gemm_attr_tciA_tciB>
                           -object $<TARGET_FILE:sample_hipblaslt_ext_op_layernorm>
                           -object $<TARGET_FILE:sample_hipblaslt_ext_op_amax>
                           -object $<TARGET_FILE:sample_hipblaslt_gemm_with_TF32>
                           -object $<TARGET_FILE:sample_hipblaslt_gemm_swizzle_a>
                           -object $<TARGET_FILE:sample_hipblaslt_gemm_bias_swizzle_a_ext>
                           -object $<TARGET_FILE:sample_hipblaslt_weight_swizzle_padding>
                           -object $<TARGET_FILE:sample_hipblaslt_gemm_swish_bias>
                           -object $<TARGET_FILE:sample_hipblaslt_basic_matmul_for_cov>
                           -instr-profile="${coverage_dir}/hipblaslt.profdata"
                           -format=html
                           -output-dir=${coverage_dir}
                           --ignore-filename-regex=${hipblaslt_coverage_ignore_regex}
  COMMAND ${CMAKE_COMMAND} -E touch "${coverage_dir}/coverage-report.stamp"
)
add_custom_target(
  coverage
  DEPENDS "${coverage_dir}/coverage-report.stamp"
)
add_custom_target(
    show-coverage
    DEPENDS "${coverage_dir}/hipblaslt.profdata"
    COMMAND ${LLVM_COV} report -object $<TARGET_FILE:hipblaslt>
                               -object $<TARGET_FILE:hipblaslt-test>
                               -object $<TARGET_FILE:sample_hipblaslt_gemm>
                               -object $<TARGET_FILE:sample_hipblaslt_gemm_ext>
                               -object $<TARGET_FILE:sample_hipblaslt_gemm_batched>
                               -object $<TARGET_FILE:sample_hipblaslt_gemm_batched_ext>
                               -object $<TARGET_FILE:sample_hipblaslt_gemm_tuning_splitk_ext>
                               -object $<TARGET_FILE:sample_hipblaslt_gemm_bias>
                               -object $<TARGET_FILE:sample_hipblaslt_gemm_bias_ext>
                               -object $<TARGET_FILE:sample_hipblaslt_gemm_get_all_algos>
                               -object $<TARGET_FILE:sample_hipblaslt_gemm_get_all_algos_ext>
                               -object $<TARGET_FILE:sample_hipblaslt_gemm_get_algo_by_index_ext>
                               -object $<TARGET_FILE:sample_hipblaslt_gemm_alphavec_ext>
                               -object $<TARGET_FILE:sample_hipblaslt_gemm_gelu_aux_bias>
                               -object $<TARGET_FILE:sample_hipblaslt_gemm_gelu_aux_bias_ext>
                               -object $<TARGET_FILE:sample_hipblaslt_gemm_amax>
                               -object $<TARGET_FILE:sample_hipblaslt_gemm_amax_ext>
                               -object $<TARGET_FILE:sample_hipblaslt_gemm_amax_with_scale>
                               -object $<TARGET_FILE:sample_hipblaslt_gemm_amax_with_scale_ext>
                               -object $<TARGET_FILE:sample_hipblaslt_gemm_bgradb>
                               -object $<TARGET_FILE:sample_hipblaslt_gemm_ext_bgradb>
                               -object $<TARGET_FILE:sample_hipblaslt_gemm_dgelu_bgrad>
                               -object $<TARGET_FILE:sample_hipblaslt_gemm_dgelu_bgrad_ext>
                               -object $<TARGET_FILE:sample_hipblaslt_gemm_is_tuned_ext>
                               -object $<TARGET_FILE:sample_hipblaslt_gemm_tuning_wgm_ext>
                               -object $<TARGET_FILE:sample_hipblaslt_gemm_with_scale_a_b>
                               -object $<TARGET_FILE:sample_hipblaslt_gemm_with_scale_a_b_ext>
                               -object $<TARGET_FILE:sample_hipblaslt_groupedgemm_ext>
                               -object $<TARGET_FILE:sample_hipblaslt_groupedgemm_fixed_mk_ext>
                               -object $<TARGET_FILE:sample_hipblaslt_groupedgemm_get_all_algos_ext>
                               -object $<TARGET_FILE:sample_hipblaslt_gemm_mix_precision>
                               -object $<TARGET_FILE:sample_hipblaslt_gemm_mix_precision_ext>
                               -object $<TARGET_FILE:sample_hipblaslt_gemm_mix_precision_with_amax_ext>
                               -object $<TARGET_FILE:sample_hipblaslt_gemm_attr_tciA_tciB>
                               -object $<TARGET_FILE:sample_hipblaslt_ext_op_layernorm>
                               -object $<TARGET_FILE:sample_hipblaslt_ext_op_amax>
                               -object $<TARGET_FILE:sample_hipblaslt_gemm_with_TF32>
                               -object $<TARGET_FILE:sample_hipblaslt_gemm_swizzle_a>
                               -object $<TARGET_FILE:sample_hipblaslt_gemm_bias_swizzle_a_ext>
                               -object $<TARGET_FILE:sample_hipblaslt_weight_swizzle_padding>
                               -object $<TARGET_FILE:sample_hipblaslt_gemm_swish_bias>
                               -object $<TARGET_FILE:sample_hipblaslt_basic_matmul_for_cov>
                               -instr-profile="${coverage_dir}/hipblaslt.profdata"
                               --ignore-filename-regex${hipblaslt_coverage_ignore_regex}
)
