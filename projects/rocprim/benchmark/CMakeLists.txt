# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

include(ConfigAutotune)
include(ConfigAutotuneSettings)

option(ROCPRIM_BENCHMARK_USE_AMDSMI "Let benchmarks use AMD SMI to output more GPU statistics" OFF)
set(ROCPRIM_BENCHMARK_TUNE_PARAM_NAMES "" CACHE STRING "Tuning parameter names")
set(ROCPRIM_BENCHMARK_TUNE_PARAMS "" CACHE STRING "Tuning parameters")

if(ROCPRIM_ENABLE_CONFIG_TUNING)
    add_custom_target(benchmark_config_tuning)
endif()

function(add_rocprim_benchmark BENCHMARK_SOURCE)
    get_filename_component(BENCHMARK_TARGET ${BENCHMARK_SOURCE} NAME_WE)

    if(ROCPRIM_USE_HIPCXX)
        set_source_files_properties(${BENCHMARK_SOURCE} PROPERTIES LANGUAGE HIP)
    endif()

    if(ROCPRIM_ENABLE_CONFIG_TUNING)
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${BENCHMARK_TARGET}.parallel.cpp.in")
            message(STATUS "Found ${BENCHMARK_TARGET}.parallel.cpp.in file, compiling in parallel.")
            read_config_autotune_settings(${BENCHMARK_TARGET} list_across_names list_across output_pattern_suffix)

            if(ROCPRIM_BENCHMARK_TUNE_PARAM_NAMES AND ROCPRIM_BENCHMARK_TUNE_PARAMS)
                set(list_across_names "${ROCPRIM_BENCHMARK_TUNE_PARAM_NAMES}")
                set(list_across "${ROCPRIM_BENCHMARK_TUNE_PARAMS}")
            endif()

            if(list_across_names)
                add_executable(${BENCHMARK_TARGET} ${BENCHMARK_SOURCE})
                target_compile_definitions(${BENCHMARK_TARGET} PRIVATE BENCHMARK_CONFIG_TUNING)
                add_matrix(
                    TARGET ${BENCHMARK_TARGET}
                    SHARDS 1
                    CURRENT_SHARD 0
                    INPUT "${BENCHMARK_TARGET}.parallel.cpp.in"
                    OUTPUT_PATTERN "${BENCHMARK_TARGET}_${output_pattern_suffix}"
                    NAMES ${list_across_names}
                    LISTS ${list_across}
                )
                add_dependencies(benchmark_config_tuning ${BENCHMARK_TARGET})
            else()
                message(WARNING "No config-tuning entry in benchmark/ConfigAutotuneSettings.cmake for ${BENCHMARK_TARGET}!")
                return()
            endif()
        else()
            return()
        endif()
    else()
        add_executable(${BENCHMARK_TARGET} ${BENCHMARK_SOURCE})
    endif()

    if(ROCPRIM_ENABLE_NAIVE_BENCHMARK)
        target_compile_definitions(${BENCHMARK_TARGET} PRIVATE BUILD_NAIVE_BENCHMARK)
    endif()

    if(ROCPRIM_ENABLE_AUTOTUNED_TYPES_ONLY)
        target_compile_definitions(${BENCHMARK_TARGET} PRIVATE BENCHMARK_AUTOTUNED_TYPES_ONLY)
    endif()

    target_link_libraries(${BENCHMARK_TARGET}
        PRIVATE
            roc::rocprim
            benchmark::benchmark
    )

    if(ROCPRIM_USE_HIPCXX)
        target_link_libraries(${BENCHMARK_TARGET}
            PRIVATE
                $<IF:$<LINK_LANGUAGE:HIP>,hip::host,hip::device>
        )
    else()
        target_link_libraries(${BENCHMARK_TARGET}
            PRIVATE
                hip::device
        )
    endif()

    if(ROCPRIM_BENCHMARK_USE_AMDSMI)
        find_library(AMDSMI_LIB NAMES amd_smi REQUIRED)
        target_link_libraries(${BENCHMARK_TARGET} PRIVATE ${AMDSMI_LIB})
        target_compile_definitions(${BENCHMARK_TARGET} PRIVATE BENCHMARK_USE_AMDSMI)
    endif()

    target_compile_options(${BENCHMARK_TARGET}
        PRIVATE
            $<$<CXX_COMPILER_ID:MSVC>:/bigobj>
    )

    set_target_properties(${BENCHMARK_TARGET}
        PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/benchmark"
    )

    if(ROCPRIM_ENABLE_INSTALL)
        rocm_install(TARGETS ${BENCHMARK_TARGET} COMPONENT benchmarks)
    endif()

    if(WIN32 AND NOT DEFINED DLLS_COPIED)
        set(DLLS_COPIED "YES" PARENT_SCOPE)
        file(GLOB third_party_dlls
            LIST_DIRECTORIES ON
            CONFIGURE_DEPENDS
            ${HIP_DIR}/bin/*.dll
            ${CMAKE_SOURCE_DIR}/rtest.*
        )
        foreach(file_i ${third_party_dlls})
            add_custom_command(
                TARGET ${BENCHMARK_TARGET} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different ${file_i} ${PROJECT_BINARY_DIR}/benchmark
            )
        endforeach()
    endif()
endfunction()

# Benchmarks
add_rocprim_benchmark(benchmark_block_adjacent_difference.cpp)
add_rocprim_benchmark(benchmark_block_discontinuity.cpp)
add_rocprim_benchmark(benchmark_block_exchange.cpp)
add_rocprim_benchmark(benchmark_block_histogram.cpp)
add_rocprim_benchmark(benchmark_block_radix_sort.cpp)
add_rocprim_benchmark(benchmark_block_radix_rank.cpp)
add_rocprim_benchmark(benchmark_block_reduce.cpp)
add_rocprim_benchmark(benchmark_block_run_length_decode.cpp)
add_rocprim_benchmark(benchmark_block_scan.cpp)
add_rocprim_benchmark(benchmark_block_sort.cpp)
add_rocprim_benchmark(benchmark_config_dispatch.cpp)
add_rocprim_benchmark(benchmark_device_adjacent_difference.cpp)
add_rocprim_benchmark(benchmark_device_adjacent_find.cpp)
add_rocprim_benchmark(benchmark_device_batch_memcpy.cpp)
add_rocprim_benchmark(benchmark_device_binary_search.cpp)
add_rocprim_benchmark(benchmark_device_find_first_of.cpp)
add_rocprim_benchmark(benchmark_device_find_end.cpp)
add_rocprim_benchmark(benchmark_device_histogram.cpp)
add_rocprim_benchmark(benchmark_device_merge.cpp)
add_rocprim_benchmark(benchmark_device_merge_inplace.cpp)
add_rocprim_benchmark(benchmark_device_merge_sort.cpp)
add_rocprim_benchmark(benchmark_device_merge_sort_block_sort.cpp)
add_rocprim_benchmark(benchmark_device_merge_sort_block_merge.cpp)
add_rocprim_benchmark(benchmark_device_nth_element.cpp)
add_rocprim_benchmark(benchmark_device_partial_sort.cpp)
add_rocprim_benchmark(benchmark_device_partial_sort_copy.cpp)
add_rocprim_benchmark(benchmark_device_partition.cpp)
add_rocprim_benchmark(benchmark_device_radix_sort.cpp)
add_rocprim_benchmark(benchmark_device_radix_sort_block_sort.cpp)
add_rocprim_benchmark(benchmark_device_radix_sort_onesweep.cpp)
add_rocprim_benchmark(benchmark_device_reduce_by_key.cpp)
add_rocprim_benchmark(benchmark_device_reduce_by_key_deterministic.cpp)
add_rocprim_benchmark(benchmark_device_reduce.cpp)
add_rocprim_benchmark(benchmark_device_run_length_encode.cpp)
add_rocprim_benchmark(benchmark_device_run_length_encode_non_trivial_runs.cpp)
add_rocprim_benchmark(benchmark_device_scan.cpp)
add_rocprim_benchmark(benchmark_device_scan_deterministic.cpp)
add_rocprim_benchmark(benchmark_device_scan_by_key.cpp)
add_rocprim_benchmark(benchmark_device_search.cpp)
add_rocprim_benchmark(benchmark_device_scan_by_key_deterministic.cpp)
add_rocprim_benchmark(benchmark_device_search_n.cpp)
add_rocprim_benchmark(benchmark_device_select.cpp)
add_rocprim_benchmark(benchmark_device_segmented_radix_sort_keys.cpp)
add_rocprim_benchmark(benchmark_device_segmented_radix_sort_pairs.cpp)
add_rocprim_benchmark(benchmark_device_segmented_reduce.cpp)
add_rocprim_benchmark(benchmark_device_transform.cpp)
add_rocprim_benchmark(benchmark_device_transform_pointer.cpp)
add_rocprim_benchmark(benchmark_predicate_iterator.cpp)
add_rocprim_benchmark(benchmark_warp_exchange.cpp)
add_rocprim_benchmark(benchmark_warp_reduce.cpp)
add_rocprim_benchmark(benchmark_warp_scan.cpp)
add_rocprim_benchmark(benchmark_warp_sort.cpp)
add_rocprim_benchmark(benchmark_device_memory.cpp)
