# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(AMDGPU_TEST_TARGETS "" CACHE STRING "List of specific device types to test for")

function(get_rocprim_test_target TEST_SOURCES TEST_TARGET)
    list(GET TEST_SOURCES 0 TEST_MAIN_SOURCE)
    get_filename_component(TEST_TARGET ${TEST_MAIN_SOURCE} NAME_WE)
    set(TEST_TARGET ${TEST_TARGET} PARENT_SCOPE)
endfunction()

function(add_rocprim_test TEST_NAME TEST_SOURCES)
    get_rocprim_test_target(${TEST_SOURCES} TEST_TARGET)
    add_rocprim_test_internal(${TEST_NAME} "${TEST_SOURCES}" ${TEST_TARGET})
    if(ROCPRIM_ENABLE_ROCRAND)
        find_package(rocrand REQUIRED)
        target_link_libraries(${TEST_TARGET} PRIVATE roc::rocrand)
        target_compile_definitions(${TEST_TARGET} PRIVATE WITH_ROCRAND)
    endif()
endfunction()

function(add_rocprim_test_internal TEST_NAME TEST_SOURCES TEST_TARGET)
    if(ROCPRIM_USE_HIPCXX)
        set_source_files_properties(${TEST_SOURCES} PROPERTIES LANGUAGE HIP)
    endif()

    add_executable(${TEST_TARGET} ${TEST_SOURCES})

    if(ROCPRIM_ENABLE_INSTALL)
        rocm_install(TARGETS ${TEST_TARGET} COMPONENT tests)
    endif()

    target_include_directories(${TEST_TARGET}
        PRIVATE
            $<BUILD_INTERFACE:${COMMON_TEST_HEADER_DIRECTORY}>
    )

    if(ROCPRIM_USE_SYSTEM_LIBS)
        target_link_libraries(${TEST_TARGET} PRIVATE roc::rocprim)
        target_include_directories(${TEST_TARGET} PRIVATE ${CMAKE_CURRENT_LIST_DIR})
    else()
        target_link_libraries(${TEST_TARGET} PRIVATE roc::rocprim)
    endif()

    target_link_libraries(${TEST_TARGET}
        PRIVATE
            GTest::GTest
            GTest::Main
    )

    if(ROCPRIM_USE_HIPCXX)
        target_link_libraries(${TEST_TARGET}
            PRIVATE
                $<IF:$<LINK_LANGUAGE:HIP>,hip::host,hip::device>
        )
    else()
        target_link_libraries(${TEST_TARGET}
            PRIVATE
                hip::device
        )
    endif()

    target_compile_options(${TEST_TARGET}
        PRIVATE
            $<$<CXX_COMPILER_ID:MSVC>:/bigobj>
            $<$<COMPILE_LANGUAGE:HIP>:$<$<HIP_COMPILER_ID:MSVC>:/bigobj>>
    )

    if(WIN32)
        target_compile_definitions(${TEST_TARGET} PRIVATE _ENABLE_EXTENDED_ALIGNED_STORAGE)
    endif()

    set_target_properties(${TEST_TARGET}
        PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test/rocprim"
    )

    if(AMDGPU_TEST_TARGETS)
        foreach(AMDGPU_TARGET IN LISTS AMDGPU_TEST_TARGETS)
            add_relative_test("${AMDGPU_TARGET}-${TEST_NAME}" ${TEST_TARGET})
            set_tests_properties("${AMDGPU_TARGET}-${TEST_NAME}"
                PROPERTIES
                    RESOURCE_GROUPS "1,${AMDGPU_TARGET}:1"
                    LABELS "hip;${AMDGPU_TARGET}"
            )
        endforeach()
    else()
        add_relative_test(${TEST_NAME} ${TEST_TARGET})
        set_tests_properties(${TEST_NAME}
            PROPERTIES
                LABELS "hip"
        )
    endif()

    if(WIN32 AND NOT DEFINED DLLS_COPIED_2)
        set(DLLS_COPIED_2 "YES" PARENT_SCOPE)
        file(GLOB third_party_dlls
            LIST_DIRECTORIES ON
            CONFIGURE_DEPENDS
            ${HIP_DIR}/bin/*.dll
            ${CMAKE_SOURCE_DIR}/rtest.*
        )
        foreach(file_i ${third_party_dlls})
            add_custom_command(
                TARGET ${TEST_TARGET} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different ${file_i} ${PROJECT_BINARY_DIR}/test/rocprim
            )
        endforeach()
    endif()
endfunction()

function(add_rocprim_test_disable_dpp TEST_NAME TEST_SOURCES)
    get_rocprim_test_target(${TEST_SOURCES} TEST_TARGET)
    set(TEST_TARGET "${TEST_TARGET}_disable_dpp")
    add_rocprim_test_internal("${TEST_NAME}_disable_dpp" "${TEST_SOURCES}" ${TEST_TARGET})
    target_compile_definitions(${TEST_TARGET} PRIVATE ROCPRIM_DISABLE_DPP=1)
endfunction()

function(get_match_list FILE_STRING WORD MATCH_LIST)
    string(REGEX MATCHALL "${WORD}[ ]*==[ ]*[0-9]*" LINE_MATCHES "${${FILE_STRING}}")
    set(${MATCH_LIST} "")
    foreach(LINE IN LISTS LINE_MATCHES)
        string(REGEX MATCH "${WORD}[ ]*==[ ]*([0-9]*)" TMP "${LINE}")
        list(APPEND ${MATCH_LIST} "${CMAKE_MATCH_1}")
    endforeach()
    set(${MATCH_LIST} ${${MATCH_LIST}} PARENT_SCOPE)
endfunction()

function(add_rocprim_test_parallel TEST_NAME TEST_SOURCE)
    get_rocprim_test_target(${TEST_SOURCE} TEST_TARGET)

    file(READ ${TEST_SOURCE} FILE_CONTENTS)

    set(SOURCES "")

    # Handle all non-typed tests
    set(ROCPRIM_TEST_SUITE_SLICE -1)
    set(ROCPRIM_TEST_TYPE_SLICE -1)
    get_match_list(FILE_CONTENTS "ROCPRIM_TEST_SLICE" TEST_SLICE_LIST)
    list(LENGTH TEST_SLICE_LIST TEST_SLICE_COUNT)
    if(TEST_SLICE_COUNT EQUAL 0)
        message(VERBOSE "Found no non-typed tests for test target ${TEST_TARGET}")
    else()
        message(VERBOSE "Found ${TEST_SLICE_COUNT} non-typed test slice(s) for test target ${TEST_TARGET}")
        foreach(ROCPRIM_TEST_SLICE IN LISTS TEST_SLICE_LIST)
            set(FILENAME "${TEST_TARGET}.parallel/${TEST_TARGET}_${ROCPRIM_TEST_SLICE}.cpp")
            configure_file(${TEST_SOURCE} ${FILENAME} @ONLY)
            list(APPEND SOURCES "${CMAKE_CURRENT_BINARY_DIR}/${FILENAME}")
        endforeach()
    endif()

    # Handle all typed tests
    set(ROCPRIM_TEST_SLICE -1)
    get_match_list(FILE_CONTENTS "ROCPRIM_TEST_SUITE_SLICE" TEST_SUITE_SLICE_LIST)
    list(LENGTH TEST_SUITE_SLICE_LIST TEST_SUITE_SLICE_COUNT)
    get_match_list(FILE_CONTENTS "ROCPRIM_TEST_TYPE_SLICE" TEST_TYPE_SLICE_LIST)
    list(LENGTH TEST_TYPE_SLICE_LIST TEST_TYPE_SLICE_COUNT)
    if(TEST_SUITE_SLICE_COUNT EQUAL 0 OR TEST_TYPE_SLICE_COUNT EQUAL 0)
        message(VERBOSE "Found no typed tests for test target ${TEST_TARGET}")
    else()
        message(VERBOSE "Found ${TEST_SUITE_SLICE_COUNT} test suite slice(s) and ${TEST_TYPE_SLICE_COUNT} test type slice(s) for test target ${TEST_TARGET}")
        foreach(ROCPRIM_TEST_SUITE_SLICE IN LISTS TEST_SUITE_SLICE_LIST)
            foreach(ROCPRIM_TEST_TYPE_SLICE IN LISTS TEST_TYPE_SLICE_LIST)
                set(FILENAME "${TEST_TARGET}.parallel/${TEST_TARGET}_typed_${ROCPRIM_TEST_SUITE_SLICE}_${ROCPRIM_TEST_TYPE_SLICE}.cpp")
                configure_file(${TEST_SOURCE} ${FILENAME} @ONLY)
                list(APPEND SOURCES "${CMAKE_CURRENT_BINARY_DIR}/${FILENAME}")
            endforeach()
        endforeach()
    endif()

    list(LENGTH SOURCES SOURCES_COUNT)
    if(${SOURCES_COUNT} EQUAL 0)
        message(FATAL_ERROR "No .cpp files generated for target ${TEST_TARGET}")
    endif()

    set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_CLEAN_FILES "${TEST_TARGET}.parallel")
    add_rocprim_test_internal(${TEST_NAME} "${SOURCES}" ${TEST_TARGET})
    target_include_directories("${TEST_TARGET}" PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${SOURCES})
endfunction()

function(add_rocprim_cpp_standard_test STANDARD EXTENSIONS TARGET_SUFFIX TEST_NAME TEST_SOURCES)
    get_rocprim_test_target(${TEST_SOURCES} TEST_TARGET)
    set(TEST_TARGET ${TEST_TARGET}${TARGET_SUFFIX})
    add_rocprim_test_internal(${TEST_NAME}${TARGET_SUFFIX} "${TEST_SOURCES}" ${TEST_TARGET})
    set_target_properties(${TEST_TARGET}
        PROPERTIES
            CXX_STANDARD ${STANDARD}
            CXX_EXTENSIONS ${EXTENSIONS}
    )
    if(ROCPRIM_USE_HIPCXX)
        set_target_properties(${TEST_TARGET}
            PROPERTIES
                HIP_STANDARD ${STANDARD}
                HIP_EXTENSIONS ${EXTENSIONS}
        )
    endif()
endfunction()

# Tests
add_rocprim_test("rocprim.internal_merge_path" "internal/test_internal_merge_path.cpp")
add_rocprim_test("rocprim.basic_test" "test_basic.cpp;detail/get_rocprim_version.cpp")
add_rocprim_test("rocprim.arg_index_iterator" test_arg_index_iterator.cpp)
add_rocprim_test("rocprim.temporary_storage_partitioning" test_temporary_storage_partitioning.cpp)

if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_rocprim_test_parallel("rocprim.block_adjacent_difference" test_block_adjacent_difference.cpp.in)
    add_rocprim_test_parallel("rocprim.block_discontinuity" test_block_discontinuity.cpp.in)
endif()

add_rocprim_test("rocprim.bit_cast" test_bit_cast.cpp)
add_rocprim_test("rocprim.block_exchange" test_block_exchange.cpp)
add_rocprim_test("rocprim.block_histogram" test_block_histogram.cpp)
add_rocprim_test("rocprim.block_load_store" test_block_load_store.cpp)
add_rocprim_test("rocprim.block_sort_merge" test_block_sort_merge.cpp)
add_rocprim_test("rocprim.block_sort_merge_stable" test_block_sort_merge_stable.cpp)
add_rocprim_test_parallel("rocprim.block_radix_rank" test_block_radix_rank.cpp.in)
add_rocprim_test_parallel("rocprim.block_radix_sort" test_block_radix_sort.cpp.in)
add_rocprim_test("rocprim.block_reduce" test_block_reduce.cpp)
add_rocprim_test("rocprim.block_run_length_decode" test_block_run_length_decode.cpp)
add_rocprim_test_parallel("rocprim.block_scan" test_block_scan.cpp.in)
add_rocprim_test("rocprim.block_shuffle" test_block_shuffle.cpp)
add_rocprim_test("rocprim.block_sort_bitonic" test_block_sort_bitonic.cpp)
add_rocprim_test("rocprim.config_dispatch" test_config_dispatch.cpp)
add_rocprim_test("rocprim.constant_iterator" test_constant_iterator.cpp)
add_rocprim_test("rocprim.counting_iterator" test_counting_iterator.cpp)
add_rocprim_test("rocprim.device_batch_memcpy" test_device_batch_memcpy.cpp)
add_rocprim_test("rocprim.device_binary_search" test_device_binary_search.cpp)
add_rocprim_test("rocprim.device_find_first_of" test_device_find_first_of.cpp)
add_rocprim_test("rocprim.device_adjacent_difference" test_device_adjacent_difference.cpp)
add_rocprim_test("rocprim.device_adjacent_find" test_device_adjacent_find.cpp)
add_rocprim_test("rocprim.device_find_end" test_device_find_end.cpp)
add_rocprim_test("rocprim.device_histogram" test_device_histogram.cpp)
add_rocprim_test("rocprim.device_merge" test_device_merge.cpp)
add_rocprim_test("rocprim.device_merge_inplace" test_device_merge_inplace.cpp)
add_rocprim_test("rocprim.device_merge_sort" test_device_merge_sort.cpp)
add_rocprim_test("rocprim.nth_element" test_device_nth_element.cpp)
add_rocprim_test("rocprim.device_partial_sort" test_device_partial_sort.cpp)
add_rocprim_test("rocprim.device_partition" test_device_partition.cpp)
add_rocprim_test_parallel("rocprim.device_radix_sort" test_device_radix_sort.cpp.in)
add_rocprim_test("rocprim.device_reduce_by_key" test_device_reduce_by_key.cpp)
add_rocprim_test("rocprim.device_reduce" test_device_reduce.cpp)
add_rocprim_test("rocprim.device_run_length_encode" test_device_run_length_encode.cpp)
add_rocprim_test("rocprim.device_scan_by_key" test_device_scan_by_key.cpp)
add_rocprim_test("rocprim.device_scan" test_device_scan.cpp)
add_rocprim_test("rocprim.device_search" test_device_search.cpp)
add_rocprim_test_parallel("rocprim.device_segmented_radix_sort" test_device_segmented_radix_sort.cpp.in)
add_rocprim_test("rocprim.device_search_n" test_device_search_n.cpp)
add_rocprim_test("rocprim.device_segmented_reduce" test_device_segmented_reduce.cpp)
add_rocprim_test("rocprim.device_segmented_scan" test_device_segmented_scan.cpp)
add_rocprim_test("rocprim.device_select" test_device_select.cpp)
add_rocprim_test("rocprim.device_transform" test_device_transform.cpp)
add_rocprim_test("rocprim.discard_iterator" test_discard_iterator.cpp)
add_rocprim_test("rocprim.lookback_reproducibility" test_lookback_reproducibility.cpp)
add_rocprim_test("rocprim.radix_key_codec" test_radix_key_codec.cpp)
add_rocprim_test("rocprim.predicate_iterator" test_predicate_iterator.cpp)
add_rocprim_test("rocprim.reverse_iterator" test_reverse_iterator.cpp)
add_rocprim_test("rocprim.rocprim_tuple" test_rocprim_tuple.cpp)
add_rocprim_test("rocprim.rocprim_types" test_rocprim_types.cpp)
add_rocprim_test("rocprim.texture_cache_iterator" test_texture_cache_iterator.cpp)
add_rocprim_test("rocprim.thread" test_thread.cpp)
add_rocprim_test("rocprim.thread_algos" test_thread_algos.cpp)
add_rocprim_test("rocprim.tuple" test_tuple.cpp)
add_rocprim_test("rocprim.utils_sort_checker" test_utils_sort_checker.cpp)
add_rocprim_test("rocprim.transform_iterator" test_transform_iterator.cpp)
add_rocprim_cpp_standard_test(17 OFF "_cpp17" "rocprim.type_traits_interface" test_type_traits_interface.cpp)
add_rocprim_cpp_standard_test(17 ON "_gnupp17" "rocprim.type_traits_interface" test_type_traits_interface.cpp)
add_rocprim_cpp_standard_test(20 OFF "_cpp20" "rocprim.type_traits_interface" test_type_traits_interface.cpp)
add_rocprim_cpp_standard_test(20 ON "_gnupp20" "rocprim.type_traits_interface" test_type_traits_interface.cpp)
add_rocprim_test("rocprim.no_half_operators" test_no_half_operators.cpp)
add_rocprim_test("rocprim.intrinsics" test_intrinsics.cpp)
add_rocprim_test("rocprim.intrinsics_atomic" test_intrinsics_atomic.cpp)
add_rocprim_test("rocprim.invoke_result" test_invoke_result.cpp)
add_rocprim_test("rocprim.warp_exchange" test_warp_exchange.cpp)
add_rocprim_test("rocprim.warp_load" test_warp_load.cpp)
add_rocprim_test("rocprim.warp_reduce" test_warp_reduce.cpp)
add_rocprim_test("rocprim.warp_scan" test_warp_scan.cpp)
add_rocprim_test_disable_dpp("rocprim.warp_scan_disable_dpp" test_warp_scan.cpp)
add_rocprim_test("rocprim.warp_sort" test_warp_sort.cpp)
add_rocprim_test("rocprim.warp_store" test_warp_store.cpp)
add_rocprim_test("rocprim.zip_iterator" test_zip_iterator.cpp)
add_rocprim_test("rocprim.accumulator_t" test_accumulator_t.cpp)

if(NOT WIN32)
    set(WITHOUT_INLINE_NAMESPACE OFF)

    if(WITHOUT_INLINE_NAMESPACE)
        set(COMMON_DEFINITIONS "ROCPRIM_DISABLE_INLINE_NAMESPACE")
    endif()

    add_library(test_linking_lib1 SHARED test_linking_lib.cpp)
    if(ROCPRIM_ENABLE_INSTALL)
        rocm_install(TARGETS test_linking_lib1 COMPONENT tests EXPORT rocprim-tests)
    endif()
    target_include_directories(test_linking_lib1
        PRIVATE
            $<BUILD_INTERFACE:${COMMON_TEST_HEADER_DIRECTORY}>
            $<TARGET_PROPERTY:GTest::GTest,INTERFACE_INCLUDE_DIRECTORIES>
    )
    target_link_libraries(test_linking_lib1 PRIVATE roc::rocprim_hip)
    set_target_properties(test_linking_lib1
        PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test/rocprim"
            COMPILE_DEFINITIONS "TEST_FUNC=test1;${COMMON_DEFINITIONS};"
    )

    add_library(test_linking_lib2 SHARED test_linking_lib.cpp)
    if(ROCPRIM_ENABLE_INSTALL)
        rocm_install(TARGETS test_linking_lib2 COMPONENT tests EXPORT rocprim-tests)
    endif()
    target_include_directories(test_linking_lib2
        PRIVATE
            $<BUILD_INTERFACE:${COMMON_TEST_HEADER_DIRECTORY}>
            $<TARGET_PROPERTY:GTest::GTest,INTERFACE_INCLUDE_DIRECTORIES>
    )
    target_link_libraries(test_linking_lib2 PRIVATE roc::rocprim_hip)
    if(NOT WITHOUT_INLINE_NAMESPACE)
        set(LIB2_NAMESPACE_DEFINITION "ROCPRIM_INLINE_NAMESPACE=ROCPRIM_300201_NS")
    endif()
    set_target_properties(test_linking_lib2
        PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test/rocprim"
            COMPILE_DEFINITIONS "TEST_FUNC=test2;TEST_LINKING_EMULATE_ANOTHER_VERSION;${COMMON_DEFINITIONS};${LIB2_NAMESPACE_DEFINITION};"
    )

    add_rocprim_test_internal("rocprim.linking" "test_linking.cpp;test_linking_lib.cpp" test_linking)
    target_link_libraries(test_linking PUBLIC test_linking_lib1 test_linking_lib2)
    set_target_properties(test_linking
        PROPERTIES
            COMPILE_DEFINITIONS "TEST_FUNC=test0;${COMMON_DEFINITIONS};"
    )

    set(NM_ARGS --demangle --defined-only --extern-only)
    string(REPLACE ";" " " NM_ARGS_STR "${NM_ARGS}")
    
    add_test(
        NAME "rocprim.linking_lib1_symbols"
        COMMAND sh -c "${CMAKE_NM} ${NM_ARGS_STR} $<TARGET_FILE:test_linking_lib1> | grep -c -P \"(rocprim::(?!ROCPRIM_${rocprim_VERSION_NUMBER}_NS))|kernel\""
    )
    add_test(
        NAME "rocprim.linking_lib2_symbols"
        COMMAND sh -c "${CMAKE_NM} ${NM_ARGS_STR} $<TARGET_FILE:test_linking_lib2> | grep -c -P \"(rocprim::(?!ROCPRIM_300201_NS))|kernel\""
    )
    add_test(
        NAME "rocprim.linking_symbols"
        COMMAND sh -c "${CMAKE_NM} ${NM_ARGS_STR} $<TARGET_FILE:test_linking> | grep -c -P \"(rocprim::(?!ROCPRIM_${rocprim_VERSION_NUMBER}_NS))\""
    )
    set_tests_properties(
        "rocprim.linking_lib1_symbols"
        "rocprim.linking_lib2_symbols"
        "rocprim.linking_symbols"
        PROPERTIES
            PASS_REGULAR_EXPRESSION "^0"
    )
endif()
