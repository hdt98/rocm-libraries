# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

function(add_configured_source)
    cmake_parse_arguments(PARSE_ARGV 0 ARG "" "INPUT;TARGET;OUTPUT_PATTERN" "NAMES;VALUES")
    list(LENGTH ARG_NAMES NAMES_LEN)
    list(LENGTH ARG_VALUES VALS_LEN)
    if(NOT NAMES_LEN EQUAL VALS_LEN)
        message(FATAL_ERROR "add_configured_source: The same number of names (${NAMES_LEN}) and values (${VALS_LEN}) must be provided!")
    endif()

    set(max ${VALS_LEN})
    math(EXPR max "${max} - 1")
    foreach(i RANGE ${max})
        list(GET ARG_NAMES ${i} curr_name)
        list(GET ARG_VALUES ${i} "${curr_name}")
    endforeach()

    string(CONFIGURE "${ARG_OUTPUT_PATTERN}" output @ONLY)
    string(MAKE_C_IDENTIFIER ${output} output)
    set(output_path "${ARG_TARGET}.parallel/${output}.cpp")
    configure_file("${ARG_INPUT}" "${output_path}" @ONLY)
    set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_CLEAN_FILES "${ARG_TARGET}.parallel")
    target_sources("${ARG_TARGET}" PRIVATE "${output_path}")
    target_include_directories("${ARG_TARGET}" PRIVATE "../benchmark")

    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${ARG_INPUT}" "${output_path}")
endfunction()

function(div_round_up dividend divisor result_var)
    math(EXPR result "(${dividend} + ${divisor} - 1) / ${divisor}")
    set("${result_var}" "${result}" PARENT_SCOPE)
endfunction()

function(add_matrix)
    set(single_value_args "TARGET" "INPUT" "OUTPUT_PATTERN" "SHARDS" "CURRENT_SHARD")
    cmake_parse_arguments(PARSE_ARGV 0 ARG "" "${single_value_args}" "NAMES;LISTS")

    list(LENGTH ARG_NAMES NAMES_LEN)
    list(LENGTH ARG_LISTS LISTS_LEN)
    if(NOT NAMES_LEN EQUAL LISTS_LEN)
        message(FATAL_ERROR "add_matrix: The same number of names (${NAMES_LEN}) and lists (${LISTS_LEN}) must be provided!")
    endif()

    set(total_len 1)
    foreach(LIST IN LISTS ARG_LISTS)
        string(REPLACE " " ";" list ${LIST})
        list(LENGTH list LIST_LEN)
        math(EXPR total_len "${total_len} * ${LIST_LEN}")
    endforeach()

    if(NOT DEFINED ARG_SHARDS)
        set(ARG_SHARDS 1)
    endif()
    div_round_up("${total_len}" "${ARG_SHARDS}" per_shard)

    math(EXPR start "${ARG_CURRENT_SHARD} * ${per_shard}")
    math(EXPR stop "${start} + ${per_shard} - 1")

    foreach(i RANGE ${start} ${stop})
        set(index ${i})
        set(values "")
        foreach(input_list IN LISTS ARG_LISTS)
            string(REPLACE " " ";" curr_list ${input_list})
            list(LENGTH curr_list curr_length)
            math(EXPR curr_index "${index} % ${curr_length}")
            list(GET curr_list ${curr_index} curr_item)
            list(APPEND values "${curr_item}")
            math(EXPR index "${index} / ${curr_length}")
        endforeach()

        add_configured_source(
            TARGET "${ARG_TARGET}"
            INPUT "${ARG_INPUT}"
            OUTPUT_PATTERN "${ARG_OUTPUT_PATTERN}"
            NAMES ${ARG_NAMES}
            VALUES ${values}
        )
    endforeach()
endfunction()

function(reject_odd_blocksize RESULT BlockSize)
    math(EXPR res "${BlockSize} % 2")
    if(res EQUAL 0)
        set("${RESULT}" ON PARENT_SCOPE)
    else()
        set("${RESULT}" OFF PARENT_SCOPE)
    endif()
endfunction()
