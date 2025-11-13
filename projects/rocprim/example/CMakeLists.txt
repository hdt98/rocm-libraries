# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

function(add_example EXAMPLE_SOURCE)
    get_filename_component(EXAMPLE_TARGET ${EXAMPLE_SOURCE} NAME_WE)

    if(ROCPRIM_USE_HIPCXX)
        set_source_files_properties(${EXAMPLE_SOURCE} PROPERTIES LANGUAGE HIP)
    endif()

    add_executable(${EXAMPLE_TARGET} ${EXAMPLE_SOURCE})

    target_link_libraries(${EXAMPLE_TARGET}
        PRIVATE
            roc::rocprim
    )

    if(ROCPRIM_USE_HIPCXX)
        target_link_libraries(${EXAMPLE_TARGET}
            PRIVATE
                $<IF:$<LINK_LANGUAGE:HIP>,hip::host,hip::device>
        )
    else()
        target_link_libraries(${EXAMPLE_TARGET}
            PRIVATE
                hip::device
        )
    endif()

    set_target_properties(${EXAMPLE_TARGET}
        PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/example"
    )

    # Use an EXAMPLE_SOURCE like "rocprim/device/example_device_search.cpp"
    # to create a test named "example.rocprim.device.example_device_search"
    get_filename_component(DIR_PATH "${EXAMPLE_SOURCE}" PATH)
    string(REPLACE "/" "." DIR_PATH_DOTS "${DIR_PATH}")
    set(TEST_NAME "example.${DIR_PATH_DOTS}.${EXAMPLE_TARGET}")
    add_test(NAME ${TEST_NAME} COMMAND ${EXAMPLE_TARGET})

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
                TARGET ${EXAMPLE_TARGET} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different ${file_i} ${PROJECT_BINARY_DIR}/example
            )
        endforeach()
    endif()
endfunction()

# rocPRIM examples
add_example(extra/example_temporary_storage.cpp)
add_example(extra/example_type_traits_interface.cpp)
add_example(rocprim/device/example_device_search.cpp)
