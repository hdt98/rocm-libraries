# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

find_package(GTest REQUIRED)

add_executable(test-header)
target_sources(test-header PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/test_header.c"
)
target_compile_features(test-header PRIVATE c_std_99)
target_link_libraries(test-header PRIVATE roc::rocsolver)

if(UNIX AND BUILD_SHARED_LIBS)
    if(TARGET rocsolver)
        add_executable(test-rocsolver-dlopen)
        target_sources(test-rocsolver-dlopen PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/test_dlopen_main.cpp"
        )
        target_compile_definitions(test-rocsolver-dlopen PRIVATE
            ROCSOLVER_LIB_NAME="$<TARGET_FILE:rocsolver>"
        )
        target_link_libraries(test-rocsolver-dlopen PRIVATE
            $<IF:$<TARGET_EXISTS:GTest::gtest>,GTest::gtest,GTest::GTest>
            ${CMAKE_DL_LIBS}
        )

        add_test(
            NAME test-rocsolver-dlopen
            COMMAND test-rocsolver-dlopen
        )
    endif()

    if(TARGET rocsolver-bench)
        find_package(Python3 COMPONENTS Interpreter)
        if(Python3_FOUND)
            add_test(
                NAME test-rocsolver-bench
                COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/test_rocsolver_bench.py"
                WORKING_DIRECTORY "$<TARGET_FILE_DIR:rocsolver-bench>"
            )
        endif()
    endif()
endif()
