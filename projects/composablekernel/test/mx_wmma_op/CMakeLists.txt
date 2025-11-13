# Only build this test for gfx125
if(GPU_TARGETS MATCHES "gfx125")
    add_gtest_executable(test_mx_coexecution mx_wmma_coexecution.cpp)
    if(result EQUAL 0)
        target_link_libraries(test_mx_coexecution PRIVATE utility)
    endif()
else()
    message(STATUS "test_mx_coexecution is only supported on gfx125x architectures")
endif()
