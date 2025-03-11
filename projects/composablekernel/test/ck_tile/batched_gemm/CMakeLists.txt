# Currently ck_tile is only built on gfx9
if(GPU_TARGETS MATCHES "gfx9" OR GPU_TARGETS MATCHES "gfx13")
    add_gtest_executable(test_ck_tile_batched_gemm test_batched_gemm.cpp)
endif()
