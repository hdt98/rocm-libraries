# Currently ck_tile is only built on gfx9
if(GPU_TARGETS MATCHES "gfx9" OR GPU_TARGETS MATCHES "gfx13")
    add_gtest_executable(test_tile_image_to_column test_tile_image_to_column.cpp)
endif()
