enable_language(HIP)

# conv3d_test links only the 3D convolution translation unit, avoiding the
# pre-existing compiler-crash in grouped_conv.cpp (grouped 2D kernel, gfx950).
add_library(hipconv_3d_only STATIC
    ../src/conv3d/conv3d.cpp
)
set_source_files_properties(../src/conv3d/conv3d.cpp PROPERTIES LANGUAGE HIP)
set_target_properties(hipconv_3d_only PROPERTIES HIP_ARCHITECTURES gfx950)
target_include_directories(hipconv_3d_only
    PUBLIC  ../include
    PRIVATE ../src
)
target_link_libraries(hipconv_3d_only PUBLIC hip::device)
target_compile_options(hipconv_3d_only PRIVATE -fvisibility=hidden -fvisibility-inlines-hidden)
target_compile_definitions(hipconv_3d_only PRIVATE HIPCONV_BUILDING_LIBRARY)

add_executable(conv3d_test conv3d_test.cpp)
set_source_files_properties(conv3d_test.cpp PROPERTIES LANGUAGE HIP)
set_target_properties(conv3d_test PROPERTIES HIP_ARCHITECTURES gfx950)
target_include_directories(conv3d_test PRIVATE ../src)
target_link_libraries(conv3d_test PRIVATE hipconv_3d_only hip::device)

add_executable(conv3d_bench conv3d_bench.cpp)
set_source_files_properties(conv3d_bench.cpp PROPERTIES LANGUAGE HIP)
set_target_properties(conv3d_bench PROPERTIES HIP_ARCHITECTURES gfx950)
target_include_directories(conv3d_bench PRIVATE ../src)
target_link_libraries(conv3d_bench PRIVATE hipconv_3d_only hip::device)
