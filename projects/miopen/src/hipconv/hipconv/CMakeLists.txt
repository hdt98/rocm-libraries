enable_language(HIP)

# ---- hipconv static library ----
add_library(hipconv STATIC
    src/registry.cpp
    src/grouped/grouped_conv.cpp
)
set_target_properties(hipconv PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    HIP_ARCHITECTURES gfx950
)
set_source_files_properties(
    src/registry.cpp
    src/grouped/grouped_conv.cpp
    PROPERTIES LANGUAGE HIP
)

target_include_directories(hipconv
    PUBLIC  include
    PRIVATE src
)
target_link_libraries(hipconv PUBLIC hip::device)
target_compile_options(hipconv PRIVATE -fvisibility=hidden -fvisibility-inlines-hidden)
target_compile_definitions(hipconv PUBLIC HIPCONV_BUILDING_LIBRARY)
