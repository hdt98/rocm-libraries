if(GPU_TARGETS MATCHES "gfx94|gfx95|gfx90a")

    # CK_Tile backend
    add_executable(example_unified_tile_00_descriptor_ck main.cpp)
    set_source_files_properties(main.cpp PROPERTIES LANGUAGE HIP)
    target_compile_options(example_unified_tile_00_descriptor_ck PRIVATE
        -DUNIFIED_TILE_BACKEND_CK_TILE
        -Wno-error
    )

    # MINT backend
    add_executable(example_unified_tile_00_descriptor_mint main.cpp)
    set_source_files_properties(main.cpp PROPERTIES LANGUAGE HIP)
    target_compile_options(example_unified_tile_00_descriptor_mint PRIVATE
        -DUNIFIED_TILE_BACKEND_MINT
        -DMINT_BACKEND_ROCM
        -fgpu-rdc
        -Wno-error
    )
    target_link_options(example_unified_tile_00_descriptor_mint PRIVATE
        -fgpu-rdc
    )

endif()
