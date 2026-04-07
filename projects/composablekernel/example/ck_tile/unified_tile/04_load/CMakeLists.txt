if(GPU_TARGETS MATCHES "gfx94|gfx95|gfx90a")
    # CK_Tile backend
    add_executable(example_unified_tile_04_load_ck main.cpp)
    target_compile_definitions(example_unified_tile_04_load_ck PRIVATE UNIFIED_TILE_BACKEND_CK_TILE)
    target_compile_options(example_unified_tile_04_load_ck PRIVATE -Wno-error)

    # MINT backend
    add_executable(example_unified_tile_04_load_mint main.cpp)
    target_compile_definitions(example_unified_tile_04_load_mint PRIVATE
        UNIFIED_TILE_BACKEND_MINT MINT_BACKEND_ROCM)
    target_compile_options(example_unified_tile_04_load_mint PRIVATE
        -fgpu-rdc -Wno-error)
    target_link_options(example_unified_tile_04_load_mint PRIVATE -fgpu-rdc)
endif()
