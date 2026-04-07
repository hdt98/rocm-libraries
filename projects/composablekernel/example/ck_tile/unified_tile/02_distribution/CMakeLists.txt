if(GPU_TARGETS MATCHES "gfx94|gfx95|gfx90a")

    # ---- CK_Tile backend ----
    set(CK_TARGET example_unified_tile_02_distribution_ck)
    add_executable(${CK_TARGET} main.cpp)
    set_source_files_properties(main.cpp PROPERTIES LANGUAGE HIP)
    target_include_directories(${CK_TARGET} PRIVATE
        ${CMAKE_SOURCE_DIR}/include)
    target_compile_definitions(${CK_TARGET} PRIVATE
        UNIFIED_TILE_BACKEND_CK_TILE)
    target_compile_options(${CK_TARGET} PRIVATE
        -Wno-error)

    # ---- MINT backend ----
    set(MINT_TARGET example_unified_tile_02_distribution_mint)
    add_executable(${MINT_TARGET} main.cpp)
    set_source_files_properties(main.cpp PROPERTIES LANGUAGE HIP)
    target_include_directories(${MINT_TARGET} PRIVATE
        ${CMAKE_SOURCE_DIR}/include)
    target_compile_definitions(${MINT_TARGET} PRIVATE
        UNIFIED_TILE_BACKEND_MINT
        MINT_BACKEND_ROCM)
    target_compile_options(${MINT_TARGET} PRIVATE
        -fgpu-rdc -Wno-error)
    target_link_options(${MINT_TARGET} PRIVATE
        -fgpu-rdc)

endif()
