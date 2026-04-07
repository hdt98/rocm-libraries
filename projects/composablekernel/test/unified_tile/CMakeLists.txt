if(GPU_TARGETS MATCHES "gfx94|gfx95|gfx90a")

    # ============================================================================
    # Descriptor tests
    # ============================================================================

    # CK_Tile backend
    add_executable(test_unified_tile_descriptor_ck test_descriptor.cpp)
    set_source_files_properties(test_descriptor.cpp PROPERTIES LANGUAGE HIP)
    target_link_libraries(test_unified_tile_descriptor_ck PRIVATE
        gtest_main hip::device
    )
    target_compile_options(test_unified_tile_descriptor_ck PRIVATE
        -DUNIFIED_TILE_BACKEND_CK_TILE
        -Wno-error
    )
    add_test(
        NAME test_unified_tile_descriptor_ck
        COMMAND test_unified_tile_descriptor_ck
    )
    set_tests_properties(test_unified_tile_descriptor_ck PROPERTIES
        LABELS "SMOKE_TEST"
    )

    # MINT backend
    add_executable(test_unified_tile_descriptor_mint test_descriptor.cpp)
    set_source_files_properties(test_descriptor.cpp PROPERTIES LANGUAGE HIP)
    target_link_libraries(test_unified_tile_descriptor_mint PRIVATE
        gtest_main hip::device
    )
    target_compile_options(test_unified_tile_descriptor_mint PRIVATE
        -DUNIFIED_TILE_BACKEND_MINT
        -DMINT_BACKEND_ROCM
        -fgpu-rdc
        -Wno-error
    )
    target_link_options(test_unified_tile_descriptor_mint PRIVATE
        -fgpu-rdc
    )
    add_test(
        NAME test_unified_tile_descriptor_mint
        COMMAND test_unified_tile_descriptor_mint
    )
    set_tests_properties(test_unified_tile_descriptor_mint PROPERTIES
        LABELS "SMOKE_TEST"
    )

    # ============================================================================
    # View tests
    # ============================================================================

    # CK_Tile backend
    add_executable(test_unified_tile_view_ck test_view.cpp)
    set_source_files_properties(test_view.cpp PROPERTIES LANGUAGE HIP)
    target_link_libraries(test_unified_tile_view_ck PRIVATE
        gtest_main hip::device
    )
    target_compile_options(test_unified_tile_view_ck PRIVATE
        -DUNIFIED_TILE_BACKEND_CK_TILE
        -Wno-error
    )
    add_test(
        NAME test_unified_tile_view_ck
        COMMAND test_unified_tile_view_ck
    )
    set_tests_properties(test_unified_tile_view_ck PROPERTIES
        LABELS "SMOKE_TEST"
    )

    # MINT backend
    add_executable(test_unified_tile_view_mint test_view.cpp)
    set_source_files_properties(test_view.cpp PROPERTIES LANGUAGE HIP)
    target_link_libraries(test_unified_tile_view_mint PRIVATE
        gtest_main hip::device
    )
    target_compile_options(test_unified_tile_view_mint PRIVATE
        -DUNIFIED_TILE_BACKEND_MINT
        -DMINT_BACKEND_ROCM
        -fgpu-rdc
        -Wno-error
    )
    target_link_options(test_unified_tile_view_mint PRIVATE
        -fgpu-rdc
    )
    add_test(
        NAME test_unified_tile_view_mint
        COMMAND test_unified_tile_view_mint
    )
    set_tests_properties(test_unified_tile_view_mint PROPERTIES
        LABELS "SMOKE_TEST"
    )

    # ============================================================================
    # Distribution tests
    # ============================================================================

    # CK_Tile backend
    add_executable(test_unified_tile_distribution_ck test_distribution.cpp)
    set_source_files_properties(test_distribution.cpp PROPERTIES LANGUAGE HIP)
    target_link_libraries(test_unified_tile_distribution_ck PRIVATE
        gtest_main hip::device
    )
    target_compile_options(test_unified_tile_distribution_ck PRIVATE
        -DUNIFIED_TILE_BACKEND_CK_TILE
        -Wno-error
    )
    add_test(
        NAME test_unified_tile_distribution_ck
        COMMAND test_unified_tile_distribution_ck
    )
    set_tests_properties(test_unified_tile_distribution_ck PROPERTIES
        LABELS "SMOKE_TEST"
    )

    # MINT backend
    add_executable(test_unified_tile_distribution_mint test_distribution.cpp)
    set_source_files_properties(test_distribution.cpp PROPERTIES LANGUAGE HIP)
    target_link_libraries(test_unified_tile_distribution_mint PRIVATE
        gtest_main hip::device
    )
    target_compile_options(test_unified_tile_distribution_mint PRIVATE
        -DUNIFIED_TILE_BACKEND_MINT
        -DMINT_BACKEND_ROCM
        -fgpu-rdc
        -Wno-error
    )
    target_link_options(test_unified_tile_distribution_mint PRIVATE
        -fgpu-rdc
    )
    add_test(
        NAME test_unified_tile_distribution_mint
        COMMAND test_unified_tile_distribution_mint
    )
    set_tests_properties(test_unified_tile_distribution_mint PROPERTIES
        LABELS "SMOKE_TEST"
    )

    # ============================================================================
    # Window tests
    # ============================================================================

    # CK_Tile backend
    add_executable(test_unified_tile_window_ck test_window.cpp)
    set_source_files_properties(test_window.cpp PROPERTIES LANGUAGE HIP)
    target_link_libraries(test_unified_tile_window_ck PRIVATE
        gtest_main hip::device
    )
    target_compile_options(test_unified_tile_window_ck PRIVATE
        -DUNIFIED_TILE_BACKEND_CK_TILE
        -Wno-error
    )
    add_test(
        NAME test_unified_tile_window_ck
        COMMAND test_unified_tile_window_ck
    )
    set_tests_properties(test_unified_tile_window_ck PROPERTIES
        LABELS "SMOKE_TEST"
    )

    # MINT backend
    add_executable(test_unified_tile_window_mint test_window.cpp)
    set_source_files_properties(test_window.cpp PROPERTIES LANGUAGE HIP)
    target_link_libraries(test_unified_tile_window_mint PRIVATE
        gtest_main hip::device
    )
    target_compile_options(test_unified_tile_window_mint PRIVATE
        -DUNIFIED_TILE_BACKEND_MINT
        -DMINT_BACKEND_ROCM
        -fgpu-rdc
        -Wno-error
    )
    target_link_options(test_unified_tile_window_mint PRIVATE
        -fgpu-rdc
    )
    add_test(
        NAME test_unified_tile_window_mint
        COMMAND test_unified_tile_window_mint
    )
    set_tests_properties(test_unified_tile_window_mint PROPERTIES
        LABELS "SMOKE_TEST"
    )

endif()
