# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

target_sources(rocroller
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/ControlFlowArgumentTracer.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/ControlFlowRWTracer.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/ControlToCoordinateMapper.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/GlobalConstraints.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/KernelGraph.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/KernelGraph_serialization.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/LastRWTracer.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/LowerFromCommand.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/RegisterTagManager.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/Reindexer.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/ScopeManager.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/Utils.cpp"
)

if(ROCROLLER_ENABLE_IMGUI)
    find_package(SDL2 REQUIRED)
    find_package(OpenGL REQUIRED)

    FetchContent_Declare(
        imgui
        GIT_REPOSITORY https://github.com/ocornut/imgui.git
        GIT_TAG 29fadad1939ab62adcc5cd9339db7a784de90120
    )
    FetchContent_MakeAvailable(imgui)

    FetchContent_Declare(
        imgui-node-editor
        GIT_REPOSITORY https://github.com/thedmd/imgui-node-editor.git
        GIT_TAG e78e447900909a051817a760efe13fe83e6e1afc
    )
    FetchContent_MakeAvailable(imgui-node-editor)

    FetchContent_GetProperties(imgui SOURCE_DIR imgui_SOURCE_DIR)
    FetchContent_GetProperties(imgui-node-editor SOURCE_DIR imgui-node-editor_SOURCE_DIR)

    target_sources(rocroller
        PRIVATE
        ${rocroller_src}
        ${imgui_SOURCE_DIR}/imgui.cpp
        ${imgui_SOURCE_DIR}/imgui_demo.cpp
        ${imgui_SOURCE_DIR}/imgui_draw.cpp
        ${imgui_SOURCE_DIR}/imgui_tables.cpp
        ${imgui_SOURCE_DIR}/imgui_widgets.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_sdl2.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl2.cpp
        ${imgui-node-editor_SOURCE_DIR}/imgui_node_editor.cpp
        ${imgui-node-editor_SOURCE_DIR}/imgui_node_editor_api.cpp
        ${imgui-node-editor_SOURCE_DIR}/imgui_canvas.cpp
        ${imgui-node-editor_SOURCE_DIR}/crude_json.cpp
        "${CMAKE_CURRENT_SOURCE_DIR}/Visualizer.cpp"
    )

    target_include_directories(rocroller PRIVATE ${imgui_SOURCE_DIR} ${imgui-node-editor_SOURCE_DIR} ${SDL2_INCLUDE_DIRS})
    target_compile_definitions(rocroller PRIVATE ROCROLLER_ENABLE_IMGUI IMGUI_USER_CONFIG="${CMAKE_SOURCE_DIR}/lib/include/rocRoller/KernelGraph/ImGUIConfig.h")
    # For Linux: SDL2 and OpenGL
    target_link_libraries(rocroller PUBLIC ${SDL2_LIBRARIES} ${OPENGL_LIBRARY})
    # Graphviz layout
    # XXXX Need to do find_package or something
    target_link_libraries(rocroller PUBLIC cgraph gvc)
endif()

add_subdirectory(Transformations)
add_subdirectory(CoordinateGraph)
add_subdirectory(ControlGraph)
