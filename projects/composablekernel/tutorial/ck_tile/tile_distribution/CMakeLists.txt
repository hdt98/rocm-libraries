# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(TUTORIAL_NAME "tile_tutorial_tile_distribution")

add_executable(${TUTORIAL_NAME} EXCLUDE_FROM_ALL tile_distribution.cpp)
target_include_directories(${TUTORIAL_NAME} PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_compile_options(${TUTORIAL_NAME} PRIVATE
  -Wno-undefined-func-template -Wno-float-equal -Wno-ctad-maybe-unsupported
)

add_dependencies(tutorials ${TUTORIAL_NAME})

set(TUTORIAL_NAME_2 "tile_tutorial_tile_distribution_2")

add_executable(${TUTORIAL_NAME_2} EXCLUDE_FROM_ALL tile_distribution_2.cpp)
target_include_directories(${TUTORIAL_NAME_2} PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_compile_options(${TUTORIAL_NAME_2} PRIVATE
  -Wno-undefined-func-template -Wno-float-equal -Wno-ctad-maybe-unsupported
)

add_dependencies(tutorials ${TUTORIAL_NAME_2})

set(TUTORIAL_NAME_3 "tile_tutorial_tile_distribution_3")

add_executable(${TUTORIAL_NAME_3} EXCLUDE_FROM_ALL tile_distribution_3.cpp)
target_include_directories(${TUTORIAL_NAME_3} PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_compile_options(${TUTORIAL_NAME_3} PRIVATE
  -Wno-undefined-func-template -Wno-float-equal -Wno-ctad-maybe-unsupported
)

add_dependencies(tutorials ${TUTORIAL_NAME_3})

set(TUTORIAL_NAME_4 "tile_tutorial_tile_distribution_4")

add_executable(${TUTORIAL_NAME_4} EXCLUDE_FROM_ALL tile_distribution_4.cpp)
target_include_directories(${TUTORIAL_NAME_4} PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_compile_options(${TUTORIAL_NAME_4} PRIVATE
  -Wno-undefined-func-template -Wno-float-equal -Wno-ctad-maybe-unsupported
)

add_dependencies(tutorials ${TUTORIAL_NAME_4})
