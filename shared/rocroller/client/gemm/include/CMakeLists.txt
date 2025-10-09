# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

target_sources(rocroller-gemm
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/client/BenchmarkSolution.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/client/DataParallelGEMMSolution.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/client/GEMMParameters.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/client/GEMMParameters_serialization.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/client/GEMMSolution.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/client/GraphInspector.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/client/GraphInspector_impl.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/client/PreSwizzle.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/client/StreamKGEMMSolution.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/client/visualize.hpp"
)
