# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
#
# Declares the `hipblaslt::mxdatagen` INTERFACE wrapper around
# roc::mxDataGenerator. There is no compiled source here any more - the
# generateMXInput implementation lives in
# shared/mxdatagenerator/lib/include/mxDataGenerator/mxDataGen.hpp; the
# adapter helpers (toDataFormat / toScaleType in testing_matmul.hpp,
# scaleTypeForDataGenerator in DataInitialization.cpp) live next to their
# consumers.
#
# This file is added by the project root via add_subdirectory(clients/common)
# only when HIPBLASLT_ENABLE_MXDATAGENERATOR is ON and at least one consumer
# subtree (hipblaslt clients, tensilelite client, or tensilelite tests) is
# being built.
#
# Consumers should
#     target_link_libraries(<tgt> PUBLIC hipblaslt::mxdatagen)
# (PRIVATE if no further consumer needs the macro). Linking propagates:
#   * HIPBLASLT_ENABLE_MXDATAGENERATOR
#   * the link to roc::mxDataGenerator (which carries the
#     <mxDataGenerator/...> include path).
if(NOT ROCM_LIBS_SUPERBUILD)
    if(HIPBLASLT_ENABLE_THEROCK)
        find_package(mxDataGenerator REQUIRED)
    else()
        # `${PROJECT_SOURCE_DIR}/../../shared/mxdatagenerator` resolves to the
        # sibling `shared/mxdatagenerator` subtree of the project. The binary
        # dir is anchored under the hipblaslt project's build tree to match
        # the upstream subdirectory's existing layout assumptions.
        add_subdirectory(
            "${PROJECT_SOURCE_DIR}/../../shared/mxdatagenerator"
            "${PROJECT_BINARY_DIR}/mxdatagenerator"
        )
    endif()
endif()

add_library(hipblaslt-mxdatagen INTERFACE)
add_library(hipblaslt::mxdatagen ALIAS hipblaslt-mxdatagen)

target_link_libraries(hipblaslt-mxdatagen
    INTERFACE
        roc::mxDataGenerator
)
# The GPU init overload in <mxDataGenerator/mxDataGen.hpp> is gated behind
# `__HIPCC__` and uses `<hip/hip_runtime.h>`. All hipblaslt consumers of this
# shim already compile as HIP (amdclang++ -x hip), so we propagate hip::device
# here to make the include path / runtime symbols available without forcing
# every consumer to wire it up themselves.
if(TARGET hip::device)
    target_link_libraries(hipblaslt-mxdatagen INTERFACE hip::device)
endif()
target_compile_features(hipblaslt-mxdatagen
    INTERFACE
        cxx_std_20
  )
target_compile_definitions(hipblaslt-mxdatagen
    INTERFACE
        HIPBLASLT_ENABLE_MXDATAGENERATOR
)
