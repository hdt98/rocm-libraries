# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

add_subdirectory(amd_detail)

target_sources(hipblaslt
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/amd_detail/include/auxiliary.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/amd_detail/include/exceptions.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/amd_detail/include/hipblaslt_ostream.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/amd_detail/include/hipblaslt_internal.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/amd_detail/include/hipblaslt-ext-op-internal.hpp"

        "${CMAKE_CURRENT_SOURCE_DIR}/amd_detail/rocblaslt/include/Debug.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/amd_detail/rocblaslt/include/rocblaslt-auxiliary.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/amd_detail/rocblaslt/include/rocblaslt-functions.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/amd_detail/rocblaslt/include/rocblaslt-types.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/amd_detail/rocblaslt/include/rocblaslt.h"

        "${CMAKE_CURRENT_SOURCE_DIR}/amd_detail/rocblaslt/src/include/rocblaslt/UserDrivenTuningParser.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/amd_detail/rocblaslt/src/include/rocblaslt/check_numerics_matrix.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/amd_detail/rocblaslt/src/include/rocblaslt/debug.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/amd_detail/rocblaslt/src/include/rocblaslt/definitions.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/amd_detail/rocblaslt/src/include/rocblaslt/handle.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/amd_detail/rocblaslt/src/include/rocblaslt/logging.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/amd_detail/rocblaslt/src/include/rocblaslt/rocblaslt_mat_utils.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/amd_detail/rocblaslt/src/include/rocblaslt/status.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/amd_detail/rocblaslt/src/include/rocblaslt/tensile_host.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/amd_detail/rocblaslt/src/include/rocblaslt/tuple_helper.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/amd_detail/rocblaslt/src/include/rocblaslt/utility.hpp"

        "${CMAKE_CURRENT_SOURCE_DIR}/amd_detail/rocblaslt/src/rocroller/include/custom_kernels.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/amd_detail/rocblaslt/src/rocroller/include/rocroller/analytical_utils.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/amd_detail/rocblaslt/src/rocroller/include/rocroller/gemm.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/amd_detail/rocblaslt/src/rocroller/include/rocroller/kernel_type.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/amd_detail/rocblaslt/src/rocroller/include/rocroller/parameter_selection.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/amd_detail/rocblaslt/src/rocroller/include/rocroller/rocroller_host.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/amd_detail/rocblaslt/src/rocroller/include/rocroller/runtime_args_selection.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/amd_detail/rocblaslt/src/rocroller/include/rocroller/solution_cache.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/amd_detail/rocblaslt/src/rocroller/include/rocroller/solution_selection.hpp"
)
