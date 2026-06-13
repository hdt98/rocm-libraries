# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

add_ops_compile_options(
        OP_NAME MoeComputeExpertTokens
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
)

target_sources(op_host_aclnnInner PRIVATE
        moe_compute_expert_tokens.cpp
)

target_sources(opapi PRIVATE
        moe_compute_expert_tokens.cpp
)
if (NOT BUILD_OPEN_PROJECT)
    target_sources(aclnn_ops_train PRIVATE
            moe_compute_expert_tokens.cpp
    )

    target_sources(aclnn_ops_infer PRIVATE
            moe_compute_expert_tokens.cpp
    )
endif ()

target_sources(optiling PRIVATE
        moe_compute_expert_tokens_tiling.cc
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../../moe_compute_expert_tokens/ophost
)

target_sources(opsproto PRIVATE
        moe_compute_expert_tokens_ops.cc
)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/aclnn_moe_compute_expert_tokens.h
        DESTINATION ${ACLNN_INC_INSTALL_DIR} OPTIONAL
)
