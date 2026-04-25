// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qx_ks_vs_custom_policy.hpp"

namespace ck_tile {

using BlockFmhaPipelineQRKSVSSageAttnDefaultPolicy =
    BlockFmhaPipelineQXKSVSCustomPolicy</* QLoadOnce = */ true,
                                        /* AsyncCopy = */ true,
                                        /* NumPrefetchK = */ 1,
                                        /* NumPrefetchV = */ 1>;

} // namespace ck_tile
