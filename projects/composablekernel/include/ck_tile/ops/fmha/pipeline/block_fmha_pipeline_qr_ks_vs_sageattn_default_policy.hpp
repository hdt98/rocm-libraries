// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_default_policy.hpp"

namespace ck_tile {

// SA3 pipeline reuses the same tiling policy as the MX pipeline.
using BlockFmhaPipelineQRKSVSSageAttnDefaultPolicy = BlockFmhaPipelineQRKSVSDefaultPolicy;

} // namespace ck_tile
