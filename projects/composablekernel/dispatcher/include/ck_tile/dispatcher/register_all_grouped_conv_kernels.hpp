// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Forward declaration of registration functions for generated backward weight kernels.
// The implementation is auto-generated and compiled in the OBJECT library.

#pragma once

#include <string>

namespace ck_tile {
namespace dispatcher {

class GroupedConvRegistry;

void register_all_grouped_conv_bwd_weight_kernels(GroupedConvRegistry& registry,
                                                  const std::string& arch);

void register_all_grouped_conv_bwd_weight_kernels(const std::string& arch);

} // namespace dispatcher
} // namespace ck_tile
