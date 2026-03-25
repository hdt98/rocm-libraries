// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_data_sdk/data_objects/graph_generated.h>

namespace sdpa_kernel_provider
{
flatbuffers::FlatBufferBuilder
    createValidSdpaFpropGraph(const std::vector<int64_t>& dims = {4, 8, 256, 128},
                              hipdnn_data_sdk::data_objects::DataType dataType
                              = hipdnn_data_sdk::data_objects::DataType::BFLOAT16,
                              hipdnn_data_sdk::data_objects::DataType computeType
                              = hipdnn_data_sdk::data_objects::DataType::FLOAT,
                              bool withAttnMask = false,
                              bool withScale = false,
                              bool withStats = false,
                              bool alibiMask = false,
                              bool paddingMask = false,
                              bool causalMask = false);
}
