// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <vector>
#include <string>
#include "ck/host/types.hpp"
#include "ck/host/device_fmha_bwd_dq_dk_dv/problem.hpp"

namespace ck {
namespace host {
namespace device_fmha_bwd_dq_dk_dv {

struct Operation
{
    DataType dtype         = DataType::Half;
    bool is_causal         = false;
    bool has_bias          = false;
    bool has_dbias         = false;
    bool has_dropout       = false;
    bool is_deterministic  = false;
    bool is_store_randval  = false;
    bool use_trload        = false;
    static std::vector<Operation> CreateOperations(const Problem& prob, const std::string& arch);
    Solution ToSolution() const;
};

} // namespace device_fmha_bwd_dq_dk_dv
} // namespace host
} // namespace ck
