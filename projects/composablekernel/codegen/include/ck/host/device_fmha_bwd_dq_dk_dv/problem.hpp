// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Phase 6 scaffolding: main backward kernel. Computes dQ, dK, dV, and
// optionally dBias / rand_val. Reads the workspace `d_ptr` produced by
// stage 1 (bwd_dot_do_o) and writes into `dq_acc_ptr` (fp32) for later
// conversion by stage 3 (bwd_convert_dq).

#include <cstdlib>
#include <vector>
#include <string>
#include "ck/host/types.hpp"

namespace ck {
namespace host {
namespace device_fmha_bwd_dq_dk_dv {

struct Problem
{
    std::size_t M = 0;
    std::size_t N = 0;
    std::size_t K = 0;
    std::size_t O = 0;
    std::size_t batch   = 0;
    std::size_t nhead_q = 0;
    std::size_t nhead_k = 0;
    DataType dtype = DataType::Half;
    bool is_causal = false;
    bool has_bias  = false;
    bool has_dbias = false;
    bool has_dropout = false;
    bool is_deterministic  = false;
    bool is_store_randval  = false;
    bool use_trload        = false;
    std::string GetIncludeHeader() const;
    std::vector<Solution> GetSolutions(const std::string& arch) const;
};

} // namespace device_fmha_bwd_dq_dk_dv
} // namespace host
} // namespace ck
