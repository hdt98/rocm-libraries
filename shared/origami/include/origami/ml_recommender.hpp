// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "origami/types.hpp"
#include "origami/hardware.hpp"

#include <string>
#include <vector>

namespace origami {
namespace ml_recommender {

bool load_weights(const std::string& bin_path);

bool weights_loaded();

std::vector<prediction_result_t> rank_configs(const problem_t& problem,
                                              const hardware_t& hardware,
                                              const std::vector<config_t>& configs);

bool cluster_uses_ml(int cluster_id);

int route_cluster_for_problem(const problem_t& problem);

}
}
