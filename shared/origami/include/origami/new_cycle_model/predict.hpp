#pragma once

#include "origami/new_cycle_model/params.hpp"
#include "origami/hardware.hpp"
#include "origami/types.hpp"

namespace origami {
namespace new_cycle_model {

double predict(const problem_t& problem,
               const hardware_t& hardware,
               const config_t& config,
               const model_params_t& params = model_params_t{});

bool is_enabled();

} // namespace new_cycle_model
} // namespace origami
