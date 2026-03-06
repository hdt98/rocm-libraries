// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <string>
#include <tuple>
#include <vector>

namespace TensileLite
{
    using expression_compute_t = float;

    expression_compute_t processExpression(
        std::string const&                                                exprStr,
        std::vector<std::tuple<std::string, expression_compute_t>> const& argValues = {});
}
