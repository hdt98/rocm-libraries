/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <miopen/origami_search.hpp>

#include <origami/types.hpp>

namespace miopen {
namespace solver {

origami::data_type_t GetOriDataType(const miopen::conv::ProblemDescription& problem)
{
    switch(problem.GetInDataType())
    {
    case miopenHalf: return origami::data_type_t::Half;
    case miopenFloat: return origami::data_type_t::Float;
    case miopenInt8: return origami::data_type_t::Int8;
    case miopenBFloat16: return origami::data_type_t::BFloat16;
    case miopenInt64: return origami::data_type_t::Int64;
    case miopenInt32: return origami::data_type_t::Int32;
    case miopenFloat8_fnuz: return origami::data_type_t::Float8_fnuz;
    case miopenBFloat8_fnuz: return origami::data_type_t::BFloat8_fnuz;
    case miopenDouble: return origami::data_type_t::Double;
    }
    return origami::data_type_t::None;
}

std::string SerializeOrigamiConfig(const origami::config_t config)
{
    std::string ret;
    ret = std::to_string(config.mt.m);
    ret += "_" + std::to_string(config.mt.n);
    ret += "_" + std::to_string(config.mt.k);
    ret += "_" + std::to_string(config.mi.m);
    ret += "_" + std::to_string(config.mi.n);
    ret += "_" + std::to_string(config.mi.k);
    ret += "_" + std::to_string(config.occupancy);
    return ret;
}

} // namespace solver
} // namespace miopen
