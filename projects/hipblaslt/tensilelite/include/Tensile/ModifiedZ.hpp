/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
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

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace TensileLite
{
    namespace ModifiedZ
    {
        inline double removeOutliersAndGetMean(std::vector<double> values, double zThreshold)
        {
            const auto mean = [](const std::vector<double>& samples) {
                if(samples.empty())
                    throw std::runtime_error("Cannot calculate mean of empty timing samples");
                return std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
            };
            const auto medianOfSortedData = [](const std::vector<double>& sortedData) {
                if(sortedData.empty())
                    throw std::runtime_error("Cannot calculate median of empty timing samples");
                const auto size = sortedData.size();
                return (size % 2 == 0) ? (sortedData[size / 2 - 1] + sortedData[size / 2]) / 2.0
                                       : sortedData[size / 2];
            };
            const auto median = [&](std::vector<double>& samples) {
                std::sort(samples.begin(), samples.end());
                return medianOfSortedData(samples);
            };
            const auto scores = [&](const std::vector<double>& sortedData) {
                const double medianValue = medianOfSortedData(sortedData);
                std::vector<double> absoluteDeviation;
                absoluteDeviation.reserve(sortedData.size());
                std::transform(sortedData.begin(),
                               sortedData.end(),
                               std::back_inserter(absoluteDeviation),
                               [&](auto value) { return std::abs(value - medianValue); });
                const double mad = median(absoluteDeviation);
                if(mad == 0.0)
                {
                    std::vector<double> result;
                    result.reserve(sortedData.size());
                    std::transform(sortedData.begin(),
                                   sortedData.end(),
                                   std::back_inserter(result),
                                   [&](auto value) {
                                       if(value == medianValue)
                                           return 0.0;
                                       return value > medianValue
                                                  ? std::numeric_limits<double>::infinity()
                                                  : -std::numeric_limits<double>::infinity();
                                   });
                    return result;
                }

                std::vector<double> result;
                result.reserve(sortedData.size());
                std::transform(sortedData.begin(),
                               sortedData.end(),
                               std::back_inserter(result),
                               [&](auto value) { return 0.6745 * (value - medianValue) / mad; });
                return result;
            };

            std::sort(values.begin(), values.end());
            const auto zScores = scores(values);
            std::vector<double> filtered;
            filtered.reserve(values.size());
            for(std::size_t i = 0; i < values.size(); ++i)
            {
                if(std::abs(zScores[i]) <= zThreshold)
                    filtered.push_back(values[i]);
            }
            if(filtered.empty())
                throw std::runtime_error("Outlier trimming removed all timing samples");
            return mean(filtered);
        }
    } // namespace ModifiedZ
} // namespace TensileLite
