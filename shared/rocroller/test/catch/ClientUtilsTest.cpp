/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2025 AMD ROCm(TM) Software
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
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <client/Utils.hpp>

#include <cmath>

using namespace Catch;

TEST_CASE("GEMM Speed calculation", "[client][utilities]")
{
    SECTION("Should be NaN if execution time is <= zero")
    {
        REQUIRE(std::isnan(gemmSpeedInGigaFLOPS(11, 22, 33, 0)));

        REQUIRE(std::isnan(gemmSpeedInGigaFLOPS(11, 22, 33, -96.024)));
    }

    SECTION("Speed should be calculated correctly")
    {
        auto [M, N, K, executionTimeInSeconds, expectedGFLOPS]
            = GENERATE(std::tuple<int, int, int, double, double>{128, 256, 512, 0.05, 6.71e-01},
                       std::tuple<int, int, int, double, double>{64, 128, 256, 0.02, 2.10e-01},
                       std::tuple<int, int, int, double, double>{100, 200, 300, 0.1, 1.20e-01},
                       std::tuple<int, int, int, double, double>{512, 1024, 2048, 0.5, 4.29e00});

        REQUIRE(gemmSpeedInGigaFLOPS(M, N, K, executionTimeInSeconds)
                == Approx(expectedGFLOPS).margin(0.01));
    }
}
