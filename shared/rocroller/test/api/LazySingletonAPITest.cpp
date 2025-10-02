/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
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

#include <catch2/catch_test_macros.hpp>

#include <rocRoller/Utilities/LazySingleton.hpp>
#include <rocRoller/Utilities/Settings.hpp>
#include <rocRoller/GPUArchitecture/GPUArchitectureLibrary.hpp>

#include <thread>
#include <vector>

TEST_CASE("API: Derived singleton works (GPUArchitectureLibrary)", "[API:LazySingleton]") {
    auto gpu1 = rocRoller::LazySingleton<rocRoller::GPUArchitectureLibrary>::getInstance();
    auto gpu2 = rocRoller::LazySingleton<rocRoller::GPUArchitectureLibrary>::getInstance();
    REQUIRE(gpu1 == gpu2);
}

TEST_CASE("API: Thread safety across dynamic linking boundary", "[API:LazySingleton]") {
    constexpr int N = 16;
    std::vector<std::shared_ptr<rocRoller::GPUArchitectureLibrary>> results(N);

    std::vector<std::thread> threads;
    for (int i = 0; i < N; ++i) {
        threads.emplace_back([&results, i] {
            results[i] = rocRoller::LazySingleton<rocRoller::GPUArchitectureLibrary>::getInstance();
        });
    }
    for (auto& t : threads) t.join();

    for (int i = 1; i < N; ++i) {
        REQUIRE(results[i] == results[0]);
    }
}

TEST_CASE("API: Reset is globally visible across threads", "[API:LazySingleton]") {
    auto first = rocRoller::LazySingleton<rocRoller::Settings>::getInstance();
    auto rawFirst = first.get();

    std::thread t([] {
        rocRoller::LazySingleton<rocRoller::Settings>::reset();
    });
    t.join();

    auto second = rocRoller::LazySingleton<rocRoller::Settings>::getInstance();
    auto rawSecond = second.get();

    REQUIRE(rawFirst != rawSecond);
}

TEST_CASE("API: Shared_ptr stability across reset", "[API:LazySingleton]") {
    auto inst1 = rocRoller::LazySingleton<rocRoller::GPUArchitectureLibrary>::getInstance();
    auto inst2 = rocRoller::LazySingleton<rocRoller::GPUArchitectureLibrary>::getInstance();

    REQUIRE(inst1 == inst2);

    // Reset should not invalidate previously held shared_ptrs
    rocRoller::LazySingleton<rocRoller::GPUArchitectureLibrary>::reset();

    auto inst3 = rocRoller::LazySingleton<rocRoller::GPUArchitectureLibrary>::getInstance();
    REQUIRE(inst1 != inst3);

    // The old shared_ptr is still valid
    REQUIRE(inst1 != nullptr);
}

TEST_CASE("API: Resetting one singleton does not affect another", "[API:LazySingleton]") {
    auto settings1 = rocRoller::LazySingleton<rocRoller::Settings>::getInstance();
    auto gpu1      = rocRoller::LazySingleton<rocRoller::GPUArchitectureLibrary>::getInstance();

    rocRoller::LazySingleton<rocRoller::Settings>::reset();

    auto settings2 = rocRoller::LazySingleton<rocRoller::Settings>::getInstance();
    auto gpu2      = rocRoller::LazySingleton<rocRoller::GPUArchitectureLibrary>::getInstance();

    REQUIRE(settings1 != settings2);  // reset affected only Settings
    REQUIRE(gpu1 == gpu2);            // GPUArchitectureLibrary unaffected
}