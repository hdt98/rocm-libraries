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

#include <rocRoller/GPUArchitecture/GPUArchitectureLibrary.hpp>
#include <rocRoller/Utilities/LazySingleton.hpp>
#include <rocRoller/Utilities/Settings.hpp>

#include <thread>
#include <vector>

TEST_CASE("LazySingletonUnit: Same instance is returned", "[utils]")
{
    auto a = rocRoller::LazySingleton<rocRoller::Settings>::getInstance();
    auto b = rocRoller::LazySingleton<rocRoller::Settings>::getInstance();
    REQUIRE(a == b); // same aliasing shared_ptr
}

TEST_CASE("LazySingletonUnit: Reset does not replace singleton instance (in-place)", "[utils]")
{
    auto instance1 = rocRoller::LazySingleton<rocRoller::Settings>::getInstance();
    REQUIRE(instance1 != nullptr);

    // In-place reset (does not change identity)
    rocRoller::LazySingleton<rocRoller::Settings>::reset();

    auto instance2 = rocRoller::LazySingleton<rocRoller::Settings>::getInstance();
    REQUIRE(instance2 != nullptr);

    // Identity is unchanged; state is reset internally by Settings
    REQUIRE(instance1 == instance2);
}

TEST_CASE("LazySingletonUnit: Different types have independent instances", "[utils]")
{
    auto settings = rocRoller::LazySingleton<rocRoller::Settings>::getInstance();
    auto gpuLib   = rocRoller::LazySingleton<rocRoller::GPUArchitectureLibrary>::getInstance();

    // Compare addresses as void* to avoid cross-type pointer comparison
    const void* s_addr = static_cast<const void*>(settings.get());
    const void* g_addr = static_cast<const void*>(gpuLib.get());
    REQUIRE(s_addr != g_addr);
}

TEST_CASE("LazySingletonUnit: Singleton persists across scopes", "[utils]")
{
    auto inst1 = rocRoller::LazySingleton<rocRoller::Settings>::getInstance();
    auto raw1  = inst1.get();

    {
        auto inst2 = rocRoller::LazySingleton<rocRoller::Settings>::getInstance();
        REQUIRE(inst2.get() == raw1);
    }

    auto inst3 = rocRoller::LazySingleton<rocRoller::Settings>::getInstance();
    REQUIRE(inst3.get() == raw1);
}

TEST_CASE("LazySingletonUnit: Multiple resets create new singleton instances", "[utils]")
{
    auto prevInstance = rocRoller::GPUArchitectureLibrary::getInstance();
    REQUIRE(prevInstance != nullptr);

    for(int i = 0; i < 5; ++i)
    {
        rocRoller::LazySingleton<rocRoller::GPUArchitectureLibrary>::reset();
        auto newInstance = rocRoller::GPUArchitectureLibrary::getInstance();
        REQUIRE(newInstance != nullptr);

        // Identity must remain the same with in-place reset
        REQUIRE(prevInstance == newInstance);

        prevInstance = newInstance;
    }
}

TEST_CASE("LazySingletonUnit: Thread safety under concurrent access", "[utils]")
{
    constexpr int                                     N = 32;
    std::vector<std::shared_ptr<rocRoller::Settings>> results(N);

    std::vector<std::thread> threads;
    threads.reserve(N);
    for(int i = 0; i < N; ++i)
    {
        threads.emplace_back([&results, i] {
            results[i] = rocRoller::LazySingleton<rocRoller::Settings>::getInstance();
        });
    }
    for(auto& t : threads)
        t.join();

    for(int i = 1; i < N; ++i)
        REQUIRE(results[i] == results[0]); // aliasing the same object
}

TEST_CASE("LazySingletonUnit: Reset under concurrent access does not crash", "[utils]")
{
    auto baseline = rocRoller::LazySingleton<rocRoller::Settings>::getInstance();
    REQUIRE(baseline != nullptr);

    constexpr int            N = 16;
    std::vector<std::thread> threads;
    threads.reserve(N);

    for(int i = 0; i < N; ++i)
    {
        threads.emplace_back([i] {
            if(i % 5 == 0)
                rocRoller::LazySingleton<rocRoller::Settings>::reset(); // in-place
            else
                (void)rocRoller::LazySingleton<rocRoller::Settings>::getInstance();
        });
    }

    for(auto& t : threads)
        t.join();

    auto after = rocRoller::LazySingleton<rocRoller::Settings>::getInstance();
    REQUIRE(after != nullptr);
    REQUIRE(after == baseline); // identity unchanged
}

TEST_CASE("LazySingletonUnit: GPUArchitectureLibrary reset keeps identity", "[utils]")
{
    auto gpu1 = rocRoller::LazySingleton<rocRoller::GPUArchitectureLibrary>::getInstance();

    // In-place reset (does not change the singleton object identity)
    rocRoller::LazySingleton<rocRoller::GPUArchitectureLibrary>::reset();

    auto gpu2 = rocRoller::LazySingleton<rocRoller::GPUArchitectureLibrary>::getInstance();

    REQUIRE(gpu1 != nullptr);
    REQUIRE(gpu2 != nullptr);
    REQUIRE(gpu1 == gpu2); // identity must be unchanged
}
