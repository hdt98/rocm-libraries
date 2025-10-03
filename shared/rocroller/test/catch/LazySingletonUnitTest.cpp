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

TEST_CASE("LazySingletonUnit: Same instance is returned", "Unit:LazySingleton")
{
    auto a = rocRoller::LazySingleton<rocRoller::Settings>::getInstance();
    auto b = rocRoller::LazySingleton<rocRoller::Settings>::getInstance();
    REQUIRE(a == b);
}

TEST_CASE("LazySingletonUnit: Reset does not break singleton instance", "Unit:LazySingleton")
{
    auto instance1 = rocRoller::Settings::getInstance();
    REQUIRE(instance1 != nullptr);

    rocRoller::Settings::reset(); // no-op in dynamic-linking version

    auto instance2 = rocRoller::Settings::getInstance();
    REQUIRE(instance2 != nullptr);

    // In dynamic-linking safe design, reset does not change identity
    REQUIRE(instance1 == instance2);
}

TEST_CASE("LazySingletonUnit: Different types have independent instances", "Unit:LazySingleton")
{
    auto settings = rocRoller::LazySingleton<rocRoller::Settings>::getInstance();
    auto gpuLib   = rocRoller::LazySingleton<rocRoller::GPUArchitectureLibrary>::getInstance();

    // Compare addresses as void* to avoid cross-type pointer comparison
    const void* s_addr = static_cast<const void*>(settings.get());
    const void* g_addr = static_cast<const void*>(gpuLib.get());
    REQUIRE(s_addr != g_addr);
}

TEST_CASE("LazySingletonUnit: Singleton persists across scopes", "Unit:LazySingleton")
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

TEST_CASE("LazySingletonUnit: Multiple resets keep the same singleton instance",
          "Unit:LazySingleton")
{
    auto instance1 = rocRoller::GPUArchitectureLibrary::getInstance();

    for(int i = 0; i < 5; ++i)
    {
        rocRoller::GPUArchitectureLibrary::reset();
        auto instance2 = rocRoller::GPUArchitectureLibrary::getInstance();
        REQUIRE(instance2 != nullptr);
        REQUIRE(instance1 == instance2); // Always the same object
    }
}

TEST_CASE("LazySingletonUnit: Thread safety under concurrent access", "Unit:LazySingleton")
{
    constexpr int                                     N = 32;
    std::vector<std::shared_ptr<rocRoller::Settings>> results(N);

    std::vector<std::thread> threads;
    for(int i = 0; i < N; ++i)
    {
        threads.emplace_back([&results, i] {
            results[i] = rocRoller::LazySingleton<rocRoller::Settings>::getInstance();
        });
    }
    for(auto& t : threads)
        t.join();

    for(int i = 1; i < N; ++i)
    {
        REQUIRE(results[i] == results[0]);
    }
}

TEST_CASE("LazySingletonUnit: Reset under concurrent access does not crash", "Unit:LazySingleton")
{
    // Stress test where one thread resets while others read
    constexpr int            N = 16;
    std::vector<std::thread> threads;

    for(int i = 0; i < N; ++i)
    {
        threads.emplace_back([i] {
            // Mix reads and resets
            if(i % 5 == 0)
            {
                rocRoller::LazySingleton<rocRoller::Settings>::reset();
            }
            else
            {
                (void)rocRoller::LazySingleton<rocRoller::Settings>::getInstance();
            }
        });
    }

    for(auto& t : threads)
        t.join();

    // Should still return a valid instance
    auto inst = rocRoller::LazySingleton<rocRoller::Settings>::getInstance();
    REQUIRE(inst != nullptr);
}
