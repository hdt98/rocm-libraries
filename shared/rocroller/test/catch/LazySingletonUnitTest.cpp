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
#include <rocRoller/Utilities/Logging.hpp>
#include <rocRoller/Utilities/Settings.hpp>

#include <string>
#include <thread>
#include <vector>

TEST_CASE("LazySingletonUnit: Same instance is returned", "[utils]")
{
    auto* a = rocRoller::LazySingleton<rocRoller::Settings>::getInstance();
    auto* b = rocRoller::LazySingleton<rocRoller::Settings>::getInstance();
    REQUIRE(a == b);
}

TEST_CASE("LazySingletonUnit: Reset does not replace singleton instance (in-place)", "[utils]")
{
    auto* instance1 = rocRoller::LazySingleton<rocRoller::Settings>::getInstance();
    REQUIRE(instance1 != nullptr);

    rocRoller::LazySingleton<rocRoller::Settings>::reset();

    auto* instance2 = rocRoller::LazySingleton<rocRoller::Settings>::getInstance();
    REQUIRE(instance2 != nullptr);
    REQUIRE(instance1 == instance2);
}

TEST_CASE("LazySingletonUnit: Different types have independent instances", "[utils]")
{
    auto* settings = rocRoller::LazySingleton<rocRoller::Settings>::getInstance();
    auto* gpuLib   = rocRoller::LazySingleton<rocRoller::GPUArchitectureLibrary>::getInstance();

    const void* s_addr = static_cast<const void*>(settings);
    const void* g_addr = static_cast<const void*>(gpuLib);
    REQUIRE(s_addr != g_addr);
}

TEST_CASE("LazySingletonUnit: Singleton persists across scopes", "[utils]")
{
    auto* inst1 = rocRoller::LazySingleton<rocRoller::Settings>::getInstance();

    {
        auto* inst2 = rocRoller::LazySingleton<rocRoller::Settings>::getInstance();
        REQUIRE(inst2 == inst1);
    }

    auto* inst3 = rocRoller::LazySingleton<rocRoller::Settings>::getInstance();
    REQUIRE(inst3 == inst1);
}

TEST_CASE("LazySingletonUnit: Multiple resets keep same singleton instance", "[utils]")
{
    auto* prevInstance = rocRoller::LazySingleton<rocRoller::GPUArchitectureLibrary>::getInstance();
    REQUIRE(prevInstance != nullptr);

    for(int i = 0; i < 5; ++i)
    {
        rocRoller::LazySingleton<rocRoller::GPUArchitectureLibrary>::reset();
        auto* newInstance
            = rocRoller::LazySingleton<rocRoller::GPUArchitectureLibrary>::getInstance();
        REQUIRE(newInstance != nullptr);

        REQUIRE(prevInstance == newInstance);

        prevInstance = newInstance;
    }
}

TEST_CASE("LazySingletonUnit: Thread safety under concurrent access", "[utils]")
{
    constexpr int                     N = 32;
    std::vector<rocRoller::Settings*> results(N);

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
        REQUIRE(results[i] == results[0]);
}

TEST_CASE("LazySingletonUnit: Reset under concurrent access does not crash", "[utils]")
{
    auto* baseline = rocRoller::LazySingleton<rocRoller::Settings>::getInstance();
    REQUIRE(baseline != nullptr);

    constexpr int            N = 16;
    std::vector<std::thread> threads;
    threads.reserve(N);

    for(int i = 0; i < N; ++i)
    {
        threads.emplace_back([i] {
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

    auto* after = rocRoller::LazySingleton<rocRoller::Settings>::getInstance();
    REQUIRE(after != nullptr);
    REQUIRE(after == baseline);
}

TEST_CASE("LazySingletonUnit: GPUArchitectureLibrary reset keeps identity", "[utils]")
{
    auto* gpu1 = rocRoller::LazySingleton<rocRoller::GPUArchitectureLibrary>::getInstance();

    rocRoller::LazySingleton<rocRoller::GPUArchitectureLibrary>::reset();

    auto* gpu2 = rocRoller::LazySingleton<rocRoller::GPUArchitectureLibrary>::getInstance();

    REQUIRE(gpu1 != nullptr);
    REQUIRE(gpu2 != nullptr);
    REQUIRE(gpu1 == gpu2);
}

TEST_CASE("LazySingletonAPI: GPUArchitectureLibrary getInstance() is stable", "[utils][API]")
{
    auto* a = rocRoller::GPUArchitectureLibrary::getInstance();
    auto* b = rocRoller::GPUArchitectureLibrary::getInstance();
    REQUIRE(a != nullptr);
    REQUIRE(b != nullptr);
    REQUIRE(a == b);
}

TEST_CASE("LazySingletonAPI: Settings change visible via Settings::Get", "[utils][API]")
{
    auto* settings = rocRoller::Settings::getInstance();
    REQUIRE(settings != nullptr);

    settings->set(rocRoller::Settings::LogConsole, false);

    REQUIRE(rocRoller::Settings::Get(rocRoller::Settings::LogConsole) == false);

    settings->set(rocRoller::Settings::LogConsole, true);
}

TEST_CASE("LazySingletonAPI: Settings string option round-trip", "[utils][API]")
{
    auto* settings = rocRoller::Settings::getInstance();
    REQUIRE(settings != nullptr);

    std::string customPath = "/tmp/rocm_custom";
    settings->set(rocRoller::Settings::ROCMPath, customPath);

    REQUIRE(settings->get(rocRoller::Settings::ROCMPath) == customPath);
    REQUIRE(rocRoller::Settings::Get(rocRoller::Settings::ROCMPath) == customPath);
}

TEST_CASE("LazySingletonAPI: Independent Settings options remain independent", "[utils][API]")
{
    auto* settings = rocRoller::Settings::getInstance();
    REQUIRE(settings != nullptr);

    settings->set(rocRoller::Settings::LogConsole, false);
    settings->set(rocRoller::Settings::ROCMPath, "/tmp/test_path");

    REQUIRE(settings->get(rocRoller::Settings::LogConsole) == false);
    REQUIRE(settings->get(rocRoller::Settings::ROCMPath) == std::string("/tmp/test_path"));

    settings->set(rocRoller::Settings::LogConsole, true);
}

TEST_CASE("LazySingletonAPI: Settings visibility across threads (public API only)", "[utils][API]")
{
    constexpr int writers = 2;
    constexpr int readers = 8;

    auto* settings = rocRoller::Settings::getInstance();
    REQUIRE(settings != nullptr);

    settings->set(rocRoller::Settings::LogConsole, true);

    std::vector<std::thread> ts;

    for(int i = 0; i < readers; ++i)
    {
        ts.emplace_back([] { (void)rocRoller::Settings::Get(rocRoller::Settings::LogConsole); });
    }

    for(int i = 0; i < writers; ++i)
    {
        ts.emplace_back([settings, i] {
            settings->set(rocRoller::Settings::LogConsole, (i % 2) == 0 ? false : true);
        });
    }

    for(auto& t : ts)
        t.join();

    REQUIRE_NOTHROW((void)rocRoller::Settings::Get(rocRoller::Settings::LogConsole));
}

TEST_CASE("LazySingletonAPI: Logging behavior reflects Settings toggles", "[utils][API]")
{
    const bool prevConsole            = rocRoller::Settings::Get(rocRoller::Settings::LogConsole);
    const rocRoller::LogLevel prevLvl = rocRoller::Settings::Get(rocRoller::Settings::LogLvl);
    const rocRoller::LogLevel prevCLvl
        = rocRoller::Settings::Get(rocRoller::Settings::LogConsoleLvl);

    auto* settings = rocRoller::Settings::getInstance();
    auto  logger   = rocRoller::Log::getLogger();

    REQUIRE(settings != nullptr);
    REQUIRE(logger != nullptr);

    settings->set(rocRoller::Settings::LogConsole, true);
    settings->set(rocRoller::Settings::LogLvl, rocRoller::LogLevel::Info);
    settings->set(rocRoller::Settings::LogConsoleLvl, rocRoller::LogLevel::Info);

    REQUIRE(logger->should_log(rocRoller::LogLevel::Warning));
    REQUIRE(logger->should_log(rocRoller::LogLevel::Error));
    REQUIRE_FALSE(logger->should_log(rocRoller::LogLevel::Debug));
    REQUIRE_FALSE(logger->should_log(rocRoller::LogLevel::Trace));

    REQUIRE(settings->get(rocRoller::Settings::LogLvl) == rocRoller::LogLevel::Info);

    settings->set(rocRoller::Settings::LogLvl, rocRoller::LogLevel::Warning);
    settings->set(rocRoller::Settings::LogConsoleLvl, rocRoller::LogLevel::Warning);

    REQUIRE(logger->should_log(rocRoller::LogLevel::Warning));
    REQUIRE(logger->should_log(rocRoller::LogLevel::Error));
    REQUIRE(logger->should_log(rocRoller::LogLevel::Critical));
    REQUIRE_FALSE(logger->should_log(rocRoller::LogLevel::Debug));
    REQUIRE_FALSE(logger->should_log(rocRoller::LogLevel::Trace));

    settings->set(rocRoller::Settings::LogConsole, prevConsole);
    settings->set(rocRoller::Settings::LogLvl, prevLvl);
    settings->set(rocRoller::Settings::LogConsoleLvl, prevCLvl);
}