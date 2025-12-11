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

namespace
{
    class DummySingletonA
    {
    public:
        int value{0};

        void reset()
        {
            value = 0;
        }
    };

    class DummySingletonB
    {
    public:
        int value{99};

        void reset()
        {
            value = 99;
        }
    };
}

TEST_CASE("LazySingletonUnit: Same instance is returned", "[utils]")
{
    auto* a = rocRoller::LazySingleton<DummySingletonA>::getInstance();
    auto* b = rocRoller::LazySingleton<DummySingletonA>::getInstance();

    REQUIRE(a != nullptr);
    REQUIRE(b != nullptr);
    REQUIRE(a == b);
}

TEST_CASE("LazySingletonUnit: Reset does not replace singleton instance (in-place)", "[utils]")
{
    auto* instance1 = rocRoller::LazySingleton<DummySingletonA>::getInstance();
    REQUIRE(instance1 != nullptr);

    instance1->value = 123; // mutate

    rocRoller::LazySingleton<DummySingletonA>::reset();

    auto* instance2 = rocRoller::LazySingleton<DummySingletonA>::getInstance();
    REQUIRE(instance2 != nullptr);

    REQUIRE(instance1 == instance2);
    REQUIRE(instance2->value == 0);
}

TEST_CASE("LazySingletonUnit: Different types have independent instances", "[utils]")
{
    auto* a = rocRoller::LazySingleton<DummySingletonA>::getInstance();
    auto* b = rocRoller::LazySingleton<DummySingletonB>::getInstance();

    REQUIRE(a != nullptr);
    REQUIRE(b != nullptr);

    REQUIRE(static_cast<const void*>(a) != static_cast<const void*>(b));
}

TEST_CASE("LazySingletonUnit: Singleton persists across scopes", "[utils]")
{
    auto* inst1 = rocRoller::LazySingleton<DummySingletonA>::getInstance();
    REQUIRE(inst1 != nullptr);

    {
        auto* inst2 = rocRoller::LazySingleton<DummySingletonA>::getInstance();
        REQUIRE(inst2 == inst1);
    }

    auto* inst3 = rocRoller::LazySingleton<DummySingletonA>::getInstance();
    REQUIRE(inst3 == inst1);
}

TEST_CASE("LazySingletonUnit: Multiple resets keep same singleton instance", "[utils]")
{
    auto* prevInstance = rocRoller::LazySingleton<DummySingletonB>::getInstance();
    REQUIRE(prevInstance != nullptr);

    prevInstance->value = 100;

    for(int i = 0; i < 5; ++i)
    {
        rocRoller::LazySingleton<DummySingletonB>::reset();
        auto* newInstance = rocRoller::LazySingleton<DummySingletonB>::getInstance();

        REQUIRE(newInstance != nullptr);
        REQUIRE(prevInstance == newInstance);
        REQUIRE(newInstance->value == 99);

        newInstance->value = 999;
    }
}

TEST_CASE("LazySingletonUnit: Thread safety under concurrent access", "[utils]")
{
    constexpr int                 N = 32;
    std::vector<DummySingletonA*> results(N);

    std::vector<std::thread> threads;
    threads.reserve(N);

    for(int i = 0; i < N; ++i)
    {
        threads.emplace_back([&results, i] {
            results[i] = rocRoller::LazySingleton<DummySingletonA>::getInstance();
        });
    }

    for(auto& t : threads)
        t.join();

    for(int i = 1; i < N; ++i)
        REQUIRE(results[i] == results[0]);
}

TEST_CASE("LazySingletonUnit: Reset under concurrent access does not crash", "[utils]")
{
    auto* baseline = rocRoller::LazySingleton<DummySingletonA>::getInstance();
    REQUIRE(baseline != nullptr);

    constexpr int            N = 16;
    std::vector<std::thread> threads;
    threads.reserve(N);

    for(int i = 0; i < N; ++i)
    {
        threads.emplace_back([i] {
            if(i % 5 == 0)
            {
                rocRoller::LazySingleton<DummySingletonA>::reset();
            }
            else
            {
                (void)rocRoller::LazySingleton<DummySingletonA>::getInstance();
            }
        });
    }

    for(auto& t : threads)
        t.join();

    auto* after = rocRoller::LazySingleton<DummySingletonA>::getInstance();
    REQUIRE(after != nullptr);
    REQUIRE(after == baseline);
}

TEST_CASE("LazySingletonUnit: DummySingletonB reset keeps identity", "[utils]")
{
    auto* obj1 = rocRoller::LazySingleton<DummySingletonB>::getInstance();
    REQUIRE(obj1 != nullptr);

    obj1->value = 777;

    rocRoller::LazySingleton<DummySingletonB>::reset();

    auto* obj2 = rocRoller::LazySingleton<DummySingletonB>::getInstance();
    REQUIRE(obj2 != nullptr);

    REQUIRE(obj1 == obj2); // identity unchanged
    REQUIRE(obj2->value == 99); // state reset
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
    const bool original = rocRoller::Settings::Get(rocRoller::Settings::LogConsole);

    auto* settings = rocRoller::Settings::getInstance();
    REQUIRE(settings != nullptr); // fundamental precondition

    const bool flipped = !original;
    settings->set(rocRoller::Settings::LogConsole, flipped);

    REQUIRE(rocRoller::Settings::Get(rocRoller::Settings::LogConsole) == flipped);

    settings->set(rocRoller::Settings::LogConsole, original);
    CHECK(rocRoller::Settings::Get(rocRoller::Settings::LogConsole) == original);
}

TEST_CASE("LazySingletonAPI: Settings string option round-trip", "[utils][API]")
{
    auto* settings = rocRoller::Settings::getInstance();
    REQUIRE(settings != nullptr);

    std::string customPath = "/tmp/rocm_custom";
    settings->set(rocRoller::Settings::ROCMPath, customPath);

    CHECK(settings->get(rocRoller::Settings::ROCMPath) == customPath);
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