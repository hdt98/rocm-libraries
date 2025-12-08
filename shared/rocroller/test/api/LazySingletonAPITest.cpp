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
#include <rocRoller/Utilities/Logging.hpp>
#include <rocRoller/Utilities/Settings.hpp>

#include <string>
#include <thread>
#include <vector>

//GPUArchitectureLibrary singleton is stable via its own API
TEST_CASE("LazySingletonAPI: GPUArchitectureLibrary getInstance() is stable", "[utils][API]")
{
    auto a = rocRoller::GPUArchitectureLibrary::getInstance();
    auto b = rocRoller::GPUArchitectureLibrary::getInstance();
    REQUIRE(a != nullptr);
    REQUIRE(b != nullptr);
    REQUIRE(a == b);
}

// Settings change is visible through library's read path (Settings::Get)
TEST_CASE("LazySingletonAPI: Settings change visible via Settings::Get", "[utils][API]")
{
    auto settings = rocRoller::Settings::getInstance();
    REQUIRE(settings != nullptr);

    // Set through instance (simulating an API user's program)
    settings->set(rocRoller::Settings::LogConsole, false);

    // Read back through the library's static read path (simulates "inside the library")
    REQUIRE(rocRoller::Settings::Get(rocRoller::Settings::LogConsole) == false);

    // Restore default to avoid cross-test pollution
    settings->set(rocRoller::Settings::LogConsole, true);
}

// Settings string option change is globally visible and consistent
TEST_CASE("LazySingletonAPI: Settings string option round-trip", "[utils][API]")
{
    auto settings = rocRoller::Settings::getInstance();
    REQUIRE(settings != nullptr);

    std::string customPath = "/tmp/rocm_custom";
    settings->set(rocRoller::Settings::ROCMPath, customPath);

    // Read via instance and static path
    REQUIRE(settings->get(rocRoller::Settings::ROCMPath) == customPath);
    REQUIRE(rocRoller::Settings::Get(rocRoller::Settings::ROCMPath) == customPath);
}

// Multiple options remain independent
TEST_CASE("LazySingletonAPI: Independent Settings options remain independent", "[utils][API]")
{
    auto settings = rocRoller::Settings::getInstance();
    REQUIRE(settings != nullptr);

    settings->set(rocRoller::Settings::LogConsole, false);
    settings->set(rocRoller::Settings::ROCMPath, "/tmp/test_path");

    REQUIRE(settings->get(rocRoller::Settings::LogConsole) == false);
    REQUIRE(settings->get(rocRoller::Settings::ROCMPath) == std::string("/tmp/test_path"));

    // Restore
    settings->set(rocRoller::Settings::LogConsole, true);
}

// Settings visibility across threads using only public API
TEST_CASE("LazySingletonAPI: Settings visibility across threads (public API only)", "[utils][API]")
{
    // Writer toggles a setting; readers check via static Get()
    constexpr int writers = 2;
    constexpr int readers = 8;

    auto settings = rocRoller::Settings::getInstance();
    REQUIRE(settings != nullptr);

    // Start with true
    settings->set(rocRoller::Settings::LogConsole, true);

    std::vector<std::thread> ts;

    // Readers
    for(int i = 0; i < readers; ++i)
    {
        ts.emplace_back([] { (void)rocRoller::Settings::Get(rocRoller::Settings::LogConsole); });
    }

    // Writers
    for(int i = 0; i < writers; ++i)
    {
        ts.emplace_back([settings, i] {
            settings->set(rocRoller::Settings::LogConsole, (i % 2) == 0 ? false : true);
        });
    }

    for(auto& t : ts)
        t.join();

    // Final state is whichever last writer set—exact truth value is not asserted here
    // we only assert that public reads do not crash and are reachable.
    REQUIRE_NOTHROW((void)rocRoller::Settings::Get(rocRoller::Settings::LogConsole));
}

TEST_CASE("LazySingletonAPI: Logging behavior reflects Settings toggles", "[utils][API]")
{
    // Snapshot so test is side-effect free
    const bool prevConsole            = rocRoller::Settings::Get(rocRoller::Settings::LogConsole);
    const rocRoller::LogLevel prevLvl = rocRoller::Settings::Get(rocRoller::Settings::LogLvl);
    const rocRoller::LogLevel prevCLvl
        = rocRoller::Settings::Get(rocRoller::Settings::LogConsoleLvl);

    auto settings = rocRoller::Settings::getInstance();
    auto logger   = rocRoller::Log::getLogger();

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