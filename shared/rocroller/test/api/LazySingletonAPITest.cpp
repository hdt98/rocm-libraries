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
    REQUIRE(settings != nullptr);

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

TEST_CASE("LazySingletonAPI: GPUArchitectureLibrary::resetState restores loaded library",
          "[utils][API]")
{
    auto* lib = rocRoller::GPUArchitectureLibrary::getInstance();
    REQUIRE(lib != nullptr);

    auto before = lib->getAllSupportedISAs();

    REQUIRE_NOTHROW(lib->resetState());

    auto after = lib->getAllSupportedISAs();
    REQUIRE(after == before);
}
