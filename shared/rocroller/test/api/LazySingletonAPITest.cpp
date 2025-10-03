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

TEST_CASE("LazySingletonAPI: Derived singleton works (GPUArchitectureLibrary)", "API:LazySingleton")
{
    auto gpu1 = rocRoller::LazySingleton<rocRoller::GPUArchitectureLibrary>::getInstance();
    auto gpu2 = rocRoller::LazySingleton<rocRoller::GPUArchitectureLibrary>::getInstance();
    REQUIRE(gpu1 == gpu2);
}

TEST_CASE("LazySingletonAPI: Thread safety across dynamic linking boundary", "API:LazySingleton")
{
    constexpr int                                                   N = 16;
    std::vector<std::shared_ptr<rocRoller::GPUArchitectureLibrary>> results(N);

    std::vector<std::thread> threads;
    for(int i = 0; i < N; ++i)
    {
        threads.emplace_back([&results, i] {
            results[i] = rocRoller::LazySingleton<rocRoller::GPUArchitectureLibrary>::getInstance();
        });
    }
    for(auto& t : threads)
        t.join();

    for(int i = 1; i < N; ++i)
    {
        REQUIRE(results[i] == results[0]);
    }
}

TEST_CASE("LazySingletonAPI: Reset is safe across threads", "API:LazySingleton")
{
    auto before = rocRoller::Settings::getInstance();

    std::thread t([] { rocRoller::Settings::reset(); });
    t.join();

    auto after = rocRoller::Settings::getInstance();

    // In dynamic-linking design, both are the same
    REQUIRE(before == after);
}
TEST_CASE("LazySingletonAPI: Shared_ptr remains valid after reset", "API:LazySingleton")
{
    auto instance = rocRoller::GPUArchitectureLibrary::getInstance();
    REQUIRE(instance != nullptr);

    rocRoller::GPUArchitectureLibrary::reset();

    // After reset, pointer should still be valid and identical
    auto instance2 = rocRoller::GPUArchitectureLibrary::getInstance();
    REQUIRE(instance2 != nullptr);
    REQUIRE(instance == instance2);
}

TEST_CASE("LazySingletonAPI: Different singletons remain independent", "API:LazySingleton")
{
    auto settings = rocRoller::Settings::getInstance();
    auto gpuLib   = rocRoller::GPUArchitectureLibrary::getInstance();

    rocRoller::Settings::reset();
    auto settings2 = rocRoller::Settings::getInstance();
    auto gpuLib2   = rocRoller::GPUArchitectureLibrary::getInstance();

    REQUIRE(settings == settings2); // Reset didn't change Settings
    REQUIRE(gpuLib == gpuLib2); // GPUArchitectureLibrary unaffected
    REQUIRE(static_cast<const void*>(settings.get())
            != static_cast<const void*>(gpuLib.get())); // Distinct singletons
}

TEST_CASE("LazySingletonAPI: Settings boolean option change is globally visible",
          "API:LazySingleton:Settings")
{
    auto settings = rocRoller::Settings::getInstance();

    // Set LogConsole to false, simulate API user program change
    settings->set(rocRoller::Settings::LogConsole, false);

    // From "library" code, fetch again through singleton
    auto settingsAgain = rocRoller::Settings::getInstance();
    REQUIRE(settingsAgain->get(rocRoller::Settings::LogConsole) == false);

    // Reset to default
    settingsAgain->set(rocRoller::Settings::LogConsole, true);
}

TEST_CASE("LazySingletonAPI: Settings string option change is globally visible",
          "API:LazySingleton:Settings")
{
    auto settings = rocRoller::Settings::getInstance();

    std::string customPath = "/tmp/rocm_custom";
    settings->set(rocRoller::Settings::ROCMPath, customPath);

    auto libView = rocRoller::Settings::getInstance();
    REQUIRE(libView->get(rocRoller::Settings::ROCMPath) == customPath);
}

TEST_CASE("LazySingletonAPI: Static Get reflects changes made via instance set",
          "API:LazySingleton:Settings")
{
    auto settings = rocRoller::Settings::getInstance();

    settings->set(rocRoller::Settings::LogConsole, false);

    // Static Get should return updated value
    REQUIRE(rocRoller::Settings::Get(rocRoller::Settings::LogConsole) == false);

    // Restore
    settings->set(rocRoller::Settings::LogConsole, true);
}

TEST_CASE("LazySingletonAPI: Independent settings options remain independent",
          "API:LazySingleton:Settings")
{
    auto settings = rocRoller::Settings::getInstance();

    settings->set(rocRoller::Settings::LogConsole, false);
    settings->set(rocRoller::Settings::ROCMPath, "/tmp/test_path");

    REQUIRE(settings->get(rocRoller::Settings::LogConsole) == false);
    REQUIRE(settings->get(rocRoller::Settings::ROCMPath) == "/tmp/test_path");
}
