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

TEST_CASE("LazySingletonAPI: Reset is safe across threads", "[API:LazySingleton]")
{
    auto before = rocRoller::Settings::getInstance();
    REQUIRE(before != nullptr);

    // Perform concurrent resets to stress test thread safety
    constexpr int            numThreads = 8;
    std::vector<std::thread> threads;
    threads.reserve(numThreads);

    for(int i = 0; i < numThreads; ++i)
    {
        threads.emplace_back([] { rocRoller::Settings::reset(); });
    }

    for(auto& t : threads)
        t.join();

    auto after = rocRoller::Settings::getInstance();
    REQUIRE(after != nullptr);

    // Because dynamic linking rebuilds instance, we expect new shared_ptr
    REQUIRE(before != after);
}

TEST_CASE("LazySingletonAPI: Shared_ptr after reset points to new instance", "[API:LazySingleton]")
{
    auto oldInstance = rocRoller::GPUArchitectureLibrary::getInstance();
    REQUIRE(oldInstance != nullptr);

    rocRoller::GPUArchitectureLibrary::reset();

    auto newInstance = rocRoller::GPUArchitectureLibrary::getInstance();
    REQUIRE(newInstance != nullptr);

    // Since reset() reinitializes, pointers must differ
    REQUIRE(oldInstance != newInstance);

    // Old pointer is still valid (shared_ptr kept last reference)
    REQUIRE(oldInstance.use_count() >= 1);
}

TEST_CASE("LazySingletonAPI: Different singletons remain independent", "[API:LazySingleton]")
{
    auto settings = rocRoller::Settings::getInstance();
    auto gpuLib   = rocRoller::GPUArchitectureLibrary::getInstance();

    REQUIRE(settings != nullptr);
    REQUIRE(gpuLib != nullptr);
    REQUIRE(static_cast<const void*>(settings.get()) != static_cast<const void*>(gpuLib.get()));

    // Reset both independently -- each should reinitialize its own singleton only
    rocRoller::Settings::reset();
    rocRoller::GPUArchitectureLibrary::reset();

    auto settings2 = rocRoller::Settings::getInstance();
    auto gpuLib2   = rocRoller::GPUArchitectureLibrary::getInstance();

    // Each reset should have produced a *new* instance of that singleton type
    REQUIRE(settings != settings2);
    REQUIRE(gpuLib != gpuLib2);

    // But still distinct from each other (type independence preserved)
    REQUIRE(static_cast<const void*>(settings2.get()) != static_cast<const void*>(gpuLib2.get()));
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
