/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2026 AMD ROCm(TM) Software
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

#include <string>
#include <thread>
#include <vector>

namespace
{
    class DummySingletonA : public rocRoller::LazySingleton<DummySingletonA>
    {
    public:
        int value{0};

        void resetState()
        {
            value = 0;
        }

    private:
        DummySingletonA() = default;
        friend class rocRoller::LazySingleton<DummySingletonA>;
    };

    class DummySingletonB : public rocRoller::LazySingleton<DummySingletonB>
    {
    public:
        int value{99};

        void resetState()
        {
            value = 99;
        }

    private:
        DummySingletonB() = default;
        friend class rocRoller::LazySingleton<DummySingletonB>;
    };
}

TEST_CASE("LazySingletonUnit: Same instance is returned", "[utils]")
{
    auto* a = DummySingletonA::getInstance();
    auto* b = DummySingletonA::getInstance();

    REQUIRE(a != nullptr);
    REQUIRE(b != nullptr);
    REQUIRE(a == b);
}

TEST_CASE("LazySingletonUnit: Reset does not replace singleton instance (in-place)", "[utils]")
{
    auto* instance1 = DummySingletonA::getInstance();
    REQUIRE(instance1 != nullptr);

    instance1->value = 123; // mutate

    DummySingletonA::reset();

    auto* instance2 = DummySingletonA::getInstance();
    REQUIRE(instance2 != nullptr);

    REQUIRE(instance1 == instance2); // identity unchanged
    REQUIRE(instance2->value == 0); // state reset
}

TEST_CASE("LazySingletonUnit: Different types have independent instances", "[utils]")
{
    auto* a = DummySingletonA::getInstance();
    auto* b = DummySingletonB::getInstance();

    REQUIRE(a != nullptr);
    REQUIRE(b != nullptr);

    REQUIRE(static_cast<const void*>(a) != static_cast<const void*>(b));
}

TEST_CASE("LazySingletonUnit: Singleton persists across scopes", "[utils]")
{
    auto* inst1 = DummySingletonA::getInstance();
    REQUIRE(inst1 != nullptr);

    {
        auto* inst2 = DummySingletonA::getInstance();
        REQUIRE(inst2 == inst1);
    }

    auto* inst3 = DummySingletonA::getInstance();
    REQUIRE(inst3 == inst1);
}

TEST_CASE("LazySingletonUnit: Multiple resets keep same singleton instance", "[utils]")
{
    auto* prevInstance = DummySingletonB::getInstance();
    REQUIRE(prevInstance != nullptr);

    prevInstance->value = 100; // mutate

    for(int i = 0; i < 5; ++i)
    {
        DummySingletonB::reset();

        auto* newInstance = DummySingletonB::getInstance();
        REQUIRE(newInstance != nullptr);

        REQUIRE(newInstance == prevInstance); // identity unchanged
        REQUIRE(newInstance->value == 99); // state reset

        newInstance->value = 999; // mutate again to prove reset works each iteration
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
        threads.emplace_back([&results, i] { results[i] = DummySingletonA::getInstance(); });
    }

    for(auto& t : threads)
        t.join();

    for(int i = 1; i < N; ++i)
        REQUIRE(results[i] == results[0]);
}

TEST_CASE("LazySingletonUnit: Reset under concurrent access does not crash", "[utils]")
{
    auto* baseline = DummySingletonA::getInstance();
    REQUIRE(baseline != nullptr);

    constexpr int            N = 16;
    std::vector<std::thread> threads;
    threads.reserve(N);

    for(int i = 0; i < N; ++i)
    {
        threads.emplace_back([i] {
            if(i % 5 == 0)
                DummySingletonA::reset();
            else
                (void)DummySingletonA::getInstance();
        });
    }

    for(auto& t : threads)
        t.join();

    auto* after = DummySingletonA::getInstance();
    REQUIRE(after != nullptr);
    REQUIRE(after == baseline); // identity unchanged
}

TEST_CASE("LazySingletonUnit: DummySingletonB reset keeps identity", "[utils]")
{
    auto* obj1 = DummySingletonB::getInstance();
    REQUIRE(obj1 != nullptr);

    obj1->value = 777; // mutate

    DummySingletonB::reset();

    auto* obj2 = DummySingletonB::getInstance();
    REQUIRE(obj2 != nullptr);

    REQUIRE(obj1 == obj2); // identity unchanged
    REQUIRE(obj2->value == 99); // state reset
}
