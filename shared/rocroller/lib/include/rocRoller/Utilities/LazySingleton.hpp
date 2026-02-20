// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <concepts>
#include <type_traits>
#include <utility>

namespace rocRoller
{
    template <typename T>
    concept ResettableState = requires(T& t)
    {
        {
            t.resetState()
            } -> std::same_as<void>;
    };

    template <typename Class>
    class LazySingleton
    {
    public:
        static Class& get() noexcept
        {
            static Class instance{};
            return instance;
        }

        static Class* getInstance() noexcept
        {
            return &get();
        }

        static void reset() requires ResettableState<Class>
        {
            get().resetState();
        }
    };
}
