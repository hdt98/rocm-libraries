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

#pragma once
#include <memory>
#include <mutex>
#include <type_traits>
#include <utility>

namespace rocRoller
{
    template <typename T, typename = void>
    struct has_reset : std::false_type
    {
    };

    template <typename T>
    struct has_reset<T, std::void_t<decltype(std::declval<T&>().reset())>> : std::true_type
    {
    };

    // No-op deleter for aliasing shared_ptr that must not delete the singleton
    struct NoopDeleter
    {
        void operator()(const void*) const noexcept {}
    };

    template <typename Class>
    class LazySingleton
    {
    public:
        // Stable reference to the single, never-replaced instance.
        static Class& get() noexcept
        {
            ensure_initialized();
            return *storage();
        }

        static std::shared_ptr<Class> getInstance() noexcept
        {
            // Alias the same object with a no-op deleter.
            static std::shared_ptr<Class> sp(&get(), NoopDeleter{});
            return sp;
        }

        // Reset singleton state in place; never re-construct the object.
        static void reset()
        {
            static_assert(has_reset<Class>::value,
                          "LazySingleton<Class>::reset() requires Class to provide void reset().");
            get().reset();
        }

    private:
        // One-time initialization using call_once.
        static void ensure_initialized() noexcept
        {
            std::call_once(init_flag(), []() { storage().reset(new Class()); });
        }

        // Storage accessor (unique_ptr into which we emplace once).
        static std::unique_ptr<Class>& storage() noexcept
        {
            static std::unique_ptr<Class> ptr;
            return ptr;
        }

        // The once_flag used to guard construction.
        static std::once_flag& init_flag() noexcept
        {
            static std::once_flag flag;
            return flag;
        }
    };
}