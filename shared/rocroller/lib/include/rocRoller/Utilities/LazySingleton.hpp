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

namespace rocRoller
{
    template <typename Class>
    class LazySingleton
    {
    public:
        static std::shared_ptr<Class> getInstance()
        {
            // function-local static ensures a single shared_ptr instance
            static std::shared_ptr<Class> instance = std::make_shared<Class>();
            return instance;
        }

        // Reset the singleton instance
        static void reset()
        {
            // Get a reference to the function-local static shared_ptr
            auto& instance = getRef();
            instance       = std::make_shared<Class>();
        }

    private:
        // Helper to access the function-local static by reference.
        static std::shared_ptr<Class>& getRef()
        {
            static std::shared_ptr<Class> instance = std::make_shared<Class>();
            return instance;
        }
    };
}