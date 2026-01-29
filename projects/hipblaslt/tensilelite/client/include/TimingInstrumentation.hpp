/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <chrono>
#include <iostream>
#include <string>

namespace TensileLite
{
    namespace Client
    {
        // Global flag to enable/disable timing instrumentation output
        // Set via command line: --timing-instrumentation
        inline bool g_timingInstrumentationEnabled = false;

        // Simple RAII timer that prints timing on destruction
        // Output format: TIMING:<category>:<duration_ms>
        // This format is easily parseable by post-processing scripts
        class ScopedTimer
        {
        public:
            using clock = std::chrono::high_resolution_clock;

            ScopedTimer(const std::string& category)
                : m_category(category)
                , m_start(clock::now())
            {
            }

            ~ScopedTimer()
            {
                if(g_timingInstrumentationEnabled)
                {
                    auto end      = clock::now();
                    auto duration = std::chrono::duration<double, std::milli>(end - m_start);
                    std::cerr << "TIMING:" << m_category << ":" << duration.count() << std::endl;
                }
            }

            // Get elapsed time without stopping
            double elapsedMs() const
            {
                auto now      = clock::now();
                auto duration = std::chrono::duration<double, std::milli>(now - m_start);
                return duration.count();
            }

        private:
            std::string                        m_category;
            std::chrono::time_point<clock> m_start;
        };

        // Manual timer for cases where RAII doesn't fit
        class ManualTimer
        {
        public:
            using clock = std::chrono::high_resolution_clock;

            void start()
            {
                m_start = clock::now();
            }

            double stopAndReport(const std::string& category)
            {
                auto end      = clock::now();
                auto duration = std::chrono::duration<double, std::milli>(end - m_start);
                double ms     = duration.count();
                if(g_timingInstrumentationEnabled)
                {
                    std::cerr << "TIMING:" << category << ":" << ms << std::endl;
                }
                return ms;
            }

            double elapsedMs() const
            {
                auto now      = clock::now();
                auto duration = std::chrono::duration<double, std::milli>(now - m_start);
                return duration.count();
            }

        private:
            std::chrono::time_point<clock> m_start;
        };

        // Report a timing value directly (for GPU timings already measured)
        inline void reportTiming(const std::string& category, double ms)
        {
            if(g_timingInstrumentationEnabled)
            {
                std::cerr << "TIMING:" << category << ":" << ms << std::endl;
            }
        }

        // Report problem context for correlation
        inline void reportProblemContext(size_t M, size_t N, size_t K, size_t batchCount,
                                         const std::string& typeA, const std::string& typeD)
        {
            if(g_timingInstrumentationEnabled)
            {
                std::cerr << "TIMING_CONTEXT:M=" << M << ",N=" << N << ",K=" << K
                          << ",batch=" << batchCount << ",typeA=" << typeA << ",typeD=" << typeD
                          << std::endl;
            }
        }

    } // namespace Client
} // namespace TensileLite
