/*! \file */
/* ************************************************************************
 * Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights Reserved.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#include "rocsparse_random.hpp"
#include "rocsparse_reproducibility.hpp"
#include <iostream>

#define RANDOM_CACHE_SIZE 1024

rocsparse::rng_t::rng_t()
    : m_rng(69069)
    , m_rng_nan(69069)
    , m_rng_seed(m_rng)
{
    this->m_rand_uniform_cache.resize(RANDOM_CACHE_SIZE);
    this->m_rand_normal_cache.resize(RANDOM_CACHE_SIZE);

    for(int i = 0; i < RANDOM_CACHE_SIZE; i++)
    {
        this->m_rand_uniform_cache[i]
            = std::uniform_real_distribution<double>(0.0, 1.0)(this->get_rng());
    }

    for(int i = 0; i < RANDOM_CACHE_SIZE; i++)
    {
        this->m_rand_normal_cache[i] = std::normal_distribution<double>(0.0, 1.0)(this->get_rng());
    }

    this->reset_seed();
}

float rocsparse::rng_t::uniform_float(float a, float b)
{
    this->m_rand_uniform_idx = (this->m_rand_uniform_idx + 1) & (RANDOM_CACHE_SIZE - 1);

    return a + this->m_rand_uniform_cache[this->m_rand_uniform_idx] * (b - a);
}

double rocsparse::rng_t::uniform_double(double a, double b)
{
    this->m_rand_uniform_idx = (this->m_rand_uniform_idx + 1) & (RANDOM_CACHE_SIZE - 1);

    return a + this->m_rand_uniform_cache[this->m_rand_uniform_idx] * (b - a);
}

int rocsparse::rng_t::uniform_int(int a, int b)
{
    return this->uniform_float(static_cast<float>(a), static_cast<float>(b));
}

double rocsparse::rng_t::normal_double()
{
    this->m_rand_normal_idx = (this->m_rand_normal_idx + 1) & (RANDOM_CACHE_SIZE - 1);

    return this->m_rand_normal_cache[this->m_rand_normal_idx];
}

void rocsparse_seedrand()
{
    rocsparse::rng_t::Instance().reset_seed();
}

template <typename T>
T rocsparse::rng_t::generator_exact(int a, int b)
{
    return std::uniform_int_distribution<int>(a, b)(get_rng());
}

template <typename T, typename std::enable_if_t<std::is_integral<T>::value, bool>>
T rocsparse::rng_t::generator(T a, T b)
{
    return generator_exact<T>(a, b);
}

template <typename T, typename std::enable_if_t<!std::is_integral<T>::value, bool>>
T rocsparse::rng_t::generator(T a, T b)
{
    return std::uniform_real_distribution<T>(a, b)(get_rng());
}

template <typename T>
T rocsparse::rng_t::cached_generator_exact(int a, int b)
{
    return this->uniform_int(a, b);
}

template <typename T, typename std::enable_if_t<std::is_integral<T>::value, bool>>
T rocsparse::rng_t::cached_generator(T a, T b)
{
    return this->cached_generator_exact<T>(a, b);
}

template <typename T, typename std::enable_if_t<!std::is_integral<T>::value, bool>>
T rocsparse::rng_t::cached_generator(T a, T b)
{
    return static_cast<T>(this->uniform_float(a, b));
}

template <typename T>
T rocsparse::rng_t::cached_generator_normal()
{
    return static_cast<T>(this->normal_double());
}

template int32_t rocsparse::rng_t::generator_exact<int32_t>(int a, int b);
template int64_t rocsparse::rng_t::generator_exact<int64_t>(int a, int b);

template int8_t   rocsparse::rng_t::cached_generator<int8_t, true>(int8_t a, int8_t b);
template int32_t  rocsparse::rng_t::cached_generator<int32_t, true>(int32_t a, int32_t b);
template int64_t  rocsparse::rng_t::cached_generator<int64_t, true>(int64_t a, int64_t b);
template uint64_t rocsparse::rng_t::cached_generator<uint64_t, true>(uint64_t a, uint64_t b);
template _Float16 rocsparse::rng_t::cached_generator<_Float16, true>(_Float16 a, _Float16 b);
template rocsparse_bfloat16
               rocsparse::rng_t::cached_generator<rocsparse_bfloat16, true>(rocsparse_bfloat16 a,
                                                                 rocsparse_bfloat16 b);
template float rocsparse::rng_t::cached_generator<float, true>(float a, float b);

template int8_t             rocsparse::rng_t::cached_generator_exact<int8_t>(int a, int b);
template int32_t            rocsparse::rng_t::cached_generator_exact<int32_t>(int a, int b);
template int64_t            rocsparse::rng_t::cached_generator_exact<int64_t>(int a, int b);
template uint64_t           rocsparse::rng_t::cached_generator_exact<uint64_t>(int a, int b);
template _Float16           rocsparse::rng_t::cached_generator_exact<_Float16>(int a, int b);
template rocsparse_bfloat16 rocsparse::rng_t::cached_generator_exact<rocsparse_bfloat16>(int a,
                                                                                         int b);

template float  rocsparse::rng_t::cached_generator_normal<float>();
template double rocsparse::rng_t::cached_generator_normal<double>();

template int32_t rocsparse::rng_t::generator<int32_t, true>(int32_t a, int32_t b);

template <>
rocsparse_float_complex rocsparse::rng_t::generator_exact<rocsparse_float_complex>(int a, int b)
{
    return rocsparse_float_complex(rocsparse::rng_t::generator_exact<float>(a, b),
                                   rocsparse::rng_t::generator_exact<float>(a, b));
}

template <>
rocsparse_double_complex rocsparse::rng_t::generator_exact<rocsparse_double_complex>(int a, int b)
{
    return rocsparse_double_complex(rocsparse::rng_t::generator_exact<double>(a, b),
                                    rocsparse::rng_t::generator_exact<double>(a, b));
}

template <>
rocsparse_float_complex
    rocsparse::rng_t::generator<rocsparse_float_complex>(rocsparse_float_complex a,
                                                         rocsparse_float_complex b)
{
    float theta = rocsparse::rng_t::generator<float>(0.0f, 2.0f * acos(-1.0f));
    float r     = rocsparse::rng_t::generator<float>(std::abs(a), std::abs(b));

    return rocsparse_float_complex(r * cos(theta), r * sin(theta));
}

template <>
rocsparse_double_complex
    rocsparse::rng_t::generator<rocsparse_double_complex>(rocsparse_double_complex a,
                                                          rocsparse_double_complex b)
{
    double theta = rocsparse::rng_t::generator<double>(0.0, 2.0 * acos(-1.0));
    double r     = rocsparse::rng_t::generator<double>(std::abs(a), std::abs(b));

    return rocsparse_double_complex(r * cos(theta), r * sin(theta));
}

template <>
float rocsparse::rng_t::cached_generator_exact(int a, int b)
{
    return static_cast<float>(rocsparse::rng_t::uniform_int(a, b));
}

template <>
double rocsparse::rng_t::cached_generator_exact(int a, int b)
{
    return static_cast<double>(rocsparse::rng_t::uniform_int(a, b));
}

template <>
rocsparse_float_complex rocsparse::rng_t::cached_generator_exact<rocsparse_float_complex>(int a,
                                                                                          int b)
{
    return rocsparse_float_complex(rocsparse::rng_t::cached_generator_exact<float>(a, b),
                                   rocsparse::rng_t::cached_generator_exact<float>(a, b));
}

template <>
rocsparse_double_complex rocsparse::rng_t::cached_generator_exact<rocsparse_double_complex>(int a,
                                                                                            int b)
{
    return rocsparse_double_complex(rocsparse::rng_t::cached_generator_exact<double>(a, b),
                                    rocsparse::rng_t::cached_generator_exact<double>(a, b));
}

template <>
double rocsparse::rng_t::cached_generator(double a, double b)
{
    return rocsparse::rng_t::uniform_double(a, b);
}

template <>
rocsparse_float_complex
    rocsparse::rng_t::cached_generator<rocsparse_float_complex>(rocsparse_float_complex a,
                                                                rocsparse_float_complex b)
{
    float theta = rocsparse::rng_t::cached_generator<float>(0.0f, 2.0f * acos(-1.0f));
    float r     = rocsparse::rng_t::cached_generator<float>(std::abs(a), std::abs(b));

    return rocsparse_float_complex(r * cos(theta), r * sin(theta));
}

template <>
rocsparse_double_complex
    rocsparse::rng_t::cached_generator<rocsparse_double_complex>(rocsparse_double_complex a,
                                                                 rocsparse_double_complex b)
{
    double theta = rocsparse::rng_t::cached_generator<double>(0.0, 2.0 * acos(-1.0));
    double r     = rocsparse::rng_t::cached_generator<double>(std::abs(a), std::abs(b));

    return rocsparse_double_complex(r * cos(theta), r * sin(theta));
}
