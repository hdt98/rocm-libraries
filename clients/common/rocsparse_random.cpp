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

double rocsparse::rng_t::uniform_double(double a, double b)
{
    this->m_rand_uniform_idx = (this->m_rand_uniform_idx + 1) & (RANDOM_CACHE_SIZE - 1);

    return a + this->m_rand_uniform_cache[this->m_rand_uniform_idx] * (b - a);
}

double rocsparse::rng_t::normal_double()
{
    this->m_rand_normal_idx = (this->m_rand_normal_idx + 1) & (RANDOM_CACHE_SIZE - 1);

    return this->m_rand_normal_cache[this->m_rand_normal_idx];
}

void rocsparse::rng_t::reset_seed()
{
    m_rand_uniform_idx = 0;
    m_rand_normal_idx  = 0;

    set_rng(m_rng_seed);
    set_rng_nan(m_rng_seed);
}

void rocsparse::rng_t::set_rng(rocsparse_rng_t a)
{
    m_rng = a;
}
void rocsparse::rng_t::set_rng_nan(rocsparse_rng_t a)
{
    m_rng_nan = a;
}
void rocsparse::rng_t::rng_seed_set(rocsparse_rng_t a)
{
    m_rng_seed = a;
}

rocsparse_rng_t& rocsparse::rng_t::get_rng()
{
    return m_rng;
}
rocsparse_rng_t& rocsparse::rng_t::get_rng_nan()
{
    return m_rng_nan;
}
rocsparse_rng_t& rocsparse::rng_t::get_rng_seed()
{
    return m_rng_seed;
}

template <typename T, typename std::enable_if_t<std::is_integral<T>::value, bool> = true>
static T std_generator(T a, T b, rocsparse_rng_t& rng)
{
    return std::uniform_int_distribution<T>(a, b)(rng);
}

template <typename T, typename std::enable_if_t<!std::is_integral<T>::value, bool> = true>
static T std_generator(T a, T b, rocsparse_rng_t& rng)
{
    return std::uniform_real_distribution<T>(a, b)(rng);
}

template <typename T>
T rocsparse::rng_t::generator(T a, T b)
{
    return std_generator<T>(a, b, this->get_rng());
}

template <typename T>
T rocsparse::rng_t::cached_generator(T a, T b)
{
    return static_cast<T>(this->uniform_double(a, b));
}

template <typename T>
T rocsparse::rng_t::cached_generator_normal()
{
    return static_cast<T>(this->normal_double());
}

template int32_t rocsparse::rng_t::generator<int32_t>(int32_t a, int32_t b);
template int64_t rocsparse::rng_t::generator<int64_t>(int64_t a, int64_t b);
template float   rocsparse::rng_t::generator<float>(float a, float b);
template double  rocsparse::rng_t::generator<double>(double a, double b);

template int8_t   rocsparse::rng_t::cached_generator<int8_t>(int8_t a, int8_t b);
template int32_t  rocsparse::rng_t::cached_generator<int32_t>(int32_t a, int32_t b);
template int64_t  rocsparse::rng_t::cached_generator<int64_t>(int64_t a, int64_t b);
template uint64_t rocsparse::rng_t::cached_generator<uint64_t>(uint64_t a, uint64_t b);
template _Float16 rocsparse::rng_t::cached_generator<_Float16>(_Float16 a, _Float16 b);
template rocsparse_bfloat16
                rocsparse::rng_t::cached_generator<rocsparse_bfloat16>(rocsparse_bfloat16 a,
                                                           rocsparse_bfloat16 b);
template float  rocsparse::rng_t::cached_generator<float>(float a, float b);
template double rocsparse::rng_t::cached_generator<double>(double a, double b);

template float  rocsparse::rng_t::cached_generator_normal<float>();
template double rocsparse::rng_t::cached_generator_normal<double>();

// template <>
// rocsparse_float_complex rocsparse::rng_t::generator<rocsparse_float_complex>(rocsparse_float_complex a, rocsparse_float_complex b)
// {
//     return rocsparse_float_complex(rocsparse::rng_t::generator<float>(std::real(a), std::real(b)),
//                                    rocsparse::rng_t::generator<float>(std::real(a), std::real(b)));
// }

// template <>
// rocsparse_double_complex rocsparse::rng_t::generator<rocsparse_double_complex>(rocsparse_double_complex a, rocsparse_double_complex b)
// {
//     return rocsparse_double_complex(rocsparse::rng_t::generator<double>(std::real(a), std::real(b)),
//                                     rocsparse::rng_t::generator<double>(std::real(a), std::real(b)));
// }

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

// template <>
// rocsparse_float_complex rocsparse::rng_t::cached_generator<rocsparse_float_complex>(rocsparse_float_complex a, rocsparse_float_complex b)
// {
//     return rocsparse_float_complex(rocsparse::rng_t::cached_generator<float>(a, b),
//                                    rocsparse::rng_t::cached_generator<float>(a, b));
// }

// template <>
// rocsparse_double_complex rocsparse::rng_t::cached_generator<rocsparse_double_complex>(rocsparse_double_complex a, rocsparse_double_complex b)
// {
//     return rocsparse_double_complex(rocsparse::rng_t::cached_generator<double>(a, b),
//                                     rocsparse::rng_t::cached_generator<double>(a, b));
// }

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

/*
* ===========================================================================
*    Wrappers
* ===========================================================================
*/

template <typename T>
T random_generator_exact(int a, int b)
{
    // return rocsparse::rng_t::Instance().generator_exact<T>(a, b);
    return rocsparse::rng_t::Instance().generator<T>(a, b);
}

template <typename T, typename std::enable_if_t<std::is_integral<T>::value, bool>>
T random_generator(T a, T b)
{
    return rocsparse::rng_t::Instance().generator<T>(a, b);
}

template <typename T, typename std::enable_if_t<!std::is_integral<T>::value, bool>>
T random_generator(T a, T b)
{
    return rocsparse::rng_t::Instance().generator<T>(a, b);
}

template <typename T>
T random_cached_generator_exact(int a, int b)
{
    // return rocsparse::rng_t::Instance().cached_generator_exact<T>(a, b);
    return rocsparse::rng_t::Instance().cached_generator<T>(a, b);
}

template <typename T, typename std::enable_if_t<std::is_integral<T>::value, bool>>
T random_cached_generator(T a, T b)
{
    return rocsparse::rng_t::Instance().cached_generator<T>(a, b);
}

template <typename T, typename std::enable_if_t<!std::is_integral<T>::value, bool>>
T random_cached_generator(T a, T b)
{
    return rocsparse::rng_t::Instance().cached_generator<T>(a, b);
}

template <typename T>
T random_cached_generator_normal()
{
    return rocsparse::rng_t::Instance().cached_generator_normal<T>();
}

void rocsparse_seedrand()
{
    rocsparse::rng_t::Instance().reset_seed();
}

template int8_t             random_cached_generator<int8_t, true>(int8_t, int8_t);
template int32_t            random_cached_generator<int32_t, true>(int32_t, int32_t);
template int64_t            random_cached_generator<int64_t, true>(int64_t, int64_t);
template uint64_t           random_cached_generator<uint64_t, true>(uint64_t, uint64_t);
template _Float16           random_cached_generator<_Float16, true>(_Float16, _Float16);
template rocsparse_bfloat16 random_cached_generator<rocsparse_bfloat16, true>(rocsparse_bfloat16,
                                                                              rocsparse_bfloat16);
template float              random_cached_generator<float, true>(float, float);
template double             random_cached_generator<double, true>(double, double);
template rocsparse_float_complex
    random_cached_generator<rocsparse_float_complex, true>(rocsparse_float_complex,
                                                           rocsparse_float_complex);
template rocsparse_double_complex
    random_cached_generator<rocsparse_double_complex, true>(rocsparse_double_complex,
                                                            rocsparse_double_complex);

template float  random_cached_generator_normal<float>();
template double random_cached_generator_normal<double>();

template int64_t                  random_generator_exact<int64_t>(int, int);
template int32_t                  random_generator_exact<int32_t>(int a, int b);
template float                    random_generator_exact<float>(int, int);
template double                   random_generator_exact<double>(int, int);
template rocsparse_float_complex  random_generator_exact<rocsparse_float_complex>(int, int);
template rocsparse_double_complex random_generator_exact<rocsparse_double_complex>(int, int);

template int    random_generator<int, true>(int, int);
template float  random_generator<float, true>(float, float);
template double random_generator<double, true>(double, double);
template rocsparse_float_complex
    random_generator<rocsparse_float_complex, true>(rocsparse_float_complex,
                                                    rocsparse_float_complex);
template rocsparse_double_complex
    random_generator<rocsparse_double_complex, true>(rocsparse_double_complex,
                                                     rocsparse_double_complex);

template int8_t                   random_cached_generator_exact<int8_t>(int, int);
template int32_t                  random_cached_generator_exact<int32_t>(int, int);
template int64_t                  random_cached_generator_exact<int64_t>(int, int);
template uint64_t                 random_cached_generator_exact<uint64_t>(int, int);
template _Float16                 random_cached_generator_exact<_Float16>(int, int);
template rocsparse_bfloat16       random_cached_generator_exact<rocsparse_bfloat16>(int, int);
template float                    random_cached_generator_exact<float>(int, int);
template double                   random_cached_generator_exact<double>(int, int);
template rocsparse_float_complex  random_cached_generator_exact<rocsparse_float_complex>(int, int);
template rocsparse_double_complex random_cached_generator_exact<rocsparse_double_complex>(int, int);
