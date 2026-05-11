/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
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

#include "rocrand_wrapper.hpp"

#include <algorithm>
#include <cstdint>
#include <functional>

#include <miopen/bfloat16.hpp>

#include <hip/hip_runtime.h>
#include <rocrand/rocrand.hpp>

namespace gpumemrand {

namespace {
template <typename T>
int gen_0_1_impl(T* buf, size_t sz)
{
    rocrand_cpp::xorwow_engine<> g;
    rocrand_cpp::uniform_real_distribution<T> d;
    d(g, buf, sz);
    return 0;
}

__global__ void float_to_bf16_rne(const float* __restrict__ src,
                                  std::uint16_t* __restrict__ dst,
                                  size_t n)
{
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(i >= n)
        return;
    // Round to nearest, ties to even — same algorithm as the host bfloat16
    // ctor in src/include/miopen/bfloat16.hpp. rocRAND uniform values lie in
    // (0, 1] so the exponent is at most 0x7F; the rounding add cannot push
    // bits into the 0xFF NaN/Inf range.
    std::uint32_t bits;
    __builtin_memcpy(&bits, src + i, sizeof(std::uint32_t));
    bits += 0x7FFFu + ((bits >> 16) & 1u);
    dst[i] = static_cast<std::uint16_t>(bits >> 16);
}
} // namespace

int gen_0_1(double* buf, size_t sz) //
{
    return gen_0_1_impl(buf, sz);
}

int gen_0_1(float* buf, size_t sz) //
{
    return gen_0_1_impl(buf, sz);
}

int gen_0_1(half_float::half* buf, size_t sz) //
{
    return gen_0_1_impl(reinterpret_cast<half*>(buf), sz);
}

int gen_0_1(bfloat16* buf, size_t sz)
{
    if(sz == 0)
        return 0;

    // rocRAND has no native bf16 distribution. Generate uniform floats into a
    // device-side scratch buffer, then convert to bf16 via a small HIP kernel.
    // Process in chunks so peak temp memory is bounded regardless of sz.
    constexpr size_t kChunkBytes = static_cast<size_t>(256) << 20; // 256 MB
    constexpr size_t kChunkElems = kChunkBytes / sizeof(float);
    const size_t first_chunk     = std::min(sz, kChunkElems);

    float* tmp = nullptr;
    if(hipMalloc(&tmp, first_chunk * sizeof(float)) != hipSuccess)
    {
        std::cout << "Warning: gpumemrand bfloat16 hipMalloc failed for " << first_chunk
                  << " floats; buffer remains uninitialized." << std::endl;
        return -1;
    }

    rocrand_cpp::xorwow_engine<> g;
    rocrand_cpp::uniform_real_distribution<float> d;

    auto* dst              = reinterpret_cast<std::uint16_t*>(buf);
    size_t remaining       = sz;
    constexpr int kBlock   = 256;
    while(remaining > 0)
    {
        const size_t n      = std::min(remaining, kChunkElems);
        const size_t blocks = (n + kBlock - 1) / kBlock;
        d(g, tmp, n);
        float_to_bf16_rne<<<dim3(blocks), dim3(kBlock), 0, nullptr>>>(tmp, dst, n);
        dst += n;
        remaining -= n;
    }

    static_cast<void>(hipFree(tmp));
    return 0;
}

} // namespace gpumemrand
