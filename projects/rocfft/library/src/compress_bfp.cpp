// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

// Inline block floating point: kBlockSize elements share an int8 exponent
// followed by signed `m`-bit mantissas; m must be in {4, 8, 16}.

#include "compress_backend.h"

#include <cstdint>
#include <cstdio>
#include <hip/hip_runtime.h>
#include <stdexcept>
#include <string>

namespace rocfft
{
    namespace compress
    {

        namespace
        {

            // one wavefront (wave32) or half-wavefront (wave64)
            constexpr int kBlockSize          = 32;
            constexpr int kThreadsPerHipBlock = kBlockSize;

            // per-block payload size in bytes (1 byte exponent + packed mantissas)
            __host__ __device__ inline size_t bfp_block_bytes(unsigned int m)
            {
                const size_t mantissa_bits  = static_cast<size_t>(kBlockSize) * m;
                const size_t mantissa_bytes = (mantissa_bits + 7) / 8;
                return 1 + mantissa_bytes;
            }

            inline size_t num_bfp_blocks(size_t count)
            {
                return (count + kBlockSize - 1) / kBlockSize;
            }

            template <typename Tin>
            __device__ inline float load_as_float(const Tin* p, size_t i, size_t n)
            {
                return (i < n) ? static_cast<float>(p[i]) : 0.0f;
            }

            // one HIP block per BFP block; staged through shared memory to avoid
            // nibble races on m=4
            template <typename Tin>
            __global__ void bfp_encode_kernel(const Tin* __restrict__ src,
                                              std::uint8_t* __restrict__ dst,
                                              size_t       n,
                                              unsigned int m,
                                              size_t       block_bytes)
            {
                const int    tid          = threadIdx.x;
                const size_t bfp_block_id = blockIdx.x;
                const size_t global_idx   = bfp_block_id * kBlockSize + tid;

                const float v     = load_as_float(src, global_idx, n);
                const float abs_v = fabsf(v);

                // tree reduction for block max-abs
                __shared__ float s_absmax[kBlockSize];
                s_absmax[tid] = abs_v;
                __syncthreads();
#pragma unroll
                for(int s = kBlockSize / 2; s > 0; s >>= 1)
                {
                    if(tid < s)
                        s_absmax[tid] = fmaxf(s_absmax[tid], s_absmax[tid + s]);
                    __syncthreads();
                }
                const float blockMaxAbs = s_absmax[0];

                // shared exponent; e = -128 sentinel for an all-zero block
                int e;
                if(blockMaxAbs <= 0.0f)
                {
                    e = -128;
                }
                else
                {
                    e = static_cast<int>(ceilf(log2f(blockMaxAbs)));
                    if(e > 127)
                        e = 127;
                    if(e < -128)
                        e = -128;
                }

                std::uint8_t* dst_block = dst + bfp_block_id * block_bytes;
                if(tid == 0)
                    dst_block[0] = static_cast<std::uint8_t>(static_cast<std::int8_t>(e));

                // quantise into the signed `m`-bit range; out-of-range threads keep mant = 0
                const int max_mant = (1 << (m - 1)) - 1;
                const int min_mant = -(1 << (m - 1));
                int       mant     = 0;
                if(global_idx < n)
                {
                    const float scale = ldexpf(1.0f, static_cast<int>(m) - 1 - e);
                    int         q     = static_cast<int>(rintf(v * scale));
                    if(q > max_mant)
                        q = max_mant;
                    if(q < min_mant)
                        q = min_mant;
                    mant = q;
                }

                // pack via shared memory (one writer thread per output byte)
                __shared__ int s_mant[kBlockSize];
                s_mant[tid] = mant;
                __syncthreads();

                std::uint8_t* mant_bytes = dst_block + 1;
                if(m == 8)
                {
                    if(tid < kBlockSize)
                        mant_bytes[tid]
                            = static_cast<std::uint8_t>(static_cast<std::int8_t>(s_mant[tid]));
                }
                else if(m == 16)
                {
                    if(tid < kBlockSize)
                    {
                        const std::int16_t s16  = static_cast<std::int16_t>(s_mant[tid]);
                        mant_bytes[2 * tid + 0] = static_cast<std::uint8_t>(s16 & 0xFF);
                        mant_bytes[2 * tid + 1] = static_cast<std::uint8_t>((s16 >> 8) & 0xFF);
                    }
                }
                else // m == 4
                {
                    if(tid < kBlockSize / 2)
                    {
                        const unsigned lo = static_cast<unsigned>(s_mant[2 * tid + 0]) & 0xFu;
                        const unsigned hi = static_cast<unsigned>(s_mant[2 * tid + 1]) & 0xFu;
                        mant_bytes[tid]   = static_cast<std::uint8_t>(lo | (hi << 4));
                    }
                }
            }

            template <typename Tout>
            __global__ void bfp_decode_kernel(const std::uint8_t* __restrict__ src,
                                              Tout* __restrict__ dst,
                                              size_t       n,
                                              unsigned int m,
                                              size_t       block_bytes)
            {
                const int    tid          = threadIdx.x;
                const size_t bfp_block_id = blockIdx.x;
                const size_t global_idx   = bfp_block_id * kBlockSize + tid;

                if(global_idx >= n)
                    return;

                const std::uint8_t* src_block  = src + bfp_block_id * block_bytes;
                const std::int8_t   e          = static_cast<std::int8_t>(src_block[0]);
                const std::uint8_t* mant_bytes = src_block + 1;

                int mant = 0;
                if(m == 8)
                {
                    mant = static_cast<int>(static_cast<std::int8_t>(mant_bytes[tid]));
                }
                else if(m == 16)
                {
                    const std::uint16_t lo = mant_bytes[2 * tid + 0];
                    const std::uint16_t hi = mant_bytes[2 * tid + 1];
                    mant = static_cast<int>(static_cast<std::int16_t>(lo | (hi << 8)));
                }
                else // m == 4
                {
                    const int          byte_idx = tid / 2;
                    const int          half     = tid & 1;
                    const std::uint8_t b        = mant_bytes[byte_idx];
                    const unsigned     nibble   = (half == 0) ? (b & 0xFu) : ((b >> 4) & 0xFu);
                    // sign-extend 4-bit value
                    mant = (nibble & 0x8u) ? static_cast<int>(nibble) - 16
                                           : static_cast<int>(nibble);
                }

                const float scale = ldexpf(1.0f, static_cast<int>(e) - (static_cast<int>(m) - 1));
                const float v     = static_cast<float>(mant) * scale;
                dst[global_idx]   = static_cast<Tout>(v);
            }

            class CompressBackendBfp : public CompressBackend
            {
            public:
                explicit CompressBackendBfp(unsigned int param)
                    : CompressBackend(rocfft_comm_precision_bfp, param == 0 ? 8u : param)
                {
                    const unsigned int m = this->param();
                    if(m != 4 && m != 8 && m != 16)
                    {
                        throw std::runtime_error("rocfft compress[bfp]: only mantissa widths in "
                                                 "{4, 8, 16} are supported "
                                                 "(received "
                                                 + std::to_string(m) + ")");
                    }
                }

                size_t compressed_bytes(rocfft_precision /*storage_precision*/,
                                        size_t count) const override
                {
                    return num_bfp_blocks(count) * bfp_block_bytes(this->param());
                }

                rocfft_status compress(const void*      src_device,
                                       rocfft_precision storage_precision,
                                       size_t           count,
                                       void*            dst_device,
                                       hipStream_t      stream) override
                {
                    if(count == 0)
                        return rocfft_status_success;

                    const unsigned int m       = this->param();
                    const size_t       blkb    = bfp_block_bytes(m);
                    const size_t       nblocks = num_bfp_blocks(count);

                    // zero output so trailing partial blocks have a defined byte pattern
                    if(auto rc = hipMemsetAsync(dst_device, 0, nblocks * blkb, stream);
                       rc != hipSuccess)
                        return rocfft_status_failure;

                    const dim3 grid(static_cast<unsigned int>(nblocks));
                    const dim3 block(static_cast<unsigned int>(kThreadsPerHipBlock));

                    switch(storage_precision)
                    {
                    case rocfft_precision_single:
                        hipLaunchKernelGGL(bfp_encode_kernel<float>,
                                           grid,
                                           block,
                                           0,
                                           stream,
                                           reinterpret_cast<const float*>(src_device),
                                           reinterpret_cast<std::uint8_t*>(dst_device),
                                           count,
                                           m,
                                           blkb);
                        break;
                    case rocfft_precision_double:
                        hipLaunchKernelGGL(bfp_encode_kernel<double>,
                                           grid,
                                           block,
                                           0,
                                           stream,
                                           reinterpret_cast<const double*>(src_device),
                                           reinterpret_cast<std::uint8_t*>(dst_device),
                                           count,
                                           m,
                                           blkb);
                        break;
                    case rocfft_precision_half:
                        hipLaunchKernelGGL(bfp_encode_kernel<_Float16>,
                                           grid,
                                           block,
                                           0,
                                           stream,
                                           reinterpret_cast<const _Float16*>(src_device),
                                           reinterpret_cast<std::uint8_t*>(dst_device),
                                           count,
                                           m,
                                           blkb);
                        break;
                    default:
                        return rocfft_status_failure;
                    }

                    return (hipPeekAtLastError() == hipSuccess) ? rocfft_status_success
                                                                : rocfft_status_failure;
                }

                rocfft_status decompress(const void*      src_device,
                                         void*            dst_device,
                                         rocfft_precision storage_precision,
                                         size_t           count,
                                         hipStream_t      stream) override
                {
                    if(count == 0)
                        return rocfft_status_success;

                    const unsigned int m       = this->param();
                    const size_t       blkb    = bfp_block_bytes(m);
                    const size_t       nblocks = num_bfp_blocks(count);

                    const dim3 grid(static_cast<unsigned int>(nblocks));
                    const dim3 block(static_cast<unsigned int>(kThreadsPerHipBlock));

                    switch(storage_precision)
                    {
                    case rocfft_precision_single:
                        hipLaunchKernelGGL(bfp_decode_kernel<float>,
                                           grid,
                                           block,
                                           0,
                                           stream,
                                           reinterpret_cast<const std::uint8_t*>(src_device),
                                           reinterpret_cast<float*>(dst_device),
                                           count,
                                           m,
                                           blkb);
                        break;
                    case rocfft_precision_double:
                        hipLaunchKernelGGL(bfp_decode_kernel<double>,
                                           grid,
                                           block,
                                           0,
                                           stream,
                                           reinterpret_cast<const std::uint8_t*>(src_device),
                                           reinterpret_cast<double*>(dst_device),
                                           count,
                                           m,
                                           blkb);
                        break;
                    case rocfft_precision_half:
                        hipLaunchKernelGGL(bfp_decode_kernel<_Float16>,
                                           grid,
                                           block,
                                           0,
                                           stream,
                                           reinterpret_cast<const std::uint8_t*>(src_device),
                                           reinterpret_cast<_Float16*>(dst_device),
                                           count,
                                           m,
                                           blkb);
                        break;
                    default:
                        return rocfft_status_failure;
                    }

                    return (hipPeekAtLastError() == hipSuccess) ? rocfft_status_success
                                                                : rocfft_status_failure;
                }

                const char* name() const override
                {
                    return "bfp";
                }
            };

        } // namespace

        std::unique_ptr<CompressBackend> make_compress_backend_bfp(unsigned int param)
        {
            return std::unique_ptr<CompressBackend>(new CompressBackendBfp(param));
        }

    } // namespace compress
} // namespace rocfft
