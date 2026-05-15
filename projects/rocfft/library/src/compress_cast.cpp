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

// Per-element narrow-precision cast backend (FP16/BF16/FP8).  FP8 paths
// require <hip/hip_fp8.h> (gfx942 and later); otherwise the FP8 kinds
// return rocfft_status_failure.

#include "compress_backend.h"

#include <cstdint>
#include <cstdio>
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <stdexcept>

#if __has_include(<hip/hip_fp8.h>)
#include <hip/hip_fp8.h>
#define ROCFFT_HAS_HIP_FP8 1
#endif

namespace rocfft
{
    namespace compress
    {

        namespace
        {

            constexpr int kThreadsPerBlock = 256;

            __global__ void
                cast_f32_to_f16(const float* __restrict__ src, _Float16* __restrict__ dst, size_t n)
            {
                const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
                if(i < n)
                    dst[i] = static_cast<_Float16>(src[i]);
            }

            __global__ void
                cast_f16_to_f32(const _Float16* __restrict__ src, float* __restrict__ dst, size_t n)
            {
                const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
                if(i < n)
                    dst[i] = static_cast<float>(src[i]);
            }

            __global__ void cast_f32_to_bf16(const float* __restrict__ src,
                                             hip_bfloat16* __restrict__ dst,
                                             size_t n)
            {
                const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
                if(i < n)
                    dst[i] = hip_bfloat16(src[i]);
            }

            __global__ void cast_bf16_to_f32(const hip_bfloat16* __restrict__ src,
                                             float* __restrict__ dst,
                                             size_t n)
            {
                const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
                if(i < n)
                    dst[i] = static_cast<float>(src[i]);
            }

            __global__ void cast_f64_to_f16(const double* __restrict__ src,
                                            _Float16* __restrict__ dst,
                                            size_t n)
            {
                const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
                if(i < n)
                    dst[i] = static_cast<_Float16>(static_cast<float>(src[i]));
            }

            __global__ void cast_f16_to_f64(const _Float16* __restrict__ src,
                                            double* __restrict__ dst,
                                            size_t n)
            {
                const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
                if(i < n)
                    dst[i] = static_cast<double>(src[i]);
            }

            __global__ void cast_f64_to_bf16(const double* __restrict__ src,
                                             hip_bfloat16* __restrict__ dst,
                                             size_t n)
            {
                const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
                if(i < n)
                    dst[i] = hip_bfloat16(static_cast<float>(src[i]));
            }

            __global__ void cast_bf16_to_f64(const hip_bfloat16* __restrict__ src,
                                             double* __restrict__ dst,
                                             size_t n)
            {
                const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
                if(i < n)
                    dst[i] = static_cast<double>(static_cast<float>(src[i]));
            }

            // FP8 paths (FNUZ variants on gfx942/gfx950).

#ifdef ROCFFT_HAS_HIP_FP8

            __global__ void cast_f32_to_fp8_e4m3(const float* __restrict__ src,
                                                 __hip_fp8_e4m3_fnuz* __restrict__ dst,
                                                 size_t n)
            {
                const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
                if(i < n)
                    dst[i] = __hip_fp8_e4m3_fnuz(src[i]);
            }

            __global__ void cast_fp8_e4m3_to_f32(const __hip_fp8_e4m3_fnuz* __restrict__ src,
                                                 float* __restrict__ dst,
                                                 size_t n)
            {
                const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
                if(i < n)
                    dst[i] = static_cast<float>(src[i]);
            }

            __global__ void cast_f32_to_fp8_e5m2(const float* __restrict__ src,
                                                 __hip_fp8_e5m2_fnuz* __restrict__ dst,
                                                 size_t n)
            {
                const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
                if(i < n)
                    dst[i] = __hip_fp8_e5m2_fnuz(src[i]);
            }

            __global__ void cast_fp8_e5m2_to_f32(const __hip_fp8_e5m2_fnuz* __restrict__ src,
                                                 float* __restrict__ dst,
                                                 size_t n)
            {
                const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
                if(i < n)
                    dst[i] = static_cast<float>(src[i]);
            }

            __global__ void cast_f64_to_fp8_e4m3(const double* __restrict__ src,
                                                 __hip_fp8_e4m3_fnuz* __restrict__ dst,
                                                 size_t n)
            {
                const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
                if(i < n)
                    dst[i] = __hip_fp8_e4m3_fnuz(static_cast<float>(src[i]));
            }

            __global__ void cast_fp8_e4m3_to_f64(const __hip_fp8_e4m3_fnuz* __restrict__ src,
                                                 double* __restrict__ dst,
                                                 size_t n)
            {
                const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
                if(i < n)
                    dst[i] = static_cast<double>(static_cast<float>(src[i]));
            }

            __global__ void cast_f64_to_fp8_e5m2(const double* __restrict__ src,
                                                 __hip_fp8_e5m2_fnuz* __restrict__ dst,
                                                 size_t n)
            {
                const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
                if(i < n)
                    dst[i] = __hip_fp8_e5m2_fnuz(static_cast<float>(src[i]));
            }

            __global__ void cast_fp8_e5m2_to_f64(const __hip_fp8_e5m2_fnuz* __restrict__ src,
                                                 double* __restrict__ dst,
                                                 size_t n)
            {
                const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
                if(i < n)
                    dst[i] = static_cast<double>(static_cast<float>(src[i]));
            }

#endif // ROCFFT_HAS_HIP_FP8

            inline dim3 grid_for(size_t n)
            {
                const size_t blocks = (n + kThreadsPerBlock - 1) / kThreadsPerBlock;
                return dim3(static_cast<unsigned int>(blocks));
            }

            inline size_t bytes_per_storage_element(rocfft_precision storage_precision)
            {
                switch(storage_precision)
                {
                case rocfft_precision_half:
                    return sizeof(uint16_t);
                case rocfft_precision_single:
                    return sizeof(float);
                case rocfft_precision_double:
                    return sizeof(double);
                }
                return 0;
            }

            inline size_t bytes_per_communication_element(rocfft_comm_precision kind)
            {
                switch(kind)
                {
                case rocfft_comm_precision_cast_fp16:
                case rocfft_comm_precision_cast_bf16:
                    return 2;
                case rocfft_comm_precision_cast_fp8_e4m3:
                case rocfft_comm_precision_cast_fp8_e5m2:
                    return 1;
                default:
                    return 0;
                }
            }

            class CompressBackendCast : public CompressBackend
            {
            public:
                CompressBackendCast(rocfft_comm_precision kind, unsigned int param)
                    : CompressBackend(kind, param)
                {
#ifndef ROCFFT_HAS_HIP_FP8
                    // FP8 requested without HIP FP8 support; first compress/decompress will fail.
                    if(kind == rocfft_comm_precision_cast_fp8_e4m3
                       || kind == rocfft_comm_precision_cast_fp8_e5m2)
                    {
                        std::fprintf(stderr,
                                     "rocfft compress[cast_fp8]: built without <hip/hip_fp8.h>\n");
                    }
#endif
                }

                size_t compressed_bytes(rocfft_precision /*storage_precision*/,
                                        size_t count) const override
                {
                    return count * bytes_per_communication_element(kind());
                }

                rocfft_status compress(const void*      src_device,
                                       rocfft_precision storage_precision,
                                       size_t           count,
                                       void*            dst_device,
                                       hipStream_t      stream) override
                {
                    if(count == 0)
                        return rocfft_status_success;

                    const dim3 g  = grid_for(count);
                    const dim3 bk = dim3(kThreadsPerBlock);

                    switch(kind())
                    {
                    case rocfft_comm_precision_cast_fp16:
                        if(storage_precision == rocfft_precision_single)
                        {
                            hipLaunchKernelGGL(cast_f32_to_f16,
                                               g,
                                               bk,
                                               0,
                                               stream,
                                               reinterpret_cast<const float*>(src_device),
                                               reinterpret_cast<_Float16*>(dst_device),
                                               count);
                        }
                        else if(storage_precision == rocfft_precision_double)
                        {
                            hipLaunchKernelGGL(cast_f64_to_f16,
                                               g,
                                               bk,
                                               0,
                                               stream,
                                               reinterpret_cast<const double*>(src_device),
                                               reinterpret_cast<_Float16*>(dst_device),
                                               count);
                        }
                        else
                        {
                            return rocfft_status_failure;
                        }
                        break;

                    case rocfft_comm_precision_cast_bf16:
                        if(storage_precision == rocfft_precision_single)
                        {
                            hipLaunchKernelGGL(cast_f32_to_bf16,
                                               g,
                                               bk,
                                               0,
                                               stream,
                                               reinterpret_cast<const float*>(src_device),
                                               reinterpret_cast<hip_bfloat16*>(dst_device),
                                               count);
                        }
                        else if(storage_precision == rocfft_precision_double)
                        {
                            hipLaunchKernelGGL(cast_f64_to_bf16,
                                               g,
                                               bk,
                                               0,
                                               stream,
                                               reinterpret_cast<const double*>(src_device),
                                               reinterpret_cast<hip_bfloat16*>(dst_device),
                                               count);
                        }
                        else
                        {
                            return rocfft_status_failure;
                        }
                        break;

                    case rocfft_comm_precision_cast_fp8_e4m3:
#ifdef ROCFFT_HAS_HIP_FP8
                        if(storage_precision == rocfft_precision_single)
                        {
                            hipLaunchKernelGGL(cast_f32_to_fp8_e4m3,
                                               g,
                                               bk,
                                               0,
                                               stream,
                                               reinterpret_cast<const float*>(src_device),
                                               reinterpret_cast<__hip_fp8_e4m3_fnuz*>(dst_device),
                                               count);
                        }
                        else if(storage_precision == rocfft_precision_double)
                        {
                            hipLaunchKernelGGL(cast_f64_to_fp8_e4m3,
                                               g,
                                               bk,
                                               0,
                                               stream,
                                               reinterpret_cast<const double*>(src_device),
                                               reinterpret_cast<__hip_fp8_e4m3_fnuz*>(dst_device),
                                               count);
                        }
                        else
                        {
                            return rocfft_status_failure;
                        }
                        break;
#else
                        return rocfft_status_failure;
#endif

                    case rocfft_comm_precision_cast_fp8_e5m2:
#ifdef ROCFFT_HAS_HIP_FP8
                        if(storage_precision == rocfft_precision_single)
                        {
                            hipLaunchKernelGGL(cast_f32_to_fp8_e5m2,
                                               g,
                                               bk,
                                               0,
                                               stream,
                                               reinterpret_cast<const float*>(src_device),
                                               reinterpret_cast<__hip_fp8_e5m2_fnuz*>(dst_device),
                                               count);
                        }
                        else if(storage_precision == rocfft_precision_double)
                        {
                            hipLaunchKernelGGL(cast_f64_to_fp8_e5m2,
                                               g,
                                               bk,
                                               0,
                                               stream,
                                               reinterpret_cast<const double*>(src_device),
                                               reinterpret_cast<__hip_fp8_e5m2_fnuz*>(dst_device),
                                               count);
                        }
                        else
                        {
                            return rocfft_status_failure;
                        }
                        break;
#else
                        return rocfft_status_failure;
#endif

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

                    const dim3 g  = grid_for(count);
                    const dim3 bk = dim3(kThreadsPerBlock);

                    switch(kind())
                    {
                    case rocfft_comm_precision_cast_fp16:
                        if(storage_precision == rocfft_precision_single)
                        {
                            hipLaunchKernelGGL(cast_f16_to_f32,
                                               g,
                                               bk,
                                               0,
                                               stream,
                                               reinterpret_cast<const _Float16*>(src_device),
                                               reinterpret_cast<float*>(dst_device),
                                               count);
                        }
                        else if(storage_precision == rocfft_precision_double)
                        {
                            hipLaunchKernelGGL(cast_f16_to_f64,
                                               g,
                                               bk,
                                               0,
                                               stream,
                                               reinterpret_cast<const _Float16*>(src_device),
                                               reinterpret_cast<double*>(dst_device),
                                               count);
                        }
                        else
                        {
                            return rocfft_status_failure;
                        }
                        break;

                    case rocfft_comm_precision_cast_bf16:
                        if(storage_precision == rocfft_precision_single)
                        {
                            hipLaunchKernelGGL(cast_bf16_to_f32,
                                               g,
                                               bk,
                                               0,
                                               stream,
                                               reinterpret_cast<const hip_bfloat16*>(src_device),
                                               reinterpret_cast<float*>(dst_device),
                                               count);
                        }
                        else if(storage_precision == rocfft_precision_double)
                        {
                            hipLaunchKernelGGL(cast_bf16_to_f64,
                                               g,
                                               bk,
                                               0,
                                               stream,
                                               reinterpret_cast<const hip_bfloat16*>(src_device),
                                               reinterpret_cast<double*>(dst_device),
                                               count);
                        }
                        else
                        {
                            return rocfft_status_failure;
                        }
                        break;

                    case rocfft_comm_precision_cast_fp8_e4m3:
#ifdef ROCFFT_HAS_HIP_FP8
                        if(storage_precision == rocfft_precision_single)
                        {
                            hipLaunchKernelGGL(
                                cast_fp8_e4m3_to_f32,
                                g,
                                bk,
                                0,
                                stream,
                                reinterpret_cast<const __hip_fp8_e4m3_fnuz*>(src_device),
                                reinterpret_cast<float*>(dst_device),
                                count);
                        }
                        else if(storage_precision == rocfft_precision_double)
                        {
                            hipLaunchKernelGGL(
                                cast_fp8_e4m3_to_f64,
                                g,
                                bk,
                                0,
                                stream,
                                reinterpret_cast<const __hip_fp8_e4m3_fnuz*>(src_device),
                                reinterpret_cast<double*>(dst_device),
                                count);
                        }
                        else
                        {
                            return rocfft_status_failure;
                        }
                        break;
#else
                        return rocfft_status_failure;
#endif

                    case rocfft_comm_precision_cast_fp8_e5m2:
#ifdef ROCFFT_HAS_HIP_FP8
                        if(storage_precision == rocfft_precision_single)
                        {
                            hipLaunchKernelGGL(
                                cast_fp8_e5m2_to_f32,
                                g,
                                bk,
                                0,
                                stream,
                                reinterpret_cast<const __hip_fp8_e5m2_fnuz*>(src_device),
                                reinterpret_cast<float*>(dst_device),
                                count);
                        }
                        else if(storage_precision == rocfft_precision_double)
                        {
                            hipLaunchKernelGGL(
                                cast_fp8_e5m2_to_f64,
                                g,
                                bk,
                                0,
                                stream,
                                reinterpret_cast<const __hip_fp8_e5m2_fnuz*>(src_device),
                                reinterpret_cast<double*>(dst_device),
                                count);
                        }
                        else
                        {
                            return rocfft_status_failure;
                        }
                        break;
#else
                        return rocfft_status_failure;
#endif

                    default:
                        return rocfft_status_failure;
                    }

                    return (hipPeekAtLastError() == hipSuccess) ? rocfft_status_success
                                                                : rocfft_status_failure;
                }

                const char* name() const override
                {
                    switch(kind())
                    {
                    case rocfft_comm_precision_cast_fp16:
                        return "cast_fp16";
                    case rocfft_comm_precision_cast_bf16:
                        return "cast_bf16";
                    case rocfft_comm_precision_cast_fp8_e4m3:
                        return "cast_fp8_e4m3";
                    case rocfft_comm_precision_cast_fp8_e5m2:
                        return "cast_fp8_e5m2";
                    default:
                        return "cast_unknown";
                    }
                }
            };

        } // namespace

        size_t CompressBackend::storage_bytes(rocfft_precision storage_precision, size_t count)
        {
            return count * bytes_per_storage_element(storage_precision);
        }

        double CompressBackend::compression_ratio(rocfft_precision storage_precision,
                                                  size_t           count) const
        {
            const size_t denom = compressed_bytes(storage_precision, count);
            if(denom == 0)
                return 1.0;
            return static_cast<double>(storage_bytes(storage_precision, count))
                   / static_cast<double>(denom);
        }

        // defined in compress_bfp.cpp / compress_zfp.cpp
        std::unique_ptr<CompressBackend> make_compress_backend_bfp(unsigned int param);
        std::unique_ptr<CompressBackend> make_compress_backend_zfp(unsigned int param);

        std::unique_ptr<CompressBackend> make_compress_backend(rocfft_comm_precision kind,
                                                               unsigned int          param)
        {
            switch(kind)
            {
            case rocfft_comm_precision_native:
                return nullptr;
            case rocfft_comm_precision_cast_fp16:
            case rocfft_comm_precision_cast_bf16:
            case rocfft_comm_precision_cast_fp8_e4m3:
            case rocfft_comm_precision_cast_fp8_e5m2:
                return std::unique_ptr<CompressBackend>(new CompressBackendCast(kind, param));
            case rocfft_comm_precision_bfp:
                return make_compress_backend_bfp(param);
            case rocfft_comm_precision_zfp_fixed_rate:
                return make_compress_backend_zfp(param);
            }
            throw std::runtime_error(
                "rocfft: unknown rocfft_comm_precision in make_compress_backend");
        }

    } // namespace compress
} // namespace rocfft
