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

// Stub for the fixed-rate ZFP-style backend.  Reports the deterministic
// wire-rate but zero-fills the payload; the HIP encoder/decoder is a TODO.

#include "compress_backend.h"
#include <cstdio>
#include <hip/hip_runtime.h>

namespace rocfft
{
    namespace compress
    {

        namespace
        {

            class CompressBackendZfpStub : public CompressBackend
            {
            public:
                explicit CompressBackendZfpStub(unsigned int param)
                    : CompressBackend(rocfft_comm_precision_zfp_fixed_rate, param == 0 ? 8u : param)
                {
                }

                // wire-rate matches the planned ZFP encoder: param bits per element rounded up
                size_t compressed_bytes(rocfft_precision /*storage_precision*/,
                                        size_t count) const override
                {
                    const unsigned bits_per_elem = param();
                    const size_t   bits_total    = count * bits_per_elem;
                    return (bits_total + 7) / 8;
                }

                rocfft_status compress(const void* /*src_device*/,
                                       rocfft_precision storage_precision,
                                       size_t           count,
                                       void*            dst_device,
                                       hipStream_t      stream) override
                {
                    std::fprintf(stderr, "rocfft compress[zfp]: stub backend\n");
                    size_t out_bytes = compressed_bytes(storage_precision, count);
                    return (hipMemsetAsync(dst_device, 0, out_bytes, stream) == hipSuccess)
                               ? rocfft_status_success
                               : rocfft_status_failure;
                }

                rocfft_status decompress(const void* /*src_device*/,
                                         void*            dst_device,
                                         rocfft_precision storage_precision,
                                         size_t           count,
                                         hipStream_t      stream) override
                {
                    size_t out_bytes = CompressBackend::storage_bytes(storage_precision, count);
                    return (hipMemsetAsync(dst_device, 0, out_bytes, stream) == hipSuccess)
                               ? rocfft_status_success
                               : rocfft_status_failure;
                }

                const char* name() const override
                {
                    return "zfp_stub";
                }
            };

        } // namespace

        std::unique_ptr<CompressBackend> make_compress_backend_zfp(unsigned int param)
        {
            return std::unique_ptr<CompressBackend>(new CompressBackendZfpStub(param));
        }

    } // namespace compress
} // namespace rocfft
