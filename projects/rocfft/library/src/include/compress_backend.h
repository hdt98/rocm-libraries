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

#ifndef ROCFFT_COMPRESS_BACKEND_H
#define ROCFFT_COMPRESS_BACKEND_H

#include "rocfft/rocfft.h"
#include <cstddef>
#include <hip/hip_runtime_api.h>
#include <memory>

namespace rocfft
{
    namespace compress
    {

        // Abstract interface for an MPI-payload compression backend.
        class CompressBackend
        {
        public:
            virtual ~CompressBackend() = default;

            // compressed payload size in bytes (deterministic, used to size MPI counts)
            virtual size_t compressed_bytes(rocfft_precision storage_precision,
                                            size_t           count) const = 0;

            // uncompressed payload size in bytes
            static size_t storage_bytes(rocfft_precision storage_precision, size_t count);

            // ratio of uncompressed to compressed bytes
            double compression_ratio(rocfft_precision storage_precision, size_t count) const;

            // launch async compress on `stream`; src_device and dst_device must be device pointers
            virtual rocfft_status compress(const void*      src_device,
                                           rocfft_precision storage_precision,
                                           size_t           count,
                                           void*            dst_device,
                                           hipStream_t      stream)
                = 0;

            // launch async decompress on `stream`; reverses compress()
            virtual rocfft_status decompress(const void*      src_device,
                                             void*            dst_device,
                                             rocfft_precision storage_precision,
                                             size_t           count,
                                             hipStream_t      stream)
                = 0;

            // short name for logging
            virtual const char* name() const = 0;

            rocfft_comm_precision kind() const
            {
                return _kind;
            }
            unsigned int param() const
            {
                return _param;
            }

        protected:
            CompressBackend(rocfft_comm_precision k, unsigned int p)
                : _kind(k)
                , _param(p)
            {
            }

        private:
            rocfft_comm_precision _kind;
            unsigned int          _param;
        };

        // returns nullptr for the native (no-op) kind; throws on unsupported hardware
        std::unique_ptr<CompressBackend> make_compress_backend(rocfft_comm_precision kind,
                                                               unsigned int          param);

    } // namespace compress
} // namespace rocfft

#endif // ROCFFT_COMPRESS_BACKEND_H
