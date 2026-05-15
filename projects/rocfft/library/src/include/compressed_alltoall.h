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

#ifndef ROCFFT_COMPRESSED_ALLTOALL_H
#define ROCFFT_COMPRESSED_ALLTOALL_H

#include "compress_backend.h"
#include "rocfft/rocfft.h"
#include "rocfft_mpi.h"
#include <hip/hip_runtime_api.h>
#include <memory>
#include <string>
#include <vector>

namespace rocfft
{
    namespace compress
    {

#ifdef ROCFFT_MPI_ENABLE

        // MPI_Ialltoall(v) wrapper that compresses the payload via a CompressBackend
        // before the exchange and decompresses it after.  Counts and offsets are
        // given in elements of the storage precision; this class rescales them to
        // bytes of the compressed payload internally.
        class CompressedAlltoallv
        {
        public:
            CompressedAlltoallv(rocfft_comm_precision comm_precision,
                                unsigned int          comm_param,
                                rocfft_precision      storage_precision,
                                rocfft_array_type     array_type);

            ~CompressedAlltoallv();

            CompressedAlltoallv(const CompressedAlltoallv&) = delete;
            CompressedAlltoallv& operator=(const CompressedAlltoallv&) = delete;

            // launch compress, post MPI_Ialltoallv on compressed bytes; caller must
            // invoke wait() before reading `recvbuf_device`
            rocfft_status execute_async(const void*                sendbuf_device,
                                        const std::vector<size_t>& sendOffsets,
                                        const std::vector<size_t>& sendCounts,
                                        void*                      recvbuf_device,
                                        const std::vector<size_t>& recvOffsets,
                                        const std::vector<size_t>& recvCounts,
                                        MPI_Comm                   comm,
                                        hipStream_t                stream);

            // drain the MPI exchange and launch per-peer-block decompress
            rocfft_status wait();

            // backend compression ratio for `element_count`; 1.0 for native
            double compression_ratio_for(size_t element_count) const;

            // last error message; empty on success
            const std::string& error_message() const
            {
                return _error_message;
            }

            // true between execute_async and wait
            bool in_flight() const
            {
                return _in_flight;
            }

        private:
            rocfft_comm_precision            _comm_precision;
            unsigned int                     _comm_param;
            rocfft_precision                 _storage_precision;
            rocfft_array_type                _array_type;
            std::unique_ptr<CompressBackend> _backend; // null for the native kind
            std::string                      _error_message;
            bool                             _in_flight = false;

            // device-side scratch, grown lazily and reused
            void*  _send_scratch_device  = nullptr;
            size_t _send_scratch_bytes   = 0;
            void*  _recv_scratch_device  = nullptr;
            size_t _recv_scratch_bytes   = 0;
            void*  _decoded_scratch_dev  = nullptr;
            size_t _decoded_scratch_size = 0;

            // state stashed by execute_async for wait() to drain
            MPI_Request         _request          = MPI_REQUEST_NULL;
            void*               _pending_recvbuf  = nullptr;
            hipStream_t         _pending_stream   = nullptr;
            bool                _pending_compress = false;
            std::vector<size_t> _pending_recv_counts;
            std::vector<size_t> _pending_recv_offsets_elements;
            std::vector<size_t> _pending_recv_offsets_bytes;

            rocfft_status ensure_scratch(size_t total_send_bytes, size_t total_recv_bytes);
        };

#else // !ROCFFT_MPI_ENABLE

        // no-op stub for non-MPI builds
        class CompressedAlltoallv
        {
        public:
            CompressedAlltoallv(rocfft_comm_precision,
                                unsigned int,
                                rocfft_precision,
                                rocfft_array_type)
            {
            }
        };

#endif // ROCFFT_MPI_ENABLE

    } // namespace compress
} // namespace rocfft

#endif // ROCFFT_COMPRESSED_ALLTOALL_H
