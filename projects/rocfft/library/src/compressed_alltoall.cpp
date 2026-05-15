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

#include "compressed_alltoall.h"
#include "../../shared/array_predicate.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace rocfft
{
    namespace compress
    {

#ifdef ROCFFT_MPI_ENABLE

        // bytes per element, accounting for the complex pair when the array type is complex
        static size_t element_bytes(rocfft_precision  storage_precision,
                                    rocfft_array_type array_type)
        {
            const bool is_complex = array_type_is_complex(array_type);
            switch(storage_precision)
            {
            case rocfft_precision_half:
                return (is_complex ? 2 : 1) * sizeof(uint16_t);
            case rocfft_precision_single:
                return (is_complex ? 2 : 1) * sizeof(float);
            case rocfft_precision_double:
                return (is_complex ? 2 : 1) * sizeof(double);
            }
            return 0;
        }

        CompressedAlltoallv::CompressedAlltoallv(rocfft_comm_precision comm_precision,
                                                 unsigned int          comm_param,
                                                 rocfft_precision      storage_precision,
                                                 rocfft_array_type     array_type)
            : _comm_precision(comm_precision)
            , _comm_param(comm_param)
            , _storage_precision(storage_precision)
            , _array_type(array_type)
        {
            if(comm_precision != rocfft_comm_precision_native)
            {
                _backend = make_compress_backend(comm_precision, comm_param);
            }
        }

        CompressedAlltoallv::~CompressedAlltoallv()
        {
            if(_send_scratch_device)
                (void)hipFree(_send_scratch_device);
            if(_recv_scratch_device)
                (void)hipFree(_recv_scratch_device);
            if(_decoded_scratch_dev)
                (void)hipFree(_decoded_scratch_dev);
        }

        rocfft_status CompressedAlltoallv::ensure_scratch(size_t total_send_bytes,
                                                          size_t total_recv_bytes)
        {
            auto grow = [](void*& ptr, size_t& cur, size_t need) -> hipError_t {
                if(need <= cur)
                    return hipSuccess;
                if(ptr)
                    (void)hipFree(ptr);
                ptr = nullptr;
                cur = 0;
                if(need == 0)
                    return hipSuccess;
                auto rc = hipMalloc(&ptr, need);
                if(rc == hipSuccess)
                    cur = need;
                return rc;
            };

            if(grow(_send_scratch_device, _send_scratch_bytes, total_send_bytes) != hipSuccess)
                return rocfft_status_failure;
            if(grow(_recv_scratch_device, _recv_scratch_bytes, total_recv_bytes) != hipSuccess)
                return rocfft_status_failure;

            return rocfft_status_success;
        }

        double CompressedAlltoallv::compression_ratio_for(size_t element_count) const
        {
            if(!_backend)
                return 1.0;
            return _backend->compression_ratio(_storage_precision, element_count);
        }

        rocfft_status CompressedAlltoallv::execute_async(const void*                sendbuf_device,
                                                         const std::vector<size_t>& sendOffsets,
                                                         const std::vector<size_t>& sendCounts,
                                                         void*                      recvbuf_device,
                                                         const std::vector<size_t>& recvOffsets,
                                                         const std::vector<size_t>& recvCounts,
                                                         MPI_Comm                   comm,
                                                         hipStream_t                stream)
        {
            if(_in_flight)
            {
                _error_message = "CompressedAlltoallv::execute_async called while a previous "
                                 "exchange is still in flight";
                return rocfft_status_failure;
            }

            _pending_recvbuf = recvbuf_device;
            _pending_stream  = stream;

            // native triple takes the legacy path at the caller; this class always owns a backend
            if(!_backend)
            {
                _error_message = "CompressedAlltoallv constructed without a backend";
                return rocfft_status_failure;
            }

            _pending_compress = true;

            const size_t nranks = sendCounts.size();
            if(recvCounts.size() != nranks || sendOffsets.size() != nranks
               || recvOffsets.size() != nranks)
            {
                _error_message = "sendCounts/recvCounts/sendOffsets/recvOffsets size mismatch";
                return rocfft_status_failure;
            }

            std::vector<int> compressed_send_counts(nranks);
            std::vector<int> compressed_send_offsets(nranks);
            std::vector<int> compressed_recv_counts(nranks);
            std::vector<int> compressed_recv_offsets(nranks);

            size_t cum_send = 0;
            size_t cum_recv = 0;
            for(size_t i = 0; i < nranks; ++i)
            {
                const size_t cs_bytes
                    = _backend->compressed_bytes(_storage_precision, sendCounts[i]);
                const size_t cr_bytes
                    = _backend->compressed_bytes(_storage_precision, recvCounts[i]);
                if(cs_bytes > static_cast<size_t>(std::numeric_limits<int>::max())
                   || cr_bytes > static_cast<size_t>(std::numeric_limits<int>::max()))
                {
                    _error_message = "compressed payload exceeds MPI int max";
                    return rocfft_status_failure;
                }
                compressed_send_counts[i]  = static_cast<int>(cs_bytes);
                compressed_send_offsets[i] = static_cast<int>(cum_send);
                compressed_recv_counts[i]  = static_cast<int>(cr_bytes);
                compressed_recv_offsets[i] = static_cast<int>(cum_recv);
                cum_send += cs_bytes;
                cum_recv += cr_bytes;
            }

            if(auto rc = ensure_scratch(cum_send, cum_recv); rc != rocfft_status_success)
            {
                _error_message = "failed to allocate compress/decompress scratch";
                return rc;
            }

            // compress per peer-block on `stream`
            auto*        src_bytes          = reinterpret_cast<const std::uint8_t*>(sendbuf_device);
            auto*        dst_bytes          = reinterpret_cast<std::uint8_t*>(_send_scratch_device);
            const size_t elem_bytes_storage = element_bytes(_storage_precision, _array_type);
            for(size_t i = 0; i < nranks; ++i)
            {
                if(sendCounts[i] == 0)
                    continue;
                const void* src = src_bytes + sendOffsets[i] * elem_bytes_storage;
                void*       dst = dst_bytes + compressed_send_offsets[i];
                auto rc = _backend->compress(src, _storage_precision, sendCounts[i], dst, stream);
                if(rc != rocfft_status_success)
                {
                    _error_message
                        = std::string("backend compress failed in peer block ") + std::to_string(i);
                    return rc;
                }
            }
            // wait for compress kernels before posting the MPI call
            if(auto rc = hipStreamSynchronize(stream); rc != hipSuccess)
            {
                _error_message = "hipStreamSynchronize before MPI_Ialltoallv failed";
                return rocfft_status_failure;
            }

            const auto rc = MPI_Ialltoallv(_send_scratch_device,
                                           compressed_send_counts.data(),
                                           compressed_send_offsets.data(),
                                           MPI_BYTE,
                                           _recv_scratch_device,
                                           compressed_recv_counts.data(),
                                           compressed_recv_offsets.data(),
                                           MPI_BYTE,
                                           comm,
                                           &_request);
            if(rc != MPI_SUCCESS)
            {
                char errmsg[MPI_MAX_ERROR_STRING];
                int  errlen = 0;
                MPI_Error_string(rc, errmsg, &errlen);
                _error_message = std::string("MPI_Ialltoallv (compressed) failed: ") + errmsg;
                return rocfft_status_failure;
            }

            // stash per-peer recv layout for wait() to drive per-peer-block decompress
            _pending_recv_counts           = recvCounts;
            _pending_recv_offsets_elements = recvOffsets;
            _pending_recv_offsets_bytes.assign(nranks, 0);
            for(size_t i = 0; i < nranks; ++i)
                _pending_recv_offsets_bytes[i] = static_cast<size_t>(compressed_recv_offsets[i]);

            _in_flight = true;
            return rocfft_status_success;
        }

        rocfft_status CompressedAlltoallv::wait()
        {
            if(!_in_flight)
                return rocfft_status_success;

            if(auto rc = MPI_Wait(&_request, MPI_STATUS_IGNORE); rc != MPI_SUCCESS)
            {
                char errmsg[MPI_MAX_ERROR_STRING];
                int  errlen = 0;
                MPI_Error_string(rc, errmsg, &errlen);
                _error_message = std::string("MPI_Wait failed: ") + errmsg;
                _in_flight     = false;
                return rocfft_status_failure;
            }
            _request = MPI_REQUEST_NULL;

            if(_pending_compress && _backend)
            {
                // per-peer-block decompress; each peer's payload is independently encoded
                auto* recv_compressed_bytes
                    = reinterpret_cast<const std::uint8_t*>(_recv_scratch_device);
                auto*        recv_dst_bytes     = reinterpret_cast<std::uint8_t*>(_pending_recvbuf);
                const size_t elem_bytes_storage = element_bytes(_storage_precision, _array_type);

                const size_t nranks = _pending_recv_counts.size();
                for(size_t i = 0; i < nranks; ++i)
                {
                    const size_t cnt = _pending_recv_counts[i];
                    if(cnt == 0)
                        continue;
                    const void* src = recv_compressed_bytes + _pending_recv_offsets_bytes[i];
                    void*       dst
                        = recv_dst_bytes + _pending_recv_offsets_elements[i] * elem_bytes_storage;
                    auto rc
                        = _backend->decompress(src, dst, _storage_precision, cnt, _pending_stream);
                    if(rc != rocfft_status_success)
                    {
                        _error_message = std::string("backend decompress failed in peer block ")
                                         + std::to_string(i);
                        _in_flight = false;
                        return rc;
                    }
                }

                if(auto hip_rc = hipStreamSynchronize(_pending_stream); hip_rc != hipSuccess)
                {
                    _error_message = "hipStreamSynchronize after decompress failed";
                    _in_flight     = false;
                    return rocfft_status_failure;
                }
            }

            _pending_recv_counts.clear();
            _pending_recv_offsets_elements.clear();
            _pending_recv_offsets_bytes.clear();
            _in_flight = false;
            return rocfft_status_success;
        }

#endif // ROCFFT_MPI_ENABLE

    } // namespace compress
} // namespace rocfft
