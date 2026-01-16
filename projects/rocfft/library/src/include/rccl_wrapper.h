// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCFFT_RCCL_WRAPPER_H
#define ROCFFT_RCCL_WRAPPER_H

#include <hip/hip_runtime.h>
#include <memory>
#include <set>
#include <vector>

#ifdef ROCFFT_RCCL_ENABLE
#include <rccl/rccl.h>
#endif

namespace rocfft_rccl
{
    // RCCL communicator wrapper for single-process multi-GPU
    class Communicator
    {
    public:
        // default ctor does not actually initialize rccl - call
        // create to init and check success
        Communicator();
        ~Communicator();

        // allow moves
        Communicator(Communicator&&);
        Communicator& operator=(Communicator&&);

        // return a communicator for the specified devices
        static std::shared_ptr<Communicator> create(const std::set<int>& devices);
        // process-wide communicator for all visible devices, created
        // on demand and destroyed at cleanup
        static std::shared_ptr<Communicator> comm_world;

        // get the RCCL communicator for a specific device
        void* get_comm(int device_id) const;

        // all-to-all with uniform counts
        bool alltoall(const void* sendbuf,
                      void*       recvbuf,
                      size_t      count,
                      int         device_id,
                      hipStream_t stream,
                      size_t      elem_size);

        // all-to-all with variable counts
        bool alltoallv(const void*                sendbuf,
                       void*                      recvbuf,
                       const std::vector<size_t>& sendcounts,
                       const std::vector<size_t>& recvcounts,
                       const std::vector<size_t>& sdispls,
                       const std::vector<size_t>& rdispls,
                       int                        device_id,
                       hipStream_t                stream,
                       size_t                     elem_size);

        // point-to-point send
        bool send(const void* sendbuf,
                  size_t      count,
                  int         peer_rank,
                  int         device_id,
                  hipStream_t stream,
                  size_t      elem_size);

        // point-to-point receive
        bool recv(void*       recvbuf,
                  size_t      count,
                  int         peer_rank,
                  int         device_id,
                  hipStream_t stream,
                  size_t      elem_size);

    private:
        // non-copyable
        Communicator(const Communicator&) = delete;
        Communicator& operator=(const Communicator&) = delete;

#ifdef ROCFFT_RCCL_ENABLE
        struct Impl;
        std::unique_ptr<Impl> pimpl;
#endif
    };

    // RAII wrapper for RCCL group operations
    class Group
    {
    public:
        Group();
        ~Group();

        // non-copyable, non-movable
        Group(const Group&) = delete;
        Group& operator=(const Group&) = delete;
        Group(Group&&)                 = delete;
        Group& operator=(Group&&) = delete;
    };

} // namespace rocfft_rccl

#endif // ROCFFT_RCCL_WRAPPER_H
