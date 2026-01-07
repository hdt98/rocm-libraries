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
#include <vector>

#ifdef ROCFFT_RCCL_ENABLE
#include <rccl/rccl.h>
#endif

namespace rocfft_rccl
{
    // RCCL communicator wrapper for single-process multi-GPU
    class RCCLCommunicator
    {
    public:
        RCCLCommunicator();
        ~RCCLCommunicator() = default;

        // singleton allocated in rocfft_setup and freed in rocfft_cleanup
        static std::unique_ptr<RCCLCommunicator> single;

        // check if RCCL is available and initialized
        bool is_available() const;

        // initialize RCCL for given devices (call once per process)
        bool initialize(const std::vector<int>& devices);

        // finalize RCCL (call at shutdown)
        void finalize();

        // get the RCCL communicator for a specific device
        void* get_comm(int device_id) const;

        // get number of ranks
        int get_nranks() const;

        // get device ID for a rank
        int get_device_for_rank(int rank) const;

        // get rank for a device ID
        int get_rank_for_device(int device_id) const;

        // get all device IDs
        const std::vector<int>& get_devices() const;

        // check if a specific device is managed
        bool has_device(int device_id) const;

    private:
        // non-copyable
        RCCLCommunicator(const RCCLCommunicator&) = delete;
        RCCLCommunicator& operator=(const RCCLCommunicator&) = delete;

        struct Impl;
        std::unique_ptr<Impl> pimpl;
    };

    // RAII wrapper for RCCL group operations
    class RCCLGroup
    {
    public:
        RCCLGroup();
        ~RCCLGroup();

        // non-copyable, non-movable
        RCCLGroup(const RCCLGroup&) = delete;
        RCCLGroup& operator=(const RCCLGroup&) = delete;
        RCCLGroup(RCCLGroup&&)                 = delete;
        RCCLGroup& operator=(RCCLGroup&&) = delete;
    };

    // helper functions for RCCL operations
    namespace ops
    {
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
    } // namespace ops

} // namespace rocfft_rccl

#endif // ROCFFT_RCCL_WRAPPER_H
