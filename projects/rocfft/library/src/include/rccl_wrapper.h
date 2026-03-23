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
    // RCCL communicator wrapper for single-node and multi-node multi-GPU
    class Communicator
    {
    public:
        // default ctor does not actually initialize rccl - call
        // create or create_multinode to init and check success
        Communicator();
        ~Communicator();

        // allow moves
        Communicator(Communicator&&);
        Communicator& operator=(Communicator&&);

        // single-node: return a communicator for the specified
        // local devices (one process manages all GPUs)
        static std::shared_ptr<Communicator> create(const std::set<int>& devices);

        // multi-node: return a communicator spanning multiple MPI
        // ranks.  Each rank manages its local devices.
        // mpi_rank and mpi_size come from MPI_Comm_rank/size.
        // local_devices are the HIP device IDs owned by this rank.
        // mpi_comm_ptr is a pointer to the MPI_Comm to broadcast
        // the ncclUniqueId (passed as void* to avoid MPI header dependency).
        static std::shared_ptr<Communicator> create_multinode(int                  mpi_rank,
                                                              int                  mpi_size,
                                                              const std::set<int>& local_devices,
                                                              void*                mpi_comm_ptr);

        // process-wide communicator for all visible devices, created
        // on demand and destroyed at cleanup
        static std::shared_ptr<Communicator> comm_world;

        // get the RCCL communicator for a specific device
        void* get_comm(int device_id) const;

        // total number of ranks in this communicator
        int num_ranks() const;

        // NCCL rank assigned to the given device, or -1 if not found
        int get_rank(int device_id) const;

        // MPI rank that owns this communicator, or 0 for single-node
        int mpi_rank() const;

        // number of MPI ranks in this communicator, or 1 for single-node
        int mpi_size() const;

        // number of local devices per MPI rank
        int devices_per_rank() const;

        // compute the global RCCL rank for a brick identified by
        // its (comm_rank, device_id).  For single-node, comm_rank
        // is ignored and device_id is the rank.  For multi-node,
        // global_rank = comm_rank * devices_per_rank + device_local_index.
        int global_rank_for(int comm_rank, int device_id) const;

        // all-to-all with uniform counts.
        // base_type_size is the size of one real component (2/4/8).
        bool alltoall(const void* sendbuf,
                      void*       recvbuf,
                      size_t      count,
                      int         device_id,
                      hipStream_t stream,
                      size_t      base_type_size,
                      bool        is_complex);

        // all-to-all with variable counts.
        // base_type_size is the size of one real component (2/4/8).
        bool alltoallv(const void*                sendbuf,
                       void*                      recvbuf,
                       const std::vector<size_t>& sendcounts,
                       const std::vector<size_t>& recvcounts,
                       const std::vector<size_t>& sdispls,
                       const std::vector<size_t>& rdispls,
                       int                        device_id,
                       hipStream_t                stream,
                       size_t                     base_type_size,
                       bool                       is_complex);

        // point-to-point send.
        // base_type_size is the size of one real component (2/4/8).
        bool send(const void* sendbuf,
                  size_t      count,
                  int         peer_rank,
                  int         device_id,
                  hipStream_t stream,
                  size_t      base_type_size,
                  bool        is_complex);

        // point-to-point receive.
        // base_type_size is the size of one real component (2/4/8).
        bool recv(void*       recvbuf,
                  size_t      count,
                  int         peer_rank,
                  int         device_id,
                  hipStream_t stream,
                  size_t      base_type_size,
                  bool        is_complex);

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
