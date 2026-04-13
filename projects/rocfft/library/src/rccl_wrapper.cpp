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

#include "rccl_wrapper.h"
#include "logging.h"
#include <algorithm>
#include <iostream>
#include <iterator>
#include <map>
#include <mutex>
#include <numeric>
#include <stdexcept>

namespace rocfft_rccl
{
    std::shared_ptr<Communicator> Communicator::comm_world;
    static std::mutex             comm_world_mutex;

#ifdef ROCFFT_RCCL_ENABLE

    struct NcclTypeInfo
    {
        ncclDataType_t dtype;
        size_t         count_multiplier; // 1 for real, 2 for complex
    };

    // Map (base_type_size, is_complex) to the NCCL datatype and a count
    // multiplier.  base_type_size is the size of one real component
    // (2 for half, 4 for float, 8 for double).  Complex data is
    // transferred as 2x the real component type.
    static NcclTypeInfo get_nccl_type_info(size_t base_type_size, bool is_complex)
    {
        ncclDataType_t dtype;
        switch(base_type_size)
        {
        case 2:
            dtype = ncclFloat16;
            break;
        case 4:
            dtype = ncclFloat32;
            break;
        case 8:
            dtype = ncclFloat64;
            break;
        default:
            // rocFFT only produces half (2), float (4), or double (8).
            // Any other size indicates a bug in the caller.  There is
            // no safe fallback since the count_multiplier assumes a
            // floating-point element width.
            throw std::runtime_error("unexpected base_type_size " + std::to_string(base_type_size)
                                     + " in RCCL datatype mapping (expected 2, 4, or 8)");
        }
        return {dtype, is_complex ? size_t{2} : size_t{1}};
    }

    // implementation details
    struct Communicator::Impl
    {
        // each comm is for one rank, and each rank has exactly one
        // device.  comms is indexed by device_id.
        std::vector<ncclComm_t> comms;

        // unique id used to bootstrap this communicator group.
        // stored so it can be broadcast via MPI for multi-node in the future.
        ncclUniqueId uniqueId{};

        // device_id -> NCCL rank mapping
        std::map<int, int> device_to_rank;
        int                nranks = 0;

        ~Impl()
        {
            for(auto comm : comms)
            {
                if(comm)
                {
                    ncclCommFinalize(comm);
                    ncclCommDestroy(comm);
                }
            }
        }
    };
#endif

    Communicator::Communicator()
#ifdef ROCFFT_RCCL_ENABLE
        : pimpl(std::make_unique<Impl>())
#endif
    {
    }

    Communicator::~Communicator() {}

    Communicator::Communicator(Communicator&& other)
#ifdef ROCFFT_RCCL_ENABLE
        : pimpl(std::move(other.pimpl))
#endif
    {
    }

    Communicator& Communicator::operator=(Communicator&& other)
    {
#ifdef ROCFFT_RCCL_ENABLE
        pimpl = std::move(other.pimpl);
#endif
        return *this;
    }

    std::shared_ptr<Communicator> Communicator::create(const std::set<int>& devices)
    {
#ifdef ROCFFT_RCCL_ENABLE
        // check if RCCL is disabled via environment variable
        const char* disable_rccl = std::getenv("ROCFFT_DISABLE_RCCL");
        if(disable_rccl && std::string(disable_rccl) == "1")
        {
            return {};
        }

        // need at least 2 devices for a meaningful communicator
        if(devices.size() < 2)
        {
            return {};
        }

        // create communicator scoped to the requested devices.
        // the communicator is cached in comm_world for reuse by
        // subsequent plans with the same device set.
        // guard with a mutex so concurrent plan creation from
        // multiple threads does not race on comm_world.
        std::lock_guard<std::mutex> lock(comm_world_mutex);

        if(!comm_world)
        {
            const int ndevices = static_cast<int>(devices.size());

            auto new_comm   = std::make_shared<Communicator>();
            new_comm->pimpl = std::make_unique<Impl>();

            // generate unique id for this communicator group.
            // for single-node this stays local; for multi-node the root
            // rank would broadcast this via MPI_Bcast.
            ncclUniqueId id;
            ncclResult_t result = ncclGetUniqueId(&id);
            if(result != ncclSuccess)
            {
                return {};
            }
            new_comm->pimpl->uniqueId = id;
            new_comm->pimpl->nranks   = ndevices;

            // save and restore the caller's active device
            int original_device = 0;
            if(hipGetDevice(&original_device) != hipSuccess)
                throw std::runtime_error("hipGetDevice failed during RCCL communicator init");

            // comms is indexed by rank (not device_id); device_to_rank
            // maps device_id -> rank for lookup in get_comm().
            new_comm->pimpl->comms.resize(ndevices, nullptr);

            // init one communicator per device using ncclCommInitRank,
            // batched inside a group call for single-process efficiency.
            // use try/catch to guarantee ncclGroupEnd is called even if
            // hipSetDevice throws between ncclGroupStart and ncclGroupEnd.
            // ranks are assigned in sorted device-id order (std::set).
            ncclGroupStart();
            try
            {
                int rank = 0;
                for(int dev : devices)
                {
                    if(hipSetDevice(dev) != hipSuccess)
                        throw std::runtime_error("hipSetDevice failed for device "
                                                 + std::to_string(dev));
                    new_comm->pimpl->device_to_rank[dev] = rank;
                    result = ncclCommInitRank(&new_comm->pimpl->comms[rank], ndevices, id, rank);
                    if(result != ncclSuccess)
                    {
                        ncclGroupEnd();
                        if(hipSetDevice(original_device) != hipSuccess)
                            throw std::runtime_error("hipSetDevice failed restoring device "
                                                     + std::to_string(original_device));
                        return {};
                    }
                    ++rank;
                }
            }
            catch(...)
            {
                ncclGroupEnd();
                throw;
            }
            result = ncclGroupEnd();

            if(hipSetDevice(original_device) != hipSuccess)
                throw std::runtime_error("hipSetDevice failed restoring device "
                                         + std::to_string(original_device));

            if(result != ncclSuccess)
            {
                return {};
            }
            std::swap(comm_world, new_comm);
        }

        return comm_world;
#else
        return {};
#endif
    }

    void* Communicator::get_comm(int device_id) const
    {
#ifdef ROCFFT_RCCL_ENABLE
        // look up the NCCL rank for this device and index into
        // comms by rank (not device_id) so this stays correct
        // even if device ids are non-contiguous or reordered.
        auto it = pimpl->device_to_rank.find(device_id);
        if(it == pimpl->device_to_rank.end())
            return nullptr;
        return &pimpl->comms[it->second];
#else
        return nullptr;
#endif
    }

    int Communicator::num_ranks() const
    {
#ifdef ROCFFT_RCCL_ENABLE
        return pimpl->nranks;
#else
        return 0;
#endif
    }

    int Communicator::get_rank(int device_id) const
    {
#ifdef ROCFFT_RCCL_ENABLE
        auto it = pimpl->device_to_rank.find(device_id);
        if(it == pimpl->device_to_rank.end())
            return -1;
        return it->second;
#else
        return -1;
#endif
    }

    // RAII group wrapper
    Group::Group()
    {
#ifdef ROCFFT_RCCL_ENABLE
        ncclGroupStart();
#endif
    }

    Group::~Group()
    {
#ifdef ROCFFT_RCCL_ENABLE
        ncclGroupEnd();
#endif
    }

    bool Communicator::alltoall(const void* sendbuf,
                                void*       recvbuf,
                                size_t      count,
                                int         device_id,
                                hipStream_t stream,
                                size_t      base_type_size,
                                bool        is_complex)
    {
#ifdef ROCFFT_RCCL_ENABLE
        ncclComm_t* comm_ptr = static_cast<ncclComm_t*>(get_comm(device_id));
        if(!comm_ptr)
            return false;

        auto [dtype, multiplier] = get_nccl_type_info(base_type_size, is_complex);

        ncclResult_t result
            = ncclAllToAll(sendbuf, recvbuf, count * multiplier, dtype, *comm_ptr, stream);

        if(result != ncclSuccess)
        {
            log_trace(__func__, "ncclAllToAll failed", result);
            return false;
        }

        return true;
#else
        return false;
#endif
    }

    bool Communicator::alltoallv(const void*                sendbuf,
                                 void*                      recvbuf,
                                 const std::vector<size_t>& sendcounts,
                                 const std::vector<size_t>& recvcounts,
                                 const std::vector<size_t>& sdispls,
                                 const std::vector<size_t>& rdispls,
                                 int                        device_id,
                                 hipStream_t                stream,
                                 size_t                     base_type_size,
                                 bool                       is_complex)
    {
#ifdef ROCFFT_RCCL_ENABLE
        ncclComm_t* comm_ptr = static_cast<ncclComm_t*>(get_comm(device_id));
        if(!comm_ptr)
            return false;

        auto [dtype, multiplier] = get_nccl_type_info(base_type_size, is_complex);

        // scale counts and displacements by the multiplier (2x for complex)
        std::vector<size_t> adj_sendcounts = sendcounts;
        std::vector<size_t> adj_recvcounts = recvcounts;
        std::vector<size_t> adj_sdispls    = sdispls;
        std::vector<size_t> adj_rdispls    = rdispls;

        if(multiplier > 1)
        {
            for(auto& c : adj_sendcounts)
                c *= multiplier;
            for(auto& c : adj_recvcounts)
                c *= multiplier;
            for(auto& d : adj_sdispls)
                d *= multiplier;
            for(auto& d : adj_rdispls)
                d *= multiplier;
        }

        ncclResult_t result = ncclAllToAllv(sendbuf,
                                            adj_sendcounts.data(),
                                            adj_sdispls.data(),
                                            recvbuf,
                                            adj_recvcounts.data(),
                                            adj_rdispls.data(),
                                            dtype,
                                            *comm_ptr,
                                            stream);

        if(result != ncclSuccess)
        {
            log_trace(__func__, "ncclAllToAllv failed", result);
            return false;
        }

        return true;
#else
        return false;
#endif
    }

    bool Communicator::send(const void* sendbuf,
                            size_t      count,
                            int         peer_rank,
                            int         device_id,
                            hipStream_t stream,
                            size_t      base_type_size,
                            bool        is_complex)
    {
#ifdef ROCFFT_RCCL_ENABLE
        ncclComm_t* comm_ptr = static_cast<ncclComm_t*>(get_comm(device_id));
        if(!comm_ptr)
            return false;

        auto [dtype, multiplier] = get_nccl_type_info(base_type_size, is_complex);

        ncclResult_t result
            = ncclSend(sendbuf, count * multiplier, dtype, peer_rank, *comm_ptr, stream);

        if(result != ncclSuccess)
        {
            log_trace(__func__, "ncclSend failed", result);
            return false;
        }

        return true;
#else
        return false;
#endif
    }

    bool Communicator::recv(void*       recvbuf,
                            size_t      count,
                            int         peer_rank,
                            int         device_id,
                            hipStream_t stream,
                            size_t      base_type_size,
                            bool        is_complex)
    {
#ifdef ROCFFT_RCCL_ENABLE
        ncclComm_t* comm_ptr = static_cast<ncclComm_t*>(get_comm(device_id));
        if(!comm_ptr)
            return false;

        auto [dtype, multiplier] = get_nccl_type_info(base_type_size, is_complex);

        ncclResult_t result
            = ncclRecv(recvbuf, count * multiplier, dtype, peer_rank, *comm_ptr, stream);

        if(result != ncclSuccess)
        {
            log_trace(__func__, "ncclRecv failed", result);
            return false;
        }

        return true;
#else
        return false;
#endif
    }

} // namespace rocfft_rccl
