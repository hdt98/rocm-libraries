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
#include "rocfft_mpi.h"
#include <algorithm>
#include <iostream>
#include <iterator>
#include <map>
#include <numeric>
#include <stdexcept>

namespace rocfft_rccl
{
    std::shared_ptr<Communicator> Communicator::comm_world;

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
            dtype = ncclInt8;
            break;
        }
        return {dtype, is_complex ? size_t{2} : size_t{1}};
    }

    // implementation details
    struct Communicator::Impl
    {
        // one ncclComm_t per local device
        std::vector<ncclComm_t> comms;

        // unique id used to bootstrap this communicator group
        ncclUniqueId uniqueId{};

        // local device_id -> global NCCL rank mapping
        std::map<int, int> device_to_rank;
        int                nranks = 0;

        // MPI topology (0/1 for single-node)
        int mpi_rank_val = 0;
        int mpi_size_val = 1;
        int nlocal       = 0; // number of local devices per rank

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

        if(devices.empty())
        {
            return {};
        }

        // create world communicator if necessary
        if(!comm_world)
        {
            int ndevices = 0;
            if(hipGetDeviceCount(&ndevices) != hipSuccess || ndevices < 2)
                return {};
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
            new_comm->pimpl->nlocal   = ndevices;

            // save and restore the caller's active device
            int original_device = 0;
            if(hipGetDevice(&original_device) != hipSuccess)
                throw std::runtime_error("hipGetDevice failed during RCCL communicator init");

            new_comm->pimpl->comms.resize(ndevices, nullptr);

            // init one communicator per device using ncclCommInitRank,
            // batched inside a group call for single-process efficiency
            ncclGroupStart();
            for(int i = 0; i < ndevices; ++i)
            {
                if(hipSetDevice(i) != hipSuccess)
                    throw std::runtime_error("hipSetDevice failed for device " + std::to_string(i));
                new_comm->pimpl->device_to_rank[i] = i;
                result = ncclCommInitRank(&new_comm->pimpl->comms[i], ndevices, id, i);
                if(result != ncclSuccess)
                {
                    ncclGroupEnd();
                    if(hipSetDevice(original_device) != hipSuccess)
                        throw std::runtime_error("hipSetDevice failed restoring device "
                                                 + std::to_string(original_device));
                    return {};
                }
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

    std::shared_ptr<Communicator> Communicator::create_multinode(int                  mpi_rank,
                                                                 int                  mpi_size,
                                                                 const std::set<int>& local_devices,
                                                                 void*                mpi_comm_ptr)
    {
#if defined(ROCFFT_RCCL_ENABLE) && defined(ROCFFT_MPI_ENABLE)
        const char* disable_rccl = std::getenv("ROCFFT_DISABLE_RCCL");
        if(disable_rccl && std::string(disable_rccl) == "1")
            return {};

        if(local_devices.empty() || mpi_size < 2 || !mpi_comm_ptr)
            return {};

        MPI_Comm mpi_comm = *static_cast<MPI_Comm*>(mpi_comm_ptr);

        int nlocal = static_cast<int>(local_devices.size());
        int ntotal = nlocal * mpi_size;

        auto new_comm                 = std::make_shared<Communicator>();
        new_comm->pimpl               = std::make_unique<Impl>();
        new_comm->pimpl->nranks       = ntotal;
        new_comm->pimpl->nlocal       = nlocal;
        new_comm->pimpl->mpi_rank_val = mpi_rank;
        new_comm->pimpl->mpi_size_val = mpi_size;

        // root rank generates the unique id, then broadcasts to all
        ncclUniqueId id{};
        if(mpi_rank == 0)
        {
            ncclResult_t result = ncclGetUniqueId(&id);
            if(result != ncclSuccess)
                return {};
        }
        MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, mpi_comm);
        new_comm->pimpl->uniqueId = id;

        int original_device = 0;
        if(hipGetDevice(&original_device) != hipSuccess)
            throw std::runtime_error("hipGetDevice failed during multi-node RCCL init");

        new_comm->pimpl->comms.resize(nlocal, nullptr);

        // each rank initializes its local devices with global ranks:
        // global_rank = mpi_rank * nlocal + local_index
        ncclGroupStart();
        int local_idx = 0;
        for(int dev : local_devices)
        {
            int global_rank = mpi_rank * nlocal + local_idx;

            if(hipSetDevice(dev) != hipSuccess)
                throw std::runtime_error("hipSetDevice failed for device " + std::to_string(dev));

            new_comm->pimpl->device_to_rank[dev] = global_rank;

            ncclResult_t result
                = ncclCommInitRank(&new_comm->pimpl->comms[local_idx], ntotal, id, global_rank);
            if(result != ncclSuccess)
            {
                ncclGroupEnd();
                if(hipSetDevice(original_device) != hipSuccess)
                    throw std::runtime_error("hipSetDevice failed restoring device "
                                             + std::to_string(original_device));
                return {};
            }
            ++local_idx;
        }
        ncclResult_t result = ncclGroupEnd();

        if(hipSetDevice(original_device) != hipSuccess)
            throw std::runtime_error("hipSetDevice failed restoring device "
                                     + std::to_string(original_device));

        if(result != ncclSuccess)
            return {};

        if(LOG_PLAN_ENABLED())
        {
            log_plan("RCCL multi-node init: mpi_rank=" + std::to_string(mpi_rank) + "/"
                     + std::to_string(mpi_size) + ", local_devices=" + std::to_string(nlocal)
                     + ", total_ranks=" + std::to_string(ntotal) + "\n");
        }

        return new_comm;
#else
        return {};
#endif
    }

    void* Communicator::get_comm(int device_id) const
    {
#ifdef ROCFFT_RCCL_ENABLE
        auto it = pimpl->device_to_rank.find(device_id);
        if(it == pimpl->device_to_rank.end())
            return nullptr;

        // comms is indexed by local position within device_to_rank.
        // find the local index for this device_id.
        int local_idx = 0;
        for(auto& kv : pimpl->device_to_rank)
        {
            if(kv.first == device_id)
                return &pimpl->comms[local_idx];
            ++local_idx;
        }
        return nullptr;
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

    int Communicator::mpi_rank() const
    {
#ifdef ROCFFT_RCCL_ENABLE
        return pimpl->mpi_rank_val;
#else
        return 0;
#endif
    }

    int Communicator::mpi_size() const
    {
#ifdef ROCFFT_RCCL_ENABLE
        return pimpl->mpi_size_val;
#else
        return 1;
#endif
    }

    int Communicator::devices_per_rank() const
    {
#ifdef ROCFFT_RCCL_ENABLE
        return pimpl->nlocal;
#else
        return 0;
#endif
    }

    int Communicator::global_rank_for(int comm_rank, int device_id) const
    {
#ifdef ROCFFT_RCCL_ENABLE
        if(pimpl->mpi_size_val <= 1)
        {
            // single-node: comm_rank is irrelevant, rank == device_id
            auto it = pimpl->device_to_rank.find(device_id);
            return (it != pimpl->device_to_rank.end()) ? it->second : -1;
        }
        // multi-node: assume uniform layout across ranks.
        // global_rank = comm_rank * devices_per_rank + device_local_index.
        // For the common 1-GPU-per-rank case this is just comm_rank.
        //
        // device_local_index is the position of device_id within the
        // ordered set of devices on that rank.  Since we only know
        // the local mapping, assume remote ranks use the same device
        // index ordering.
        if(comm_rank == pimpl->mpi_rank_val)
        {
            // local rank: use the actual mapping
            auto it = pimpl->device_to_rank.find(device_id);
            return (it != pimpl->device_to_rank.end()) ? it->second : -1;
        }
        else
        {
            // remote rank: compute based on uniform layout assumption
            // device_local_index = 0 for the 1-GPU-per-rank case
            return comm_rank * pimpl->nlocal;
        }
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
