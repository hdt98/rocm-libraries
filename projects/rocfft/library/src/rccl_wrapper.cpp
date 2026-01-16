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

#include "rccl_wrapper.h"
#include "logging.h"
#include <algorithm>
#include <iostream>
#include <iterator>
#include <map>
#include <numeric>
#include <stdexcept>

namespace rocfft_rccl
{
#ifdef ROCFFT_RCCL_ENABLE

    std::shared_ptr<Communicator> Communicator::comm_world;

    // RCCL data type mapping - returns base NCCL type for given element size.
    // for complex types (8 or 16 bytes), callers must double the count.
    static ncclDataType_t get_nccl_datatype(size_t elem_size)
    {
        switch(elem_size)
        {
        case 1:
            return ncclInt8;
        case 2:
            return ncclFloat16;
        case 4:
            return ncclFloat32;
        case 8:
            // complex<float> (8 bytes) - use float32, caller doubles count
            return ncclFloat32;
        case 16:
            // complex<double> (16 bytes) - use float64, caller doubles count
            return ncclFloat64;
        default:
            return ncclInt8;
        }
    }

    // implementation details
    struct Communicator::Impl
    {
        std::vector<ncclComm_t> comms; // one per device
        std::vector<int>        devices; // device IDs
        std::map<int, int>      device_to_rank;

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

        // build device-rank mappings
        void init_maps()
        {
            for(int rank = 0; rank < static_cast<int>(devices.size()); ++rank)
            {
                int dev_id             = devices[rank];
                device_to_rank[dev_id] = rank;
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

            // init with devices 0..N
            new_comm->pimpl->devices.resize(ndevices);
            std::iota(new_comm->pimpl->devices.begin(), new_comm->pimpl->devices.end(), 0);
            new_comm->pimpl->comms.resize(ndevices);
            ncclResult_t result = ncclCommInitAll(
                new_comm->pimpl->comms.data(), ndevices, new_comm->pimpl->devices.data());
            if(result != ncclSuccess)
            {
                return {};
            }
            new_comm->pimpl->init_maps();
            std::swap(comm_world, new_comm);
        }

        // if we need a communicator for all devices, just use the world communicator
        if(devices.size() == comm_world->pimpl->devices.size())
            return comm_world;

        try
        {
            // create sub-communicator from comm_world for specified devices
            auto ret   = std::make_shared<Communicator>();
            ret->pimpl = std::make_unique<Impl>();

            std::copy(devices.begin(), devices.end(), std::back_inserter(ret->pimpl->devices));

            // for each rank in comm_world, split it into a
            // sub-communicator.  use NOCOLOR for ranks that aren't
            // supposed to be in this sub-communicator
            for(size_t rank = 0; rank < comm_world->pimpl->comms.size(); ++rank)
            {
                bool rank_in_sub = devices.find(comm_world->pimpl->devices[rank]) != devices.end();
                ncclComm_t   rank_comm = nullptr;
                ncclResult_t result    = ncclCommSplit(comm_world->pimpl->comms[rank],
                                                    rank_in_sub ? 0 : NCCL_SPLIT_NOCOLOR,
                                                    comm_world->pimpl->devices[rank],
                                                    &rank_comm,
                                                    nullptr);
                if(result != ncclSuccess)
                    return {};
                ret->pimpl->comms.push_back(rank_comm);
            }

            ret->pimpl->init_maps();
            return ret;
        }
        catch(const std::exception& e)
        {
            log_trace(__func__, "rccl initialization failed", e.what());
            return {};
        }
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

        int rank = it->second;
        if(rank < 0 || rank >= static_cast<int>(pimpl->comms.size()))
            return nullptr;

        return static_cast<void*>(&pimpl->comms[rank]);
#else
        return nullptr;
#endif
    }

    int Communicator::get_rank_for_device(int device_id) const
    {
#ifdef ROCFFT_RCCL_ENABLE
        auto it = pimpl->device_to_rank.find(device_id);
        return (it != pimpl->device_to_rank.end()) ? it->second : -1;
#else
        return -1;
#endif
    }

    bool Communicator::has_device(int device_id) const
    {
#ifdef ROCFFT_RCCL_ENABLE
        return pimpl->device_to_rank.find(device_id) != pimpl->device_to_rank.end();
#else
        return false;
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
                                size_t      elem_size)
    {
#ifdef ROCFFT_RCCL_ENABLE
        ncclComm_t* comm_ptr = static_cast<ncclComm_t*>(get_comm(device_id));
        if(!comm_ptr)
            return false;

        ncclDataType_t dtype = get_nccl_datatype(elem_size);

        // for complex types, adjust count
        size_t nccl_count = count;
        if(elem_size == 8)
        {
            // complex float: treat as 2x float
            nccl_count = count * 2;
            dtype      = ncclFloat32;
        }
        else if(elem_size == 16)
        {
            // complex double: treat as 2x double
            nccl_count = count * 2;
            dtype      = ncclFloat64;
        }

        ncclResult_t result = ncclAllToAll(sendbuf, recvbuf, nccl_count, dtype, *comm_ptr, stream);

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
                                 size_t                     elem_size)
    {
#ifdef ROCFFT_RCCL_ENABLE
        ncclComm_t* comm_ptr = static_cast<ncclComm_t*>(get_comm(device_id));
        if(!comm_ptr)
            return false;

        ncclDataType_t dtype = get_nccl_datatype(elem_size);

        // for complex types, adjust counts and displacements
        std::vector<size_t> adj_sendcounts = sendcounts;
        std::vector<size_t> adj_recvcounts = recvcounts;
        std::vector<size_t> adj_sdispls    = sdispls;
        std::vector<size_t> adj_rdispls    = rdispls;

        if(elem_size == 8)
        {
            // complex float
            dtype = ncclFloat32;
            for(auto& c : adj_sendcounts)
                c *= 2;
            for(auto& c : adj_recvcounts)
                c *= 2;
            for(auto& d : adj_sdispls)
                d *= 2;
            for(auto& d : adj_rdispls)
                d *= 2;
        }
        else if(elem_size == 16)
        {
            // complex double
            dtype = ncclFloat64;
            for(auto& c : adj_sendcounts)
                c *= 2;
            for(auto& c : adj_recvcounts)
                c *= 2;
            for(auto& d : adj_sdispls)
                d *= 2;
            for(auto& d : adj_rdispls)
                d *= 2;
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
                            size_t      elem_size)
    {
#ifdef ROCFFT_RCCL_ENABLE
        ncclComm_t* comm_ptr = static_cast<ncclComm_t*>(get_comm(device_id));
        if(!comm_ptr)
            return false;

        ncclDataType_t dtype      = get_nccl_datatype(elem_size);
        size_t         nccl_count = count;

        if(elem_size == 8)
        {
            nccl_count = count * 2;
            dtype      = ncclFloat32;
        }
        else if(elem_size == 16)
        {
            nccl_count = count * 2;
            dtype      = ncclFloat64;
        }

        ncclResult_t result = ncclSend(sendbuf, nccl_count, dtype, peer_rank, *comm_ptr, stream);

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
                            size_t      elem_size)
    {
#ifdef ROCFFT_RCCL_ENABLE
        ncclComm_t* comm_ptr = static_cast<ncclComm_t*>(get_comm(device_id));
        if(!comm_ptr)
            return false;

        ncclDataType_t dtype      = get_nccl_datatype(elem_size);
        size_t         nccl_count = count;

        if(elem_size == 8)
        {
            nccl_count = count * 2;
            dtype      = ncclFloat32;
        }
        else if(elem_size == 16)
        {
            nccl_count = count * 2;
            dtype      = ncclFloat64;
        }

        ncclResult_t result = ncclRecv(recvbuf, nccl_count, dtype, peer_rank, *comm_ptr, stream);

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
