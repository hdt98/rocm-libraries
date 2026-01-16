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
        // each comm is for one rank, and each rank has exactly one
        // device
        std::vector<ncclComm_t> comms;

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

            // init with devices 0..N
            new_comm->pimpl->comms.resize(ndevices);
            ncclResult_t result = ncclCommInitAll(new_comm->pimpl->comms.data(), ndevices, nullptr);
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
        return &pimpl->comms[device_id];
#else
        return nullptr;
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
