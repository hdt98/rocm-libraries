/* **************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * *************************************************************************/

#pragma once

#include "common_host_helpers.hpp"
#include "lib_host_helpers.hpp"
#include "rocsolver/rocsolver.h"

#include <vector>

ROCSOLVER_BEGIN_NAMESPACE

#define MIN_CHUNK_SIZE 64

/* ROCSOLVER_WORKSPACE_HELPER provides a wrapper for workspace arrays, allowing workspace arrays to be passed
   between functions using a single object rather than individually, and improving device memory utilization. */
class rocsolver_workspace_helper
{
private:
    size_t num_excl, size_excl, size_shared;
    std::vector<uint8_t*> pointers;
    std::vector<size_t> sizes;
    std::vector<rocsolver_workspace_helper*> nested;

    void assign_buffer(void* buffer)
    {
        pointers.reserve(sizes.size());

        uint8_t* curr_ptr = (uint8_t*)buffer;
        uint8_t* nested_ptr = curr_ptr;
        for(int i = 0; i < sizes.size(); i++)
        {
            pointers.push_back(curr_ptr);
            curr_ptr += sizes[i];
            if(i < num_excl)
                nested_ptr = curr_ptr;
        }

        for(int i = 0; i < nested.size(); i++)
            nested[i]->assign_buffer(nested_ptr);
    }

public:
    /* Constructor */
    rocsolver_workspace_helper()
        : num_excl(0)
        , size_excl(0)
        , size_shared(0)
        , pointers(0)
        , sizes(0)
        , nested(0)
    {
    }

    /* Disallow copying. */
    rocsolver_workspace_helper(const rocsolver_workspace_helper&) = delete;

    /* Disallow assigning. */
    rocsolver_workspace_helper& operator=(const rocsolver_workspace_helper&) = delete;

    /* Destructor */
    ~rocsolver_workspace_helper()
    {
        for(int i = 0; i < nested.size(); i++)
            delete nested[i];
    }

    /* Assigns the given workspace sizes to the workspace helper. Workspace sizes are passed in two separate lists:
       sizes_excl for workspaces that must not be overwritten by nested functions, and sizes_shared for workspaces that
       may be overwritten by nested functions. Also rounds the sizes up to the nearest MIN_CHUNK_SIZE for better
       alignment. */
    void assign_sizes(std::initializer_list<size_t> sizes_excl,
                      std::initializer_list<size_t> sizes_shared = {})
    {
        this->num_excl = sizes_excl.size();
        this->size_excl = 0;
        this->size_shared = 0;

        this->sizes.clear();
        this->sizes.reserve(sizes_excl.size() + sizes_shared.size());

        this->sizes.insert(this->sizes.end(), sizes_excl);
        this->sizes.insert(this->sizes.end(), sizes_shared);

        // round up sizes to nearest MIN_CHUNK_SIZE
        for(int i = 0; i < sizes.size(); i++)
        {
            this->sizes[i]
                = ((this->sizes[i] + MIN_CHUNK_SIZE - 1) / MIN_CHUNK_SIZE) * MIN_CHUNK_SIZE;

            if(i < this->num_excl)
                this->size_excl += this->sizes[i];
            else
                this->size_shared += this->sizes[i];
        }
    }
    /* Gets the size of the workspace array at position i, rounded up to a multiple of MIN_CHUNK_SIZE. */
    size_t get_size(int i)
    {
        return sizes[i];
    }
    /* Gets a pointer to the workspace array at position i, whose size is >= the assigned size at position i.
       Only called after assign_buffer. */
    void* operator[](int i)
    {
        return pointers[i];
    }

    /* Sets the capacity of the internal vector that holds workspace helpers for nested functions. Optional,
       but should be called before add_nested. */
    void set_nested_capacity(int capacity)
    {
        nested.reserve(capacity);
    }
    /* Adds a nested workspace helper to manage workspaces for a nested function. May be called before or
       after assign_sizes. */
    rocsolver_workspace_helper* add_nested()
    {
        rocsolver_workspace_helper* result = new rocsolver_workspace_helper();
        nested.push_back(result);
        return result;
    }
    /* Gets a pointer to the nested workspace helper at position i, which was created by the ith call to
       add_nested. */
    rocsolver_workspace_helper* get_nested(int i)
    {
        return nested[i];
    }

    /* Assigns device memory to the workspace helper, to be partitioned into individual workspace arrays
       based on the assigned sizes. Only called after assign_sizes (and, if applicable, add_nested). */
    rocblas_status assign_buffer(rocblas_handle handle, void* buffer)
    {
        // zero the workspace
        hipStream_t stream;
        rocblas_get_stream(handle, &stream);

        HIP_CHECK(hipMemsetAsync(buffer, 0, get_total_size(), stream));

        // assign pointers
        assign_buffer(buffer);

        return rocblas_status_success;
    }

    /* Returns the total amount of device memory required by the workspaces assigned to this helper and its
       nested helpers. Only called after assign_sizes (and, if applicable, add_nested). */
    size_t get_total_size()
    {
        size_t shared = this->size_shared;

        for(int i = 0; i < nested.size(); i++)
            shared = std::max(shared, nested[i]->get_total_size());

        return this->size_excl + shared;
    }

    void print_debug()
    {
        std::cerr << "Num Sizes: " << sizes.size() << std::endl;
        for(int i = 0; i < sizes.size(); i++)
        {
            std::cerr << sizes[i] << " bytes";
            if(i < pointers.size())
                std::cerr << " (" << (void*)pointers[i] << ")";
            if(i >= num_excl)
                std::cerr << " shared";
            std::cerr << std::endl;
        }
        std::cerr << "Total Size (Self): " << size_excl + size_shared << std::endl;
        if(nested.size() > 0)
            std::cerr << "Total Size (Self+Nested): " << get_total_size() << std::endl;
    }
};

#undef MIN_CHUNK_SIZE

ROCSOLVER_END_NAMESPACE
