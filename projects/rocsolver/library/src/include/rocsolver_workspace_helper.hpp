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
    size_t reserved;
    std::vector<uint8_t*> pointers;
    std::vector<size_t> sizes;
    std::vector<rocsolver_workspace_helper*> nested;

public:
    /* Constructor */
    rocsolver_workspace_helper()
        : reserved(0)
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

    /* Assigns the given workspace sizes to the workspace helper. Also rounds the sizes up to the nearest
       MIN_CHUNK_SIZE for better alignment. */
    void assign_sizes(std::initializer_list<size_t> sizes)
    {
        this->sizes.assign(sizes);

        // round up sizes to nearest MIN_CHUNK_SIZE
        for(int i = 0; i < sizes.size(); i++)
            this->sizes[i]
                = ((this->sizes[i] + MIN_CHUNK_SIZE - 1) / MIN_CHUNK_SIZE) * MIN_CHUNK_SIZE;
    }
    /* Assigns device memory to the workspace helper, to be partitioned into individual workspace arrays
       based on the assigned sizes. Only called after assign_sizes (and, if applicable, add_nested). */
    void assign_buffer(uint8_t* buffer)
    {
        pointers.reserve(sizes.size());

        uint8_t* curr_ptr = buffer + reserved;
        for(int i = 0; i < sizes.size(); i++)
        {
            pointers.push_back(curr_ptr);
            curr_ptr += sizes[i];
        }

        for(int i = 0; i < nested.size(); i++)
            nested[i]->assign_buffer(buffer + reserved);
    }
    /* Gets a pointer to the workspace array at position i, whose size is >= the assigned size at position i.
       Only called after assign_buffer. */
    uint8_t* operator[](int i)
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
       after assign_sizes. add_nested accepts a (possibly empty) list of sizes representing the amount of
       reserved memory that the nested function should not overwrite. In the standard use case, for n reserved
       sizes, this list should contain the first n sizes passed to assign_sizes.

       For example, suppose a function requires arrays foo, bar, and qux, where foo and qux must not be
       overwritten by a nested function. Then, we pass {size_foo, size_qux, size_bar} to assign_sizes, and
       pass {size_foo, size_qux} to add_nested. */
    rocsolver_workspace_helper* add_nested(std::initializer_list<size_t> reserved_sizes)
    {
        // round up sizes to nearest MIN_CHUNK_SIZE
        size_t bytes_reserved = 0;
        for(size_t const& size : reserved_sizes)
            bytes_reserved += ((size + MIN_CHUNK_SIZE - 1) / MIN_CHUNK_SIZE) * MIN_CHUNK_SIZE;

        rocsolver_workspace_helper* result = new rocsolver_workspace_helper();
        nested.push_back(result);
        result->reserved = bytes_reserved;
        return result;
    }
    /* Gets a pointer to the nested workspace helper at position i, which was created by the ith call to
       add_nested. */
    rocsolver_workspace_helper* get_nested(int i)
    {
        return nested[i];
    }

    /* Returns the total amount of device memory required by the workspaces assigned to this helper and its
       nested helpers. Only called after assign_sizes (and, if applicable, add_nested). */
    size_t get_total_size()
    {
        size_t result = 0;
        for(int i = 0; i < sizes.size(); i++)
            result += sizes[i];

        for(int i = 0; i < nested.size(); i++)
            result = std::max(result, nested[i]->get_total_size());

        return reserved + result;
    }

    void print_debug()
    {
        std::cout << "Reserved Bytes: " << reserved << std::endl;

        std::cout << "Num Sizes: " << sizes.size() << std::endl;
        for(int i = 0; i < sizes.size(); i++)
        {
            std::cout << sizes[i];
            if(i < sizes.size() - 1)
                std::cout << ", ";
        }
        std::cout << std::endl;

        std::cout << "Num Pointers: " << pointers.size() << std::endl;
        for(int i = 0; i < pointers.size(); i++)
        {
            std::cout << pointers[i];
            if(i < pointers.size() - 1)
                std::cout << ", ";
        }
        std::cout << std::endl;
    }
};

#undef MIN_CHUNK_SIZE

ROCSOLVER_END_NAMESPACE
