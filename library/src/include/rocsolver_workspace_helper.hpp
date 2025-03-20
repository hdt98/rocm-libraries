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

    void assign_sizes(std::initializer_list<size_t> sizes)
    {
        this->sizes.assign(sizes);
    }
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
    uint8_t* operator[](int i)
    {
        return pointers[i];
    }

    void set_nested_capacity(int capacity)
    {
        nested.reserve(capacity);
    }
    rocsolver_workspace_helper* add_nested(size_t reserved)
    {
        rocsolver_workspace_helper* result = new rocsolver_workspace_helper();
        nested.push_back(result);
        result->reserved = reserved;
        return result;
    }
    rocsolver_workspace_helper* get_nested(int i)
    {
        return nested[i];
    }

    size_t get_total_size()
    {
        size_t result = 0;
        for(int i = 0; i < sizes.size(); i++)
            result += sizes[i];

        for(int i = 0; i < nested.size(); i++)
            result = std::max(result, nested[i]->get_total_size());

        return reserved + result;
    }
};

ROCSOLVER_END_NAMESPACE
