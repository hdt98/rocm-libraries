// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <sstream>
#include <hip/hip_runtime.h>

namespace ck {

inline void hip_check_error(hipError_t x,
                            const char* file = __builtin_FILE(),
                            int line = __builtin_LINE(),
                            const char* func = __builtin_FUNCTION())
{
    if(x != hipSuccess)
    {
        std::ostringstream ss;
        ss << "HIP runtime error: " << hipGetErrorString(x) << ". " << file << ": "
           << line << " in function: " << func;
        throw std::runtime_error(ss.str());
    }
}

#define HIP_CHECK_ERROR(retval_or_funcall)                                         \
    do                                                                             \
    {                                                                              \
        hipError_t _tmpVal = retval_or_funcall;                                    \
        if(_tmpVal != hipSuccess)                                                  \
        {                                                                          \
            std::ostringstream ostr;                                               \
            ostr << "HIP Function Failed (" << __FILE__ << "," << __LINE__ << ") " \
                 << hipGetErrorString(_tmpVal);                                    \
            throw std::runtime_error(ostr.str());                                  \
        }                                                                          \
    } while(0)

} // namespace ck
