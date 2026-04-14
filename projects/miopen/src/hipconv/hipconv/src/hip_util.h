#pragma once

#include <hip/hip_runtime.h>
#include <sstream>
#include <stdexcept>

inline void hip_check(hipError_t err, const char* file, int line)
{
    if(err != hipSuccess)
    {
        std::ostringstream s;
        s << "HIP error at " << file << ":" << line << ": " << hipGetErrorString(err);
        throw std::runtime_error(s.str());
    }
}

#define HIP_CHECK(call) hip_check(call, __FILE__, __LINE__)
