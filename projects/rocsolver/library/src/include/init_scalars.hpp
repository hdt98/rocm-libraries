/* ************************************************************************
 * Copyright (C) 2020-2024 Advanced Micro Devices, Inc.
 * ************************************************************************/

#pragma once

#include <cassert>

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include "lib_macros.hpp"
#include "rocsolver_logger.hpp"

ROCSOLVER_BEGIN_NAMESPACE

#define IOTA_MAX_THDS 32

// Fills the given range with sequentially increasing values.
// The name and interface is based on std::iota

// Device overload — launched as a kernel; fills [first, first+count) on the GPU.
template <typename T>
ROCSOLVER_KERNEL void __launch_bounds__(IOTA_MAX_THDS) iota_n(T* first, uint32_t count, T value)
{
    const auto idx = hipThreadIdx_x;
    if(idx < count)
    {
        first[idx] = T(idx) + value;
    }
}

// Host overload — fills a caller-supplied stack/host array with the same
// arithmetic as the device kernel, without any GPU interaction.
template <typename T>
void iota_n(T* first, uint32_t count, T value, /*host_tag*/ std::nullptr_t)
{
    for(uint32_t idx = 0; idx < count; ++idx)
        first[idx] = T(idx) + value;
}

// Initializes scalars on the device.
template <typename T>
void init_scalars(rocblas_handle handle, T* scalars)
{
    assert(scalars != nullptr);

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);
    ROCSOLVER_LAUNCH_KERNEL(iota_n<T>, dim3(1), dim3(IOTA_MAX_THDS), 0, stream, scalars, 3, -1);
}

ROCSOLVER_END_NAMESPACE
