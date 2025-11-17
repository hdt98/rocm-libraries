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

#include <hip/hip_runtime.h>

// Identity load callback - simply returns the input value unchanged.
//
// This function signature is for single-precision complex-to-complex transforms.
// The function name "load_callback" must match the symbol name passed to
// rocfft_plan_description_set_load_callback().
//
// Parameters:
//   buffer         - Pointer to the input buffer
//   offset         - Element offset to read from
//   callback_data  - User-provided callback data (can be nullptr)
//   shared_memory  - Shared memory pointer (currently unused, must be nullptr)
//
// Returns:
//   The complex value at buffer[offset]

extern "C" __device__ float2 load_callback(float2* buffer,
                                           size_t  offset,
                                           void*   callback_data,
                                           void*   shared_memory)
{
    // simple identity to demonstrate the callback mechanism without modifying data
    return buffer[offset];
}