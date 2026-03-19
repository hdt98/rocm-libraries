#pragma once

/// This file implements some utilities for memory operations.
///
/// Implements functions for constructing buffer resource descriptors with bounds checking
/// and for waiting for memory operations to complete.

#include <hip/hip_runtime.h>

// Wait for all but the last "count" loads.
//
// One BUFFER_LOADS_*_LDS instruction counts as vmcnt of 1, regardless
// of how many memory transcations it generates.
template <int Count>
__device__ inline void wait_vmcnt()
{
    static_assert(Count >= 0 && Count <= 15,
                  "vmcnt must be in range [0, 15] (4-bit field)");

    // s_waitcnt encoding: bits [3:0] = vmcnt, bits [9:4] = expcnt, bits [15:10] = lgkmcnt
    // 0x0F70 = lgkmcnt(0), expcnt(7), vmcnt(0)
    // We want to modify only vmcnt field
    constexpr unsigned int waitcnt_value = 0x0F70 | (Count & 0xF);
    __builtin_amdgcn_s_waitcnt(waitcnt_value);
}

// Convenience alias for common case
__device__ __forceinline__ void wait_vmcnt_all()
{
    wait_vmcnt<0>();
}
