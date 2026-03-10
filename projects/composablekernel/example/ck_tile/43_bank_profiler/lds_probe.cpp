// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include <hip/hip_runtime.h>

#include <cstdint>
#include <iostream>
#include <vector>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "lds_probe_kernel.hpp"

int main(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("thread_a", "0", "First active thread index (0-63)")
        .insert("thread_b", "1", "Second active thread index (0-63)")
        .insert("offset_a", "0", "LDS byte offset for thread A")
        .insert("offset_b", "0", "LDS byte offset for thread B")
        .insert("mode", "2", "Instruction: 0=ds_read_b64, 1=ds_read_b96, 2=ds_read_b128, 3=ds_write_b64")
        .insert("repeat", "1", "Number of kernel launches");

    if(!arg_parser.parse(argc, argv))
        return 1;

    int thread_a      = arg_parser.get_int("thread_a");
    int thread_b      = arg_parser.get_int("thread_b");
    uint32_t offset_a = arg_parser.get_uint32("offset_a");
    uint32_t offset_b = arg_parser.get_uint32("offset_b");
    int mode          = arg_parser.get_int("mode");
    int repeat        = arg_parser.get_int("repeat");

    // Host arrays
    std::vector<uint32_t> h_should_read(64, 0);
    std::vector<uint32_t> h_offset(64, 0);

    h_should_read[thread_a] = 1;
    if(thread_a != thread_b)
    {
        h_should_read[thread_b] = 1;
    }
    h_offset[thread_a] = offset_a;
    h_offset[thread_b] = offset_b;

    // Device memory
    ck_tile::DeviceMem d_should_read(64 * sizeof(uint32_t));
    ck_tile::DeviceMem d_offset(64 * sizeof(uint32_t));
    d_should_read.ToDevice(h_should_read.data());
    d_offset.ToDevice(h_offset.data());

    auto kargs = ck_tile::LdsProbeKernel::MakeKargs(
        d_should_read.GetDeviceBuffer(), d_offset.GetDeviceBuffer(), mode);

    constexpr dim3 grids  = ck_tile::LdsProbeKernel::GridSize();
    constexpr dim3 blocks = ck_tile::LdsProbeKernel::BlockSize();

    // Launch without timing (profiling is done externally via rocprofv3)
    // Kernel uses static __shared__ memory (32KB), no hipFuncSetAttribute needed
    ck_tile::stream_config s{nullptr, false, 0, 0, repeat};

    for(int i = 0; i < repeat; ++i)
    {
        ck_tile::launch_kernel(
            s,
            ck_tile::make_kernel<1>(
                ck_tile::LdsProbeKernel{}, grids, blocks, 0, kargs));
    }

    HIP_CHECK_ERROR(hipDeviceSynchronize());

    return 0;
}
