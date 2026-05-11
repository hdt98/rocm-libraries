// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "unit_conv_solver_group_xdlops.hpp"

namespace miopen {
namespace unit_tests {

// Shapes derived from the PyTorch reproducer in rocm-23997:
//   torch.nn.Conv3d(96, 32, kernel_size=3, padding=1)
//   x = torch.empty((1, 96, Nx, Ny, Z))
//   for (Nx,Ny) in {64,128,256,512,1024}^2 (Nx==Ny) and Z in {16,32,64,84,86,88,128,256,512,1024}.
// 17 of the 50 configurations have an input element-stride 96*Nx*Ny*Z that exceeds INT_MAX,
// which currently causes PyTorch to skip the MIOpen dispatch. Including them here gives the
// underlying CK 3D grouped xdlops solver coverage for when the int32 limit is lifted upstream.
inline std::vector<GroupXdlopsNumericData> GetLargeStride3DFullShapes(bool tf32_compute)
{
    constexpr std::array<size_t, 5> spatial_xy = {64, 128, 256, 512, 1024};
    constexpr std::array<size_t, 10> z_values  = {16, 32, 64, 84, 86, 88, 128, 256, 512, 1024};

    std::vector<GroupXdlopsNumericData> cases;
    cases.reserve(spatial_xy.size() * z_values.size());
    for(size_t nxy : spatial_xy)
    {
        for(size_t z : z_values)
        {
            cases.push_back(GroupXdlopsNumericData{
                {1, 96, nxy, nxy, z},  // x: {N, C, D, H, W}
                {32, 96, 3, 3, 3},     // w: {K, C/group, Z, Y, X}
                {1, 1, 1},             // pad
                {1, 1, 1},             // stride
                {1, 1, 1},             // dilation
                1,                     // group_count
                false,                 // deterministic
                tf32_compute,
            });
        }
    }
    return cases;
}

// Canary shape promoted into Standard (PR validation) coverage. Chosen as the largest
// shape PyTorch currently dispatches: 96*512*512*84 = 2,113,929,216 — just under INT_MAX.
inline std::vector<GroupXdlopsNumericData> GetLargeStride3DStandardShapes(bool tf32_compute)
{
    return {
        GroupXdlopsNumericData{
            {1, 96, 512, 512, 84},
            {32, 96, 3, 3, 3},
            {1, 1, 1},
            {1, 1, 1},
            {1, 1, 1},
            1,
            false,
            tf32_compute,
        },
    };
}

} // namespace unit_tests
} // namespace miopen
