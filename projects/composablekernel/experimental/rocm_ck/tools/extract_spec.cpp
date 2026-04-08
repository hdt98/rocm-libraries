// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Generic spec extractor — compiled once per kernel variant via
// -DVARIANT_HEADER="path/to/variant.hip". Outputs the variant's spec
// as JSON to stdout.
//
// Build: cmake compiles this as a host C++20 program per variant.
// Run:   at build time, output captured to ${variant}.spec.json.
//
// The included .hip file defines three static constexpr variables:
//   kName    — variant name (const char*)
//   kTargets — TargetSet
//   kSpec    — GemmSpec or ElementwiseSpec

#ifndef VARIANT_HEADER
#error "Define VARIANT_HEADER to the .hip file path (e.g., -DVARIANT_HEADER=\"gemm_fp16.hip\")"
#endif

#include VARIANT_HEADER

#include <rocm_ck/spec_json.hpp>

#include <cstdio>

int main()
{
    std::puts(rocm_ck::to_json(kName, kSpec, kTargets).c_str());
    return 0;
}
