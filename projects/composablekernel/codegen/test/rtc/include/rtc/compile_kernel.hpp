// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef GUARD_HOST_TEST_RTC_INCLUDE_RTC_COMPILE_KERNEL
#define GUARD_HOST_TEST_RTC_INCLUDE_RTC_COMPILE_KERNEL

#include <rtc/kernel.hpp>
#include <rtc/filesystem.hpp>
#include <string>

namespace rtc {

struct src_file
{
    src_file(std::filesystem::path p, std::string c) : path{std::move(p)}, content{std::move(c)} {}
    fs::path path;
    std::string content;
};

struct compile_options
{
    std::string flags       = "";
    std::string kernel_name = "main";
};

kernel compile_kernel(const std::vector<src_file>& srcs,
                      compile_options options = compile_options{});

/// Like `compile_kernel`, but also returns the raw HIP code-object byte
/// blob the hiprtc driver produced. Callers persist this blob to disk to
/// build a cross-process compile cache: the next invocation reads the
/// blob back and uses `kernel_from_code_object` (below) to reconstitute
/// the `kernel` without re-invoking the compiler. The second returned
/// vector is empty when the clang-based backend is in use (the
/// codegen test suite exercises hiprtc; production uses hiprtc).
struct compiled_artifact
{
    kernel            k;
    std::vector<char> code_object;
};

compiled_artifact compile_kernel_with_code_object(const std::vector<src_file>& srcs,
                                                  compile_options options = compile_options{});

/// Reconstruct a `kernel` from a previously-persisted HIP code-object
/// blob. Equivalent to `rtc::kernel(code_object_bytes, kernel_name)`
/// but named explicitly so the cache-hit call site reads clearly.
kernel kernel_from_code_object(const std::vector<char>& code_object,
                               const std::string&       kernel_name);

} // namespace rtc

#endif
