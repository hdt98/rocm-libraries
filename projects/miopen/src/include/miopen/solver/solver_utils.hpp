/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#pragma once

namespace miopen {
namespace solver {

// Wrapper usable in any solver member function:
#define MIOPEN_SOLVER_INAPPLICABLE_IF(cond, ...)                                 \
    if(cond)                                                                     \
    {                                                                            \
        MIOPEN_LOG_I2("[" << SolverDbId() << "] Inapplicable: " << __VA_ARGS__); \
        return false;                                                            \
    }

namespace a_msg {

inline constexpr const char* NonPacked         = "Non-packed tensors are not supported";
inline constexpr const char* AsymmPad          = "Asymmetric padding is not supported";
inline constexpr const char* DimsLargerThanInt = "All tensor dimensions do not fit into int";
inline constexpr const char* NegPad            = "Negative padding not supported";
inline constexpr const char* EnvDisable        = "Disabled by env var";
inline constexpr const char* DType             = "Unsupported data type";
inline constexpr const char* Layout            = "Unsupported tensor layout";
inline constexpr const char* UnsupportedGPU    = "Unsupported GPU";
inline constexpr const char* ConvDir           = "Convolution direction is not supported";
inline constexpr const char* ASMDisabled       = "Assembly kernels are disabled";
inline constexpr const char* Not2DConv         = "Only 2D convolution is supported";
inline constexpr const char* MetaData          = "Unsupported ROCM metadata version";
inline constexpr const char* TensorCast        = "Casted tensors are not supported";
inline constexpr const char* Xnack             = "Xnack requirement is not met";
inline constexpr const char* NoOCLConv         = "OpenCL convolutions are disabled";
inline constexpr const char* Workaround        = "Disabled as a workaround for known issues";
inline constexpr const char* NoKernel          = "No kernel available for the given configuration";

} // namespace a_msg
} // namespace solver
} // namespace miopen
