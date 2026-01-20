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

/**
 * @brief Macros for marking solver inapplicability with logging.
 *
 * These macros are used in solver classes to check conditions that make the solver inapplicable.
 * When a condition is true, they log an inapplicable message and return false.
 *
 * MIOPEN_SOLVER_INAPPLICABLE_IF is for use within solver member functions, automatically using
 * SolverDbId(). MIOPEN_SOLVER_INAPPLICABLE_IF_CONTEXT is for use outside solver member functions,
 * requiring a context string.
 *
 * @section Usage Guidelines
 * - **Purpose**: Use these macros to validate solver applicability by checking for conditions
 *   (e.g., unsupported data types, hardware limitations, or configuration mismatches) that would
 *   prevent the solver from running. Each check logs a descriptive message and returns `false` if
 *   the condition fails.
 * - **Best Practices**:
 *   - Before adding a new message, search for similar existing checks to ensure
 *     consistency. If a matching predefined message exists in `inapplicable_msg`, use it;
 *     otherwise, add a new one or use a custom string.
 *   - Prefer predefined messages from `inapplicable_msg` over custom strings to maintain uniformity
 *     and ease of maintenance.
 *   - For the final check in a solver's `IsApplicable` method, if it is a complex check use
 *     `inapplicable_msg::NoKernelForConfig` as the message. This indicates that no suitable kernel
 *     exists for the given configuration, providing a clear fallback reason.
 * - **Examples**:
 *   - Basic check with predefined message:
 *     @code
 *     MIOPEN_SOLVER_INAPPLICABLE_IF(!problem.Is2d(), inapplicable_msg::Is2d);
 *     @endcode
 *   - Check with custom message:
 *     @code
 *     MIOPEN_SOLVER_INAPPLICABLE_IF(<some_condition>, "Custom reason for inapplicability");
 *     @endcode
 *   - Context-specific check (outside solver member functions):
 *     @code
 *     MIOPEN_SOLVER_INAPPLICABLE_IF_CONTEXT(<some_condition>, "CustomContext",
 * inapplicable_msg::UnsupportedDevice);
 *     @endcode
 *   - Final complex check:
 *     @code
 *     MIOPEN_SOLVER_INAPPLICABLE_IF(<complex_validation_logic>,
 * inapplicable_msg::NoKernelForConfig);
 *     @endcode
 */
// Wrapper usable in any solver member function:
#define MIOPEN_SOLVER_INAPPLICABLE_IF(cond, ...)                                     \
    do                                                                               \
    {                                                                                \
        if((cond))                                                                   \
        {                                                                            \
            MIOPEN_LOG_I2("[" << SolverDbId() << "] Inapplicable: " << __VA_ARGS__); \
            return false;                                                            \
        }                                                                            \
    } while(false)

// Wrapper for when not in a solver member function
#define MIOPEN_SOLVER_INAPPLICABLE_IF_CONTEXT(cond, context, ...)                 \
    do                                                                            \
    {                                                                             \
        if((cond))                                                                \
        {                                                                         \
            MIOPEN_LOG_I2("[" << (context) << "] Inapplicable: " << __VA_ARGS__); \
            return false;                                                         \
        }                                                                         \
    } while(false)

namespace inapplicable_msg {

inline constexpr const char* HasNonPackedTensors = "Non-packed tensors are not supported.";
inline constexpr const char* IsAsymmetricPad     = "Asymmetric padding is not supported.";
inline constexpr const char* AllTensorsDimsFitIntoInt =
    "Tensor dimensions/lengths do not fit into int.";
inline constexpr const char* NegPad            = "Negative padding is not supported.";
inline constexpr const char* EnvDisabled       = "Disabled by environment variable.";
inline constexpr const char* DataType          = "Unsupported data type.";
inline constexpr const char* Layout            = "Unsupported tensor layout.";
inline constexpr const char* UnsupportedDevice = "Unsupported GPU.";
inline constexpr const char* Direction         = "Unsupported direction.";
inline constexpr const char* UseAsmKernels     = "Assembly kernels are disabled.";
inline constexpr const char* HIPDisabled       = "HIP kernels are disabled.";
inline constexpr const char* Is2d              = "Only 2D operations are supported.";
inline constexpr const char* Is3d              = "Only 3D operations are supported.";
inline constexpr const char* Not2Dor3D         = "Only 2D or 3D operations are supported.";
inline constexpr const char* MetaData          = "Unsupported ROCm metadata version.";
inline constexpr const char* IsTensorsCasted   = "Casted tensors are not supported.";
inline constexpr const char* isXnackEnabled    = "Xnack requirement is not met.";
inline constexpr const char* NoOCLConv         = "OpenCL convolutions are disabled.";
inline constexpr const char* Workaround        = "Disabled as a workaround for known issues.";
inline constexpr const char* NoKernelForConfig =
    "No kernel is available for the given configuration.";
inline constexpr const char* IsGfx90aFp16altRequired =
    "Solver does not implement ALT FP16 kernels on GFX90a.";
inline constexpr const char* Generic       = "Shape or convolution parameters are not supported.";
inline constexpr const char* GetGroupCount = "Unsupported group count.";
inline constexpr const char* Deterministic = "Solver does not support the deterministic attribute.";
inline constexpr const char* ZeroSplitBatchSize = "IGemm split batch size is zero.";
inline constexpr const char* MixedDatatype      = "Mixed data type is not supported.";
inline constexpr const char* TransposeNonPacked =
    "Transpose kernel does not support non-packed tensors.";
inline constexpr const char* CKWhitelist      = "GPU is not in CK whitelist.";
inline constexpr const char* NoCKSupport      = "Not on CK's supported hardware list.";
inline constexpr const char* CKNotApplicable  = "CK solver is not applicable for the given shape.";
inline constexpr const char* Xdlops           = "No XDLops support.";
inline constexpr const char* IndexRange       = "Index range is not large enough.";
inline constexpr const char* MissingIntrinsic = "Hardware is missing intrinsic.";
inline constexpr const char* InvalidGridGemm  = "Invalid grid GEMM for GEMM XDLops.";
inline constexpr const char* NoMLIRSupport    = "MLIR solvers are not supported on this hardware.";
inline constexpr const char* CastOnly8FNUZ =
    "Casting is only supported for the miopenFloat8_fnuz and miopenBFloat8_fnuz data types.";
inline constexpr const char* NotContiguous = "Requires all tensors to be contiguous.";
inline constexpr const char* IsAllPacked   = "Requires all tensors to be packed.";
inline constexpr const char* NotEnoughLDS =
    "Device does not have enough local memory for this solver.";
inline constexpr const char* DataTypeMismatch = "Data types do not match.";
inline constexpr const char* LengthMismatch   = "Tensor lengths do not match.";
inline constexpr const char* RightNormDim =
    "Normalized dim is greater than 0 and less than or equal to Tensor dimension "
    "length.";
inline constexpr const char* IsImprovementOverROCm =
    "No performance improvement over ROCm implementation.";
inline constexpr const char* InvalidDim       = "Invalid dimension.";
inline constexpr const char* IsValidLength    = "Invalid length.";
inline constexpr const char* ReduceLenInvalid = "Reduction length is not valid.";
inline constexpr const char* NotLastDim = "Reduction dimension must not be the last dimension.";
inline constexpr const char* GridTooLarge =
    "Grid size required by the problem exceeds the maximum grid size supported by the device.";
inline constexpr const char* NotFWDFusion  = "First operation is not miopenFusionOpConvForward.";
inline constexpr const char* FusionOpCount = "Not the expected number of fusion operations.";
inline constexpr const char* ElementSizeMismatch = "Element sizes do not match.";
inline constexpr const char* FirstFusionOp =
    "First fusion operation is not what the solver expects.";
inline constexpr const char* SecondFusionOp =
    "Second fusion operation is not what the solver expects.";
inline constexpr const char* ThirdFusionOp =
    "Third fusion operation is not what the solver expects.";
inline constexpr const char* InvalidActivation = "Activation mode is not supported.";

} // namespace inapplicable_msg
} // namespace solver
} // namespace miopen
