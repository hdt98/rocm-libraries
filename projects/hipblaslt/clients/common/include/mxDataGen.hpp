/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
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

// All content below is gated so the header is harmless to include when the
// feature is disabled (the entire translation unit collapses to nothing).
// Callers that actually invoke `generateMXInput` must guard their use sites
// on the same macro.
#if HIPBLASLT_ENABLE_MXDATAGENERATOR

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-export.h>
#include <hipblaslt/hipblaslt-types.h>
#include <stdint.h>

#include <string_view>
#include <vector>

// Selects whether `generateMXInput` populates the packed `data`/`scale`
// buffers via the host (CPU PRNG + std::memcpy) or writes them straight
// into device memory via `DGen::DataGeneratorGPU<DT>`.
//
// MXInitDevice::Cpu  -- host PRNG path. `data` and `scale` are interpreted
//                       as host pointers. Deterministic and byte-stable
//                       across hardware; useful as a regression baseline.
//
// MXInitDevice::Gpu  -- on-device PRNG path. `data` and `scale` MUST be
//                       device pointers (typically from `hipMalloc`). The
//                       on-device generator writes the packed bytes
//                       directly; the returned reference float vector is
//                       materialised on the host by reading those bytes
//                       back and dequantising via `toFloatPacked<DT>`, so
//                       the float reference is consistent with whatever
//                       ended up in device memory regardless of any PRNG
//                       differences between CPU and GPU implementations.
//                       For the small set of matrix layouts that need the
//                       PRNG output rearranged (anything that goes through
//                       `getAlignedFloat`), the GPU overload silently falls
//                       back to the CPU path internally. Only the easy
//                       layout (`isMatrixA && isTranspose` or
//                       `!isMatrixA && !isTranspose`) actually exercises
//                       the GPU PRNG.
enum class MXInitDevice
{
    Cpu = 0,
    Gpu = 1,
};

// Architecture-flavoured scale tensor memory layout that `generateMXInput`
// will leave behind in the `scale` buffer. mxDataGenerator natively writes
// the natural (non-swizzled) layout; specific architectures' kernels expect
// a permuted view of that buffer instead. Add new entries when more layouts
// land -- the per-arch swizzle is otherwise self-contained inside
// `generateMXInput`, so call sites only need to pick the enum value.
//
//   kNone    -- natural mxDataGenerator scale layout, no swizzle.
//   kGFX950  -- AITER (preSwizzleScalesGFX950) layout used by gfx950
//               subtile MX kernels.
//   kGFX1250 -- dimk (preSwizzleScalesGFX1250) layout used by gfx1250 MX
//               kernels.
enum class MXScaleLayout
{
    kNone    = 0,
    kGFX950  = 1,
    kGFX1250 = 2,
};

// CPU-only overload (kept for callers that don't want to opt into the
// GPU backend). Equivalent to passing `MXInitDevice::Cpu` to the overload
// below. `scaleLayout` selects the post-generation scale memory layout
// (see `MXScaleLayout`); the default leaves the natural layout in place.
std::vector<float> generateMXInput(hipDataType            dataType,
                                   hipDataType            scaleType,
                                   void*                  data,
                                   void*                  scale,
                                   uint64_t               row,
                                   uint64_t               col,
                                   uint64_t               stride,
                                   bool                   isTranspose,
                                   int const              scaleBlockRowSize,
                                   int const              scaleBlockColSize,
                                   bool                   isMatrixA,
                                   MXScaleLayout          scaleLayout = MXScaleLayout::kNone,
                                   std::string_view const initMethod  = "Bounded",
                                   float                  min_val     = -1.0f,
                                   float                  max_val     = 1.0f);

// Backend-selectable overload. `initDevice == MXInitDevice::Cpu` delegates
// to the host PRNG path above. `initDevice == MXInitDevice::Gpu` runs the
// kernel directly into the device buffers (data/scale must be device
// pointers in that case); see `MXInitDevice` for the full semantics.
// `scaleLayout` selects the post-generation scale memory layout (see
// `MXScaleLayout`).
std::vector<float> generateMXInput(hipDataType            dataType,
                                   hipDataType            scaleType,
                                   void*                  data,
                                   void*                  scale,
                                   uint64_t               row,
                                   uint64_t               col,
                                   uint64_t               stride,
                                   bool                   isTranspose,
                                   int const              scaleBlockRowSize,
                                   int const              scaleBlockColSize,
                                   bool                   isMatrixA,
                                   MXInitDevice           initDevice,
                                   MXScaleLayout          scaleLayout = MXScaleLayout::kNone,
                                   std::string_view const initMethod  = "Bounded",
                                   float                  min_val     = -1.0f,
                                   float                  max_val     = 1.0f);

#endif
