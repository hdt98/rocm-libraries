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
#include <mxDataGenerator/DataGenerator.hpp>
#include <mxDataGenerator/PreSwizzle.hpp>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>
namespace DGen
{
    enum class DataFormat
    {
        Fp4,
        Fp6E2M3,
        Fp6E3M2,
        Fp8E4M3,
        Fp8E5M2,
    };
    namespace detail
    {
        template <typename DT>
        std::vector<uint8_t> unpackData(std::vector<uint8_t> const& packedBytes,
                                        size_t                      elementCount)
        {
            static_assert(std::is_same_v<DT, ocp_e2m1_mxfp4>
                          || std::is_same_v<DT, ocp_e2m1_mxfp4_e5m3>
                          || std::is_same_v<DT, ocp_e2m1_mxfp4_e4m3>
                          || std::is_same_v<DT, ocp_e3m2_mxfp6>
                          || std::is_same_v<DT, ocp_e2m3_mxfp6>);
            if constexpr(std::is_same_v<DT, ocp_e3m2_mxfp6>
                         || std::is_same_v<DT, ocp_e2m3_mxfp6>)
            {
                std::vector<uint8_t> unpackedDataBytes(elementCount);
                for(size_t i = 0; i < elementCount; ++i)
                {
                    size_t const bitOffset = i * 6;
                    size_t const byteIndex = bitOffset / 8;
                    size_t const bitIndex  = bitOffset % 8;
                    uint16_t word = 0;
                    if(byteIndex < packedBytes.size())
                        word |= static_cast<uint16_t>(packedBytes[byteIndex]);
                    if(byteIndex + 1 < packedBytes.size())
                        word |= static_cast<uint16_t>(packedBytes[byteIndex + 1]) << 8;
                    unpackedDataBytes[i] = static_cast<uint8_t>((word >> bitIndex) & 0x3F);
                }
                return unpackedDataBytes;
            }
            else
            {
                std::vector<uint8_t> unpackedDataBytes(elementCount);
                for(size_t i = 0; i < elementCount; ++i)
                {
                    size_t const  byteIndex = i / 2;
                    uint8_t const b
                        = (byteIndex < packedBytes.size()) ? packedBytes[byteIndex] : 0;
                    unpackedDataBytes[i]
                        = static_cast<uint8_t>((i % 2 == 0) ? (b & 0x0F) : ((b >> 4) & 0x0F));
                }
                return unpackedDataBytes;
            }
        }
        template <typename DT>
        void packData(std::vector<uint8_t> const& dataBytes, uint8_t* packedData)
        {
            static_assert(std::is_same_v<DT, ocp_e2m1_mxfp4>
                          || std::is_same_v<DT, ocp_e2m1_mxfp4_e5m3>
                          || std::is_same_v<DT, ocp_e2m1_mxfp4_e4m3>
                          || std::is_same_v<DT, ocp_e3m2_mxfp6>
                          || std::is_same_v<DT, ocp_e2m3_mxfp6>);
            if constexpr(std::is_same_v<DT, ocp_e3m2_mxfp6>
                         || std::is_same_v<DT, ocp_e2m3_mxfp6>)
            {
                size_t const elementCount = dataBytes.size();
                size_t const packedSize   = (elementCount * 6 + 7) / 8;
                std::memset(packedData, 0, packedSize);
                for(size_t i = 0; i < elementCount; ++i)
                {
                    uint16_t const v         = static_cast<uint16_t>(dataBytes[i] & 0x3F);
                    size_t const   bitOffset = i * 6;
                    size_t const   byteIndex = bitOffset / 8;
                    size_t const   bitIndex  = bitOffset % 8;
                    if(byteIndex >= packedSize)
                        break;
                    uint16_t word = static_cast<uint16_t>(packedData[byteIndex]);
                    if(byteIndex + 1 < packedSize)
                        word |= static_cast<uint16_t>(packedData[byteIndex + 1]) << 8;
                    uint16_t const mask = static_cast<uint16_t>(0x3F) << bitIndex;
                    word                = static_cast<uint16_t>((word & ~mask) | (v << bitIndex));
                    packedData[byteIndex] = static_cast<uint8_t>(word & 0xFF);
                    if(byteIndex + 1 < packedSize)
                        packedData[byteIndex + 1] = static_cast<uint8_t>((word >> 8) & 0xFF);
                }
            }
            else
            {
                size_t const elementCount = dataBytes.size();
                size_t const packedSize   = (elementCount + 1) / 2;
                std::memset(packedData, 0, packedSize);
                for(size_t i = 0; i < elementCount; ++i)
                {
                    size_t const  byteIndex = i / 2;
                    uint8_t const v         = static_cast<uint8_t>(dataBytes[i] & 0x0F);
                    if(i % 2 == 0)
                        packedData[byteIndex]
                            = static_cast<uint8_t>((packedData[byteIndex] & 0xF0) | v);
                    else
                        packedData[byteIndex]
                            = static_cast<uint8_t>((packedData[byteIndex] & 0x0F) | (v << 4));
                }
            }
        }
        template <typename DT>
        std::vector<float> getAlignedFloat(std::vector<uint8_t>&        dataBytes,
                                           std::vector<uint8_t> const&  scaleBytes,
                                           std::array<index_t, 2> const sizes,
                                           int                          elementsPerMXBlock,
                                           bool                         isMatrixA)
        {
            std::vector<float>   refFloat(sizes[0] * sizes[1], 0.0);
            std::vector<uint8_t> alignedDataBytes(dataBytes.size());
            if(isMatrixA)
            {
                int M = sizes[0];
                int K = sizes[1];
#pragma omp parallel for
                for(size_t mk = 0; mk < M * K; ++mk)
                {
                    auto m        = mk % M;
                    auto k        = mk / M;
                    auto scale_id = (k / elementsPerMXBlock) * M + m;
                    auto data_id         = scale_id * elementsPerMXBlock + k % elementsPerMXBlock;
                    alignedDataBytes[mk] = dataBytes[data_id];
                    refFloat[mk]
                        = toFloat<DT>(scaleBytes.data(), dataBytes.data(), scale_id, data_id);
                }
                std::swap(dataBytes, alignedDataBytes);
            }
            else
            {
                int N = sizes[0];
                int K = sizes[1];
#pragma omp parallel for
                for(size_t kn = 0; kn < K * N; ++kn)
                {
                    auto k        = kn / N;
                    auto n        = kn % N;
                    auto scale_id = (k / elementsPerMXBlock) * N + n;
                    auto data_id         = scale_id * elementsPerMXBlock + k % elementsPerMXBlock;
                    alignedDataBytes[kn] = dataBytes[data_id];
                    refFloat[kn]
                        = toFloat<DT>(scaleBytes.data(), dataBytes.data(), scale_id, data_id);
                }
                std::swap(dataBytes, alignedDataBytes);
            }
            return refFloat;
        }
        template <typename T, typename DT>
        std::vector<float> generateData(T                          dgen,
                                        void*                      data,
                                        void*                      scale,
                                        std::vector<index_t>       sizes,
                                        std::vector<index_t>       strides,
                                        uint32_t                   seed,
                                        DataGeneratorOptions&      opt,
                                        int                        elementsPerMXBlock,
                                        bool                       isTranspose,
                                        bool                       isMatrixA,
                                        std::vector<size_t> const& preSwizzleTile,
                                        std::vector<size_t> const& /*preTile*/,
                                        bool                       gfx1250Swizzle = false)
        {
            dgen.setSeed(seed);
            dgen.generate(sizes, strides, opt);
            std::vector<uint8_t> dataBytes = dgen.getDataBytes();
            std::memcpy(data, dataBytes.data(), dataBytes.size() * sizeof(uint8_t));
            std::vector<uint8_t> scaleBytes = dgen.getScaleBytes();
            size_t scaleRows = sizes[0] / elementsPerMXBlock;
            size_t scaleCols = sizes[1];
            if(preSwizzleTile.size() == 3)
            {
                scaleBytes = preSwizzleScalesGFX950(scaleBytes, {scaleCols, scaleRows});
            }
            else if(gfx1250Swizzle && elementsPerMXBlock > 0)
            {
                // gfx1250 / non-rocroller WMMA dimk swizzle. The natural-layout
                // scale buffer the MX generator produced has scaleRows along the
                // fast (stride-1) dim of the column-major scale tensor, so for the
                // common TN-style tests `(isMatrixA == isTranspose) == true` we get
                // `slowDim = scaleCols (M or N)` and `fastDim = scaleRows (K/MX)`.
                // The dimk axis is then split out of the K-block dim.
                scaleBytes = preSwizzleScalesGFX1250(scaleBytes,
                                                     /*slowDim=*/scaleCols,
                                                     /*fastDim=*/scaleRows,
                                                     /*mxBlock=*/static_cast<size_t>(
                                                         elementsPerMXBlock));
            }
            std::memcpy(scale, scaleBytes.data(), scaleBytes.size() * sizeof(uint8_t));
            if((isMatrixA && isTranspose) || (!isMatrixA && !isTranspose))
            {
                return dgen.getReferenceFloat();
            }
            if constexpr(std::is_same_v<DT, ocp_e5m2_mxfp8>
                         || std::is_same_v<DT, ocp_e4m3_mxfp8>)
            {
                auto ret = getAlignedFloat<DT>(
                    dataBytes, scaleBytes, {sizes[0], sizes[1]}, elementsPerMXBlock, isMatrixA);
                std::memcpy(data, dataBytes.data(), dataBytes.size() * sizeof(uint8_t));
                return ret;
            }
            else if constexpr(std::is_same_v<DT, ocp_e3m2_mxfp6>
                              || std::is_same_v<DT, ocp_e2m3_mxfp6>)
            {
                size_t const elementCount
                    = static_cast<size_t>(sizes[0]) * static_cast<size_t>(sizes[1]);
                auto unpackedDataBytes = unpackData<DT>(dataBytes, elementCount);
                auto ret               = getAlignedFloat<DT>(unpackedDataBytes,
                                                             scaleBytes,
                                                             {sizes[0], sizes[1]},
                                                             elementsPerMXBlock,
                                                             isMatrixA);
                packData<DT>(unpackedDataBytes, static_cast<uint8_t*>(data));
                return ret;
            }
            else if constexpr(std::is_same_v<DT, ocp_e2m1_mxfp4>
                              || std::is_same_v<DT, ocp_e2m1_mxfp4_e5m3>
                              || std::is_same_v<DT, ocp_e2m1_mxfp4_e4m3>)
            {
                size_t const elementCount
                    = static_cast<size_t>(sizes[0]) * static_cast<size_t>(sizes[1]);
                auto unpackedDataBytes = unpackData<DT>(dataBytes, elementCount);
                auto ret               = getAlignedFloat<DT>(unpackedDataBytes,
                                                             scaleBytes,
                                                             {sizes[0], sizes[1]},
                                                             elementsPerMXBlock,
                                                             isMatrixA);
                packData<DT>(unpackedDataBytes, static_cast<uint8_t*>(data));
                return ret;
            }
            else
            {
                throw std::runtime_error("Unsupported data types in MX data generation!");
            }
        }
    } // namespace detail
    inline std::vector<float> generateMXInput(DataFormat                 dataType,
                                              ScaleType                  scaleType,
                                              void*                      data,
                                              void*                      scale,
                                              uint64_t                   row,
                                              uint64_t                   col,
                                              uint64_t                   stride,
                                              bool                       isTranspose,
                                              const std::vector<size_t>& preSwizzleTile,
                                              const std::vector<size_t>& preTile,
                                              int const                  scaleBlockRowSize,
                                              int const                  scaleBlockColSize,
                                              bool                       isMatrixA,
                                              std::string_view const     initMethod = "Bounded",
                                              float                      min_val    = -1.0f,
                                              float                      max_val    = 1.0f,
                                              // When true, apply the gfx1250 / non-rocroller WMMA
                                              // dimk swizzle to scales (instead of, never in
                                              // addition to, the gfx950 AITER swizzle implied by a
                                              // non-empty `preSwizzleTile`). See
                                              // `preSwizzleScalesGFX1250`.
                                              bool gfx1250Swizzle = false)
    {
        DataGeneratorOptions opt;
        opt.min          = initMethod == "uniform_01" ? 0. : (initMethod == "hpl" ? -.5 : min_val);
        opt.max          = initMethod == "uniform_01" ? 1. : (initMethod == "hpl" ? .5 : max_val);
        opt.blockScaling = scaleBlockRowSize * scaleBlockColSize;
        opt.forceDenorm  = false;
        if(initMethod == "Sequential")
            opt.initMode = DataInitMode(Sequential{});
        else if(initMethod == "RowIndex")
            opt.initMode = DataInitMode(RowIndex{});
        else if(initMethod == "ColIndex")
            opt.initMode = DataInitMode(ColIndex{});
        else if(initMethod == "Checkerboard")
            opt.initMode = DataInitMode(Checkerboard{});
        else if(initMethod == "ScaledDiagonal")
            opt.initMode = DataInitMode(ScaledDiagonal{});
        else if(initMethod == "Identity")
            opt.initMode = DataInitMode(Identity{});
        else if(initMethod == "Ones")
            opt.initMode = DataInitMode(Ones{});
        else if(initMethod == "Zeros")
            opt.initMode = DataInitMode(Zeros{});
        else if(initMethod == "Bounded" || initMethod == "uniform_01")
            opt.initMode = DataInitMode(Bounded{});
        else
            opt.initMode = DataInitMode(TrigonometricFromFloat{});
        const uint32_t seed = 1713573849;
        std::vector<index_t> sizes
            = {static_cast<index_t>(row), static_cast<index_t>(col)};
        std::vector<index_t> strides;
        strides.push_back(1);
        strides.push_back(static_cast<index_t>(stride));
        auto const elementsPerMXBlock = scaleBlockRowSize * scaleBlockColSize;
        if(preSwizzleTile.size() == 3 && gfx1250Swizzle)
            throw std::runtime_error("generateMXInput: cannot combine the gfx950 AITER scale "
                                     "swizzle (non-empty preSwizzleTile) with the gfx1250 "
                                     "dimk scale swizzle. Choose at most one.");
        if(dataType == DataFormat::Fp8E5M2)
        {
            DataGenerator<ocp_e5m2_mxfp8> dgen;
            return detail::generateData<decltype(dgen), ocp_e5m2_mxfp8>(
                dgen, data, scale, sizes, strides, seed, opt, elementsPerMXBlock,
                isTranspose, isMatrixA, preSwizzleTile, preTile, gfx1250Swizzle);
        }
        else if(dataType == DataFormat::Fp8E4M3)
        {
            DataGenerator<ocp_e4m3_mxfp8> dgen;
            return detail::generateData<decltype(dgen), ocp_e4m3_mxfp8>(
                dgen, data, scale, sizes, strides, seed, opt, elementsPerMXBlock,
                isTranspose, isMatrixA, preSwizzleTile, preTile, gfx1250Swizzle);
        }
        else if(dataType == DataFormat::Fp6E2M3)
        {
            DataGenerator<ocp_e2m3_mxfp6> dgen;
            return detail::generateData<decltype(dgen), ocp_e2m3_mxfp6>(
                dgen, data, scale, sizes, strides, seed, opt, elementsPerMXBlock,
                isTranspose, isMatrixA, preSwizzleTile, preTile, gfx1250Swizzle);
        }
        else if(dataType == DataFormat::Fp6E3M2)
        {
            DataGenerator<ocp_e3m2_mxfp6> dgen;
            return detail::generateData<decltype(dgen), ocp_e3m2_mxfp6>(
                dgen, data, scale, sizes, strides, seed, opt, elementsPerMXBlock,
                isTranspose, isMatrixA, preSwizzleTile, preTile, gfx1250Swizzle);
        }
        else if(dataType == DataFormat::Fp4)
        {
            if(scaleType == ScaleType::E4M3)
            {
                DataGenerator<ocp_e2m1_mxfp4_e4m3> dgen;
                return detail::generateData<decltype(dgen), ocp_e2m1_mxfp4_e4m3>(
                    dgen, data, scale, sizes, strides, seed, opt, elementsPerMXBlock,
                    isTranspose, isMatrixA, preSwizzleTile, preTile, gfx1250Swizzle);
            }
            else if(scaleType == ScaleType::E5M3)
            {
                DataGenerator<ocp_e2m1_mxfp4_e5m3> dgen;
                return detail::generateData<decltype(dgen), ocp_e2m1_mxfp4_e5m3>(
                    dgen, data, scale, sizes, strides, seed, opt, elementsPerMXBlock,
                    isTranspose, isMatrixA, preSwizzleTile, preTile, gfx1250Swizzle);
            }
            else
            {
                DataGenerator<ocp_e2m1_mxfp4> dgen;
                return detail::generateData<decltype(dgen), ocp_e2m1_mxfp4>(
                    dgen, data, scale, sizes, strides, seed, opt, elementsPerMXBlock,
                    isTranspose, isMatrixA, preSwizzleTile, preTile, gfx1250Swizzle);
            }
        }
        throw std::runtime_error("Unsupported data types in MX data generation!");
    }
} // namespace DGen

// =============================================================================
// Optional GPU init backend.
//
// Visible only to consumers that compile their TU through hipcc/clang in HIP
// mode (as signalled by `__HIPCC__`). A host-only TU that just wants the
// HIP-free CPU `generateMXInput` above is unaffected.
//
// Adds:
//   * `DGen::MXInitDevice` enum (Cpu/Gpu).
//   * An overload of `generateMXInput` taking `MXInitDevice initDevice`.
//
// MXInitDevice::Cpu  -- delegates to the host overload above (CPU PRNG +
//                       std::memcpy). Deterministic and byte-stable across
//                       hardware; useful as a regression baseline.
//
// MXInitDevice::Gpu  -- runs `DataGeneratorGPU<DT>` on the device to produce
//                       the packed bytes directly into the caller's device
//                       buffers (no host PRNG). The reference float vector
//                       is then materialised on the host by reading those
//                       bytes back and dequantising via `toFloatPacked<DT>`,
//                       so the returned reference is, by construction,
//                       consistent with whatever ended up in device memory
//                       (we don't require the CPU and GPU PRNGs to agree
//                       bit-for-bit). For matrix layouts that need the data
//                       re-aligned (non-TN combinations) we fall back to the
//                       CPU path -- the GPU PRNG fast path is wired only for
//                       the common transposed-A / non-transposed-B layout
//                       (which is what FP4 subtile and the bulk of MX
//                       benchmarks use).
// =============================================================================
#if defined(__HIPCC__)

#include <hip/hip_runtime.h>

#include <mxDataGenerator/DataGeneratorGPU.hpp>
#include <mxDataGenerator/dataTypeInfo.hpp>

namespace DGen
{
    enum class MXInitDevice
    {
        Cpu = 0,
        Gpu = 1,
    };

    namespace detail
    {
        // Compute the reference float vector from packed device bytes that
        // have just been read back to host. Linear walk over `arraySize`
        // elements; each element dequantises against its block scale.
        template <typename DT>
        std::vector<float>
            referenceFromPackedBytes(std::vector<uint8_t> const& dataPacked,
                                     std::vector<uint8_t> const& scaleBytes,
                                     size_t                      arraySize,
                                     int                         elementsPerMXBlock)
        {
            std::vector<float> ref(arraySize, 0.0f);
            int const          blockSize
                = (elementsPerMXBlock > 0) ? elementsPerMXBlock : 1;
#pragma omp parallel for
            for(size_t i = 0; i < arraySize; ++i)
            {
                size_t const scaleIdx = i / static_cast<size_t>(blockSize);
                ref[i]                = toFloatPacked<DT>(scaleBytes.data(),
                                                          dataPacked.data(),
                                                          static_cast<index_t>(scaleIdx),
                                                          static_cast<index_t>(i));
            }
            return ref;
        }

        // GPU PRNG fast path. Generates straight into the caller's device
        // buffers, reads the packed bytes back to host to (a) recover a
        // reference float vector and (b) feed the host-side preSwizzle when
        // the scales need re-laying out for gfx950 / gfx1250. Only the
        // (re-laid-out) scales are re-uploaded to the device; the data buffer
        // (large) stays on the device exactly as the PRNG kernel wrote it.
        // At most one of `preSwizzleTile.size() == 3` (gfx950 AITER swizzle)
        // and `gfx1250Swizzle == true` may be set.
        template <typename DT>
        std::vector<float>
            generateOnDevice(void*                       data,
                             void*                       scale,
                             std::vector<index_t> const& sizes,
                             std::vector<index_t> const& strides,
                             DataGeneratorOptions&       opt,
                             uint32_t                    seed,
                             int                         elementsPerMXBlock,
                             std::vector<size_t> const&  preSwizzleTile,
                             bool                        gfx1250Swizzle = false)
        {
            DataGeneratorGPU<DT> dgen;
            dgen.setSeed(seed);
            dgen.generateInto(data, scale, sizes, strides, opt);
            (void)hipDeviceSynchronize();

            size_t const dataBytes
                = DataGeneratorGPU<DT>::getDataBufferBytes(sizes, opt);
            size_t const naturalScaleBytes
                = DataGeneratorGPU<DT>::getScaleBufferBytes(sizes, opt);

            std::vector<uint8_t> dataHost(dataBytes);
            std::vector<uint8_t> scaleHostNatural(naturalScaleBytes);
            if(dataBytes > 0)
                (void)hipMemcpy(dataHost.data(),
                                data,
                                dataBytes,
                                hipMemcpyDeviceToHost);
            if(naturalScaleBytes > 0)
                (void)hipMemcpy(scaleHostNatural.data(),
                                scale,
                                naturalScaleBytes,
                                hipMemcpyDeviceToHost);

            // `DataGenerator` and `DataGeneratorGPU` both populate
            // `array_size = strides[N-1] * sizes[N-1]` worth of logical
            // elements (which includes any leading-dim padding rolled into
            // the stride). Use the same value here so the returned reference
            // vector lines up element-for-element with what the CPU overload
            // would have produced via `dgen.getReferenceFloat()`.
            size_t const arraySize
                = gpu_detail::computeArraySize<DT>(sizes, strides);
            auto refFloat = referenceFromPackedBytes<DT>(
                dataHost, scaleHostNatural, arraySize, elementsPerMXBlock);

            // Re-emit the scales in the swizzled layout if the caller asked
            // for one. The natural-packed scales the kernel just wrote are
            // overwritten with the swizzled version (which is at least as
            // large), so the kernel sees the same bytes the CPU path would
            // have produced.
            size_t const scaleRows
                = (elementsPerMXBlock > 0)
                      ? static_cast<size_t>(sizes[0])
                            / static_cast<size_t>(elementsPerMXBlock)
                      : 0;
            size_t const scaleCols = static_cast<size_t>(sizes[1]);
            if(preSwizzleTile.size() == 3 && naturalScaleBytes > 0)
            {
                auto scaleSwizzled
                    = preSwizzleScalesGFX950(scaleHostNatural, {scaleCols, scaleRows});
                (void)hipMemcpy(scale,
                                scaleSwizzled.data(),
                                scaleSwizzled.size(),
                                hipMemcpyHostToDevice);
            }
            else if(gfx1250Swizzle && naturalScaleBytes > 0 && elementsPerMXBlock > 0)
            {
                auto scaleSwizzled
                    = preSwizzleScalesGFX1250(scaleHostNatural,
                                              /*slowDim=*/scaleCols,
                                              /*fastDim=*/scaleRows,
                                              /*mxBlock=*/static_cast<size_t>(
                                                  elementsPerMXBlock));
                (void)hipMemcpy(scale,
                                scaleSwizzled.data(),
                                scaleSwizzled.size(),
                                hipMemcpyHostToDevice);
            }

            return refFloat;
        }
    } // namespace detail

    inline std::vector<float>
        generateMXInput(DataFormat                 dataType,
                        ScaleType                  scaleType,
                        void*                      data,
                        void*                      scale,
                        uint64_t                   row,
                        uint64_t                   col,
                        uint64_t                   stride,
                        bool                       isTranspose,
                        std::vector<size_t> const& preSwizzleTile,
                        std::vector<size_t> const& preTile,
                        int const                  scaleBlockRowSize,
                        int const                  scaleBlockColSize,
                        bool                       isMatrixA,
                        MXInitDevice               initDevice,
                        std::string_view const     initMethod = "Bounded",
                        float                      min_val    = -1.0f,
                        float                      max_val    = 1.0f,
                        // gfx1250 / non-rocroller WMMA dimk scale swizzle. Mutually
                        // exclusive with the gfx950 AITER swizzle implied by a
                        // 3-element `preSwizzleTile`; passing both throws.
                        bool                       gfx1250Swizzle = false)
    {
        if(preSwizzleTile.size() == 3 && gfx1250Swizzle)
            throw std::runtime_error("generateMXInput(GPU): cannot combine the gfx950 AITER "
                                     "scale swizzle with the gfx1250 dimk scale swizzle.");

        // CPU init: straight delegation to the host overload.
        if(initDevice == MXInitDevice::Cpu)
        {
            return generateMXInput(dataType,
                                   scaleType,
                                   data,
                                   scale,
                                   row,
                                   col,
                                   stride,
                                   isTranspose,
                                   preSwizzleTile,
                                   preTile,
                                   scaleBlockRowSize,
                                   scaleBlockColSize,
                                   isMatrixA,
                                   initMethod,
                                   min_val,
                                   max_val,
                                   gfx1250Swizzle);
        }

        // GPU init fast path is wired for the layout combinations where the
        // CPU path returns `dgen.getReferenceFloat()` directly (i.e. the
        // PRNG-natural data layout already matches what the kernel expects).
        // Other layouts go through `getAlignedFloat`, which rearranges the
        // packed buffer; rather than reproduce that on the device we fall
        // back to the CPU path here.
        bool const easyLayout
            = (isMatrixA && isTranspose) || (!isMatrixA && !isTranspose);
        if(!easyLayout)
        {
            return generateMXInput(dataType,
                                   scaleType,
                                   data,
                                   scale,
                                   row,
                                   col,
                                   stride,
                                   isTranspose,
                                   preSwizzleTile,
                                   preTile,
                                   scaleBlockRowSize,
                                   scaleBlockColSize,
                                   isMatrixA,
                                   MXInitDevice::Cpu,
                                   initMethod,
                                   min_val,
                                   max_val,
                                   gfx1250Swizzle);
        }

        // Build the same DataGeneratorOptions the CPU host overload would
        // build, then dispatch to a templated on-device generator that
        // reuses chunxlin's option mapping rules.
        DataGeneratorOptions opt;
        opt.min          = initMethod == "uniform_01"
                               ? 0.
                               : (initMethod == "hpl" ? -.5 : min_val);
        opt.max          = initMethod == "uniform_01"
                               ? 1.
                               : (initMethod == "hpl" ? .5 : max_val);
        opt.blockScaling = scaleBlockRowSize * scaleBlockColSize;
        opt.forceDenorm  = false;
        if(initMethod == "Sequential")
            opt.initMode = DataInitMode(Sequential{});
        else if(initMethod == "RowIndex")
            opt.initMode = DataInitMode(RowIndex{});
        else if(initMethod == "ColIndex")
            opt.initMode = DataInitMode(ColIndex{});
        else if(initMethod == "Checkerboard")
            opt.initMode = DataInitMode(Checkerboard{});
        else if(initMethod == "ScaledDiagonal")
            opt.initMode = DataInitMode(ScaledDiagonal{});
        else if(initMethod == "Identity")
            opt.initMode = DataInitMode(Identity{});
        else if(initMethod == "Ones")
            opt.initMode = DataInitMode(Ones{});
        else if(initMethod == "Zeros")
            opt.initMode = DataInitMode(Zeros{});
        else if(initMethod == "Bounded" || initMethod == "uniform_01")
            opt.initMode = DataInitMode(Bounded{});
        else
            opt.initMode = DataInitMode(TrigonometricFromFloat{});

        constexpr uint32_t kSeed = 1713573849;
        std::vector<index_t> const sizes
            = {static_cast<index_t>(row), static_cast<index_t>(col)};
        std::vector<index_t> const strides
            = {static_cast<index_t>(1), static_cast<index_t>(stride)};
        int const elementsPerMXBlock = scaleBlockRowSize * scaleBlockColSize;

        if(dataType == DataFormat::Fp8E5M2)
            return detail::generateOnDevice<ocp_e5m2_mxfp8>(
                data, scale, sizes, strides, opt, kSeed,
                elementsPerMXBlock, preSwizzleTile, gfx1250Swizzle);
        if(dataType == DataFormat::Fp8E4M3)
            return detail::generateOnDevice<ocp_e4m3_mxfp8>(
                data, scale, sizes, strides, opt, kSeed,
                elementsPerMXBlock, preSwizzleTile, gfx1250Swizzle);
        if(dataType == DataFormat::Fp6E2M3)
            return detail::generateOnDevice<ocp_e2m3_mxfp6>(
                data, scale, sizes, strides, opt, kSeed,
                elementsPerMXBlock, preSwizzleTile, gfx1250Swizzle);
        if(dataType == DataFormat::Fp6E3M2)
            return detail::generateOnDevice<ocp_e3m2_mxfp6>(
                data, scale, sizes, strides, opt, kSeed,
                elementsPerMXBlock, preSwizzleTile, gfx1250Swizzle);
        if(dataType == DataFormat::Fp4)
        {
            if(scaleType == ScaleType::E4M3)
                return detail::generateOnDevice<ocp_e2m1_mxfp4_e4m3>(
                    data, scale, sizes, strides, opt, kSeed,
                    elementsPerMXBlock, preSwizzleTile, gfx1250Swizzle);
            if(scaleType == ScaleType::E5M3)
                return detail::generateOnDevice<ocp_e2m1_mxfp4_e5m3>(
                    data, scale, sizes, strides, opt, kSeed,
                    elementsPerMXBlock, preSwizzleTile, gfx1250Swizzle);
            return detail::generateOnDevice<ocp_e2m1_mxfp4>(
                data, scale, sizes, strides, opt, kSeed,
                elementsPerMXBlock, preSwizzleTile, gfx1250Swizzle);
        }
        throw std::runtime_error("Unsupported data type in GPU MX data generation");
    }
} // namespace DGen

#endif // __HIPCC__
