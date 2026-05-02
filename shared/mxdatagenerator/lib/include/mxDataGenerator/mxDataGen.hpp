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
                                        std::vector<size_t> const& /*preTile*/)
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
                                              float                      max_val    = 1.0f)
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
        if(dataType == DataFormat::Fp8E5M2)
        {
            DataGenerator<ocp_e5m2_mxfp8> dgen;
            return detail::generateData<decltype(dgen), ocp_e5m2_mxfp8>(
                dgen, data, scale, sizes, strides, seed, opt, elementsPerMXBlock,
                isTranspose, isMatrixA, preSwizzleTile, preTile);
        }
        else if(dataType == DataFormat::Fp8E4M3)
        {
            DataGenerator<ocp_e4m3_mxfp8> dgen;
            return detail::generateData<decltype(dgen), ocp_e4m3_mxfp8>(
                dgen, data, scale, sizes, strides, seed, opt, elementsPerMXBlock,
                isTranspose, isMatrixA, preSwizzleTile, preTile);
        }
        else if(dataType == DataFormat::Fp6E2M3)
        {
            DataGenerator<ocp_e2m3_mxfp6> dgen;
            return detail::generateData<decltype(dgen), ocp_e2m3_mxfp6>(
                dgen, data, scale, sizes, strides, seed, opt, elementsPerMXBlock,
                isTranspose, isMatrixA, preSwizzleTile, preTile);
        }
        else if(dataType == DataFormat::Fp6E3M2)
        {
            DataGenerator<ocp_e3m2_mxfp6> dgen;
            return detail::generateData<decltype(dgen), ocp_e3m2_mxfp6>(
                dgen, data, scale, sizes, strides, seed, opt, elementsPerMXBlock,
                isTranspose, isMatrixA, preSwizzleTile, preTile);
        }
        else if(dataType == DataFormat::Fp4)
        {
            if(scaleType == ScaleType::E4M3)
            {
                DataGenerator<ocp_e2m1_mxfp4_e4m3> dgen;
                return detail::generateData<decltype(dgen), ocp_e2m1_mxfp4_e4m3>(
                    dgen, data, scale, sizes, strides, seed, opt, elementsPerMXBlock,
                    isTranspose, isMatrixA, preSwizzleTile, preTile);
            }
            else if(scaleType == ScaleType::E5M3)
            {
                DataGenerator<ocp_e2m1_mxfp4_e5m3> dgen;
                return detail::generateData<decltype(dgen), ocp_e2m1_mxfp4_e5m3>(
                    dgen, data, scale, sizes, strides, seed, opt, elementsPerMXBlock,
                    isTranspose, isMatrixA, preSwizzleTile, preTile);
            }
            else
            {
                DataGenerator<ocp_e2m1_mxfp4> dgen;
                return detail::generateData<decltype(dgen), ocp_e2m1_mxfp4>(
                    dgen, data, scale, sizes, strides, seed, opt, elementsPerMXBlock,
                    isTranspose, isMatrixA, preSwizzleTile, preTile);
            }
        }
        throw std::runtime_error("Unsupported data types in MX data generation!");
    }
} // namespace DGen
