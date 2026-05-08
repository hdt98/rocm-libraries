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

#include "mxDataGen.hpp"
#include <hip/hip_runtime.h>
#include <mxDataGenerator/DataGenerator.hpp>
#include <mxDataGenerator/DataGeneratorGPU.hpp>
#include <mxDataGenerator/PreSwizzle.hpp>
#include <mxDataGenerator/dataTypeInfo.hpp>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <stdexcept>


template <typename DT>
std::vector<uint8_t> unpackData(std::vector<uint8_t> const& packedBytes, size_t elementCount)
{
    // Only F4 and F6 need to unpack data.
    static_assert(std::is_same_v<DT, DGen::ocp_e2m1_mxfp4>
                  || std::is_same_v<DT, DGen::ocp_e2m1_mxfp4_e5m3>
                  || std::is_same_v<DT, DGen::ocp_e2m1_mxfp4_e4m3>
                  || std::is_same_v<DT, DGen::ocp_e3m2_mxfp6>
                  || std::is_same_v<DT, DGen::ocp_e2m3_mxfp6>);

    if constexpr(std::is_same_v<DT, DGen::ocp_e3m2_mxfp6>
                 || std::is_same_v<DT, DGen::ocp_e2m3_mxfp6>)
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
            uint8_t const b = (byteIndex < packedBytes.size()) ? packedBytes[byteIndex] : 0;
            unpackedDataBytes[i]
                = static_cast<uint8_t>((i % 2 == 0) ? (b & 0x0F) : ((b >> 4) & 0x0F));
        }
        return unpackedDataBytes;
    }
}

template <typename DT>
void packData(std::vector<uint8_t> const& dataBytes, uint8_t* packedData)
{
    // Only F4 and F6 need to unpack data.
    static_assert(std::is_same_v<DT, DGen::ocp_e2m1_mxfp4>
                  || std::is_same_v<DT, DGen::ocp_e2m1_mxfp4_e5m3>
                  || std::is_same_v<DT, DGen::ocp_e2m1_mxfp4_e4m3>
                  || std::is_same_v<DT, DGen::ocp_e3m2_mxfp6>
                  || std::is_same_v<DT, DGen::ocp_e2m3_mxfp6>);

    if constexpr(std::is_same_v<DT, DGen::ocp_e3m2_mxfp6>
                 || std::is_same_v<DT, DGen::ocp_e2m3_mxfp6>)
    {
        size_t const elementCount = dataBytes.size();
        size_t const packedSize   = (elementCount * 6 + 7) / 8;
        std::memset(packedData, 0, packedSize);

        for(size_t i = 0; i < elementCount; ++i)
        {
            uint16_t const v = static_cast<uint16_t>(dataBytes[i] & 0x3F);
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
                packedData[byteIndex] = static_cast<uint8_t>((packedData[byteIndex] & 0xF0) | v);
            else
                packedData[byteIndex]
                    = static_cast<uint8_t>((packedData[byteIndex] & 0x0F) | (v << 4));
        }
    }
}

/**
 * @brief Align data with scale and return reference floats
 *
 * mxDataGenerator returns data and scale in which every consecutive
 * 32 data share a scale (i.e., data 0-31 use scale 0, data 32-63 use
 * scale 1, etc.). But when doing matrix multiplication with non-transpose
 * matrix A or transpose matrix B, the data and scale are accessed in a
 * different order (see the example in comment below). This function
 * re-arranges the data to let the data use the correct scale.
 * Note, the passed-in dataBytes will be changed due to the rearrangement.
 *
 * @return float values of generated MX type data aligned with scale
 */
template <typename DT>
std::vector<float> getAlignedFloat(std::vector<uint8_t>&              dataBytes,
                                   std::vector<uint8_t> const&        scaleBytes,
                                   std::array<DGen::index_t, 2> const sizes,
                                   int                                elementsPerMXBlock,
                                   bool                               isMatrixA)
{
    std::vector<float>   refFloat(sizes[0] * sizes[1], 0.0);
    std::vector<uint8_t> alignedDataBytes(dataBytes.size());

    if(isMatrixA) // non-transpose
    {
        int M = sizes[0];
        int K = sizes[1];

        // For example, assume matrix A is 128x128 and elementsPerMXBlock is 32.
        // Before aligned,
        //
        //  mk     m     k       scale ID
        //  0      0     0           0
        //  1      1     0           1     (data at index 1 use scale 1 not 0)
        //  2      2     0           2
        //            ...
        //  127   127    0          127
        //
        //  128    0     1           0
        //  129    1     1           1
        //            ...
        //  255   127    1          127
        //            ...
        //
        // To align data with scale,
        //
        //  mk     m     k       scale ID      data id
        //  0      0     0           0            0
        //  1      1     0           1           32
        //  2      2     0           2           64
        //            ...
        // 127    127    0          127        4064 (127 x 32)
        //
        // We move data at index 32 to index 1 (because the index 1
        // is using scale 1), data at index 64 to index 2, and so on.

#pragma omp parallel for
        for(size_t mk = 0; mk < M * K; ++mk)
        {
            auto m        = mk % M;
            auto k        = mk / M;
            auto scale_id = (k / elementsPerMXBlock) * M + m;

            auto data_id         = scale_id * elementsPerMXBlock + k % elementsPerMXBlock;
            alignedDataBytes[mk] = dataBytes[data_id];
            refFloat[mk]
                = DGen::toFloat<DT>(scaleBytes.data(), dataBytes.data(), scale_id, data_id);
        }
        std::swap(dataBytes, alignedDataBytes);
    }
    else // transpose matrixB
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
                = DGen::toFloat<DT>(scaleBytes.data(), dataBytes.data(), scale_id, data_id);
        }
        std::swap(dataBytes, alignedDataBytes);
    }
    return refFloat;
}

template <typename T, typename DT>
std::vector<float> generateData(T                           dgen,
                                void*                       data,
                                void*                       scale,
                                std::vector<DGen::index_t>  sizes,
                                std::vector<DGen::index_t>  strides,
                                uint32_t                    seed,
                                DGen::DataGeneratorOptions& opt,
                                int                         elementsPerMXBlock,
                                bool                        isTranspose,
                                bool                        isMatrixA,
                                std::vector<size_t> const&  preSwizzleTile,
                                std::vector<size_t> const&  preTile,
                                bool                        gfx1250Swizzle = false)
{
    using namespace DGen;

    dgen.setSeed(seed);
    dgen.generate(sizes, strides, opt);

    std::vector<uint8_t> dataBytes = dgen.getDataBytes();
    std::memcpy(data, dataBytes.data(), dataBytes.size() * sizeof(uint8_t));

    std::vector<uint8_t> scaleBytes = dgen.getScaleBytes();

    // Apply pre-swizzle to scale data. The two swizzles target different
    // architectures and are mutually exclusive: gfx950 expects the AITER
    // (preSwizzleScalesGFX950) layout, gfx1250 expects the dimk layout.
    size_t scaleRows = sizes[0] / elementsPerMXBlock;
    size_t scaleCols = sizes[1];

    if(preSwizzleTile.size() == 3)
    {
        scaleBytes = DGen::preSwizzleScalesGFX950(scaleBytes, {scaleCols, scaleRows});
    }
    else if(gfx1250Swizzle && elementsPerMXBlock > 0)
    {
        scaleBytes = DGen::preSwizzleScalesGFX1250(scaleBytes,
                                                   /*slowDim=*/scaleCols,
                                                   /*fastDim=*/scaleRows,
                                                   /*mxBlock=*/static_cast<size_t>(
                                                       elementsPerMXBlock));
    }

    std::memcpy(scale, scaleBytes.data(), scaleBytes.size() * sizeof(uint8_t));

    if((isMatrixA && isTranspose) || (!isMatrixA && !isTranspose))
    {
        // For (1) transposed matrixA and (2) non-transposed matrixB,
        // return the reference float directly since they are aligned already.
        return dgen.getReferenceFloat();
    }

    // For types smaller than 8-bit, mxDataGenerator returns packed data (i.e., two FP4 will be
    // stored in a uint8_t), so unpacking the data is required before converting them to float
    if constexpr(std::is_same_v<DT, DGen::ocp_e5m2_mxfp8>
                 || std::is_same_v<DT, DGen::ocp_e4m3_mxfp8>)
    {
        auto ret = getAlignedFloat<DT>(
            dataBytes, scaleBytes, {sizes[0], sizes[1]}, elementsPerMXBlock, isMatrixA);
        std::memcpy(data, dataBytes.data(), dataBytes.size() * sizeof(uint8_t));
        return ret;
    }
    else if constexpr(std::is_same_v<DT, DGen::ocp_e3m2_mxfp6>
                      || std::is_same_v<DT, DGen::ocp_e2m3_mxfp6>)
    {
        size_t const elementCount = static_cast<size_t>(sizes[0]) * static_cast<size_t>(sizes[1]);
        auto         unpackedDataBytes = unpackData<DT>(dataBytes, elementCount);
        auto ret               = getAlignedFloat<DT>(
            unpackedDataBytes, scaleBytes, {sizes[0], sizes[1]}, elementsPerMXBlock, isMatrixA);
        // GPU expects the data are packed
        packData<DT>(unpackedDataBytes, static_cast<uint8_t*>(data));
        return ret;
    }
    else if constexpr(std::is_same_v<DT, DGen::ocp_e2m1_mxfp4>
                      || std::is_same_v<DT, DGen::ocp_e2m1_mxfp4_e5m3>
                      || std::is_same_v<DT, DGen::ocp_e2m1_mxfp4_e4m3>)
    {
        size_t const elementCount = static_cast<size_t>(sizes[0]) * static_cast<size_t>(sizes[1]);
        auto         unpackedDataBytes = unpackData<DT>(dataBytes, elementCount);
        auto ret               = getAlignedFloat<DT>(
            unpackedDataBytes, scaleBytes, {sizes[0], sizes[1]}, elementsPerMXBlock, isMatrixA);
        // GPU expects the data are packed
        packData<DT>(unpackedDataBytes, static_cast<uint8_t*>(data));
        return ret;
    }
    else
    {
        throw std::runtime_error("Unsupported data types in MX data generation!");
    }
}

/**
 * @brief Generate random data for OCP (MX) F8/F6/F4 types
 *
 * The generated data consist of data part and scale part,
 * and the corresponding float values (combine data and scale)
 * will be returned.
 *
 * @return float values of generated MX type data
 */
std::vector<float> generateMXInput(hipDataType                dataType,
                                   hipDataType                scaleType,
                                   void*                      data,
                                   void*                      scale,
                                   DGen::index_t              rowSize,
                                   DGen::index_t              colSize,
                                   DGen::index_t              stride,
                                   bool                       isTranspose,
                                   const std::vector<size_t>& preSwizzleTile,
                                   const std::vector<size_t>& preTile,
                                   int const                  scaleBlockRowSize,
                                   int const                  scaleBlockColSize,
                                   bool                       isMatrixA,
                                   std::string_view const     initMethod,
                                   float                      min_val,
                                   float                      max_val,
                                   bool                       gfx1250Swizzle)
{
    using namespace DGen;
    if(preSwizzleTile.size() == 3 && gfx1250Swizzle)
        throw std::runtime_error("generateMXInput(CPU): cannot combine the gfx950 AITER "
                                 "scale swizzle with the gfx1250 dimk scale swizzle.");

    DataGeneratorOptions opt;
    opt.min          = initMethod == "uniform_01" ? 0. : (initMethod == "hpl" ? -.5 : min_val);
    opt.max          = initMethod == "uniform_01" ? 1. : (initMethod == "hpl" ? .5 : max_val);
    opt.blockScaling = scaleBlockRowSize * scaleBlockColSize;
    opt.forceDenorm  = false;

    // Map string initMethod to DataInitMode
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
        // TODO initMethod == "hpl" should also be Bounded, but fails some tests
        opt.initMode = DataInitMode(TrigonometricFromFloat{});

    const uint32_t seed = 1713573849;

    std::vector<index_t> sizes = {rowSize, colSize};
    std::vector<index_t> strides;

    strides.push_back(1);
    strides.push_back(stride);

    auto const elementsPerMXBlock = scaleBlockRowSize * scaleBlockColSize;

    if(dataType == HIP_R_8F_E5M2)
    {
        DGen::DataGenerator<DGen::ocp_e5m2_mxfp8> dgen;
        return generateData<decltype(dgen), DGen::ocp_e5m2_mxfp8>(dgen,
                                                                  data,
                                                                  scale,
                                                                  sizes,
                                                                  strides,
                                                                  seed,
                                                                  opt,
                                                                  elementsPerMXBlock,
                                                                  isTranspose,
                                                                  isMatrixA,
                                                                  preSwizzleTile,
                                                                  preTile,
                                                                  gfx1250Swizzle);
    }
    else if(dataType == HIP_R_8F_E4M3)
    {
        DGen::DataGenerator<DGen::ocp_e4m3_mxfp8> dgen;
        return generateData<decltype(dgen), DGen::ocp_e4m3_mxfp8>(dgen,
                                                                  data,
                                                                  scale,
                                                                  sizes,
                                                                  strides,
                                                                  seed,
                                                                  opt,
                                                                  elementsPerMXBlock,
                                                                  isTranspose,
                                                                  isMatrixA,
                                                                  preSwizzleTile,
                                                                  preTile,
                                                                  gfx1250Swizzle);
    }
    else if(static_cast<hipDataType>(dataType) == HIP_R_6F_E2M3)
    {
        DGen::DataGenerator<DGen::ocp_e2m3_mxfp6> dgen;
        return generateData<decltype(dgen), DGen::ocp_e2m3_mxfp6>(dgen,
                                                                  data,
                                                                  scale,
                                                                  sizes,
                                                                  strides,
                                                                  seed,
                                                                  opt,
                                                                  elementsPerMXBlock,
                                                                  isTranspose,
                                                                  isMatrixA,
                                                                  preSwizzleTile,
                                                                  preTile,
                                                                  gfx1250Swizzle);
    }
    else if(static_cast<hipDataType>(dataType) == HIP_R_6F_E3M2)
    {
        DGen::DataGenerator<DGen::ocp_e3m2_mxfp6> dgen;
        return generateData<decltype(dgen), DGen::ocp_e3m2_mxfp6>(dgen,
                                                                  data,
                                                                  scale,
                                                                  sizes,
                                                                  strides,
                                                                  seed,
                                                                  opt,
                                                                  elementsPerMXBlock,
                                                                  isTranspose,
                                                                  isMatrixA,
                                                                  preSwizzleTile,
                                                                  preTile,
                                                                  gfx1250Swizzle);
    }
    else if(static_cast<hipDataType>(dataType) == HIP_R_4F_E2M1)
    {
        if(scaleType == HIP_R_8F_E4M3)
        {
            DGen::DataGenerator<DGen::ocp_e2m1_mxfp4_e4m3> dgen;
            return generateData<decltype(dgen), DGen::ocp_e2m1_mxfp4_e4m3>(dgen,
                                                                          data,
                                                                          scale,
                                                                          sizes,
                                                                          strides,
                                                                          seed,
                                                                          opt,
                                                                          elementsPerMXBlock,
                                                                          isTranspose,
                                                                          isMatrixA,
                                                                          preSwizzleTile,
                                                                          preTile,
                                                                          gfx1250Swizzle);
        }
        else if(scaleType == static_cast<hipDataType>(HIP_R_8F_E5M3_EXT))
        {
            DGen::DataGenerator<DGen::ocp_e2m1_mxfp4_e5m3> dgen;
            return generateData<decltype(dgen), DGen::ocp_e2m1_mxfp4_e5m3>(dgen,
                                                                          data,
                                                                          scale,
                                                                          sizes,
                                                                          strides,
                                                                          seed,
                                                                          opt,
                                                                          elementsPerMXBlock,
                                                                          isTranspose,
                                                                          isMatrixA,
                                                                          preSwizzleTile,
                                                                          preTile,
                                                                          gfx1250Swizzle);
        }
        else
        {
            DGen::DataGenerator<DGen::ocp_e2m1_mxfp4> dgen;
            return generateData<decltype(dgen), DGen::ocp_e2m1_mxfp4>(dgen,
                                                                      data,
                                                                      scale,
                                                                      sizes,
                                                                      strides,
                                                                      seed,
                                                                      opt,
                                                                      elementsPerMXBlock,
                                                                      isTranspose,
                                                                      isMatrixA,
                                                                      preSwizzleTile,
                                                                      preTile,
                                                                      gfx1250Swizzle);
        }
    }
    else
    {
        throw std::runtime_error("Unsupported data types in MX data generation!");
    }
}

// ----------------------------------------------------------------------------
// GPU PRNG fast path. Helpers + template dispatch + the public
// `MXInitDevice`-taking overload of `generateMXInput`.
//
// The fast path is wired only for the matrix layouts where the host CPU
// overload would have returned `dgen.getReferenceFloat()` directly --
// i.e. transposed-A (`isMatrixA && isTranspose`) and non-transposed-B
// (`!isMatrixA && !isTranspose`). Other layouts go through `getAlignedFloat`
// (above), which rearranges the packed buffer; the GPU path falls back to
// the host overload for those.
// ----------------------------------------------------------------------------

namespace
{
    // Compute the reference float vector from packed device bytes that have
    // just been read back to host. Linear walk over `arraySize` elements;
    // each element dequantises against its block scale.
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
            ref[i]
                = DGen::toFloatPacked<DT>(scaleBytes.data(),
                                          dataPacked.data(),
                                          static_cast<DGen::index_t>(scaleIdx),
                                          static_cast<DGen::index_t>(i));
        }
        return ref;
    }

    // Generate straight into the caller's device buffers, then read the
    // packed bytes back to host to (a) recover a reference float vector
    // and (b) feed the host-side preSwizzle when the scales need re-laying
    // out for gfx950 / gfx1250. Only the (re-laid-out) scales are
    // re-uploaded to the device; the data buffer (large) stays on the
    // device exactly as the PRNG kernel wrote it.
    template <typename DT>
    std::vector<float>
        generateOnDevice(void*                             data,
                         void*                             scale,
                         std::vector<DGen::index_t> const& sizes,
                         std::vector<DGen::index_t> const& strides,
                         DGen::DataGeneratorOptions&       opt,
                         uint32_t                          seed,
                         int                               elementsPerMXBlock,
                         std::vector<size_t> const&        preSwizzleTile,
                         bool                              gfx1250Swizzle)
    {
        DGen::DataGeneratorGPU<DT> dgen;
        dgen.setSeed(seed);
        dgen.generateInto(data, scale, sizes, strides, opt);
        (void)hipDeviceSynchronize();

        size_t const dataBytes
            = DGen::DataGeneratorGPU<DT>::getDataBufferBytes(sizes, opt);
        size_t const naturalScaleBytes
            = DGen::DataGeneratorGPU<DT>::getScaleBufferBytes(sizes, opt);

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
        // elements (which includes any leading-dim padding rolled into the
        // stride). Use the same value here so the returned reference vector
        // lines up element-for-element with what the CPU overload would
        // have produced via `dgen.getReferenceFloat()`.
        size_t const arraySize
            = DGen::gpu_detail::computeArraySize<DT>(sizes, strides);
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
                = DGen::preSwizzleScalesGFX950(scaleHostNatural, {scaleCols, scaleRows});
            (void)hipMemcpy(scale,
                            scaleSwizzled.data(),
                            scaleSwizzled.size(),
                            hipMemcpyHostToDevice);
        }
        else if(gfx1250Swizzle && naturalScaleBytes > 0 && elementsPerMXBlock > 0)
        {
            auto scaleSwizzled
                = DGen::preSwizzleScalesGFX1250(scaleHostNatural,
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
} // namespace

std::vector<float> generateMXInput(hipDataType                dataType,
                                   hipDataType                scaleType,
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
                                   MXInitDevice               initDevice,
                                   std::string_view const     initMethod,
                                   float                      min_val,
                                   float                      max_val,
                                   bool                       gfx1250Swizzle)
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
    // CPU path returns `dgen.getReferenceFloat()` directly. Other layouts
    // go through `getAlignedFloat`, which rearranges the packed buffer;
    // rather than reproduce that on the device we fall back to the CPU
    // path. (Note: when we fall back, `data`/`scale` MUST be host pointers,
    // i.e. the caller must not hand us device buffers for non-easy layouts.)
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
                               initMethod,
                               min_val,
                               max_val,
                               gfx1250Swizzle);
    }

    // Build the same DataGeneratorOptions the host overload would build,
    // then dispatch to a templated on-device generator.
    DGen::DataGeneratorOptions opt;
    opt.min          = initMethod == "uniform_01"
                           ? 0.
                           : (initMethod == "hpl" ? -.5 : min_val);
    opt.max          = initMethod == "uniform_01"
                           ? 1.
                           : (initMethod == "hpl" ? .5 : max_val);
    opt.blockScaling = scaleBlockRowSize * scaleBlockColSize;
    opt.forceDenorm  = false;
    if(initMethod == "Sequential")
        opt.initMode = DGen::DataInitMode(DGen::Sequential{});
    else if(initMethod == "RowIndex")
        opt.initMode = DGen::DataInitMode(DGen::RowIndex{});
    else if(initMethod == "ColIndex")
        opt.initMode = DGen::DataInitMode(DGen::ColIndex{});
    else if(initMethod == "Checkerboard")
        opt.initMode = DGen::DataInitMode(DGen::Checkerboard{});
    else if(initMethod == "ScaledDiagonal")
        opt.initMode = DGen::DataInitMode(DGen::ScaledDiagonal{});
    else if(initMethod == "Identity")
        opt.initMode = DGen::DataInitMode(DGen::Identity{});
    else if(initMethod == "Ones")
        opt.initMode = DGen::DataInitMode(DGen::Ones{});
    else if(initMethod == "Zeros")
        opt.initMode = DGen::DataInitMode(DGen::Zeros{});
    else if(initMethod == "Bounded" || initMethod == "uniform_01")
        opt.initMode = DGen::DataInitMode(DGen::Bounded{});
    else
        opt.initMode = DGen::DataInitMode(DGen::TrigonometricFromFloat{});

    constexpr uint32_t               kSeed = 1713573849;
    std::vector<DGen::index_t> const sizes
        = {static_cast<DGen::index_t>(row), static_cast<DGen::index_t>(col)};
    std::vector<DGen::index_t> const strides
        = {static_cast<DGen::index_t>(1), static_cast<DGen::index_t>(stride)};
    int const elementsPerMXBlock = scaleBlockRowSize * scaleBlockColSize;

    if(dataType == HIP_R_8F_E5M2)
        return generateOnDevice<DGen::ocp_e5m2_mxfp8>(
            data, scale, sizes, strides, opt, kSeed,
            elementsPerMXBlock, preSwizzleTile, gfx1250Swizzle);
    if(dataType == HIP_R_8F_E4M3)
        return generateOnDevice<DGen::ocp_e4m3_mxfp8>(
            data, scale, sizes, strides, opt, kSeed,
            elementsPerMXBlock, preSwizzleTile, gfx1250Swizzle);
    if(static_cast<hipDataType>(dataType) == HIP_R_6F_E2M3)
        return generateOnDevice<DGen::ocp_e2m3_mxfp6>(
            data, scale, sizes, strides, opt, kSeed,
            elementsPerMXBlock, preSwizzleTile, gfx1250Swizzle);
    if(static_cast<hipDataType>(dataType) == HIP_R_6F_E3M2)
        return generateOnDevice<DGen::ocp_e3m2_mxfp6>(
            data, scale, sizes, strides, opt, kSeed,
            elementsPerMXBlock, preSwizzleTile, gfx1250Swizzle);
    if(static_cast<hipDataType>(dataType) == HIP_R_4F_E2M1)
    {
        if(scaleType == HIP_R_8F_E4M3)
            return generateOnDevice<DGen::ocp_e2m1_mxfp4_e4m3>(
                data, scale, sizes, strides, opt, kSeed,
                elementsPerMXBlock, preSwizzleTile, gfx1250Swizzle);
        if(scaleType == static_cast<hipDataType>(HIP_R_8F_E5M3_EXT))
            return generateOnDevice<DGen::ocp_e2m1_mxfp4_e5m3>(
                data, scale, sizes, strides, opt, kSeed,
                elementsPerMXBlock, preSwizzleTile, gfx1250Swizzle);
        return generateOnDevice<DGen::ocp_e2m1_mxfp4>(
            data, scale, sizes, strides, opt, kSeed,
            elementsPerMXBlock, preSwizzleTile, gfx1250Swizzle);
    }
    throw std::runtime_error("Unsupported data type in GPU MX data generation");
}
