// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <tuple>
#include <vector>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/utility/algorithm.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/convolution_parameter.hpp"
#include "ck/library/utility/convolution_host_tensor_descriptor_helper.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_conv_fwd.hpp"
#include "ck/tensor_operation/gpu/warp/wconv_conv.hpp"

using InElementOp  = ck::tensor_operation::element_wise::PassThrough;
using WeiElementOp = ck::tensor_operation::element_wise::PassThrough;
using OutElementOp = ck::tensor_operation::element_wise::UnaryConvert;

#define USE_ABSOLUTE_SIZE 1
#ifdef USE_ABSOLUTE_SIZE
#define DEFAULT_K 16
#define DEFAULT_C 32
#define DEFAULT_W 8
#define DEFAULT_H 16
#define DEFAULT_W_1K 48
#define DEFAULT_H_1K 32
#else
#define DEFAULT_C_REPEAT 1
#define DEFAULT_K_REPEAT 1
#define DEFAULT_W_REPEAT 1
#define DEFAULT_H_REPEAT 1
#endif

enum ShapeType
{
    Shape_4X2,
    Shape_4X4,
    Shape_8X4,
};

enum FilterType
{
    Filter_1X1,
    Filter_3X3,
    Filter_2X2,
};

namespace ck {

namespace conv_op_util {

//#define LOAD_DATA_PER_TILE 0

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wundefined-reinterpret-cast"
template <typename InDataType,
          typename WeiDataType,
          typename AccDataType,
          bool UseF32I32,
          index_t HPerWconv,
          index_t WPerWconv,
          index_t FilterSize,
          index_t DilationX,
          index_t DilationY,
          index_t Width,
          index_t Height,
          index_t UnpackedInputChannels,
          index_t UnpackedOutputChannels>
__global__ void __launch_bounds__(64, 1)
    conv_fwd(const InDataType* in_, const WeiDataType* wei_, AccDataType* c_)
{
    constexpr auto wconvConv = ck::WconvConv<WeiDataType,
                                             InDataType,
                                             AccDataType,
                                             HPerWconv,
                                             WPerWconv,
                                             FilterSize,
                                             DilationX,
                                             DilationY,
                                             1,
                                             false,
                                             false,
                                             UseF32I32>();

    auto in  = reinterpret_cast<const typename decltype(wconvConv)::KernelInDataType*>(in_);
    auto wei = reinterpret_cast<const typename decltype(wconvConv)::KernelWeightDataType*>(wei_);

    static_assert(Width % WPerWconv == 0, "");
    static_assert(Height % HPerWconv == 0, "");
    static_assert(UnpackedInputChannels % wconvConv.GetUnpackedNumInputChannels() == 0, "");
    static_assert(UnpackedOutputChannels % wconvConv.GetUnpackedNumOutputChannels() == 0, "");

    constexpr index_t DataTileHeight = 4;
    constexpr index_t InputChannels  = UnpackedInputChannels * wconvConv.GetNumInputChannels() /
                                      wconvConv.GetUnpackedNumInputChannels();
    constexpr index_t OutputChannels = UnpackedOutputChannels * wconvConv.GetNumOutputChannels() /
                                       wconvConv.GetUnpackedNumOutputChannels();

    constexpr index_t HRepeat = Height / HPerWconv;
    constexpr index_t WRepeat = Width / WPerWconv;
    constexpr index_t CRepeat = UnpackedInputChannels / wconvConv.GetUnpackedNumInputChannels();
    constexpr index_t KRepeat = UnpackedOutputChannels / wconvConv.GetUnpackedNumOutputChannels();
#ifdef LOAD_DATA_PER_TILE
    constexpr index_t AccVectorCount = 1;
#else
    constexpr index_t AccVectorCount = HRepeat * WRepeat * KRepeat;
#endif

    using InDataVec      = typename decltype(wconvConv)::InDataVec::type;
    using InDataTileVec  = typename decltype(wconvConv)::InDataTileVec::type;
    using WeiDataVec     = typename decltype(wconvConv)::WeiDataVec::type;
    using WeiDataTileVec = typename decltype(wconvConv)::WeiDataTileVec::type;
    using AccDataVec     = typename decltype(wconvConv)::AccDataVec;
    using AccVec         = StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr,
                                             AccDataType,
                                             AccVectorCount,
                                             wconvConv.GetNumAccumComponents(),
                                             true>;

    const int lIdx = threadIdx.x;
#if !defined(LOAD_DATA_PER_TILE)
    InDataVec inData[HRepeat * WRepeat * CRepeat] = {};
    AccVec c_thread_buf_                          = {};
#endif
    WeiDataVec weiData[KRepeat * CRepeat] = {};

    // Data layout: HWC, unit: InDataType
    constexpr index_t data_H_stride = Width * InputChannels;
    constexpr index_t data_W_stride = InputChannels;
    constexpr index_t data_C_stride = wconvConv.GetNumInputChannels();

    // Filter layout: KYXC, unit: WeiDataType
    constexpr index_t weight_stride = InputChannels;

    // Acc layout: HWK, unit: AccDataType
    constexpr index_t acc_H_stride = Width * OutputChannels;
    constexpr index_t acc_W_stride = OutputChannels;
    constexpr index_t acc_K_stride = wconvConv.GetNumOutputChannels();

    // Load inData
    const index_t compIdx =
        lIdx * wconvConv.GetNumDataComponents() / wconvConv.GetNumImageSubTilesInVertical();

    const index_t accCompIdx =
        lIdx * wconvConv.GetNumAccumComponents() / wconvConv.GetNumSubTilesPerImageTile();

    auto load_in_data = [&](index_t h, index_t w, index_t c) {
        const index_t subC = compIdx % wconvConv.GetNumInputChannels();
        const index_t subW = (compIdx / wconvConv.GetNumInputChannels()) % WPerWconv;
        const index_t subH = (compIdx / wconvConv.GetNumInputChannels()) / WPerWconv;

        const index_t offset = (h * HPerWconv + subH) * data_H_stride +
                               (w * WPerWconv + subW) * data_W_stride + (c * data_C_stride + subC);
        if constexpr(wconvConv.GetNumImageSubTilesInVertical() > 1)
        {
            // shape 8x4
            typename decltype(wconvConv)::InDataVec inDataVec;

            static_for<0, wconvConv.GetNumImageSubTilesInVertical(), 1>{}([&](auto tileId) {
                inDataVec.template AsType<InDataTileVec>()(Number<tileId>{}) =
                    *reinterpret_cast<const InDataTileVec*>(
                        in + offset + tileId * DataTileHeight * data_H_stride);
            });
            return inDataVec.template AsType<InDataVec>()(Number<0>{});
        }
        else
        {
            return *reinterpret_cast<const InDataVec*>(in + offset);
        }
    };

    auto store_acc_data = [&](index_t h, index_t w, index_t k, AccDataVec& c_vec) {
        const index_t subW = (accCompIdx / wconvConv.GetNumOutputChannels()) % WPerWconv;
        const index_t subH = (accCompIdx / wconvConv.GetNumOutputChannels()) / WPerWconv;

        if constexpr(wconvConv.GetNumAccumComponents() == 4)
        {
            const index_t subK   = accCompIdx % wconvConv.GetNumOutputChannels();
            const index_t offset = (h * HPerWconv + subH) * acc_H_stride +
                                   (w * WPerWconv + subW) * acc_W_stride +
                                   (k * acc_K_stride + subK);
            *reinterpret_cast<typename AccDataVec::type*>(c_ + offset) =
                c_vec.template AsType<typename AccDataVec::type>()(Number<0>{});
        }
        else
        {
            static_assert(wconvConv.GetNumAccumComponents() == 8, "unexpected value");
            // ACO = 0, do swizzle after 4 channels.
            using AccSwizzleVec = typename vector_type<AccDataType, 4>::type;
            const index_t subK  = accCompIdx % wconvConv.GetNumOutputChannels() /
                                 (wconvConv.GetNumAccumComponents() * 2) *
                                 (wconvConv.GetNumAccumComponents() * 2);
            const index_t offset = (h * HPerWconv + subH) * acc_H_stride +
                                   (w * WPerWconv + subW) * acc_W_stride +
                                   (k * acc_K_stride + subK);
            index_t secOffset = 8;
            if constexpr(wconvConv.GetNumSubTilesPerImageTile() > 1)
            {
                secOffset = HPerWconv / wconvConv.GetNumSubTilesPerImageTile() * acc_H_stride;
            }
            const index_t swizzleOffset = (lIdx & 1) * 4;
            *reinterpret_cast<AccSwizzleVec*>(c_ + offset + swizzleOffset) =
                c_vec.template AsType<AccSwizzleVec>()(Number<0>{});
            *reinterpret_cast<AccSwizzleVec*>(c_ + offset + swizzleOffset + secOffset) =
                c_vec.template AsType<AccSwizzleVec>()(Number<1>{});
        }
    };

#if !defined(LOAD_DATA_PER_TILE)
    static_for<0, HRepeat, 1>{}([&](auto h) {
        static_for<0, WRepeat, 1>{}([&](auto w) {
            static_for<0, CRepeat, 1>{}([&](auto c) {
                constexpr index_t tileOffset = h * WRepeat * CRepeat + w * CRepeat + c;
                inData[tileOffset]           = load_in_data(h, w, c);
            });
        });
    });
#endif

    // Load weight
    constexpr index_t numWeightTile = wconvConv.GetWeightRegSize();
    const index_t weiCompIdx =
        lIdx / numWeightTile * numWeightTile * wconvConv.GetNumWeightComponents();
    static_for<0, KRepeat, 1>{}([&](auto k) {
        static_for<0, CRepeat, 1>{}([&](auto c) {
            constexpr index_t tileOffset = k * CRepeat + c;

            const index_t subC   = weiCompIdx % wconvConv.GetNumInputChannels();
            const index_t subK   = weiCompIdx / wconvConv.GetNumInputChannels();
            const index_t offset = (k * wconvConv.GetNumOutputChannels() + subK) * weight_stride +
                                   c * wconvConv.GetNumInputChannels() + subC;
            if constexpr(wconvConv.GetNumWeightComponents() * 32 >
                         wconvConv.GetNumInputChannels() * wconvConv.GetNumOutputChannels())
            {
                // only shape 8x4 consume 16 lanes
                static_assert(numWeightTile == 1, "");
                static_assert(HPerWconv == 8 && WPerWconv == 4, "unexpected shape!");
                if(weiCompIdx < wconvConv.GetNumInputChannels() * wconvConv.GetNumOutputChannels())
                {
                    weiData[tileOffset] = *reinterpret_cast<const WeiDataVec*>(wei + offset);
                }
            }
            else if constexpr(numWeightTile == 2)
            {
                // shape 4x2
                typename decltype(wconvConv)::WeiDataVec weiDataTmp;

                const index_t offsetBase =
                    offset + (lIdx % 2) * wconvConv.GetNumWeightComponents() / 2;
                weiDataTmp.template AsType<WeiDataTileVec>()(Number<0>{}) =
                    *reinterpret_cast<const WeiDataTileVec*>(wei + offsetBase);

                weiDataTmp.template AsType<WeiDataTileVec>()(Number<1>{}) =
                    *reinterpret_cast<const WeiDataTileVec*>(wei + offsetBase +
                                                             wconvConv.GetNumWeightComponents());

                weiData[tileOffset] = weiDataTmp.template AsType<WeiDataVec>()(Number<0>{});
            }
            else
            {
                // shape 4x4
                static_assert(numWeightTile == 1, "");
                weiData[tileOffset] = *reinterpret_cast<const WeiDataVec*>(wei + offset);
            }
        });
    });

    __syncthreads();

    // Do convolution
    static_for<0, KRepeat, 1>{}([&](auto k) {
        static_for<0, HRepeat, 1>{}([&](auto h) {
            static_for<0, WRepeat, 1>{}([&](auto w) {
#ifdef LOAD_DATA_PER_TILE
                AccVec c_thread_buf_ = {};
                auto& c_vec          = c_thread_buf_.GetVectorTypeReference(Number<0>{});
#else
                constexpr index_t tileOffset = h * WRepeat * KRepeat + w * KRepeat + k;
                auto& c_vec                  = c_thread_buf_.GetVectorTypeReference(
                    Number<tileOffset * wconvConv.GetNumAccumComponents()>{});
#endif
                static_for<0, CRepeat, 1>{}([&](auto c) {
#ifdef LOAD_DATA_PER_TILE
                    InDataVec in_tile_data = load_in_data(h, w, c);
#else
                    InDataVec& in_tile_data = inData[h * WRepeat * CRepeat + w * CRepeat + c];
#endif
                    const InDataVec* p_in_tile_data[1] = {&in_tile_data};
                    wconvConv.wconv_instr.Run(
                        weiData[k * CRepeat + c], p_in_tile_data, c_vec, Number<0>{});
                });
#ifdef LOAD_DATA_PER_TILE
                store_acc_data(h, w, k, c_vec);
#endif
            });
        });
    });

    __syncthreads();

#if !defined(LOAD_DATA_PER_TILE)
    // Output accum data
    static_for<0, HRepeat, 1>{}([&](auto h) {
        static_for<0, WRepeat, 1>{}([&](auto w) {
            static_for<0, KRepeat, 1>{}([&](auto k) {
                constexpr index_t tileOffset = h * WRepeat * KRepeat + w * KRepeat + k;
                auto& c_vec                  = c_thread_buf_.GetVectorTypeReference(
                    Number<tileOffset * wconvConv.GetNumAccumComponents()>{});
                store_acc_data(h, w, k, c_vec);
            });
        });
    });
#endif
}

template <typename InDataType,
          typename WeiDataType,
          typename AccDataType,
          bool UseF32I32,
          index_t HPerWconv,
          index_t WPerWconv,
          index_t FilterSize,
          index_t DilationX,
          index_t DilationY,
          index_t Width,
          index_t Height,
          index_t UnpackedInputChannels,
          index_t UnpackedOutputChannels>
__global__ void __launch_bounds__(64, 1)
    conv3_fwd(const InDataType* in_, const WeiDataType* wei_, AccDataType* c_)
{
    constexpr index_t DataTileHeight = 4;

    // conv3 always LOAD_DATA_PER_TILE. so marco LOAD_DATA_PER_TILE is ignored.
    constexpr auto wconvConv = ck::WconvConv<WeiDataType,
                                             InDataType,
                                             AccDataType,
                                             HPerWconv,
                                             WPerWconv,
                                             FilterSize,
                                             DilationX,
                                             DilationY,
                                             1,
                                             false,
                                             false,
                                             UseF32I32>();

    auto in  = reinterpret_cast<const typename decltype(wconvConv)::KernelInDataType*>(in_);
    auto wei = reinterpret_cast<const typename decltype(wconvConv)::KernelWeightDataType*>(wei_);

    static_assert(Width % WPerWconv == 0, "");
    static_assert(Height % HPerWconv == 0, "");
    static_assert(UnpackedInputChannels % wconvConv.GetUnpackedNumInputChannels() == 0, "");
    static_assert(UnpackedOutputChannels % wconvConv.GetUnpackedNumOutputChannels() == 0, "");

    constexpr index_t InputChannels = UnpackedInputChannels * wconvConv.GetNumInputChannels() /
                                      wconvConv.GetUnpackedNumInputChannels();
    constexpr index_t OutputChannels = UnpackedOutputChannels * wconvConv.GetNumOutputChannels() /
                                       wconvConv.GetUnpackedNumOutputChannels();
    constexpr index_t HRepeat = Height / HPerWconv;
    constexpr index_t WRepeat = Width / WPerWconv;
    constexpr index_t CRepeat = UnpackedInputChannels / wconvConv.GetUnpackedNumInputChannels();
    constexpr index_t KRepeat = UnpackedOutputChannels / wconvConv.GetUnpackedNumOutputChannels();

    using InDataVec     = typename decltype(wconvConv)::InDataVec::type;
    using InDataTileVec = typename decltype(wconvConv)::InDataTileVec::type;
    using WeiDataVec    = typename decltype(wconvConv)::WeiDataVec::type;

    using AccDataVec = typename decltype(wconvConv)::AccDataVec;
    using AccVec     = StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr,
                                             AccDataType,
                                             1,
                                             wconvConv.GetNumAccumComponents(),
                                             true>;
    const int lIdx   = threadIdx.x;

    // Data layout: HWC, unit: InDataType
    constexpr index_t data_H_stride = Width * InputChannels;
    constexpr index_t data_W_stride = InputChannels;
    constexpr index_t data_C_stride = wconvConv.GetNumInputChannels();

    // Filter layout: KYXC, unit: WeiDataType
    constexpr index_t weight_X_stride = InputChannels;
    constexpr index_t weight_Y_stride = FilterSize * InputChannels;
    constexpr index_t weight_K_stride = FilterSize * FilterSize * InputChannels;

    // Acc layout: HWK, unit: AccDataType
    constexpr index_t acc_H_stride = Width * OutputChannels;
    constexpr index_t acc_W_stride = OutputChannels;
    constexpr index_t acc_K_stride = wconvConv.GetNumOutputChannels();

    // Load inData
    const index_t compIdx =
        lIdx * wconvConv.GetNumDataComponents() / wconvConv.GetNumImageSubTilesInVertical();

    auto load_in_data = [&](index_t h, index_t w, index_t c, index_t tileOffset) {
        const index_t subC = compIdx % wconvConv.GetNumInputChannels();
        const index_t subW = (compIdx / wconvConv.GetNumInputChannels()) % WPerWconv;
        const index_t subH = (compIdx / wconvConv.GetNumInputChannels()) / WPerWconv;

        const index_t offset = (h * HPerWconv - DataTileHeight + subH) * data_H_stride +
                               (w * WPerWconv + tileOffset * WPerWconv + subW) * data_W_stride +
                               (c * data_C_stride + subC);

        static_assert(wconvConv.GetNumImageSubTilesInVertical() > 1);

        typename decltype(wconvConv)::InDataVec inData;

        static_for<0, wconvConv.GetNumImageSubTilesInVertical(), 1>{}([&](auto tileId) {
            const index_t tileH = h * HPerWconv - DataTileHeight + tileId * DataTileHeight;
            const index_t tileW = w * WPerWconv + tileOffset * WPerWconv;

            if((tileH >= 0) && (tileH + DataTileHeight <= Height) && (tileW >= 0) &&
               (tileW + WPerWconv <= Width))
            {
                inData.template AsType<InDataTileVec>()(Number<tileId>{}) =
                    *reinterpret_cast<const InDataTileVec*>(
                        in + offset + tileId * DataTileHeight * data_H_stride);
            }
            else
            {
                inData.template AsType<InDataTileVec>()(Number<tileId>{}) = {};
            }
        });
        return inData.template AsType<InDataVec>()(Number<0>{});
    };

    constexpr index_t num_acc_tile = wconvConv.GetNumSubTilesPerImageTile();
    const index_t accCompIdx       = lIdx * wconvConv.GetNumAccumComponents() / num_acc_tile;
    auto store_acc_data            = [&](index_t h, index_t w, index_t k, AccDataVec& c_vec) {
        const index_t subW = (accCompIdx / wconvConv.GetNumOutputChannels()) % WPerWconv;
        const index_t subH = (accCompIdx / wconvConv.GetNumOutputChannels()) / WPerWconv;

        if constexpr(wconvConv.GetNumAccumComponents() == 4)
        {
            const index_t subK   = accCompIdx % wconvConv.GetNumOutputChannels();
            const index_t offset = (h * HPerWconv + subH) * acc_H_stride +
                                   (w * WPerWconv + subW) * acc_W_stride +
                                   (k * acc_K_stride + subK);
            *reinterpret_cast<typename AccDataVec::type*>(c_ + offset) =
                c_vec.template AsType<typename AccDataVec::type>()(Number<0>{});
        }
        else
        {
            static_assert(wconvConv.GetNumAccumComponents() == 8, "unexpected value");
            using acc_swizzle_vec = typename vector_type<AccDataType, 4>::type;
            const index_t subK    = accCompIdx % wconvConv.GetNumOutputChannels() /
                                 (wconvConv.GetNumAccumComponents() * 2) *
                                 (wconvConv.GetNumAccumComponents() * 2);
            const index_t offset = (h * HPerWconv + subH) * acc_H_stride +
                                   (w * WPerWconv + subW) * acc_W_stride +
                                   (k * acc_K_stride + subK);

            constexpr index_t secOffset =
                (num_acc_tile > 1) ? HPerWconv / num_acc_tile * acc_H_stride : 8;
            const index_t swizzleOffset = (lIdx % 2) * 4;
            *reinterpret_cast<acc_swizzle_vec*>(c_ + offset + swizzleOffset) =
                c_vec.template AsType<acc_swizzle_vec>()(Number<0>{});
            *reinterpret_cast<acc_swizzle_vec*>(c_ + offset + swizzleOffset + secOffset) =
                c_vec.template AsType<acc_swizzle_vec>()(Number<1>{});
        }
    };

    // Load weight
    constexpr index_t numWeightTile = wconvConv.GetWeightRegSize();
    using WeiDataTileVec            = typename decltype(wconvConv)::WeiDataTileVec::type;

    auto load_weight = [&](index_t k, index_t c) {
        constexpr index_t tapeIdx[]                      = {4, 1, 0, 3, 6, 7, 8, 5, 2};
        typename decltype(wconvConv)::WeiDataVec weiData = {};
        if constexpr(HPerWconv == 8)
        {
            // shape 8 x 4
            const index_t weiCompIdx =
                (lIdx % 16) * wconvConv.GetNumWeightComponents() / numWeightTile;
            const index_t subC = weiCompIdx % wconvConv.GetNumInputChannels();
            const index_t subK = weiCompIdx / wconvConv.GetNumInputChannels();

            if(lIdx < 16)
            {
                static_for<0, 5, 1>{}([&](auto id) {
                    constexpr index_t tapeId = 2 * id;
                    constexpr index_t subX   = tapeIdx[tapeId] % 3;
                    constexpr index_t subY   = tapeIdx[tapeId] / 3;
                    const index_t offset =
                        (k * wconvConv.GetNumOutputChannels() + subK) * weight_K_stride +
                        subY * weight_Y_stride + subX * weight_X_stride + subC +
                        c * wconvConv.GetNumInputChannels();

                    weiData.template AsType<WeiDataTileVec>()(Number<tapeId / 2>{}) =
                        *reinterpret_cast<const WeiDataTileVec*>(wei + offset);
                });
            }
            else
            {
                static_for<0, 4, 1>{}([&](auto id) {
                    constexpr index_t tapeId = 2 * id + 1;
                    constexpr index_t subX   = tapeIdx[tapeId] % 3;
                    constexpr index_t subY   = tapeIdx[tapeId] / 3;
                    const index_t offset =
                        (k * wconvConv.GetNumOutputChannels() + subK) * weight_K_stride +
                        subY * weight_Y_stride + subX * weight_X_stride + subC +
                        c * wconvConv.GetNumInputChannels();

                    weiData.template AsType<WeiDataTileVec>()(Number<tapeId / 2>{}) =
                        *reinterpret_cast<const WeiDataTileVec*>(wei + offset);
                });
            }
        }
        else if constexpr(WPerWconv == 2)
        {
            // shape 4 x2
            static_assert(wconvConv.GetNumWeightComponents() % numWeightTile == 0, "");
            const index_t weiCompIdx = lIdx / 2 * 2 * wconvConv.GetNumWeightComponents() / 9;
            const index_t subC       = weiCompIdx % wconvConv.GetNumInputChannels();
            const index_t subK       = weiCompIdx / wconvConv.GetNumInputChannels();

            static_for<0, 9, 1>{}([&](auto tapeId) {
                constexpr index_t subX = tapeIdx[tapeId] % 3;
                constexpr index_t subY = tapeIdx[tapeId] / 3;
                const index_t offset =
                    (k * wconvConv.GetNumOutputChannels() + subK) * weight_K_stride +
                    subY * weight_Y_stride + subX * weight_X_stride + subC +
                    c * wconvConv.GetNumInputChannels() +
                    (lIdx % 2) * wconvConv.GetNumWeightComponents() / 9 / 2;

                weiData.template AsType<WeiDataTileVec>()(Number<tapeId * 2>{}) =
                    *reinterpret_cast<const WeiDataTileVec*>(wei + offset);
                weiData.template AsType<WeiDataTileVec>()(Number<tapeId * 2 + 1>{}) =
                    *reinterpret_cast<const WeiDataTileVec*>(
                        wei + offset + wconvConv.GetNumWeightComponents() / numWeightTile * 2);
            });
        }
        else
        {
            const index_t weiCompIdx = lIdx * wconvConv.GetNumWeightComponents() / numWeightTile;
            const index_t subC       = weiCompIdx % wconvConv.GetNumInputChannels();
            const index_t subK       = weiCompIdx / wconvConv.GetNumInputChannels();

            static_for<0, 9, 1>{}([&](auto tapeId) {
                constexpr index_t subX = tapeIdx[tapeId] % 3;
                constexpr index_t subY = tapeIdx[tapeId] / 3;
                const index_t offset =
                    (k * wconvConv.GetNumOutputChannels() + subK) * weight_K_stride +
                    subY * weight_Y_stride + subX * weight_X_stride + subC +
                    c * wconvConv.GetNumInputChannels();

                weiData.template AsType<WeiDataTileVec>()(Number<tapeId>{}) =
                    *reinterpret_cast<const WeiDataTileVec*>(wei + offset);
            });
        }

        return weiData.template AsType<WeiDataVec>()(Number<0>{});
    };

    __syncthreads();

    // Do convolution
    static_for<0, KRepeat, 1>{}([&](auto k) {
        WeiDataVec weiData[CRepeat];
        static_for<0, CRepeat, 1>{}([&](auto c) { weiData[c] = load_weight(k, c); });
        static_for<0, HRepeat, 1>{}([&](auto h) {
            static_for<0, WRepeat, 1>{}([&](auto w) {
                AccVec c_thread_buf_ = {};
                auto& c_vec          = c_thread_buf_.GetVectorTypeReference(Number<0>{});
                static_for<0, CRepeat, 1>{}([&](auto c) {
                    InDataVec inData[3];
                    inData[0]                         = load_in_data(h, w, c, -1);
                    inData[1]                         = load_in_data(h, w, c, 0);
                    inData[2]                         = load_in_data(h, w, c, 1);
                    const InDataVec* p_in_data_vec[3] = {&inData[0], &inData[1], &inData[2]};
                    wconvConv.wconv_instr.Run(weiData[c], p_in_data_vec, c_vec, Number<0>{});
                });
                store_acc_data(h, w, k, c_vec);
            });
        });
    });
}
#pragma clang diagnostic pop
}; // namespace conv_op_util
}; // namespace ck

struct ExecutionConfig final
{
    uint32_t test_mask   = 0xffffffff;
    int init_method      = 1;
    bool do_verification = true;
    bool dump_tensor     = true;
};
static ExecutionConfig config;

template <typename DataType>
void DumpTensor(const Tensor<DataType>& tensor, const char* str)
{
    if(config.dump_tensor == false)
        return;
    assert(tensor.GetNumOfDimension() == 5);
    auto lengths = tensor.GetLengths();
    std::cout << str << "  [ " << std::endl;
    for(uint32_t i0 = 0; i0 < lengths[0]; i0++)
    {
        if(lengths[1] > 1)
        {
            std::cout << "  [";
        }
        for(uint32_t i1 = 0; i1 < lengths[1]; i1++)
        {
            if(lengths[2] > 1)
            {
                std::cout << "  [";
            }
            for(uint32_t i2 = 0; i2 < lengths[2]; i2++)
            {
                if(lengths[3] > 1)
                {
                    std::cout << "  [";
                }
                for(uint32_t i3 = 0; i3 < lengths[3]; i3++)
                {
                    if(lengths[4] > 1)
                    {
                        std::cout << "  [";
                    }
                    for(uint32_t i4 = 0; i4 < lengths[4]; i4++)
                    {
                        std::vector<std::size_t> idx({i0, i1, i2, i3, i4});
                        std::cout << ck::type_convert<float>(tensor(idx)) << ", ";
                    }
                    if(lengths[4] > 1)
                    {
                        std::cout << "]";
                    }
                    if(lengths[4] > 3)
                    {
                        std::cout << std::endl;
                    }
                }
                if(lengths[3] > 1)
                {
                    std::cout << "]" << std::endl;
                }
            }
            if(lengths[2] > 1)
            {
                std::cout << "]" << std::endl;
            }
        }
        if(lengths[1] > 1)
        {
            std::cout << "]" << std::endl;
        }
    }
    std::cout << "]" << std::endl;
}

template <typename DataType>
inline constexpr double get_rtol()
{
    if constexpr(std::is_same_v<DataType, float>)
    {
        return 1e-3;
    }
    else if constexpr(std::is_same_v<DataType, double>)
    {
        return 1e-6;
    }
    else if constexpr(std::is_same_v<DataType, ck::half_t>)
    {
        return 1e-3;
    }
    else if constexpr(std::is_same_v<DataType, ck::bhalf_t>)
    {
        return 5e-2;
    }
    else if constexpr(std::is_same_v<DataType, int32_t>)
    {
        return 1e-1;
    }
    else if constexpr(std::is_same_v<DataType, int8_t>)
    {
        return 1e-1;
    }
    else if constexpr(std::is_same_v<DataType, ck::f8_t>)
    {
        return 1e-1; // 240 and 224 are acceptable
    }
    else if constexpr(std::is_same_v<DataType, ck::bf8_t>)
    {
        return 1.5e-1; // 57344 and 49152 are acceptable
    }
    else
    {
        return 1e-3;
    }
}

template <typename DataType>
inline constexpr double get_atol()
{
    if constexpr(std::is_same_v<DataType, float>)
    {
        return 1e-3;
    }
    else if constexpr(std::is_same_v<DataType, double>)
    {
        return 1e-6;
    }
    else if constexpr(std::is_same_v<DataType, ck::half_t>)
    {
        return 1e-3;
    }
    else if constexpr(std::is_same_v<DataType, ck::bhalf_t>)
    {
        return 5e-2;
    }
    else if constexpr(std::is_same_v<DataType, int32_t>)
    {
        return 1e-1;
    }
    else if constexpr(std::is_same_v<DataType, int8_t>)
    {
        return 1e-1;
    }
    else if constexpr(std::is_same_v<DataType, ck::f8_t>)
    {
        return 16.1; // 240 and 224 are acceptable
    }
    else if constexpr(std::is_same_v<DataType, ck::bf8_t>)
    {
        return 8192.1; // 57344 and 49152 are acceptable
    }
    else
    {
        return 1e-3;
    }
}

template <typename Type>
const char* get_string()
{
    if constexpr(std::is_same<Type, ck::half_t>::value)
    {
        return "half_t";
    }

    if constexpr(std::is_same<Type, float>::value)
    {
        return "float";
    }

    if constexpr(std::is_same<Type, ck::bhalf_t>::value)
    {
        return "bhalf_t";
    }

    if constexpr(std::is_same<Type, ck::f8_t>::value)
    {
        return "f8_t";
    }

    if constexpr(std::is_same<Type, ck::bf8_t>::value)
    {
        return "bf8_t";
    }

    if constexpr(std::is_same<Type, int8_t>::value)
    {
        return "int8_t";
    }

    if constexpr(std::is_same<Type, int32_t>::value)
    {
        return "int32_t";
    }

    if constexpr(std::is_same<Type, uint8_t>::value)
    {
        return "uint8_t";
    }

    if constexpr(std::is_same<Type, uint32_t>::value)
    {
        return "uint32_t";
    }

#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
    if constexpr(std::is_same<Type, ck::int4_t>::value)
    {
        return "int4_t";
    }

    if constexpr(std::is_same<Type, ck::uint4_t>::value)
    {
        return "uint4_t";
    }
#endif
}

const char* get_string(ShapeType type)
{
    switch(type)
    {
    case Shape_4X2: return "Shape_4x2";
    case Shape_4X4: return "Shape_4x4";
    case Shape_8X4: return "Shape_8x4";
    }
}

const char* get_string(FilterType filter)
{
    switch(filter)
    {
    case Filter_1X1: return "Filter_1X1";
    case Filter_3X3: return "Filter_3X3";
    case Filter_2X2: return "Filter_2X2";
    }
}

template <typename InDataType,
          typename WeiDataType,
          typename GPUAccType,
          bool UseF32I32,
          ShapeType Shape,
          FilterType Filter,
          bool Dilation,
          int32_t TestMask>
bool run_test()
{
    if((config.test_mask & 0xFFFF0000 & TestMask) == 0)
    {
        return true;
    }
    constexpr ck::index_t FilterSize   = (Filter == Filter_1X1) ? 1 : 3;
    constexpr ck::index_t HPerWconv    = (Shape == Shape_8X4) ? 8 : 4;
    constexpr ck::index_t WPerWconv    = (Shape == Shape_4X2) ? 2 : 4;
    constexpr ck::index_t DilationSize = Dilation ? 2 : 1;

#ifdef USE_ABSOLUTE_SIZE
    constexpr bool CheckVgpr1024         = (TestMask & 0xC000) != 0;
    constexpr ck::index_t Width          = CheckVgpr1024 ? DEFAULT_W_1K : DEFAULT_W;
    constexpr ck::index_t Height         = CheckVgpr1024 ? DEFAULT_H_1K : DEFAULT_H;
    constexpr ck::index_t InputChannels  = DEFAULT_C;
    constexpr ck::index_t OutputChannels = DEFAULT_K;
#else
    constexpr auto wconvConv     = ck::WconvConv<WeiDataType,
                                             InDataType,
                                             GPUAccType,
                                             HPerWconv,
                                             WPerWconv,
                                             FilterSize,
                                             DilationSize,
                                             DilationSize,
                                             1,
                                             false,
                                             false,
                                             UseF32I32>();
    constexpr ck::index_t Width  = WPerWconv * DEFAULT_W_REPEAT;
    constexpr ck::index_t Height = HPerWconv * DEFAULT_H_REPEAT;
    constexpr ck::index_t InputChannels =
        wconvConv.GetUnpackedNumInputChannels() * DEFAULT_C_REPEAT;
    constexpr ck::index_t OutputChannels =
        wconvConv.GetUnpackedNumOutputChannels() * DEFAULT_K_REPEAT;
#endif
    constexpr ck::index_t n_dim          = 2;
    constexpr ck::index_t group_count    = 1;
    constexpr ck::index_t n_batch        = 1;
    constexpr ck::index_t n_out_channels = OutputChannels;
    constexpr ck::index_t n_in_channels  = InputChannels;

    const std::vector<ck::index_t> filters_1x1{1, 1};
    const std::vector<ck::index_t> filters_3x3{3, 3};
    const std::vector<ck::index_t> dilations_1{1, 1};
    const std::vector<ck::index_t> dilations_2{2, 2};
    const std::vector<ck::index_t> pads_0{0, 0};
    const std::vector<ck::index_t> pads_1{1, 1};
    const std::vector<ck::index_t> pads_2{2, 2};

    const std::vector<ck::index_t>& filters_len =
        (Filter == Filter_1X1) ? filters_1x1 : filters_3x3;
    const std::vector<ck::index_t> input_len = {Height, Width};
    const std::vector<ck::index_t> strides{1, 1};
    const std::vector<ck::index_t>& dilations  = Dilation ? dilations_2 : dilations_1;
    const std::vector<ck::index_t>& left_pads  = (Filter == Filter_1X1) ? pads_0
                                                 : Dilation             ? pads_2
                                                                        : pads_1;
    const std::vector<ck::index_t>& right_pads = (Filter == Filter_1X1) ? pads_0
                                                 : Dilation             ? pads_2
                                                                        : pads_1;

    ck::utils::conv::ConvParam conv_param{n_dim,
                                          group_count,
                                          n_batch,
                                          n_out_channels,
                                          n_in_channels,
                                          filters_len,
                                          input_len,
                                          strides,
                                          dilations,
                                          left_pads,
                                          right_pads};

    constexpr auto NDimSpatial = ck::Number<n_dim>{};
    const auto in_element_op   = InElementOp{};
    const auto wei_element_op  = WeiElementOp{};
    const auto out_element_op  = OutElementOp{};

    namespace ctc   = ck::tensor_layout::convolution;
    auto in_layout  = ctc::GNHWC{};
    auto wei_layout = ctc::GKYXC{};
    auto out_layout = ctc::GNHWK{};
    using InLayout  = decltype(in_layout);
    using WeiLayout = decltype(wei_layout);
    using OutLayout = decltype(out_layout);

    const auto in_g_n_c_wis_desc =
        ck::utils::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<InLayout>(conv_param);

    const auto wei_g_k_c_xs_desc =
        ck::utils::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<WeiLayout>(conv_param);

    const auto out_g_n_k_wos_desc =
        ck::utils::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<OutLayout>(conv_param);

    Tensor<InDataType> in(in_g_n_c_wis_desc);
    Tensor<WeiDataType> wei(wei_g_k_c_xs_desc);
    Tensor<GPUAccType> out_host(out_g_n_k_wos_desc);
    Tensor<GPUAccType> out_device(out_g_n_k_wos_desc);

    std::cout << "in: " << in.mDesc << std::endl;
    std::cout << "wei: " << wei.mDesc << std::endl;
    std::cout << "out: " << out_host.mDesc << std::endl;

    switch(config.init_method)
    {
    case 0: break;
    case 1:
        in.GenerateTensorValue(GeneratorTensor_2<InDataType>{-5, 5});
        wei.GenerateTensorValue(GeneratorTensor_2<WeiDataType>{-5, 5});
        break;
    default:
        in.GenerateTensorValue(GeneratorTensor_3<InDataType>{0.0, 1.0});
        wei.GenerateTensorValue(GeneratorTensor_3<WeiDataType>{-0.5, 0.5});
        break;
    }

    std::array<ck::index_t, NDimSpatial + 3> a_g_n_c_wis_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> a_g_n_c_wis_strides{};
    std::array<ck::index_t, NDimSpatial + 3> b_g_k_c_xs_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> b_g_k_c_xs_strides{};
    std::array<ck::index_t, NDimSpatial + 3> e_g_n_k_wos_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> e_g_n_k_wos_strides{};
    std::array<ck::index_t, NDimSpatial> conv_filter_strides{};
    std::array<ck::index_t, NDimSpatial> conv_filter_dilations{};
    std::array<ck::index_t, NDimSpatial> input_left_pads{};
    std::array<ck::index_t, NDimSpatial> input_right_pads{};

    auto copy = [](const auto& x, auto& y) { ck::ranges::copy(x, y.begin()); };

    copy(in_g_n_c_wis_desc.GetLengths(), a_g_n_c_wis_lengths);
    copy(in_g_n_c_wis_desc.GetStrides(), a_g_n_c_wis_strides);
    copy(wei_g_k_c_xs_desc.GetLengths(), b_g_k_c_xs_lengths);
    copy(wei_g_k_c_xs_desc.GetStrides(), b_g_k_c_xs_strides);
    copy(out_g_n_k_wos_desc.GetLengths(), e_g_n_k_wos_lengths);
    copy(out_g_n_k_wos_desc.GetStrides(), e_g_n_k_wos_strides);
    copy(conv_param.conv_filter_strides_, conv_filter_strides);
    copy(conv_param.conv_filter_dilations_, conv_filter_dilations);
    copy(conv_param.input_left_pads_, input_left_pads);
    copy(conv_param.input_right_pads_, input_right_pads);

    auto ref_conv = ck::tensor_operation::host::ReferenceConvFwd<NDimSpatial,
                                                                 InDataType,
                                                                 WeiDataType,
                                                                 GPUAccType,
                                                                 InElementOp,
                                                                 WeiElementOp,
                                                                 OutElementOp>();

    auto ref_invoker  = ref_conv.MakeInvoker();
    auto ref_argument = ref_conv.MakeArgument(in,
                                              wei,
                                              out_host,
                                              conv_param.conv_filter_strides_,
                                              conv_param.conv_filter_dilations_,
                                              conv_param.input_left_pads_,
                                              conv_param.input_right_pads_,
                                              in_element_op,
                                              wei_element_op,
                                              out_element_op);

    if(config.do_verification)
    {
        ref_invoker.Run(ref_argument);
    }

    DumpTensor(in, "Input");
    DumpTensor(wei, "Weight");
    DumpTensor(out_host, "Accum");

    DeviceMem in_device_buf(sizeof(InDataType) * in.mDesc.GetElementSpaceSize());
    DeviceMem wei_device_buf(sizeof(WeiDataType) * wei.mDesc.GetElementSpaceSize());
    DeviceMem out_device_buf(sizeof(GPUAccType) * out_device.mDesc.GetElementSpaceSize());

    if constexpr(std::is_same<ck::int4_t, InDataType>::value)
    {
        std::vector<uint8_t> in_packed;
        std::vector<uint8_t> wei_packed;
        in_packed.resize(in.mData.size());
        wei_packed.resize(wei.mData.size());
        for(size_t i = 0; i < in.mData.size(); i += 2)
        {
            uint8_t val0 = (in.mData[i] & 0xf);
            uint8_t val1 = (in.mData[i + 1] & 0xf);

            in_packed[i / 2] = val0 | (val1 << 4);
        }
        for(size_t i = 0; i < wei.mData.size(); i += 2)
        {
            uint8_t val0      = (wei.mData[i] & 0xf);
            uint8_t val1      = (wei.mData[i + 1] & 0xf);
            wei_packed[i / 2] = val0 | (val1 << 4);
        }
        in_device_buf.ToDevice(in_packed.data());
        wei_device_buf.ToDevice(wei_packed.data());
    }
    else
    {
        in_device_buf.ToDevice(in.mData.data());
        wei_device_buf.ToDevice(wei.mData.data());
    }

    if constexpr(FilterSize == 1)
    {
        auto conv_kernel = ck::conv_op_util::conv_fwd<InDataType,
                                                      WeiDataType,
                                                      GPUAccType,
                                                      UseF32I32,
                                                      HPerWconv,
                                                      WPerWconv,
                                                      FilterSize,
                                                      DilationSize,
                                                      DilationSize,
                                                      Width,
                                                      Height,
                                                      InputChannels,
                                                      OutputChannels>;
        conv_kernel<<<1, 32>>>(static_cast<const InDataType*>(in_device_buf.GetDeviceBuffer()),
                               static_cast<const WeiDataType*>(wei_device_buf.GetDeviceBuffer()),
                               static_cast<GPUAccType*>(out_device_buf.GetDeviceBuffer()));
    }
    else
    {
        auto conv_kernel = ck::conv_op_util::conv3_fwd<InDataType,
                                                       WeiDataType,
                                                       GPUAccType,
                                                       UseF32I32,
                                                       HPerWconv,
                                                       WPerWconv,
                                                       FilterSize,
                                                       DilationSize,
                                                       DilationSize,
                                                       Width,
                                                       Height,
                                                       InputChannels,
                                                       OutputChannels>;
        conv_kernel<<<1, 32>>>(static_cast<const InDataType*>(in_device_buf.GetDeviceBuffer()),
                               static_cast<const WeiDataType*>(wei_device_buf.GetDeviceBuffer()),
                               static_cast<GPUAccType*>(out_device_buf.GetDeviceBuffer()));
    }

    out_device_buf.FromDevice(out_device.mData.data());

    DumpTensor(out_device, "Accum_Device");
    std::cout << "conv_op<In/Wei:" << get_string<InDataType>()
              << ", Out:" << get_string<GPUAccType>() << ", " << get_string(Shape) << ", "
              << get_string(Filter) << ", Dilation:" << DilationSize << ", Id:0x" << std::hex
              << TestMask << ">: Status: ";

    if(config.do_verification)
    {
        bool ret = ck::utils::check_err(out_device,
                                        out_host,
                                        "Error: incorrect results!",
                                        get_rtol<GPUAccType>(),
                                        get_atol<GPUAccType>());
        if(ret)
        {
            std::cout << "Passed\n";
        }
        else
        {
            std::cout << "Failed\n";
        }

        return ret;
    }
    else
    {
        return true;
    }
}

template <typename InDataType,
          typename WeiDataType,
          typename GPUAccType,
          bool UseF32I32,
          int32_t TestMask>
bool run_test_fmt()
{
    if((config.test_mask & TestMask) == 0)
    {
        return true;
    }
    bool pass = true;
    // clang-format off
    //                                                                  |ShapeType |FilterType |Dilation |TestMask
    if constexpr(std::is_same<GPUAccType, float>::value || std::is_same<GPUAccType, int32_t>::value)
    {
        pass &= run_test<InDataType, WeiDataType, GPUAccType, UseF32I32, Shape_4X2, Filter_1X1, false, TestMask | 0x10000  >();
        pass &= run_test<InDataType, WeiDataType, GPUAccType, UseF32I32, Shape_4X2, Filter_3X3, false, TestMask | 0x20000  >();
        pass &= run_test<InDataType, WeiDataType, GPUAccType, UseF32I32, Shape_4X2, Filter_3X3, true,  TestMask | 0x40000  >();
    }
    else
    {
        pass &= run_test<InDataType, WeiDataType, GPUAccType, UseF32I32, Shape_4X2, Filter_1X1, false, TestMask | 0x80000  >();
        pass &= run_test<InDataType, WeiDataType, GPUAccType, UseF32I32, Shape_4X4, Filter_1X1, false, TestMask | 0x100000 >();
        pass &= run_test<InDataType, WeiDataType, GPUAccType, UseF32I32, Shape_8X4, Filter_1X1, false, TestMask | 0x200000 >();
        pass &= run_test<InDataType, WeiDataType, GPUAccType, UseF32I32, Shape_4X2, Filter_3X3, false, TestMask | 0x400000 >();
        pass &= run_test<InDataType, WeiDataType, GPUAccType, UseF32I32, Shape_4X4, Filter_3X3, false, TestMask | 0x800000 >();
        pass &= run_test<InDataType, WeiDataType, GPUAccType, UseF32I32, Shape_8X4, Filter_3X3, false, TestMask | 0x1000000>();
        pass &= run_test<InDataType, WeiDataType, GPUAccType, UseF32I32, Shape_4X2, Filter_3X3, true,  TestMask | 0x2000000>();
        pass &= run_test<InDataType, WeiDataType, GPUAccType, UseF32I32, Shape_4X4, Filter_3X3, true,  TestMask | 0x4000000>();
        pass &= run_test<InDataType, WeiDataType, GPUAccType, UseF32I32, Shape_8X4, Filter_3X3, true,  TestMask | 0x8000000>();
    }
    // clang-format on

    return pass;
}

inline void print_help_msg()
{
    std::cerr << "arg1: test mask (hex)\n"
              << "arg2: verification (0=no, 1=yes)\n"
              << "arg3: dump tensor (0=no, 1=yes)\n"
              << "arg4: initialization (0=no init, 1=integer value, 2=decimal value)\n";
}

inline bool parse_cmd_args(int argc, char* argv[], ExecutionConfig& cfg)
{
    if(argc == 1)
    {
        // use default
    }
    else if(argc <= 5)
    {
        if(argc > 1)
        {
            cfg.test_mask = std::stoul(argv[1], nullptr, 0);
        }
        if(argc > 2)
        {
            cfg.do_verification = std::stoi(argv[2]);
        }
        if(argc > 3)
        {
            cfg.dump_tensor = std::stoi(argv[3]);
        }
        if(argc > 4)
        {
            cfg.init_method = std::stoi(argv[4]);
        }
    }
    else
    {
        print_help_msg();
        return false;
    }

    return true;
}

int main(int argc, char* argv[])
{
    bool pass = true;
    if(parse_cmd_args(argc, argv, config) == false)
    {
        return -1;
    }

    // clang-format off
    //                  |InDataType  |WeiDataType |GPUAccType F32I32
    pass &= run_test_fmt<ck::half_t,  ck::half_t, float,       false, 0x1   >();
    pass &= run_test_fmt<ck::bhalf_t, ck::bhalf_t,float,       false, 0x2   >();
    pass &= run_test_fmt<ck::f8_t,    ck::f8_t,   float,       false, 0x4   >();
    pass &= run_test_fmt<ck::bf8_t,   ck::f8_t,   float,       false, 0x4   >();
    pass &= run_test_fmt<ck::bf8_t,   ck::bf8_t,  float,       false, 0x8   >();
    pass &= run_test_fmt<ck::f8_t,    ck::bf8_t,  float,       false, 0x8   >();
    pass &= run_test_fmt<int8_t,      int8_t,     float,       false, 0x10  >();
    //crash Cannot select: intrinsic %llvm.amdgcn.convolve.f32i32.iu8.1x1
    //pass &= run_test_fmt<int8_t,      int8_t,     float,       true,  0x10  >();
    pass &= run_test_fmt<int8_t,      int8_t,     int32_t,     false, 0x20  >();

    pass &= run_test_fmt<ck::half_t,  ck::half_t, ck::half_t,  false, 0x40  >();
    pass &= run_test_fmt<ck::bhalf_t, ck::bhalf_t,ck::bhalf_t, false, 0x80  >();
    pass &= run_test_fmt<ck::f8_t,    ck::f8_t,   ck::half_t,  false, 0x100 >();
    pass &= run_test_fmt<ck::bf8_t,   ck::f8_t,   ck::half_t,  false, 0x100 >();
    pass &= run_test_fmt<ck::bf8_t,   ck::bf8_t,  ck::half_t,  false, 0x200 >();
    pass &= run_test_fmt<ck::f8_t,    ck::bf8_t,  ck::half_t,  false, 0x200 >();
    pass &= run_test_fmt<int8_t,      int8_t,     ck::half_t,  false, 0x400 >();
    pass &= run_test_fmt<ck::half_t,  ck::half_t, float,       false, 0x4000>();
    pass &= run_test_fmt<ck::half_t,  ck::half_t, ck::half_t,  false, 0x8000>();
#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
    pass &= run_test_fmt<ck::int4_t,  ck::int4_t, float,       false, 0x800 >();
    //crash intrinsic %llvm.amdgcn.convolve.f32i32.iu4.1x1
    //pass &= run_test_fmt<ck::int4_t,  ck::int4_t, float,       true,  0x800 >();
    pass &= run_test_fmt<ck::int4_t,  ck::int4_t, int32_t,     false, 0x1000>();
    pass &= run_test_fmt<ck::int4_t,  ck::int4_t, ck::half_t,  false, 0x2000>();
#endif
    // clang-format on

    std::cout << "conv_op: ..... " << (pass ? "SUCCESS" : "FAILURE") << std::endl;
    return pass ? 0 : 1;
}
