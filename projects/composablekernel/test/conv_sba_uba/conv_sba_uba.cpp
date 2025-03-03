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
#include "ck/tensor_operation/gpu/warp/acc_sba.hpp"
#include "ck/tensor_operation/gpu/warp/acc_cvt_tensor.hpp"
#include <iostream>
using namespace std;

using InElementOp         = ck::tensor_operation::element_wise::PassThrough;
using WeiElementOp        = ck::tensor_operation::element_wise::PassThrough;
using OutElementNoneOp    = ck::tensor_operation::element_wise::MultiplyAdd;
using OutElementReluOp    = ck::tensor_operation::element_wise::MultiplyAddRelu<>;
using OutElementTanhOp    = ck::tensor_operation::element_wise::MultiplyAddHardTanh;
using ActivationOp        = ck::tensor_operation::element_wise::PassThrough;
using OutElementConvertOp = ck::tensor_operation::element_wise::Activation_Mul_Clamp<ActivationOp>;

//#define ENABLE_FULL_TEST 1
//
// #define USE_ABSOLUTE_SIZE 1
#ifdef USE_ABSOLUTE_SIZE
#define DEFAULT_K 16
#define DEFAULT_C 32
#define DEFAULT_W 8
#define DEFAULT_H 16
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

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wundefined-reinterpret-cast"
template <typename InDataType,
          typename WeiDataType,
          typename AccDataType,
          typename EDataType,
          index_t HPerWconv,
          index_t WPerWconv,
          index_t FilterSize,
          index_t DilationX,
          index_t DilationY,
          index_t Width,
          index_t Height,
          index_t UnpackedInputChannels,
          index_t UnpackedOutputChannels,
          index_t activateFunc,
          index_t convertScale,
          bool scaleBiasPacked,
          bool uniformScale,
          bool convert_to_tensor>
__global__ void __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
    conv_fwd(const InDataType* in_,
             const WeiDataType* wei_,
             AccDataType* c_,
             AccDataType* scale_,
             AccDataType* bias_,
             EDataType* outTensor_)
{
    constexpr auto wconvConv = ck::WconvConv<WeiDataType,
                                             InDataType,
                                             AccDataType,
                                             HPerWconv,
                                             WPerWconv,
                                             FilterSize,
                                             DilationX,
                                             DilationY>();

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
    constexpr index_t AccVectorCount = HRepeat * WRepeat * KRepeat;

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

    const int lIdx                                = threadIdx.x;
    InDataVec inData[HRepeat * WRepeat * CRepeat] = {};
    AccVec c_thread_buf_                          = {};
    WeiDataVec weiData[KRepeat * CRepeat]         = {};

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
            typename decltype(wconvConv)::InDataVec inDataLocal;

            static_for<0, wconvConv.GetNumImageSubTilesInVertical(), 1>{}([&](auto tileId) {
                inDataLocal.template AsType<InDataTileVec>()(Number<tileId>{}) =
                    *reinterpret_cast<const InDataTileVec*>(
                        in + offset + tileId * DataTileHeight * data_H_stride);
            });
            return inDataLocal.template AsType<InDataVec>()(Number<0>{});
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

    static_for<0, HRepeat, 1>{}([&](auto h) {
        static_for<0, WRepeat, 1>{}([&](auto w) {
            static_for<0, CRepeat, 1>{}([&](auto c) {
                constexpr index_t tileOffset = h * WRepeat * CRepeat + w * CRepeat + c;
                inData[tileOffset]           = load_in_data(h, w, c);
            });
        });
    });

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

    auto load_bias_32bit_data = [&](auto k0) {
        // To do calculate stride with acc_k.
        index_t offset = 0;
        if constexpr(std::is_same<float, AccDataType>::value)
        {
            if(scaleBiasPacked)
            {
                offset = lIdx / 2;
                if(lIdx % 2)
                {
                    return *reinterpret_cast<const AccDataType*>(
                        scale_ + offset + k0 * wconvConv.GetNumOutputChannels());
                }
                else
                {
                    return *reinterpret_cast<const AccDataType*>(
                        bias_ + offset + k0 * wconvConv.GetNumOutputChannels());
                }
            }
            else
            {
                offset = lIdx;
                return *reinterpret_cast<const AccDataType*>(bias_ + offset +
                                                             k0 * wconvConv.GetNumOutputChannels());
            }
        }
    };

    auto load_bias_16bit_data = [&](auto k0) {
        auto bias_16bit_unpack_data = vector_type<AccDataType, 2>{};
        if constexpr(std::is_same<half_t, AccDataType>::value ||
                     std::is_same<bhalf_t, AccDataType>::value)
        {
            if constexpr(scaleBiasPacked)
            {
                // workaround for load_16bit is incorrect in ffm currently.
                index_t offset      = lIdx;
                uint16_t scale_data = *reinterpret_cast<const uint16_t*>(
                    scale_ + offset + k0 * wconvConv.GetNumOutputChannels());
                uint16_t bias_data = *reinterpret_cast<const uint16_t*>(
                    bias_ + offset + k0 * wconvConv.GetNumOutputChannels());
                uint32_t scale_bias    = (((scale_data & 0xffff) << 16) | (bias_data & 0xffff));
                bias_16bit_unpack_data = bit_cast<vector_type<AccDataType, 2>>(scale_bias);
            }
            else
            {
                index_t offset = lIdx * 2;
                bias_16bit_unpack_data.template AsType<AccDataType>()(Number<0>{}) =
                    *reinterpret_cast<const AccDataType*>(bias_ + offset +
                                                          k0 * wconvConv.GetNumOutputChannels());
                bias_16bit_unpack_data.template AsType<AccDataType>()(Number<1>{}) =
                    *reinterpret_cast<const AccDataType*>(bias_ + offset + 1 +
                                                          k0 * wconvConv.GetNumOutputChannels());
            }
        }
        return bias_16bit_unpack_data;
    };

    __syncthreads();

    // load scale and bias
    AccDataType scale_data[KRepeat]                      = {};
    AccDataType bias_32bit_data[KRepeat]                 = {};
    vector_type<AccDataType, 2> bias_16bit_data[KRepeat] = {};

    static_for<0, KRepeat, 1>{}([&](auto k) {
        constexpr index_t tileOffset = k;
        if constexpr(std::is_same<half_t, AccDataType>::value ||
                     std::is_same<bhalf_t, AccDataType>::value)
        {
            // workaround for 16bit by load_global_b32 and bitOp
            bias_16bit_data[tileOffset] = load_bias_16bit_data(k);
            // uniform scale
            scale_data[tileOffset] = bit_cast<AccDataType>(
                type_convert<ushort>((*reinterpret_cast<const int*>(scale_) & 0xffff)));
        }
        else if constexpr(std::is_same<float, AccDataType>::value)
        {
            bias_32bit_data[tileOffset] = load_bias_32bit_data(k);
            // uniform scale
            scale_data[tileOffset] = *reinterpret_cast<const AccDataType*>(scale_);
        }
    });

    __syncthreads();

    // Do convolution
    static_for<0, KRepeat, 1>{}([&](auto k) {
        static_for<0, HRepeat, 1>{}([&](auto h) {
            static_for<0, WRepeat, 1>{}([&](auto w) {
                constexpr index_t tileOffset = h * WRepeat * KRepeat + w * KRepeat + k;
                auto& c_vec                  = c_thread_buf_.GetVectorTypeReference(
                    Number<tileOffset * wconvConv.GetNumAccumComponents()>{});
                static_for<0, CRepeat, 1>{}([&](auto c) {
                    InDataVec& in_tile_data = inData[h * WRepeat * CRepeat + w * CRepeat + c];
                    const InDataVec* p_in_tile_data[1] = {&in_tile_data};
                    wconvConv.wconv_instr.Run(
                        weiData[k * CRepeat + c], p_in_tile_data, c_vec, Number<0>{});
                });
            });
        });
    });

    __syncthreads();

    constexpr bool IsInt4 =
        std::is_same<ck::int4_t, InDataType>::value || std::is_same<ck::uint4_t, InDataType>::value;
    constexpr index_t NumLanePerPair = (WPerWconv == 2) ? 4 : 2;
    // Output accum data
    constexpr auto accSbaInstance = ck::
        AccSba<AccDataType, HPerWconv, WPerWconv, activateFunc, scaleBiasPacked, uniformScale>();
    using SbaOutVec         = StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr,
                                                AccDataType,
                                                AccVectorCount,
                                                accSbaInstance.GetNumSbaOutComponents(),
                                                true>;
    SbaOutVec d_thread_buf_ = {};
    constexpr auto I0       = Number<0>{};
    constexpr auto I1       = Number<1>{};
    static_for<0, HRepeat, 1>{}([&](auto h) {
        static_for<0, WRepeat, 1>{}([&](auto w) {
            static_for<0, KRepeat, 1>{}([&](auto k) {
                constexpr index_t tileOffset = h * WRepeat * KRepeat + w * KRepeat + k;
                auto& c_vec                  = c_thread_buf_.GetVectorTypeReference(
                    Number<tileOffset * wconvConv.GetNumAccumComponents()>{});
                auto& d_vec = d_thread_buf_.GetVectorTypeReference(
                    Number<tileOffset * accSbaInstance.GetNumSbaOutComponents()>{});
                AccDataType& scale_local = scale_data[tileOffset];

                if constexpr(std::is_same<float, AccDataType>::value)
                {
                    const AccDataType& bias_data32bit_v = bias_32bit_data[tileOffset];
                    accSbaInstance.sba_instr.Run(c_vec, scale_local, bias_data32bit_v, d_vec);
                }
                else if constexpr(std::is_same<half_t, AccDataType>::value)
                {
                    const vector_type<AccDataType, 2>& bias_data16bit_v =
                        bias_16bit_data[tileOffset];
                    if constexpr(HPerWconv == 4 && WPerWconv == 2)
                    {
                        half2_t out0;
                        half2_t out1;
                        accSbaInstance.sba_instr.Run(
                            c_vec, scale_local, bit_cast<half2_t>(bias_data16bit_v), out0, out1);
                        d_vec.template AsType<half2_t>()(I0) = out0;
                        d_vec.template AsType<half2_t>()(I1) = out1;
                    }
                    else
                    {
                        accSbaInstance.sba_instr.Run(c_vec,
                                                     type_convert<float>(scale_local),
                                                     bit_cast<half2_t>(bias_data16bit_v),
                                                     d_vec);
                    }
                }
                else if constexpr(std::is_same<bhalf_t, AccDataType>::value)
                {
                    const vector_type<AccDataType, 2>& bias_data16bit_v =
                        bias_16bit_data[tileOffset];
                    if constexpr(HPerWconv == 4 && WPerWconv == 2)
                    {
                        bhalf2_t out0;
                        bhalf2_t out1;
                        accSbaInstance.sba_instr.Run(
                            c_vec, scale_local, bit_cast<bhalf2_t>(bias_data16bit_v), out0, out1);
                        d_vec.template AsType<half2_t>()(I0) = out0;
                        d_vec.template AsType<half2_t>()(I1) = out1;
                    }
                    else
                    {

                        accSbaInstance.sba_instr.Run(c_vec,
                                                     type_convert<float>(scale_local),
                                                     bit_cast<bhalf2_t>(bias_data16bit_v),
                                                     d_vec);
                    }
                }

                if constexpr(!convert_to_tensor)
                {
                    store_acc_data(h, w, k, d_vec);
                }
                else
                {
                    constexpr auto accCvtInstance = ck::AccCvtTensor<EDataType,
                                                                     AccDataType,
                                                                     HPerWconv,
                                                                     WPerWconv,
                                                                     activateFunc,
                                                                     true>();

                    if constexpr(!IsInt4)
                    {
                        using outTensorVec =
                            StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr,
                                                      EDataType,
                                                      1,
                                                      wconvConv.GetNumOutTensorComponents(),
                                                      true>;
                        outTensorVec out_tensor_thread_buf = {};

                        auto& outVec = out_tensor_thread_buf.GetVectorTypeReference(Number<0>{});

                        accCvtInstance.cvtTensor_instr.Run(d_vec, convertScale, outVec);

                        using outTensorDataVec     = typename decltype(wconvConv)::outTensorDataVec;
                        auto store_out_tensor_data = [&](index_t h0,
                                                         index_t w0,
                                                         index_t k0,
                                                         outTensorDataVec& cvt_out_vec) {
                            const index_t subW =
                                (accCompIdx / wconvConv.GetNumOutputChannels()) % WPerWconv;
                            const index_t subH =
                                (accCompIdx / wconvConv.GetNumOutputChannels()) / WPerWconv;

                            if constexpr(wconvConv.GetNumAccumComponents() == 4)
                            {
                                const index_t subK = accCompIdx % wconvConv.GetNumOutputChannels();
                                const index_t offset = (h0 * HPerWconv + subH) * acc_H_stride +
                                                       (w0 * WPerWconv + subW) * acc_W_stride +
                                                       (k0 * acc_K_stride + subK);
                                *reinterpret_cast<typename outTensorDataVec::type*>(outTensor_ +
                                                                                    offset) =
                                    cvt_out_vec.template AsType<typename outTensorDataVec::type>()(
                                        Number<0>{});
                            }
                            else
                            {
                                static_assert(wconvConv.GetNumAccumComponents() == 8,
                                              "unexpected value");
                                using out_tensor_swizzle_vec =
                                    typename vector_type<InDataType, 4>::type;
                                const index_t subK = accCompIdx % wconvConv.GetNumOutputChannels() /
                                                     (wconvConv.GetNumOutTensorComponents() * 2) *
                                                     (wconvConv.GetNumOutTensorComponents() * 2);
                                const index_t offset = (h0 * HPerWconv + subH) * acc_H_stride +
                                                       (w0 * WPerWconv + subW) * acc_W_stride +
                                                       (k0 * acc_K_stride + subK);

                                index_t secOffset = 8;
                                if constexpr(wconvConv.GetNumSubTilesPerImageTile() > 1)
                                {
                                    secOffset = HPerWconv / wconvConv.GetNumSubTilesPerImageTile() *
                                                acc_H_stride;
                                }
                                const index_t swizzleOffset = (lIdx & 1) * 4;

                                *reinterpret_cast<out_tensor_swizzle_vec*>(outTensor_ + offset +
                                                                           swizzleOffset) =
                                    cvt_out_vec.template AsType<out_tensor_swizzle_vec>()(
                                        Number<0>{});
                                *reinterpret_cast<out_tensor_swizzle_vec*>(
                                    outTensor_ + offset + swizzleOffset + secOffset) =
                                    cvt_out_vec.template AsType<out_tensor_swizzle_vec>()(
                                        Number<1>{});
                            }
                        };

                        store_out_tensor_data(h, w, k, outVec);
                    }
#if CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
                    else
                    {
                        static_assert(KRepeat % 2 == 0);
                        // int4_t
                        using KernelEDataType =
                            decltype(wconvConv.template GetKernelDataType<EDataType>());
                        using outTensorVecFor4bit =
                            StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr,
                                                      KernelEDataType,
                                                      1,
                                                      wconvConv.GetNumOutTensorComponentsFor4bit(),
                                                      true>;
                        outTensorVecFor4bit out_4bit_tensor_thread_buf = {};

                        auto& outVec =
                            out_4bit_tensor_thread_buf.GetVectorTypeReference(Number<0>{});

                        using out4bitTensorDataVec =
                            typename decltype(wconvConv)::out4bitTensorDataVec;
                        auto store_out_tensor_4bit_data = [&](index_t h0,
                                                              index_t w0,
                                                              index_t k0,
                                                              index_t laneMask,
                                                              out4bitTensorDataVec& cvt_out_vec) {
                            auto outTensor = reinterpret_cast<KernelEDataType*>(outTensor_);
                            constexpr index_t NumCompPerOutType =
                                wconvConv.template SizeOfBits<KernelEDataType>() /
                                wconvConv.template SizeOfBits<EDataType>();
                            if constexpr(HPerWconv == 8 && WPerWconv == 4)
                            {
                                const index_t subW = (lIdx / NumLanePerPair) % WPerWconv;
                                const index_t subH = (lIdx / NumLanePerPair) / WPerWconv;
                                const index_t subK = lIdx % NumLanePerPair;
                                const index_t offset =
                                    (h0 * HPerWconv + subH) * acc_H_stride / NumCompPerOutType +
                                    (w0 * WPerWconv + subW) * acc_W_stride / NumCompPerOutType +
                                    (k0 * acc_K_stride / NumCompPerOutType + subK);
                                const index_t sec_offset =
                                    offset + 4 * acc_H_stride / NumCompPerOutType;
                                if(laneMask & (1 << lIdx))
                                {
                                    outTensor[offset] =
                                        cvt_out_vec.template AsType<KernelEDataType>()(Number<0>{});
                                    outTensor[sec_offset] =
                                        cvt_out_vec.template AsType<KernelEDataType>()(Number<1>{});
                                }
                            }
                            else
                            {
                                const index_t subW = (lIdx / NumLanePerPair) % WPerWconv;
                                const index_t subH = (lIdx / NumLanePerPair) / WPerWconv;
                                const index_t subK = lIdx % NumLanePerPair;
                                const index_t offset =
                                    (h0 * HPerWconv + subH) * acc_H_stride / NumCompPerOutType +
                                    (w0 * WPerWconv + subW) * acc_W_stride / NumCompPerOutType +
                                    (k0 * acc_K_stride / NumCompPerOutType + subK);
                                if(laneMask & (1 << lIdx))
                                {
                                    *reinterpret_cast<typename out4bitTensorDataVec::type*>(
                                        outTensor + offset) =
                                        cvt_out_vec
                                            .template AsType<typename out4bitTensorDataVec::type>()(
                                                Number<0>{});
                                }
                            }
                        };
#ifdef LLVM_OPT_ISSUE
                        if constexpr(k % 2 == 1)
                        {
                            constexpr auto accCvtInstance2 = ck::AccCvtTensor<EDataType,
                                                                              AccDataType,
                                                                              HPerWconv,
                                                                              WPerWconv,
                                                                              activateFunc,
                                                                              true,
                                                                              true>();
                            constexpr index_t tileOffset2 =
                                h * WRepeat * KRepeat + w * KRepeat + k - 1;
                            auto& d_vec2 = d_thread_buf_.GetVectorTypeReference(
                                Number<tileOffset2 * accSbaInstance.GetNumSbaOutComponents()>{});
                            accCvtInstance.cvtTensor_instr.Run(d_vec2, convertScale, outVec);
                            accCvtInstance2.cvtTensor_instr.Run(d_vec, convertScale, outVec);
                            store_out_tensor_4bit_data(h, w, k - 1, 0xffffffff, outVec);
                        }
#else
                        if constexpr(WPerWconv == 4 && HPerWconv == 4)
                        {
                            accCvtInstance.cvtTensor_instr.Run(d_vec, convertScale, outVec);
                            store_out_tensor_4bit_data(h, w, k, 0xffffffff, outVec);
                        }
                        else if constexpr(k % 2 == 0)
                        {
                            constexpr index_t laneMask = (WPerWconv == 2) ? 0x33333333 : 0x55555555;
                            accCvtInstance.cvtTensor_instr.Run(d_vec, convertScale, outVec);
                            store_out_tensor_4bit_data(h, w, k, laneMask, outVec);
                        }
                        else
                        {
                            constexpr index_t laneMask = (WPerWconv == 2) ? 0xcccccccc : 0xaaaaaaaa;
                            constexpr auto accCvtInstance2 = ck::AccCvtTensor<EDataType,
                                                                              AccDataType,
                                                                              HPerWconv,
                                                                              WPerWconv,
                                                                              activateFunc,
                                                                              true,
                                                                              true>();
                            accCvtInstance2.cvtTensor_instr.Run(d_vec, convertScale, outVec);
                            store_out_tensor_4bit_data(h, w, k - 1, laneMask, outVec);
                        }
#endif
                    }
#endif
                }
            });
        });
    });
}

template <typename InDataType,
          typename WeiDataType,
          typename AccDataType,
          typename EDataType,
          index_t HPerWconv,
          index_t WPerWconv,
          index_t FilterSize,
          index_t DilationX,
          index_t DilationY,
          index_t Width,
          index_t Height,
          index_t UnpackedInputChannels,
          index_t UnpackedOutputChannels,
          index_t activateFunc,
          index_t convertScale,
          bool scaleBiasPacked,
          bool uniformScale,
          bool convert_to_tensor>
__global__ void __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
    conv3_fwd(const InDataType* in_,
              const WeiDataType* wei_,
              AccDataType* c_,
              AccDataType* scale_,
              AccDataType* bias_,
              EDataType* outTensor_)
{
    constexpr index_t DataTileHeight = 4;

    constexpr auto wconvConv = ck::WconvConv<WeiDataType,
                                             InDataType,
                                             AccDataType,
                                             HPerWconv,
                                             WPerWconv,
                                             FilterSize,
                                             DilationX,
                                             DilationY>();

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

    auto load_bias_32bit_data = [&]() {
        // To do calculate stride with acc_k.
        index_t offset = 0;
        if constexpr(std::is_same<float, AccDataType>::value)
        {
            if(scaleBiasPacked)
            {
                offset = lIdx / 2;
                if(lIdx % 2)
                {
                    return *reinterpret_cast<const AccDataType*>(scale_ + offset);
                }
                else
                {
                    return *reinterpret_cast<const AccDataType*>(bias_ + offset);
                }
            }
            else
            {
                offset = lIdx;
                return *reinterpret_cast<const AccDataType*>(bias_ + offset);
            }
        }
    };

    auto load_bias_16bit_data = [&]() {
        auto bias_16bit_unpack_data = vector_type<AccDataType, 2>{};
        if constexpr(std::is_same<half_t, AccDataType>::value ||
                     std::is_same<bhalf_t, AccDataType>::value)
        {
            if constexpr(scaleBiasPacked)
            {
                index_t offset         = lIdx;
                uint16_t scale_data    = *reinterpret_cast<const uint16_t*>(scale_ + offset);
                uint16_t bias_data     = *reinterpret_cast<const uint16_t*>(bias_ + offset);
                uint32_t scale_bias    = (((scale_data & 0xffff) << 16) | (bias_data & 0xffff));
                bias_16bit_unpack_data = bit_cast<vector_type<AccDataType, 2>>(scale_bias);
            }
            else
            {
                index_t offset = lIdx * 2;
                bias_16bit_unpack_data.template AsType<AccDataType>()(Number<0>{}) =
                    *reinterpret_cast<const AccDataType*>(bias_ + offset);
                bias_16bit_unpack_data.template AsType<AccDataType>()(Number<1>{}) =
                    *reinterpret_cast<const AccDataType*>(bias_ + offset + 1);
            }
        }
        return bias_16bit_unpack_data;
    };

    AccDataType scale_data[KRepeat]                      = {};
    AccDataType bias_32bit_data[KRepeat]                 = {};
    vector_type<AccDataType, 2> bias_16bit_data[KRepeat] = {};

    static_for<0, KRepeat, 1>{}([&](auto k) {
        constexpr index_t tileOffset = k;
        if constexpr(std::is_same<half_t, AccDataType>::value ||
                     std::is_same<bhalf_t, AccDataType>::value)
        {
            bias_16bit_data[tileOffset] = load_bias_16bit_data();
        }
        else if constexpr(std::is_same<float, AccDataType>::value)
        {
            bias_32bit_data[tileOffset] = load_bias_32bit_data();
        }

        scale_data[tileOffset] =
            *reinterpret_cast<const AccDataType*>(scale_); // uniform scale in SGPR
    });

    __syncthreads();

    constexpr auto accSbaInstance = ck::
        AccSba<AccDataType, HPerWconv, WPerWconv, activateFunc, scaleBiasPacked, uniformScale>();

    using SbaOutVec         = StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr,
                                                AccDataType,
                                                1,
                                                accSbaInstance.GetNumSbaOutComponents(),
                                                true>;
    SbaOutVec d_thread_buf_ = {};

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
                    wconvConv.wconv_instr.Run(weiData[c], inData, c_vec, Number<0>{});
                });

                constexpr index_t tileOffset = h * WRepeat * KRepeat + w * KRepeat + k;
                auto& d_vec                  = d_thread_buf_.GetVectorTypeReference(
                    Number<tileOffset * accSbaInstance.GetNumSbaOutComponents()>{});
                AccDataType& scale = scale_data[tileOffset];

                if constexpr(std::is_same<float, AccDataType>::value)
                {
                    const AccDataType& bias_data32bit_v = bias_32bit_data[tileOffset];
                    accSbaInstance.sba_instr.Run(c_vec, scale, bias_data32bit_v, d_vec);
                }
                else if constexpr(std::is_same<half_t, AccDataType>::value)
                {
                    const vector_type<AccDataType, 2>& bias_data16bit_v =
                        bias_16bit_data[tileOffset];
                    accSbaInstance.sba_instr.Run(
                        c_vec, scale, bit_cast<half2_t>(bias_data16bit_v), d_vec);
                }
                else if constexpr(std::is_same<bhalf_t, AccDataType>::value)
                {
                    const vector_type<AccDataType, 2>& bias_data16bit_v =
                        bias_16bit_data[tileOffset];
                    accSbaInstance.sba_instr.Run(
                        c_vec, scale, bit_cast<bhalf2_t>(bias_data16bit_v), d_vec);
                }

                if constexpr(!convert_to_tensor)
                {
                    store_acc_data(h, w, k, d_vec);
                }
                else
                {
                    static_assert(0,
                                  "not implement for 3x3 as the output both 3x3 and 1x1 is same "
                                  "for cvt_tensor.");
                }
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
          typename EDataType,
          ShapeType Shape,
          FilterType Filter,
          bool Dilation,
          ck::index_t activeFunc,
          typename OutElementOp,
          bool scaleBiasPacked,
          bool uniformScale,
          bool convert_to_tensor,
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
    constexpr ck::index_t convertScale = 1;
    constexpr auto wconvConv           = ck::WconvConv<WeiDataType,
                                             InDataType,
                                             GPUAccType,
                                             HPerWconv,
                                             WPerWconv,
                                             FilterSize,
                                             DilationSize,
                                             DilationSize>();
    constexpr bool IsInt4 =
        std::is_same<ck::int4_t, InDataType>::value || std::is_same<ck::uint4_t, InDataType>::value;
#ifdef USE_ABSOLUTE_SIZE
    constexpr ck::index_t Width         = DEFAULT_W;
    constexpr ck::index_t Height        = DEFAULT_H;
    constexpr ck::index_t InputChannels = DEFAULT_C;
    constexpr ck::index_t OutputChannels =
        (IsInt4 && convert_to_tensor) ? DEFAULT_K * 2 : DEFAULT_K;
#else
    constexpr ck::index_t Width  = WPerWconv * DEFAULT_W_REPEAT;
    constexpr ck::index_t Height = HPerWconv * DEFAULT_H_REPEAT;
    constexpr ck::index_t InputChannels =
        wconvConv.GetUnpackedNumInputChannels() * DEFAULT_C_REPEAT;
    constexpr ck::index_t OutputChannels =
        (IsInt4 && convert_to_tensor)
            ? wconvConv.GetUnpackedNumOutputChannels() * DEFAULT_K_REPEAT * 2
            : wconvConv.GetUnpackedNumOutputChannels() * DEFAULT_K_REPEAT;
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

    constexpr ck::index_t NumDs = 2;

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

    conv_param.C_ = OutputChannels;
    const auto cvt_out_g_n_c_wis_desc =
        ck::utils::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<InLayout>(conv_param);

    Tensor<InDataType> in(in_g_n_c_wis_desc);
    Tensor<WeiDataType> wei(wei_g_k_c_xs_desc);
    Tensor<GPUAccType> out_host(out_g_n_k_wos_desc);
    Tensor<EDataType> cvt_out_host(cvt_out_g_n_c_wis_desc);
    Tensor<GPUAccType> out_device(out_g_n_k_wos_desc);
    Tensor<EDataType> out_tensor_device(cvt_out_g_n_c_wis_desc);

    // Logical broadcast bias (we have to pass bias lengths in the same format as output - GNKDHW)
    const ck::index_t G = out_g_n_k_wos_desc.GetLengths()[0];
    const ck::index_t K = out_g_n_k_wos_desc.GetLengths()[2];
    std::array<ck::index_t, NDimSpatial + 3> scalebias_g_k_lengths;
    std::array<ck::index_t, NDimSpatial + 3> scalebias_g_k_strides;
    // Fill other lenghts than G,K with 1 and strides with 0
    scalebias_g_k_lengths.fill(1);
    scalebias_g_k_strides.fill(0);
    scalebias_g_k_lengths[0] = G;
    scalebias_g_k_strides[0] = K; // stride to G
    scalebias_g_k_lengths[2] = K;
    scalebias_g_k_strides[2] = 1; // stride to K
    const auto broadcasted_bias_desc =
        HostTensorDescriptor(scalebias_g_k_lengths, scalebias_g_k_strides);
    const auto broadcasted_scale_desc =
        HostTensorDescriptor(scalebias_g_k_lengths, scalebias_g_k_strides);

    std::array<Tensor<GPUAccType>, NumDs> d_tensors = {Tensor<GPUAccType>(broadcasted_scale_desc),
                                                       Tensor<GPUAccType>(broadcasted_bias_desc)};

    std::cout << "in: " << in.mDesc << std::endl;
    std::cout << "wei: " << wei.mDesc << std::endl;
    std::cout << "out: " << out_host.mDesc << std::endl;
    std::cout << "scale_tensor: " << d_tensors[0].mDesc << std::endl;
    std::cout << "bias_tensor: " << d_tensors[1].mDesc << std::endl;
    if constexpr(convert_to_tensor)
    {
        std::cout << "cvt_tensor_out: " << cvt_out_host.mDesc << std::endl;
    }

    switch(config.init_method)
    {
    case 0: break;
    case 1:
        in.GenerateTensorValue(GeneratorTensor_2<InDataType>{-5, 5});
        wei.GenerateTensorValue(GeneratorTensor_2<WeiDataType>{-5, 5});
        d_tensors[0].GenerateTensorValue(GeneratorTensor_1<GPUAccType>{0.5}); // scale
        if constexpr(uniformScale)
        {
            d_tensors[1].GenerateTensorValue(GeneratorTensor_0<GPUAccType>{});
        }
        else
        {
            d_tensors[1].GenerateTensorValue(GeneratorTensor_2<GPUAccType>{-2, 2});
        }
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
    std::array<ck::index_t, NDimSpatial + 3> cvt_e_g_n_k_wos_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> cvt_e_g_n_k_wos_strides{};
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

    if constexpr(convert_to_tensor)
    {
        copy(cvt_out_g_n_c_wis_desc.GetLengths(), cvt_e_g_n_k_wos_lengths);
        copy(cvt_out_g_n_c_wis_desc.GetStrides(), cvt_e_g_n_k_wos_strides);
    }

    auto ref_conv = ck::tensor_operation::host::ReferenceConvFwd<NDimSpatial,
                                                                 InDataType,
                                                                 WeiDataType,
                                                                 GPUAccType,
                                                                 InElementOp,
                                                                 WeiElementOp,
                                                                 OutElementOp,
                                                                 0, /*Num A Elementwise Tensors*/
                                                                 0, /*Num B Elementwise Tensors*/
                                                                 NumDs>();

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
                                              out_element_op,
                                              {},
                                              {},
                                              d_tensors);

    ref_invoker.Run(ref_argument);

    DumpTensor(in, "Input");
    DumpTensor(wei, "Weight");
    DumpTensor(out_host, "Accum");
    DumpTensor(d_tensors[0], "scale");
    DumpTensor(d_tensors[1], "bias");

    DeviceMem in_device_buf(sizeof(InDataType) * in.mDesc.GetElementSpaceSize());
    DeviceMem wei_device_buf(sizeof(WeiDataType) * wei.mDesc.GetElementSpaceSize());
    DeviceMem scale_device_buf(sizeof(GPUAccType) * d_tensors[0].mDesc.GetElementSpaceSize());
    DeviceMem bias_device_buf(sizeof(GPUAccType) * d_tensors[1].mDesc.GetElementSpaceSize());
    DeviceMem out_device_buf(sizeof(GPUAccType) * out_device.mDesc.GetElementSpaceSize());
    DeviceMem out_tensor_device_buf(sizeof(EDataType) *
                                    out_tensor_device.mDesc.GetElementSpaceSize());

    if constexpr(IsInt4)
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
    scale_device_buf.ToDevice(d_tensors[0].mData.data());
    bias_device_buf.ToDevice(d_tensors[1].mData.data());

    if constexpr(FilterSize == 1)
    {
        auto conv_kernel = ck::conv_op_util::conv_fwd<InDataType,
                                                      WeiDataType,
                                                      GPUAccType,
                                                      EDataType,
                                                      HPerWconv,
                                                      WPerWconv,
                                                      FilterSize,
                                                      DilationSize,
                                                      DilationSize,
                                                      Width,
                                                      Height,
                                                      InputChannels,
                                                      OutputChannels,
                                                      activeFunc,
                                                      convertScale,
                                                      scaleBiasPacked,
                                                      uniformScale,
                                                      convert_to_tensor>;
        conv_kernel<<<1, 32>>>(static_cast<const InDataType*>(in_device_buf.GetDeviceBuffer()),
                               static_cast<const WeiDataType*>(wei_device_buf.GetDeviceBuffer()),
                               static_cast<GPUAccType*>(out_device_buf.GetDeviceBuffer()),
                               static_cast<GPUAccType*>(scale_device_buf.GetDeviceBuffer()),
                               static_cast<GPUAccType*>(bias_device_buf.GetDeviceBuffer()),
                               static_cast<EDataType*>(out_tensor_device_buf.GetDeviceBuffer()));
    }
    else
    {
        auto conv_kernel = ck::conv_op_util::conv3_fwd<InDataType,
                                                       WeiDataType,
                                                       GPUAccType,
                                                       EDataType,
                                                       HPerWconv,
                                                       WPerWconv,
                                                       FilterSize,
                                                       DilationSize,
                                                       DilationSize,
                                                       Width,
                                                       Height,
                                                       InputChannels,
                                                       OutputChannels,
                                                       activeFunc,
                                                       convertScale,
                                                       scaleBiasPacked,
                                                       uniformScale,
                                                       convert_to_tensor>;
        conv_kernel<<<1, 32>>>(static_cast<const InDataType*>(in_device_buf.GetDeviceBuffer()),
                               static_cast<const WeiDataType*>(wei_device_buf.GetDeviceBuffer()),
                               static_cast<GPUAccType*>(out_device_buf.GetDeviceBuffer()),
                               static_cast<GPUAccType*>(scale_device_buf.GetDeviceBuffer()),
                               static_cast<GPUAccType*>(bias_device_buf.GetDeviceBuffer()),
                               static_cast<EDataType*>(out_tensor_device_buf.GetDeviceBuffer()));
    }

    if constexpr(convert_to_tensor)
    {
        if constexpr(IsInt4)
        {
            std::vector<uint8_t> out_packed;
            out_packed.resize(out_tensor_device.mData.size());
            out_tensor_device_buf.FromDevice(out_packed.data());

            for(size_t i = 0; i < out_packed.size() / 2; i++)
            {
                out_tensor_device.mData[2 * i]     = (out_packed[i] & 0xf);
                out_tensor_device.mData[2 * i + 1] = ((out_packed[i] >> 4) & 0xf);
            }
        }
        else
        {
            out_tensor_device_buf.FromDevice(out_tensor_device.mData.data());
        }

        DumpTensor(out_tensor_device, "out_tensor_Device");
    }
    else
    {
        out_device_buf.FromDevice(out_device.mData.data());
        DumpTensor(out_device, "Accum_Device");
    }

    std::cout << "conv_sba_uba<In/Wei:" << get_string<InDataType>()
              << ", Out:" << get_string<GPUAccType>() << ", " << get_string(Shape) << ", "
              << get_string(Filter) << ", Dilation:" << DilationSize << ", activeFun:" << activeFunc
              << ", ConvertToTensor:" << convert_to_tensor
              << ", scalebiaspacked:" << scaleBiasPacked << " ,uniformscale: " << uniformScale
              << ", Id:0x" << std::hex << TestMask << ">: Status: ";

    if(config.do_verification)
    {
        bool ret = false;
        if constexpr(convert_to_tensor)
        {
            const auto cvtTensorScale         = 1.0; // Scale=1 to update
            const auto out_element_convert_op = OutElementConvertOp{
                std::powf(static_cast<float>(2), cvtTensorScale), ActivationOp{}};
            out_host.ForEach([&](auto&, auto idx) { // out_host always smaller than cvt_out_host
                out_element_convert_op(cvt_out_host(idx), out_host(idx));
            });
            ret = ck::utils::check_err(out_tensor_device,
                                       cvt_out_host,
                                       "Error: incorrect results for cvt_tensor!",
                                       get_rtol<EDataType>(),
                                       get_atol<EDataType>());
            if(ret == false)
            {
                DumpTensor(out_host, "Ref_output_before_cvt");
                DumpTensor(cvt_out_host, "Ref_output");
            }
        }
        else
        {
            ret = ck::utils::check_err(out_device,
                                       out_host,
                                       "Error: incorrect results!",
                                       get_rtol<GPUAccType>(),
                                       get_atol<GPUAccType>());
        }
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

template <typename SrcType,
          typename GPUAccType,
          bool scaleBiasPacked,
          bool uniformScale,
          bool convert_to_tensor,
          int32_t TestMask>
bool run_test_fmt()
{
    if((config.test_mask & TestMask) == 0)
    {
        return true;
    }
    bool pass = true;
    // clang-format off
    //                                                        |ShapeType |FilterType |Dilation | ActiveFunc | OutElementOp | scaleBiasPacked | uniformScale
    if constexpr(std::is_same<GPUAccType, float>::value || std::is_same<GPUAccType, int32_t>::value)
    {
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_1X1, false, 0, OutElementNoneOp, scaleBiasPacked, uniformScale, convert_to_tensor, TestMask | 0x10000  >();
#ifdef ENABLE_FULL_TEST
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_3X3, false, 0, OutElementNoneOp, scaleBiasPacked, uniformScale, convert_to_tensor, TestMask | 0x10000  >();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_3X3, true,  0, OutElementNoneOp, scaleBiasPacked, uniformScale, convert_to_tensor, TestMask | 0x10000  >();
#endif
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_1X1, false, 1, OutElementReluOp, scaleBiasPacked, uniformScale, convert_to_tensor, TestMask | 0x20000  >();
#ifdef ENABLE_FULL_TEST
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_3X3, false, 1, OutElementReluOp, scaleBiasPacked, uniformScale, convert_to_tensor, TestMask | 0x20000  >();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_3X3, true,  1, OutElementReluOp, scaleBiasPacked, uniformScale, convert_to_tensor, TestMask | 0x20000  >();
#endif
        // ActiveFunc:Tanh is not used for cvt_to_tensor
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_1X1, false, 2, OutElementTanhOp, scaleBiasPacked, uniformScale, 0, TestMask | 0x40000  >();
#ifdef ENABLE_FULL_TEST
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_3X3, false, 2, OutElementTanhOp, scaleBiasPacked, uniformScale, 0, TestMask | 0x40000  >();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_3X3, true,  2, OutElementTanhOp, scaleBiasPacked, uniformScale, 0, TestMask | 0x40000  >();
#endif
    }
    else
    {
        // issue@llvm: https://ontrack-internal.amd.com/browse/LWPSCGFX13-478 for v_scale_bias_activate_f16 which will impact the all the accType=half case which will impact 4x4
#if 0
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X4, Filter_1X1, false, 0, OutElementNoneOp, scaleBiasPacked, uniformScale, convert_to_tensor, TestMask | 0x80000  >();
#ifdef ENABLE_FULL_TEST
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X4, Filter_3X3, false, 0, OutElementNoneOp, scaleBiasPacked, uniformScale, 0, TestMask | 0x80000  >();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X4, Filter_3X3, true,  0, OutElementNoneOp, scaleBiasPacked, uniformScale, 0, TestMask | 0x80000  >();
#endif
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X4, Filter_1X1, false, 1, OutElementReluOp, scaleBiasPacked, uniformScale, convert_to_tensor, TestMask | 0x100000 >();
#ifdef ENABLE_FULL_TEST
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X4, Filter_3X3, false, 1, OutElementReluOp, scaleBiasPacked, uniformScale, 0, TestMask | 0x100000 >();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X4, Filter_3X3, true,  1, OutElementReluOp, scaleBiasPacked, uniformScale, 0, TestMask | 0x100000 >();
#endif
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X4, Filter_1X1, false, 2, OutElementTanhOp, scaleBiasPacked, uniformScale, 0, TestMask | 0x200000 >();
#endif
#ifdef ENABLE_FULL_TEST
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X4, Filter_3X3, false, 2, OutElementTanhOp, scaleBiasPacked, uniformScale, 0, TestMask | 0x200000 >();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X4, Filter_3X3, true,  2, OutElementTanhOp, scaleBiasPacked, uniformScale, 0, TestMask | 0x200000 >();
#endif
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_8X4, Filter_1X1, false, 0, OutElementNoneOp, scaleBiasPacked, uniformScale, convert_to_tensor, TestMask | 0x400000 >();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_8X4, Filter_1X1, false, 1, OutElementReluOp, scaleBiasPacked, uniformScale, convert_to_tensor, TestMask | 0x800000 >();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_8X4, Filter_1X1, false, 2, OutElementTanhOp, scaleBiasPacked, uniformScale, 0, TestMask | 0x1000000>();
#ifdef ENABLE_FULL_TEST
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_8X4, Filter_3X3, false, 0, OutElementNoneOp, scaleBiasPacked, uniformScale, convert_to_tensor, TestMask | 0x400000 >();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_8X4, Filter_3X3, true,  0, OutElementNoneOp, scaleBiasPacked, uniformScale, convert_to_tensor, TestMask | 0x400000 >();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_8X4, Filter_3X3, false, 1, OutElementReluOp,  scaleBiasPacked, uniformScale, convert_to_tensor, TestMask | 0x800000 >();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_8X4, Filter_3X3, true,  1, OutElementReluOp,  scaleBiasPacked, uniformScale, convert_to_tensor, TestMask | 0x800000 >();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_8X4, Filter_3X3, false, 2, OutElementTanhOp,  scaleBiasPacked, uniformScale, 0, TestMask | 0x1000000 >();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_8X4, Filter_3X3, true,  2, OutElementTanhOp,  scaleBiasPacked, uniformScale, 0, TestMask | 0x1000000 >();
#endif

        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_1X1, false, 0, OutElementNoneOp, scaleBiasPacked, uniformScale, convert_to_tensor, TestMask | 0x2000000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_1X1, false, 1, OutElementReluOp,  scaleBiasPacked, uniformScale, convert_to_tensor, TestMask | 0x4000000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_1X1, false, 2, OutElementTanhOp,  scaleBiasPacked, uniformScale, 0,                 TestMask | 0x8000000>();
#ifdef ENABLE_FULL_TEST
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_3X3, false, 0, OutElementNoneOp, scaleBiasPacked, uniformScale, convert_to_tensor, TestMask | 0x2000000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_3X3, true,  0, OutElementNoneOp, scaleBiasPacked, uniformScale, convert_to_tensor, TestMask | 0x2000000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_3X3, false, 1, OutElementReluOp, scaleBiasPacked, uniformScale, convert_to_tensor, TestMask | 0x4000000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_3X3, true,  1, OutElementReluOp, scaleBiasPacked, uniformScale, convert_to_tensor, TestMask | 0x4000000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_3X3, false, 2, OutElementTanhOp, scaleBiasPacked, uniformScale, 0, TestMask | 0x8000000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, SrcType, Shape_4X2, Filter_3X3, true,  2, OutElementTanhOp, scaleBiasPacked, uniformScale, 0, TestMask | 0x8000000>();
#endif


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
    //                  |SrcType     |GPUAccType  |scaleBiasPacked  |uniformScale | convert_to_tensor
    pass &= run_test_fmt<ck::half_t,  float,       0, 0, 0, 0x1  >();
    pass &= run_test_fmt<ck::half_t,  float,       1, 0, 0, 0x2  >();
    pass &= run_test_fmt<ck::half_t,  float,       0, 1, 0, 0x4  >();
#ifdef ENABLE_FULL_TEST
    pass &= run_test_fmt<ck::bhalf_t, float,       0, 0, 0, 0x1  >();
    pass &= run_test_fmt<ck::bhalf_t, float,       1, 0, 0, 0x2  >();
    pass &= run_test_fmt<ck::bhalf_t, float,       0, 1, 0, 0x4  >();

    pass &= run_test_fmt<ck::f8_t,    float,       0, 0, 0, 0x1  >();
    pass &= run_test_fmt<ck::f8_t,    float,       1, 0, 0, 0x2  >();
    pass &= run_test_fmt<ck::f8_t,    float,       0, 1, 0, 0x4  >();
    pass &= run_test_fmt<ck::bf8_t,   float,       0, 0, 0, 0x1  >();
    pass &= run_test_fmt<ck::bf8_t,   float,       1, 0, 0, 0x2  >();
    pass &= run_test_fmt<ck::bf8_t,   float,       0, 1, 0, 0x4  >();
    pass &= run_test_fmt<int8_t,      float,       0, 0, 0, 0x1  >();
    pass &= run_test_fmt<int8_t,      float,       1, 0, 0, 0x2  >();
    pass &= run_test_fmt<int8_t,      float,       0, 1, 0, 0x4  >();

    pass &= run_test_fmt<ck::half_t,  ck::half_t,  0, 0, 0, 0x8  >();
    pass &= run_test_fmt<ck::half_t,  ck::half_t,  1, 0, 0, 0x10 >();
    pass &= run_test_fmt<ck::half_t,  ck::half_t,  0, 1, 0, 0x20 >();
    pass &= run_test_fmt<ck::bhalf_t, ck::bhalf_t, 0, 0, 0, 0x40 >();
    pass &= run_test_fmt<ck::bhalf_t, ck::bhalf_t, 0, 1, 0, 0x80 >();
    pass &= run_test_fmt<ck::bhalf_t, ck::bhalf_t, 1, 0, 0, 0x100>();

    pass &= run_test_fmt<ck::f8_t,    ck::half_t,  0, 0, 0, 0x8  >();
    pass &= run_test_fmt<ck::f8_t,    ck::half_t,  1, 0, 0, 0x10 >();
    pass &= run_test_fmt<ck::f8_t,    ck::half_t,  0, 1, 0, 0x20 >();
#endif
    pass &= run_test_fmt<int8_t,      ck::half_t,  0, 0, 0, 0x8  >();
    pass &= run_test_fmt<int8_t,      ck::half_t,  1, 0, 0, 0x10 >();
    pass &= run_test_fmt<int8_t,      ck::half_t,  0, 1, 0, 0x20 >();
    pass &= run_test_fmt<ck::bf8_t,   ck::half_t,  0, 0, 0, 0x40 >();
    pass &= run_test_fmt<ck::bf8_t,   ck::half_t,  0, 1, 0, 0x80 >();
    pass &= run_test_fmt<ck::bf8_t,   ck::half_t,  1, 0, 0, 0x100>();
    // cvt to tensor
    pass &= run_test_fmt<ck::half_t,  float,       0, 0, 1, 0x1  >();
    pass &= run_test_fmt<ck::bhalf_t, float,       0, 0, 1, 0x1  >();
    pass &= run_test_fmt<ck::f8_t,    float,       0, 0, 1, 0x1  >();
    pass &= run_test_fmt<ck::bf8_t,   float,       0, 0, 1, 0x1  >();
    pass &= run_test_fmt<int8_t,      float,       0, 0, 1, 0x1  >();
    pass &= run_test_fmt<ck::half_t,  ck::half_t,  0, 0, 1, 0x1  >();
    pass &= run_test_fmt<ck::f8_t,    ck::half_t,  0, 0, 1, 0x1  >();
    pass &= run_test_fmt<ck::bf8_t,   ck::half_t,  0, 0, 1, 0x1  >();
    pass &= run_test_fmt<int8_t,      ck::half_t,  0, 0, 1, 0x1  >();
#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
    pass &= run_test_fmt<ck::int4_t,  float,       0, 0, 1, 0x800 >();
    pass &= run_test_fmt<ck::int4_t,  ck::half_t,  0, 0, 1, 0x2000>();
#endif
    // clang-format on

    std::cout << "conv_sba_uba ..... " << (pass ? "SUCCESS" : "FAILURE") << std::endl;
    return pass ? 0 : 1;
}
