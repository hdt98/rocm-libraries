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
#include "ck/utility/amd_semaphore.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_convsuba_wconvsuba.hpp"

//#include "windows.h"

using InElementOp  = ck::tensor_operation::element_wise::PassThrough;
using WeiElementOp = ck::tensor_operation::element_wise::PassThrough;
using OutElementOp = ck::tensor_operation::element_wise::UnaryConvert;

#define DEFAULT_H 8
#define DEFAULT_W 8
#define DEFAULT_C 16
#define DEFAULT_K 16

#define DEFAULT_H_PERBLOCK 8
#define DEFAULT_W_PERBLOCK 8
#define DEFAULT_C_PERBLOCK 16
#define DEFAULT_K_PERBLOCK 16
#define DEFAULT_BLOCKSIZE 32

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

template <typename T>
struct Debug;

namespace ck {

namespace conv_op_util {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wundefined-reinterpret-cast"
template <typename InDataType,
          typename WeiDataType,
          typename AccDataType,
          index_t HPerBlock,
          index_t WPerBlock,
          index_t CPerBlock,
          index_t KPerBlock,
          index_t HRepeat,
          index_t WRepeat,
          index_t HPerWconv,
          index_t WPerWconv,
          index_t FilterSize,
          index_t DilationX,
          index_t DilationY,
          index_t Width,
          index_t Height,
          index_t InputChannels,
          index_t OutputChannels>
__global__ void __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
    conv_block_fwd(const InDataType* in, const WeiDataType* wei, AccDataType* c)
{
    constexpr auto wconv_conv = ck::WconvConv<WeiDataType,
                                              InDataType,
                                              AccDataType,
                                              HPerWconv,
                                              WPerWconv,
                                              FilterSize,
                                              DilationX,
                                              DilationY>();

    constexpr index_t LDS_SIZE  = 32 * 1024;
    constexpr index_t BlockSize = DEFAULT_BLOCKSIZE;

    static_assert(Width % WPerBlock == 0, "");
    static_assert(Height % HPerBlock == 0, "");
    static_assert(HPerBlock % (HRepeat * HPerWconv) == 0, "");
    static_assert(WPerBlock % (WRepeat * WPerWconv) == 0, "");

    __shared__ char p_shared[LDS_SIZE];
    constexpr index_t W_BlockTile = Width / WPerBlock;
    constexpr index_t C_BlockTile = InputChannels / CPerBlock;
    constexpr index_t K_BlockTile = OutputChannels / KPerBlock;
    const index_t lIdx            = threadIdx.x;
    const index_t h_block_start   = (blockIdx.x / W_BlockTile) * HPerBlock;
    const index_t w_block_start   = (blockIdx.x % W_BlockTile) * WPerBlock;

    constexpr index_t DataTileHeight = 4;
    constexpr index_t H_Pad          = (FilterSize == 3) ? DataTileHeight : 0;
    constexpr index_t W_Pad          = (FilterSize == 3) ? WPerWconv : 0;
    constexpr index_t HPerBlockIn    = (FilterSize == 3) ? HPerBlock + H_Pad * 2 : HPerBlock;
    constexpr index_t WPerBlockIn    = (FilterSize == 3) ? WPerBlock + W_Pad * 2 : WPerBlock;

    constexpr index_t InDataBlockSize = HPerBlockIn * WPerBlockIn * CPerBlock * sizeof(InDataType);
    constexpr index_t WeiDataBlockSize =
        CPerBlock * FilterSize * FilterSize * KPerBlock * sizeof(WeiDataType);

    static_assert(InDataBlockSize + WeiDataBlockSize * C_BlockTile < LDS_SIZE, "");

    constexpr auto WeiDataBlockDesc = make_naive_tensor_descriptor(
        make_tuple(Number<KPerBlock>{}, Number<FilterSize * FilterSize>{}, Number<CPerBlock>{}),
        make_tuple(Number<CPerBlock>{}, Number<KPerBlock * CPerBlock>{}, Number<1>{}));

    static constexpr index_t KPerWconv = wconv_conv.GetNumOutputChannels();

    // HWC
    constexpr auto InDataBlockDesc = make_naive_tensor_descriptor_packed(
        make_tuple(Number<HPerBlockIn>{}, Number<WPerBlockIn>{}, Number<CPerBlock>{}));
    using EmptyTuple                = ck::Tuple<>;
    using AccBlockwiseOperation     = ck::convolution::BlockwiseElementOpPassThrough;
    using AccBlockwiseNextOperation = ck::convolution::BlockwiseElementOpPassThrough;
    using ThisThreadBlock           = ThisThreadBlock<BlockSize>;
    auto blockwise_conv =
        BlockwiseSubaConvWconv<ThisThreadBlock,
                               WeiDataType,
                               InDataType,
                               EmptyTuple,
                               AccDataType,
                               AccDataType,
                               AccBlockwiseOperation,
                               AccBlockwiseNextOperation,
                               decltype(WeiDataBlockDesc),
                               decltype(InDataBlockDesc),
                               EmptyTuple,
                               HPerBlock,
                               WPerBlock,
                               CPerBlock,
                               KPerBlock,
                               HRepeat,
                               WRepeat,
                               HPerWconv,
                               WPerWconv,
                               FilterSize,
                               DilationX,
                               DilationY,
                               true,
                               true>{AccBlockwiseOperation{}, AccBlockwiseNextOperation{}};
    auto accum_thread_buf  = blockwise_conv.template MakeAccumThreadBuffer<false>(c);
    auto output_thread_buf = blockwise_conv.MakeOutThreadBuffer(c);
    // Data layout: HWC, unit: InDataType
    constexpr index_t data_H_stride = Width * InputChannels;
    constexpr index_t data_W_stride = InputChannels;

    // Filter layout: KYXC, unit: WeiDataType
    constexpr index_t weight_X_stride = InputChannels;
    constexpr index_t weight_Y_stride = InputChannels * FilterSize;
    constexpr index_t weight_K_stride = InputChannels * FilterSize * FilterSize;

    // Acc layout: HWK, unit: AccDataType
    constexpr index_t acc_H_stride = Width * OutputChannels;
    constexpr index_t acc_W_stride = OutputChannels;

    // KXYC -> XYKC
    auto update_weight_block_buf = [&](index_t k) {
        const WeiDataType* p_weight_block = wei + k * KPerBlock * weight_K_stride;
        WeiDataType* p_block_buf = reinterpret_cast<WeiDataType*>(&p_shared[InDataBlockSize]);
        constexpr index_t CompCountPerBlock  = KPerBlock * CPerBlock;
        constexpr index_t CompCountPerIter   = CompCountPerBlock * C_BlockTile;
        constexpr index_t CompCountPerVector = wconv_conv.GetNumWeightCompPerTap();
        constexpr index_t CompCountPerThread =
            (CompCountPerIter / BlockSize + CompCountPerVector - 1) / CompCountPerVector *
            CompCountPerVector;

        static_assert(CompCountPerThread % CompCountPerVector == 0, "");
        using WeiDataVec = typename vector_type<WeiDataType, CompCountPerVector>::type;
        static_for<0, CompCountPerThread, CompCountPerVector>{}([&](auto i) {
            const index_t totalIdx = lIdx * CompCountPerThread + i;
            const index_t idx      = totalIdx % CompCountPerBlock;
            const index_t c0       = totalIdx / CompCountPerBlock;
            const index_t c_offset = idx % CPerBlock;
            const index_t k_offset = idx / CPerBlock;
            const WeiDataVec* pIn  = reinterpret_cast<const WeiDataVec*>(
                p_weight_block + k_offset * weight_K_stride + c_offset + c0 * CPerBlock);
            WeiDataVec* pOut =
                reinterpret_cast<WeiDataVec*>(p_block_buf + c0 * CompCountPerBlock + idx);
            if(totalIdx < CompCountPerIter)
            {
                *pOut = *pIn;
            }
        });
    };

    auto update_weight_block_buf3 = [&](index_t k) {
        // 0, 1, 2, 3, 4, 5, 6, 7, 8
        constexpr index_t TapMapRev[] = {2, 1, 8, 3, 0, 7, 4, 5, 6};
        static_for<0, C_BlockTile, 1>{}([&](auto c0) {
            static_for<0, FilterSize, 1>{}([&](auto y0) {
                static_for<0, FilterSize, 1>{}([&](auto x0) {
                    constexpr index_t CompCountPerBlock  = KPerBlock * CPerBlock;
                    constexpr index_t CompCountPerVector = wconv_conv.GetNumWeightCompPerTap();
                    constexpr index_t CompCountPerThread =
                        (CompCountPerBlock / BlockSize + CompCountPerVector - 1) /
                        CompCountPerVector * CompCountPerVector;
                    const WeiDataType* p_weight_block = wei + k * KPerBlock * weight_K_stride +
                                                        x0 * weight_X_stride +
                                                        y0 * weight_Y_stride + c0 * CPerBlock;

                    WeiDataType* p_block_buf =
                        reinterpret_cast<WeiDataType*>(
                            &p_shared[InDataBlockSize + c0 * WeiDataBlockSize]) +
                        TapMapRev[y0 * FilterSize + x0] * CompCountPerBlock;

                    static_assert(CompCountPerThread % CompCountPerVector == 0, "");

                    using WeiDataVec = typename vector_type<WeiDataType, CompCountPerVector>::type;
                    static_for<0, CompCountPerThread, CompCountPerVector>{}([&](auto i) {
                        const index_t idx      = lIdx * CompCountPerThread + i;
                        const index_t c_offset = idx % CPerBlock;
                        const index_t k_offset = idx / CPerBlock;
                        const WeiDataVec* pIn  = reinterpret_cast<const WeiDataVec*>(
                            p_weight_block + k_offset * weight_K_stride + c_offset +
                            c0 * CPerBlock);
                        WeiDataVec* pOut = reinterpret_cast<WeiDataVec*>(p_block_buf + idx);
                        if(idx < CompCountPerBlock)
                        {
                            *pOut = *pIn;
                        }
                    });
                });
            });
        });
    };

    // HWC
    auto update_indata_block_buf = [&](index_t c0) {
        const InDataType* p_indata_block     = in + c0 * CPerBlock;
        InDataType* p_block_buf              = reinterpret_cast<InDataType*>(&p_shared[0]);
        constexpr index_t CompCountPerBlock  = WPerBlockIn * HPerBlockIn * CPerBlock;
        constexpr index_t CompCountPerThread = CompCountPerBlock / BlockSize;
        constexpr index_t CompCountPerVector = wconv_conv.GetNumDataCompPerTile();
        static_assert(CompCountPerBlock % BlockSize == 0, "");
        static_assert(CompCountPerThread % CompCountPerVector == 0, "");

        using InDataVec = typename vector_type<InDataType, CompCountPerVector>::type;
        static_for<0, CompCountPerThread, CompCountPerVector>{}([&](auto i) {
            const index_t idx      = lIdx * CompCountPerThread + i;
            const index_t c_offset = idx % CPerBlock;
            const index_t w_offset = (idx / CPerBlock) % WPerBlockIn;
            const index_t h_offset = (idx / CPerBlock) / WPerBlockIn;
            const index_t h        = h_block_start + h_offset - H_Pad;
            const index_t w        = w_block_start + w_offset - W_Pad;

            const InDataVec* pIn = reinterpret_cast<const InDataVec*>(
                p_indata_block + h * data_H_stride + w * data_W_stride + c_offset);
            InDataVec* pOut = reinterpret_cast<InDataVec*>(p_block_buf + idx);
            if constexpr(H_Pad > 0 || W_Pad > 0)
            {
                // TODO: avoid check for each copy, we only need to do it for edge buffer
                if(h >= 0 && w >= 0 && w < Width && h < Height)
                {
                    *pOut = *pIn;
                }
                else
                {
                    *pOut = 0;
                }
            }
            else
            {
                *pOut = *pIn;
            }
        });
    };

    // HWK
    const index_t laneId = lIdx % 32;
    const index_t accCompIdx =
        laneId * wconv_conv.GetNumAccumComponents() / wconv_conv.GetNumSubTilesPerImageTile();
    auto store_accum_buf = [&](index_t k) {
        static constexpr auto I0 = Number<0>{};
        static constexpr auto I1 = Number<1>{};
        static constexpr auto I2 = Number<2>{};

        const index_t subW     = (accCompIdx / wconv_conv.GetNumOutputChannels()) % WPerWconv;
        const index_t subH     = (accCompIdx / wconv_conv.GetNumOutputChannels()) / WPerWconv;
        using AccDataVec       = typename decltype(wconv_conv)::AccDataVec;
        constexpr auto KRepeat = blockwise_conv.KRepeat;
        const auto waveId      = blockwise_conv.GetWaveIdx();
        constexpr bool Aco     = blockwise_conv.Aco;
        static_for<0, HRepeat, 1>{}([&](auto h1) {
            static_for<0, WRepeat, 1>{}([&](auto w1) {
                static_for<0, KRepeat, 1>{}([&](auto k1) {
                    constexpr auto accum_offset = (h1 * WRepeat * KRepeat + w1 * KRepeat + k1) *
                                                  wconv_conv.GetNumAccumComponents();
                    auto& c_vec = output_thread_buf.GetVectorTypeReference(Number<accum_offset>{});
                    static_assert(wconv_conv.GetNumAccumComponents() >= 4);
                    if constexpr(Aco)
                    {
                        // Aco = 1, do swizzle after 2 channels.
                        using AccSwizzleVec = typename vector_type<AccDataType, 2>::type;
                        constexpr index_t NumLanePerPair =
                            (WPerWconv == 2) && (HPerWconv == 4) ? 4 : 2;

                        const index_t subK = accCompIdx % wconv_conv.GetNumOutputChannels() /
                                             (wconv_conv.GetNumAccumComponents() * NumLanePerPair) *
                                             (wconv_conv.GetNumAccumComponents() * NumLanePerPair);
                        const index_t offset = (h_block_start + waveId[I0] * HRepeat * HPerWconv +
                                                h1 * HPerWconv + subH) *
                                                   acc_H_stride +
                                               (w_block_start + waveId[I1] * WRepeat * WPerWconv +
                                                w1 * WPerWconv + subW) *
                                                   acc_W_stride +
                                               (k * KPerBlock + waveId[I2] * KRepeat * KPerWconv +
                                                k1 * KPerWconv + subK);
                        constexpr index_t secOffset =
                            NumLanePerPair * 2; // 8 channel for 4x2 and 4 channel for others.
                        const index_t swizzleOffset = (lIdx & (NumLanePerPair - 1)) * 2;
                        *reinterpret_cast<AccSwizzleVec*>(c + offset + swizzleOffset) =
                            c_vec.template AsType<AccSwizzleVec>()(Number<0>{});
                        *reinterpret_cast<AccSwizzleVec*>(c + offset + swizzleOffset + secOffset) =
                            c_vec.template AsType<AccSwizzleVec>()(Number<1>{});
                        if constexpr(wconv_conv.GetNumAccumComponents() == 8)
                        {
                            constexpr index_t tileOffset =
                                wconv_conv.GetNumSubTilesPerImageTile() == 2 // 8x4
                                    ? HPerWconv / wconv_conv.GetNumSubTilesPerImageTile() *
                                          acc_H_stride
                                    : 8;

                            *reinterpret_cast<AccSwizzleVec*>(c + offset + swizzleOffset +
                                                              tileOffset) =
                                c_vec.template AsType<AccSwizzleVec>()(Number<2>{});
                            *reinterpret_cast<AccSwizzleVec*>(c + offset + swizzleOffset +
                                                              secOffset + tileOffset) =
                                c_vec.template AsType<AccSwizzleVec>()(Number<3>{});
                        }
                    }
                    else
                    {
                        if constexpr(wconv_conv.GetNumAccumComponents() == 4)
                        {
                            const index_t subK = accCompIdx % wconv_conv.GetNumOutputChannels();

                            const index_t offset =
                                (h_block_start + waveId[I0] * HRepeat * HPerWconv + h1 * HPerWconv +
                                 subH) *
                                    acc_H_stride +
                                (w_block_start + waveId[I1] * WRepeat * WPerWconv + w1 * WPerWconv +
                                 subW) *
                                    acc_W_stride +
                                (k * KPerBlock + waveId[I2] * KRepeat * KPerWconv + k1 * KPerWconv +
                                 subK);
                            *reinterpret_cast<typename AccDataVec::type*>(c + offset) =
                                c_vec.template AsType<typename AccDataVec::type>()(Number<0>{});
                        }
                        else
                        {
                            static_assert(wconv_conv.GetNumAccumComponents() == 8,
                                          "unexpected value");
                            // ACO = 0, do swizzle after 4 channels.
                            using AccSwizzleVec = typename vector_type<AccDataType, 4>::type;
                            const index_t subK  = accCompIdx % wconv_conv.GetNumOutputChannels() /
                                                 (wconv_conv.GetNumAccumComponents() * 2) *
                                                 (wconv_conv.GetNumAccumComponents() * 2);
                            const index_t offset =
                                (h_block_start + waveId[I0] * HRepeat * HPerWconv + h1 * HPerWconv +
                                 subH) *
                                    acc_H_stride +
                                (w_block_start + waveId[I1] * WRepeat * WPerWconv + w1 * WPerWconv +
                                 subW) *
                                    acc_W_stride +
                                (k * KPerBlock + waveId[I2] * KRepeat * KPerWconv + k1 * KPerWconv +
                                 subK);

                            index_t secOffset = 8;
                            if constexpr(wconv_conv.GetNumSubTilesPerImageTile() > 1)
                            {
                                secOffset = HPerWconv / wconv_conv.GetNumSubTilesPerImageTile() *
                                            acc_H_stride;
                            }
                            const index_t swizzleOffset = (lIdx & 1) * 4;
                            *reinterpret_cast<AccSwizzleVec*>(c + offset + swizzleOffset) =
                                c_vec.template AsType<AccSwizzleVec>()(Number<0>{});
                            *reinterpret_cast<AccSwizzleVec*>(c + offset + swizzleOffset +
                                                              secOffset) =
                                c_vec.template AsType<AccSwizzleVec>()(Number<1>{});
                        }
                    }
                });
            });
        });
    };

    // Main loop
    static_for<0, K_BlockTile, 1>{}([&](auto k) {
        if constexpr(FilterSize == 3)
        {
            update_weight_block_buf3(k);
        }
        else
        {
            update_weight_block_buf(k);
        }
        using EmptySemas = Tuple<>;
        EmptySemas emptySemas;

        static_for<0, C_BlockTile, 1>{}([&](auto c0) {
            __syncthreads();
            update_indata_block_buf(c0);
            __syncthreads();
            auto indata_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
                reinterpret_cast<InDataType*>(&p_shared[0]), InDataBlockSize);
            auto weight_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
                reinterpret_cast<WeiDataType*>(&p_shared[InDataBlockSize + c0 * WeiDataBlockSize]),
                WeiDataBlockSize);
            blockwise_conv.Run(weight_block_buf,
                               indata_block_buf,
                               ck::Tuple<>{},
                               accum_thread_buf,
                               output_thread_buf,
                               emptySemas,
                               Number<0>{},
                               Number<1>{});
        });
        store_accum_buf(k);
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
    constexpr ck::index_t FilterSize = (Filter == Filter_1X1) ? 1 : 3;
    constexpr ck::index_t HPerWconv  = (Shape == Shape_8X4) ? 8 : 4;
    constexpr ck::index_t WPerWconv  = (Shape == Shape_4X2) ? 2 : 4;

    constexpr ck::index_t Width          = DEFAULT_W;
    constexpr ck::index_t Height         = DEFAULT_H;
    constexpr ck::index_t InputChannels  = DEFAULT_C;
    constexpr ck::index_t OutputChannels = DEFAULT_K;

    constexpr ck::index_t HPerBlock  = DEFAULT_H_PERBLOCK;
    constexpr ck::index_t WPerBlock  = DEFAULT_W_PERBLOCK;
    constexpr ck::index_t CPerBlock  = DEFAULT_C_PERBLOCK;
    constexpr ck::index_t KPerBlock  = DEFAULT_K_PERBLOCK;
    constexpr ck::index_t HRepeat    = DEFAULT_H_PERBLOCK / HPerWconv;
    constexpr ck::index_t WRepeat    = DEFAULT_W_PERBLOCK / WPerWconv;
    constexpr ck::index_t BlockCount = Width * Height / HPerBlock / WPerBlock;

    constexpr ck::index_t n_dim          = 2;
    constexpr ck::index_t group_count    = 1;
    constexpr ck::index_t n_batch        = 1;
    constexpr ck::index_t n_out_channels = OutputChannels;
    constexpr ck::index_t n_in_channels  = InputChannels;

    constexpr ck::index_t DilationSize = Dilation ? 2 : 1;

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

    in_device_buf.ToDevice(in.mData.data());
    wei_device_buf.ToDevice(wei.mData.data());

    auto conv_kernel = ck::conv_op_util::conv_block_fwd<InDataType,
                                                        WeiDataType,
                                                        GPUAccType,
                                                        HPerBlock,
                                                        WPerBlock,
                                                        CPerBlock,
                                                        KPerBlock,
                                                        HRepeat,
                                                        WRepeat,
                                                        HPerWconv,
                                                        WPerWconv,
                                                        FilterSize,
                                                        DilationSize,
                                                        DilationSize,
                                                        Width,
                                                        Height,
                                                        InputChannels,
                                                        OutputChannels>;
    conv_kernel<<<BlockCount, DEFAULT_BLOCKSIZE>>>(
        static_cast<const InDataType*>(in_device_buf.GetDeviceBuffer()),
        static_cast<const WeiDataType*>(wei_device_buf.GetDeviceBuffer()),
        static_cast<GPUAccType*>(out_device_buf.GetDeviceBuffer()));

    out_device_buf.FromDevice(out_device.mData.data());

    DumpTensor(out_device, "Accum_Device");
    std::cout << "conv_block<In/Wei:" << get_string<InDataType>()
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

template <typename SrcType, typename GPUAccType, int32_t TestMask>
bool run_test_fmt()
{
    if((config.test_mask & TestMask) == 0)
    {
        return true;
    }
    bool pass = true;
    // clang-format off
    //                                                        |ShapeType |FilterType |Dilation |TestMask
    if constexpr(std::is_same<GPUAccType, float>::value || std::is_same<GPUAccType, int32_t>::value)
    {
        pass &= run_test<SrcType, SrcType, GPUAccType, Shape_4X2, Filter_1X1, false, TestMask | 0x10000  >();
        pass &= run_test<SrcType, SrcType, GPUAccType, Shape_4X2, Filter_3X3, false, TestMask | 0x20000  >();
        pass &= run_test<SrcType, SrcType, GPUAccType, Shape_4X2, Filter_3X3, true,  TestMask | 0x40000  >();
    }
    else
    {
        pass &= run_test<SrcType, SrcType, GPUAccType, Shape_4X2, Filter_1X1, false, TestMask | 0x80000  >();
        pass &= run_test<SrcType, SrcType, GPUAccType, Shape_4X4, Filter_1X1, false, TestMask | 0x100000 >();
        pass &= run_test<SrcType, SrcType, GPUAccType, Shape_8X4, Filter_1X1, false, TestMask | 0x200000 >();
        pass &= run_test<SrcType, SrcType, GPUAccType, Shape_4X2, Filter_3X3, false, TestMask | 0x400000 >();
        pass &= run_test<SrcType, SrcType, GPUAccType, Shape_4X4, Filter_3X3, false, TestMask | 0x800000 >();
        pass &= run_test<SrcType, SrcType, GPUAccType, Shape_8X4, Filter_3X3, false, TestMask | 0x1000000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, Shape_4X2, Filter_3X3, true,  TestMask | 0x2000000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, Shape_4X4, Filter_3X3, true,  TestMask | 0x4000000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, Shape_8X4, Filter_3X3, true,  TestMask | 0x8000000>();
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
    //                   |SrcType    |GPUAccType  | TestMask
    pass &= run_test_fmt<ck::half_t,  float,       0x1  >();
    pass &= run_test_fmt<ck::bhalf_t, float,       0x2  >();
    pass &= run_test_fmt<ck::f8_t,    float,       0x4  >();
    pass &= run_test_fmt<ck::bf8_t,   float,       0x8  >();
    pass &= run_test_fmt<int8_t,      float,       0x10 >();
    pass &= run_test_fmt<int8_t,      int32_t,     0x20 >();

    pass &= run_test_fmt<ck::half_t,  ck::half_t,  0x40 >();
    pass &= run_test_fmt<ck::bhalf_t, ck::bhalf_t, 0x80 >();
    pass &= run_test_fmt<ck::f8_t,    ck::half_t,  0x100>();
    pass &= run_test_fmt<ck::bf8_t,   ck::half_t,  0x200>();
    pass &= run_test_fmt<int8_t,      ck::half_t,  0x400>();
    // clang-format on

    std::cout << "conv_block: ..... " << (pass ? "SUCCESS" : "FAILURE") << std::endl;
    return pass ? 0 : 1;
}
