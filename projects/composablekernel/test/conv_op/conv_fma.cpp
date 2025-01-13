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
#include "ck/tensor_operation/gpu/warp/wconv_fma_tensor.hpp"
#include "ck/tensor_operation/gpu/warp/wconv_conv.hpp"

template <typename A>
struct Debug;

using InElementOp  = ck::tensor_operation::element_wise::PassThrough;
using WeiElementOp = ck::tensor_operation::element_wise::PassThrough;
using OutElementOp = ck::tensor_operation::element_wise::UnaryConvert;

#define DEFAULT_K 32
#define DEFAULT_C 32
#define DEFAULT_W 8
#define DEFAULT_H 16

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
          typename AccDataType,
          index_t HPerWconv,
          index_t WPerWconv,
          index_t Width,
          index_t Height,
          index_t UnpackedInputChannels,
          index_t UnpackedOutputChannels>
__global__ void __launch_bounds__(64, 1) conv_fma(const InDataType* in_, AccDataType* c_)
{
    constexpr ck::WconvFmaFromTensor<InDataType, AccDataType, HPerWconv, WPerWconv> wconv_fma;
    constexpr index_t Aco    = wconv_fma.GetAccumChannelOrder();
    constexpr auto wconvConv = ck::WconvConv<InDataType,
                                             InDataType,
                                             AccDataType,
                                             HPerWconv,
                                             WPerWconv,
                                             1,
                                             1,
                                             1,
                                             1,
                                             false,
                                             Aco>();

    auto in = reinterpret_cast<const typename decltype(wconvConv)::KernelInDataType*>(in_);

    static_assert(Width % WPerWconv == 0, "");
    static_assert(Height % HPerWconv == 0, "");
    static_assert(UnpackedInputChannels % wconvConv.GetUnpackedNumInputChannels() == 0, "");
    static_assert(UnpackedOutputChannels % wconvConv.GetUnpackedNumOutputChannels() == 0, "");
    static_assert(UnpackedInputChannels == UnpackedOutputChannels, "");

    constexpr index_t DataTileHeight = 4;
    constexpr index_t InputChannels  = UnpackedInputChannels * wconvConv.GetNumInputChannels() /
                                      wconvConv.GetUnpackedNumInputChannels();
    constexpr index_t OutputChannels = UnpackedOutputChannels * wconvConv.GetNumOutputChannels() /
                                       wconvConv.GetUnpackedNumOutputChannels();

    constexpr index_t HRepeat = Height / HPerWconv;
    constexpr index_t WRepeat = Width / WPerWconv;
    constexpr index_t CRepeat = UnpackedInputChannels / wconvConv.GetUnpackedNumInputChannels();
    constexpr index_t KRepeat = UnpackedOutputChannels / wconvConv.GetUnpackedNumOutputChannels();
    constexpr index_t AccVectorCount = 1;

    using InDataVec     = typename decltype(wconvConv)::InDataVec::type;
    using InDataTileVec = typename decltype(wconvConv)::InDataTileVec::type;
    using AccDataVec    = typename decltype(wconvConv)::AccDataVec;
    using AccVec        = StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr,
                                             AccDataType,
                                             AccVectorCount,
                                             wconvConv.GetNumAccumComponents(),
                                             true>;
    const int lIdx      = threadIdx.x;

    // Data layout: HWC, unit: InDataType
    constexpr index_t data_H_stride = Width * InputChannels;
    constexpr index_t data_W_stride = InputChannels;
    constexpr index_t data_C_stride = wconvConv.GetNumInputChannels();

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

        static_assert(wconvConv.GetNumAccumComponents() >= 4);
        if constexpr(Aco)
        {
            // Aco = 1, do swizzle after 2 channels.
            using AccSwizzleVec              = typename vector_type<AccDataType, 2>::type;
            constexpr index_t NumLanePerPair = (WPerWconv == 2) && (HPerWconv == 4) ? 4 : 2;

            const index_t subK = accCompIdx % wconvConv.GetNumOutputChannels() /
                                 (wconvConv.GetNumAccumComponents() * NumLanePerPair) *
                                 (wconvConv.GetNumAccumComponents() * NumLanePerPair);
            const index_t offset = (h * HPerWconv + subH) * acc_H_stride +
                                   (w * WPerWconv + subW) * acc_W_stride +
                                   (k * acc_K_stride + subK);
            constexpr index_t secOffset =
                NumLanePerPair * 2; // 8 channel for 4x2 and 4 channel for others.
            const index_t swizzleOffset = (lIdx & (NumLanePerPair - 1)) * 2;
            *reinterpret_cast<AccSwizzleVec*>(c_ + offset + swizzleOffset) =
                c_vec.template AsType<AccSwizzleVec>()(Number<0>{});
            *reinterpret_cast<AccSwizzleVec*>(c_ + offset + swizzleOffset + secOffset) =
                c_vec.template AsType<AccSwizzleVec>()(Number<1>{});
            if constexpr(wconvConv.GetNumAccumComponents() == 8)
            {
                constexpr index_t tileOffset =
                    wconvConv.GetNumSubTilesPerImageTile() == 2 // 8x4
                        ? HPerWconv / wconvConv.GetNumSubTilesPerImageTile() * acc_H_stride
                        : 8;

                *reinterpret_cast<AccSwizzleVec*>(c_ + offset + swizzleOffset + tileOffset) =
                    c_vec.template AsType<AccSwizzleVec>()(Number<2>{});
                *reinterpret_cast<AccSwizzleVec*>(c_ + offset + swizzleOffset + secOffset +
                                                  tileOffset) =
                    c_vec.template AsType<AccSwizzleVec>()(Number<3>{});
            }
        }
        else
        {
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
                // Aco = 0, do swizzle after 4 channels.
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
        }
    };

    __syncthreads();
    constexpr index_t numAccum = wconv_fma.GetNumAccum();
    static_assert(KRepeat % numAccum == 0);
    // Do conv fma
    static_for<0, HRepeat, 1>{}([&](auto h) {
        static_for<0, WRepeat, 1>{}([&](auto w) {
            static_for<0, KRepeat, numAccum>{}([&](auto k) {
                AccVec c_thread_buf_            = {};
                auto& c_vec                     = c_thread_buf_.GetVectorTypeReference(Number<0>{});
                typename AccDataVec::type accum = {};
                typename AccDataVec::type scale = {};
                static_for<0, wconvConv.GetNumAccumComponents(), 1>{}(
                    [&](auto s) { scale[s.value] = 1.0; });

                InDataVec in_tile_data1 = {};
                InDataVec in_tile_data2 = {};
                if constexpr(wconv_fma.GetNumResidual() == 4)
                {
                    InDataVec in_tile_data3 = {};
                    InDataVec in_tile_data4 = {};
                    static_assert(CRepeat == 4 * KRepeat);
                    constexpr index_t c = k * 4;
                    in_tile_data1       = load_in_data(h, w, c);
                    in_tile_data2       = load_in_data(h, w, c + 1);
                    in_tile_data3       = load_in_data(h, w, c + 2);
                    in_tile_data4       = load_in_data(h, w, c + 3);
                    wconv_fma.fma_instr.Run(accum,
                                            in_tile_data1,
                                            in_tile_data2,
                                            in_tile_data3,
                                            in_tile_data4,
                                            scale,
                                            c_vec);
                }
                else if constexpr(wconv_fma.GetNumResidual() == 2)
                {
                    static_assert(CRepeat == 2 * KRepeat);
                    constexpr index_t c = k * 2;
                    in_tile_data1       = load_in_data(h, w, c);
                    in_tile_data2       = load_in_data(h, w, c + 1);
                    wconv_fma.fma_instr.Run(accum, in_tile_data1, in_tile_data2, scale, c_vec);
                }
                else if constexpr(wconv_fma.GetNumResidual() == 1)
                {
                    constexpr index_t c = k / numAccum;
                    in_tile_data1       = load_in_data(h, w, c);
                    wconv_fma.fma_instr.Run(accum, in_tile_data1, scale, c_vec);
                }

                store_acc_data(h, w, k, c_vec);

                if constexpr(numAccum == 2)
                {
                    constexpr index_t ChanOff = (HPerWconv == 8) && (WPerWconv == 4) ? 8 : 16;

                    constexpr ck::
                        WconvFmaFromTensor<InDataType, AccDataType, HPerWconv, WPerWconv, ChanOff>
                            wconvFma2;
                    wconvFma2.fma_instr.Run(accum, in_tile_data1, scale, c_vec);
                    store_acc_data(h, w, k + 1, c_vec);
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
          typename GPUAccType,
          typename CPUAccType,
          ShapeType Shape,
          int32_t TestMask>
bool run_test()
{
    if((config.test_mask & 0xFFFF0000 & TestMask) == 0)
    {
        return true;
    }
    constexpr ck::index_t HPerWconv = (Shape == Shape_8X4) ? 8 : 4;
    constexpr ck::index_t WPerWconv = (Shape == Shape_4X2) ? 2 : 4;

    constexpr ck::index_t Width          = DEFAULT_W;
    constexpr ck::index_t Height         = DEFAULT_H;
    constexpr ck::index_t InputChannels  = DEFAULT_C;
    constexpr ck::index_t OutputChannels = DEFAULT_K;

    constexpr ck::index_t n_dim          = 2;
    constexpr ck::index_t group_count    = 1;
    constexpr ck::index_t n_batch        = 1;
    constexpr ck::index_t n_out_channels = OutputChannels;
    constexpr ck::index_t n_in_channels  = InputChannels;

    const std::vector<ck::index_t> filters_1x1{1, 1};
    const std::vector<ck::index_t> dilations_1{1, 1};
    const std::vector<ck::index_t> pads_0{0, 0};

    const std::vector<ck::index_t> input_len = {Height, Width};
    const std::vector<ck::index_t> strides{1, 1};

    ck::utils::conv::ConvParam conv_param{n_dim,
                                          group_count,
                                          n_batch,
                                          n_out_channels,
                                          n_in_channels,
                                          filters_1x1,
                                          input_len,
                                          strides,
                                          dilations_1,
                                          pads_0,
                                          pads_0};

    constexpr auto NDimSpatial = ck::Number<n_dim>{};

    namespace ctc   = ck::tensor_layout::convolution;
    auto in_layout  = ctc::GNHWC{};
    auto out_layout = ctc::GNHWK{};
    using InLayout  = decltype(in_layout);
    using OutLayout = decltype(out_layout);

    const auto in_g_n_c_wis_desc =
        ck::utils::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<InLayout>(conv_param);
    const auto out_g_n_k_wos_desc =
        ck::utils::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<OutLayout>(conv_param);

    Tensor<InDataType> in(in_g_n_c_wis_desc);
    Tensor<CPUAccType> out_host(out_g_n_k_wos_desc);
    Tensor<GPUAccType> out_device(out_g_n_k_wos_desc);

    std::cout << "in: " << in.mDesc << std::endl;
    std::cout << "out: " << out_host.mDesc << std::endl;

    switch(config.init_method)
    {
    case 0: break;
    case 1: in.GenerateTensorValue(GeneratorTensor_2<InDataType>{-5, 5}); break;
    default: in.GenerateTensorValue(GeneratorTensor_3<InDataType>{0.0, 1.0}); break;
    }

    std::array<ck::index_t, NDimSpatial + 3> a_g_n_c_wis_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> a_g_n_c_wis_strides{};
    std::array<ck::index_t, NDimSpatial + 3> e_g_n_k_wos_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> e_g_n_k_wos_strides{};
    std::array<ck::index_t, NDimSpatial> conv_filter_strides{};
    std::array<ck::index_t, NDimSpatial> conv_filter_dilations{};
    std::array<ck::index_t, NDimSpatial> input_left_pads{};
    std::array<ck::index_t, NDimSpatial> input_right_pads{};

    auto copy = [](const auto& x, auto& y) { ck::ranges::copy(x, y.begin()); };

    copy(in_g_n_c_wis_desc.GetLengths(), a_g_n_c_wis_lengths);
    copy(in_g_n_c_wis_desc.GetStrides(), a_g_n_c_wis_strides);
    copy(out_g_n_k_wos_desc.GetLengths(), e_g_n_k_wos_lengths);
    copy(out_g_n_k_wos_desc.GetStrides(), e_g_n_k_wos_strides);
    copy(conv_param.conv_filter_strides_, conv_filter_strides);
    copy(conv_param.conv_filter_dilations_, conv_filter_dilations);
    copy(conv_param.input_left_pads_, input_left_pads);
    copy(conv_param.input_right_pads_, input_right_pads);

    DumpTensor(in, "Input");

    DeviceMem in_device_buf(sizeof(InDataType) * in.mDesc.GetElementSpaceSize());
    DeviceMem out_device_buf(sizeof(GPUAccType) * out_device.mDesc.GetElementSpaceSize());

    if constexpr(std::is_same<ck::int4_t, InDataType>::value)
    {
        std::vector<uint8_t> in_packed;
        in_packed.resize(in.mData.size());
        for(size_t i = 0; i < in.mData.size(); i += 2)
        {
            uint8_t val0 = (in.mData[i] & 0xf);
            uint8_t val1 = (in.mData[i + 1] & 0xf);

            in_packed[i / 2] = val0 | (val1 << 4);
        }
        in_device_buf.ToDevice(in_packed.data());
    }
    else
    {
        in_device_buf.ToDevice(in.mData.data());
    }

    auto conv_kernel = ck::conv_op_util::conv_fma<InDataType,
                                                  GPUAccType,
                                                  HPerWconv,
                                                  WPerWconv,
                                                  Width,
                                                  Height,
                                                  InputChannels,
                                                  OutputChannels>;
    conv_kernel<<<1, 32>>>(static_cast<const InDataType*>(in_device_buf.GetDeviceBuffer()),
                           static_cast<GPUAccType*>(out_device_buf.GetDeviceBuffer()));

    out_device_buf.FromDevice(out_device.mData.data());

    DumpTensor(out_device, "Fma_Device");
    std::cout << "conv_fma<In/Wei:" << get_string<InDataType>()
              << ", Out:" << get_string<GPUAccType>() << ", " << get_string(Shape) << ", Id:0x"
              << std::hex << TestMask << ">: Status: ";

    if(config.do_verification)
    {
        if constexpr(std::is_same<GPUAccType, ck::bhalf_t>::value)
        {
            // check_err doesn't support bhalf_t
            std::cout << " Ignored\n";
            return true;
        }
        else
        {
            auto in2 = in.template CopyAsType<GPUAccType>();
            bool ret = ck::utils::check_err(out_device,
                                            in2,
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
    }
    else
    {
        return true;
    }
}

template <typename SrcType, typename GPUAccType, typename CPUAccType, int32_t TestMask>
bool run_test_fmt()
{
    if((config.test_mask & TestMask) == 0)
    {
        return true;
    }
    bool pass = true;
    // clang-format off
    //                                                        |ShapeType |TestMask
    if constexpr(std::is_same<GPUAccType, float>::value || std::is_same<GPUAccType, int32_t>::value)
    {
        pass &= run_test<SrcType, GPUAccType, CPUAccType, Shape_4X2, TestMask | 0x10000  >();
    }
    else
    {
        pass &= run_test<SrcType, GPUAccType, CPUAccType, Shape_4X2, TestMask | 0x80000  >();
        // llvm issue
        if constexpr (std::is_same<ck::int4_t, SrcType>::value == false)
        {
        pass &= run_test<SrcType, GPUAccType, CPUAccType, Shape_4X4, TestMask | 0x100000 >();
        pass &= run_test<SrcType, GPUAccType, CPUAccType, Shape_8X4, TestMask | 0x200000 >();
        }

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
    //                  |SrcType     |GPUAccType  |CPUAccType
    pass &= run_test_fmt<ck::half_t,  float,       float,      0x1   >();
    pass &= run_test_fmt<ck::bhalf_t, float,       float,      0x2   >();
    pass &= run_test_fmt<ck::f8_t,    float,       float,      0x4   >();
    pass &= run_test_fmt<ck::bf8_t,   float,       float,      0x8   >();
    pass &= run_test_fmt<int8_t,      float,       float,      0x10  >();

    pass &= run_test_fmt<ck::half_t,  ck::half_t,  ck::half_t, 0x40  >();
    pass &= run_test_fmt<ck::bhalf_t, ck::bhalf_t, ck::half_t, 0x80  >();
    pass &= run_test_fmt<ck::f8_t,    ck::half_t,  ck::half_t, 0x100 >();
    pass &= run_test_fmt<ck::bf8_t,   ck::half_t,  ck::half_t, 0x200 >();
    pass &= run_test_fmt<int8_t,      ck::half_t,  ck::half_t, 0x400 >();

#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
    pass &= run_test_fmt<ck::int4_t,  float,       float     , 0x800 >();
    pass &= run_test_fmt<ck::int4_t,  ck::bhalf_t, ck::half_t, 0x1000>();
    pass &= run_test_fmt<ck::int4_t,  ck::half_t,  ck::half_t, 0x2000>();
#endif

    // clang-format on
    std::cout << "conv_fma: ..... " << (pass ? "SUCCESS" : "FAILURE") << std::endl;
    return pass ? 0 : 1;
}
