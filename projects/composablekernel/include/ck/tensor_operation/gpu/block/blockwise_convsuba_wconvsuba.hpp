// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor_operation/gpu/warp/wconv_conv.hpp"
#include "ck/tensor_operation/gpu/warp/wconv_fma_tensor.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"
#include "ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/warp/acc_sba.hpp"
#include "ck/tensor_operation/gpu/warp/acc_cvt_tensor.hpp"

namespace ck {

namespace convolution {
struct BaseBlockwiseElementOp
{
};

template <index_t ActiveFunc, bool UniformScale, bool ScaleBiasPacked>
struct BlockwiseElementOpScaleAndBias : public BaseBlockwiseElementOp
{
    static constexpr const char* name   = "BlockwiseOp_Sba_";
    static constexpr bool IsSuba        = true;
    static constexpr bool IsFma         = false;
    static constexpr index_t activeFunc = ActiveFunc;
    static constexpr index_t uniform    = UniformScale;
    static constexpr index_t packed     = ScaleBiasPacked;
    __host__ __device__ BlockwiseElementOpScaleAndBias(const float scale = 1.0f) : scale_(scale) {}
    static constexpr index_t GetNumDTensor()
    {
        return ScaleBiasPacked ? 2 : (UniformScale ? 0 : 1);
    }
    float scale_ = 1.0f;
};

template <bool UniformScale>
struct BlockwiseElementOpFma : public BaseBlockwiseElementOp
{
    static constexpr const char* name = "BlockwiseOp_Fma";
    static constexpr bool IsSuba      = false;
    static constexpr bool IsFma       = true;
    static constexpr bool uniform     = UniformScale;
    float scale_                      = 1.0f;
    __host__ __device__ BlockwiseElementOpFma(const float scale = 1.0f) : scale_(scale) {}
    static constexpr index_t GetNumDTensor() { return UniformScale ? 1 : 2; }
};

template <index_t ActiveFunc, bool UniformScale0, bool ScaleBiasPacked, bool UniformScale1>
struct BlockwiseElementOpSbaFma : public BaseBlockwiseElementOp
{
    static constexpr const char* name = "BlockwiseOp_Sba_Fma";
    static constexpr bool IsSuba      = true;
    static constexpr bool IsFma       = true;
    using Sba = BlockwiseElementOpScaleAndBias<ActiveFunc, UniformScale0, ScaleBiasPacked>;
    using Fma = BlockwiseElementOpFma<UniformScale1>;
    Sba sba_;
    Fma fma_;
    __host__ __device__ BlockwiseElementOpSbaFma(const float scale0 = 1.0f,
                                                 const float scale1 = 1.0f)
        : sba_(scale0), fma_(scale1)
    {
    }
    static constexpr index_t GetNumDTensor() { return Sba::GetNumDTensor() + Fma::GetNumDTensor(); }
};

template <bool convert_to_tensor, index_t ActiveFunc, index_t convertScale, bool Clamp>
struct BlockwiseElementOpCvtTensor : public BaseBlockwiseElementOp
{
    static constexpr const char* name   = "BlockwiseOp_cvt_tensor_";
    static constexpr index_t activeFunc = ActiveFunc;
    static constexpr index_t scale      = convertScale;
    static constexpr bool cvt_to_tensor = convert_to_tensor;
    static constexpr bool clamp         = Clamp;
};

struct BlockwiseElementOpPassThrough
{
    static constexpr const char* name   = "BlockwiseOp_passthrough";
    static constexpr bool IsSuba        = false;
    static constexpr bool IsFma         = false;
    static constexpr bool cvt_to_tensor = false;
};

}; // namespace convolution

template <typename ThisThreadBlock,
          index_t HPerBlock,
          index_t WPerBlock,
          index_t HRepeat,
          index_t WRepeat,
          index_t HPerWconv,
          index_t WPerWconv>
__device__ static auto GetWconvWaveIdx()
{
    static constexpr index_t WaveSize = 32;
    constexpr index_t HWaves          = HPerBlock / (HRepeat * HPerWconv);
    constexpr index_t WWaves          = WPerBlock / (WRepeat * WPerWconv);
    constexpr index_t KWaves = ThisThreadBlock::GetNumOfThread() / WaveSize / HWaves / WWaves;

    const index_t thread_id = ThisThreadBlock::GetThreadId();

    constexpr auto threadid_to_wave_idx_adaptor = make_single_stage_tensor_adaptor(
        make_tuple(make_merge_transform(make_tuple(HWaves, WWaves, KWaves, WaveSize))),
        make_tuple(Sequence<0, 1, 2, 3>{}),
        make_tuple(Sequence<0>{}));

    return threadid_to_wave_idx_adaptor.CalculateBottomIndex(make_multi_index(thread_id));
}

template <typename WeiDataType,
          typename InDataType,
          typename AccDataType,
          index_t CPerBlock,
          index_t HPerWconv,
          index_t WPerWconv,
          index_t FilterSize,
          bool Transposed>
static constexpr index_t GetFilterIters()
{
    if constexpr(FilterSize == 2)
    {
        return Transposed ? 1 : 4;
    }
    else if constexpr(FilterSize == 3)
    {
        return 1;
    }
    else if constexpr(FilterSize == 1)
    {
        // Enable Iters if CPerBlock is multiple of CPerWconv
        constexpr index_t CPerWconv = WconvConv<WeiDataType,
                                                InDataType,
                                                AccDataType,
                                                HPerWconv,
                                                WPerWconv,
                                                FilterSize,
                                                1,
                                                1>::GetNumInputChannels();
        constexpr index_t CRepeat   = CPerBlock / CPerWconv;
        if constexpr(CRepeat % 4 == 0)
        {
            return 4;
        }
        else if constexpr(CRepeat % 2 == 0)
        {
            return 2;
        }
        else
        {
            return 1;
        }
    }
    else
    {
        static_assert(false, "never called");
    }
}

template <typename ThisThreadBlock,
          typename AccDataType,
          typename DsBlockDesc,
          index_t HPerBlock,
          index_t WPerBlock,
          index_t KPerBlock,
          index_t HPerWconv,
          index_t WPerWconv,
          index_t KPerWconv,
          index_t HRepeat,
          index_t WRepeat,
          index_t KRepeat,
          bool DsEnableLds,
          bool Aco,
          typename BlockwiseElementOp>
struct Blockwise_element_wconv_suba
{
    static constexpr auto I0              = Number<0>{};
    static constexpr auto I1              = Number<1>{};
    static constexpr auto I2              = Number<2>{};
    static constexpr index_t WaveSize     = 32;
    static constexpr index_t NumDTensor   = DsBlockDesc::Size();
    static constexpr bool EnableWaveGroup = ThisThreadBlock::InWaveGroup();
    static constexpr auto acc_sba         = AccSba<AccDataType,
                                           HPerWconv,
                                           WPerWconv,
                                           BlockwiseElementOp::activeFunc,
                                           BlockwiseElementOp::packed,
                                           BlockwiseElementOp::uniform,
                                           Aco>{};
    static constexpr auto NumBias         = acc_sba.GetNumBiasComponents();
    static constexpr auto NumScale        = acc_sba.GetNumScaleComponents();

    __device__ static auto GetWaveIdx()
    {
        return GetWconvWaveIdx<ThisThreadBlock,
                               HPerBlock,
                               WPerBlock,
                               HRepeat,
                               WRepeat,
                               HPerWconv,
                               WPerWconv>();
    }

    template <typename DsGridDesc>
    __host__ __device__ static constexpr auto
    MakeDsGridBlockDescriptor(const DsGridDesc& ds_grid_desc)
    {
        auto ds_grid_packed_desc = generate_tuple(
            [&](auto i) {
                const auto K = ds_grid_desc[i].GetLength(I2);
                return transform_tensor_descriptor(
                    ds_grid_desc[i],
                    make_tuple(make_freeze_transform(I0),
                               make_freeze_transform(I0),
                               make_pass_through_transform(K)),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                    make_tuple(Sequence<>{}, Sequence<>{}, Sequence<0>{}));
            },
            Number<NumDTensor>{});

        if constexpr(DsEnableLds)
        {
            return ds_grid_packed_desc;
        }
        else
        {
            // K0 x K1 X K2
            constexpr auto DsPerThread = acc_sba.GetNumBiasComponents();
            return generate_tuple(
                [&](auto i) {
                    const auto K = ds_grid_packed_desc[i].GetLength(I0);
                    return transform_tensor_descriptor(ds_grid_packed_desc[i],
                                                       make_tuple(make_unmerge_transform(make_tuple(
                                                           K / KPerWconv,
                                                           Number<KPerWconv / DsPerThread>{},
                                                           Number<DsPerThread>{}))),
                                                       make_tuple(Sequence<0>{}),
                                                       make_tuple(Sequence<0, 1, 2>{}));
                },
                Number<NumDTensor>{});
        }
    }

    // Describe how data read from (LDS/VGPR) buffer, used by Block level classes
    template <typename DsBlockDesc_>
    __host__ __device__ static constexpr auto MakeDsWaveDescriptor(const DsBlockDesc_&)
    {
        constexpr auto DsPerThread  = acc_sba.GetNumBiasComponents();
        constexpr auto ds_wave_desc = [&]() {
            if constexpr(DsEnableLds)
            {
                return generate_tuple(
                    [&](auto i) {
                        return transform_tensor_descriptor(
                            DsBlockDesc_{}[Number<i>{}],
                            make_tuple(
                                make_unmerge_transform(make_tuple(Number<KPerBlock / KPerWconv>{},
                                                                  Number<KPerWconv / DsPerThread>{},
                                                                  Number<DsPerThread>{}))),
                            make_tuple(Sequence<0>{}),
                            make_tuple(Sequence<0, 1, 2>{}));
                    },
                    Number<NumDTensor>{});
            }
            else
            {
                return generate_tuple(
                    [&](auto) {
                        // KRepeat x K1 x K2
                        return make_naive_tensor_descriptor_packed(
                            make_tuple(Number<KRepeat>{}, I1, Number<DsPerThread>{}));
                    },
                    Number<NumDTensor>{});
            }
        }();

        return ds_wave_desc;
    }

    using DsWaveDesc = decltype(MakeDsWaveDescriptor(DsBlockDesc{}));

    static constexpr DsWaveDesc ds_wave_desc_;

    __device__ __host__ static auto CalculateDsThreadOriginDataIndex()
    {
#ifdef __HIP_DEVICE_COMPILE__
        const auto wave_idx     = GetWaveIdx();
        const auto waveId_k     = wave_idx[I2];
        const auto wconv_ds_idx = acc_sba.CalculateDsThreadOriginDataIndex(KPerWconv);

        return make_tuple(waveId_k * KRepeat, wconv_ds_idx[I0], wconv_ds_idx[I1]);
#else
        return make_tuple(0, 0, 0);
#endif
    }
    using TupleDsData = decltype(CalculateDsThreadOriginDataIndex());
    __host__ __device__
    Blockwise_element_wconv_suba(const BlockwiseElementOp& blockOp,
                                 TupleDsData dsData_origin = CalculateDsThreadOriginDataIndex())
        : ds_thread_copy_(dsData_origin), scale_(type_convert<AccDataType>(blockOp.scale_))
    {
    }

    __host__ __device__ static auto CalculateThreadOriginDataIndex()
    {
        return generate_tuple([&](auto) { return CalculateDsThreadOriginDataIndex(); },
                              Number<NumDTensor>{});
    }

    __host__ __device__ static constexpr auto GetDsWaveDescLength()
    {
        constexpr auto DsPerThread = acc_sba.GetNumBiasComponents();
        return generate_tuple(
            [&](auto) { return Sequence<Number<KRepeat>{}, I1, Number<DsPerThread>{}>{}; },
            Number<NumDTensor>{});
    }

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wundefined-reinterpret-cast"
    template <typename DsBlockBuffer,
              typename AccDataVec,
              typename NumberH0,
              typename NumberW0,
              typename NumberK0>
    __device__ void
    Run(const DsBlockBuffer& ds_block_buf, AccDataVec& acc_vec, NumberH0, NumberW0, NumberK0) const
    {
        using BiasVec      = typename decltype(acc_sba)::BiasVec::type;
        using BiasScaleVec = typename decltype(acc_sba)::BiasScaleVec::type;

        // TOOD: move ds_thread_buf as class member and avoid copy ds data for each Run.
        auto ds_thread_buf = generate_tuple(
            [&](auto i) {
                return make_static_buffer<AddressSpaceEnum::Vgpr, AccDataType>(
                    ds_thread_desc_[Number<i>{}].GetElementSpaceSize());
            },
            Number<NumDTensor>{});

        constexpr auto calcOffset = [&](auto i) {
            if constexpr(DsEnableLds)
            {
                return ds_thread_desc_[i].CalculateOffset(make_tuple(I0, I0, I0));
            }
            else
            {
                return ds_wave_desc_[i].CalculateOffset(make_tuple(NumberK0{}, I0, I0));
            }
        };
        auto getBufPtr = [&](auto i, auto offset) {
            if constexpr(DsEnableLds)
            {
                return &ds_thread_buf[i][offset];
            }
            else
            {
                return &ds_block_buf[i][offset];
            }
        };

        // Load scale/bias data
        if constexpr(DsEnableLds)
        {
            static_for<0, NumDTensor, 1>{}([&](auto i) {
                ds_thread_copy_.Run(ds_wave_desc_[Number<i>{}],
                                    make_tuple(NumberK0{}, I0, I0),
                                    ds_block_buf[Number<i>{}],
                                    ds_thread_desc_[Number<i>{}],
                                    make_tuple(I0, I0, I0),
                                    ds_thread_buf(i));
            });
        }

        if constexpr(BlockwiseElementOp::packed)
        {
            constexpr index_t scale_offset = calcOffset(I0);
            constexpr index_t bias_offset  = calcOffset(I1);

            if constexpr(std::is_same<float, AccDataType>::value)
            {
                BiasVec bias;
                auto laneId = ThisThreadBlock::GetThreadId() & (WaveSize - 1);
                if(laneId % 2)
                {
                    bias = *reinterpret_cast<const BiasVec*>(getBufPtr(I0, Number<scale_offset>{}));
                }
                else
                {
                    bias = *reinterpret_cast<const BiasVec*>(getBufPtr(I1, Number<bias_offset>{}));
                }
                acc_sba.sba_instr.Run(acc_vec, 0, bias, acc_vec);
            }
            else if constexpr(std::is_same<ck::half_t, AccDataType>::value ||
                              std::is_same<ck::bhalf_t, AccDataType>::value)
            {
                BiasScaleVec bias_scale;
                bias_scale[0] =
                    *reinterpret_cast<const AccDataType*>(getBufPtr(I1, Number<bias_offset>{}));
                bias_scale[1] =
                    *reinterpret_cast<const AccDataType*>(getBufPtr(I0, Number<scale_offset>{}));

                if constexpr((HPerWconv == 4) && (WPerWconv == 2))
                {
                    // 4xhalf for output of convolve
                    // 4xhalf for input of sba/uba
                    using sba_out_vector_t = typename vector_type_maker_t<AccDataType, 2>::type;
                    sba_out_vector_t& sba_uba_output0 =
                        acc_vec.template AsType<sba_out_vector_t>()(Number<0>{});
                    sba_out_vector_t& sba_uba_output1 =
                        acc_vec.template AsType<sba_out_vector_t>()(Number<1>{});
                    acc_sba.sba_instr.Run(acc_vec, 0, bias_scale, sba_uba_output0, sba_uba_output1);
                }
                else
                {
                    acc_sba.sba_instr.Run(acc_vec, 0, bias_scale, acc_vec);
                }
            }
        }
        else
        {
            constexpr index_t bias_offset = calcOffset(I0);
            BiasVec bias = *reinterpret_cast<const BiasVec*>(getBufPtr(I0, Number<bias_offset>{}));
            if constexpr(std::is_same<float, AccDataType>::value)
            {
                acc_sba.sba_instr.Run(acc_vec, scale_, bias, acc_vec);
            }
            else if constexpr(std::is_same<half_t, AccDataType>::value ||
                              std::is_same<bhalf_t, AccDataType>::value)
            {
                if constexpr((HPerWconv == 4) && (WPerWconv == 2))
                {
                    // 4xhalf for output of convolve
                    // 4xhalf for input of sba/uba
                    using sba_out_vector_t = typename vector_type_maker_t<AccDataType, 2>::type;
                    sba_out_vector_t& sba_uba_output0 =
                        acc_vec.template AsType<sba_out_vector_t>()(Number<0>{});
                    sba_out_vector_t& sba_uba_output1 =
                        acc_vec.template AsType<sba_out_vector_t>()(Number<1>{});
                    acc_sba.sba_instr.Run(acc_vec, scale_, bias, sba_uba_output0, sba_uba_output1);
                }
                else
                {
                    acc_sba.sba_instr.Run(acc_vec, scale_, bias, acc_vec);
                }
            }
        }
    }
#pragma clang diagnostic pop
    // D[H, W, K, NumDsComp]
    template <index_t Components>
    static constexpr auto MakeDsDescriptor()
    {
        return make_naive_tensor_descriptor_packed(make_tuple(I1, I1, Number<Components>{}));
    }

    static constexpr auto ds_bias_thread_desc_  = MakeDsDescriptor<NumBias>();
    static constexpr auto ds_scale_thread_desc_ = MakeDsDescriptor<NumScale>();
    static constexpr auto ds_thread_desc_ = make_tuple(ds_scale_thread_desc_, ds_bias_thread_desc_);

    static_assert(NumBias == NumScale);
    using DsDataThreadLdsCopy =
        ThreadwiseTensorSliceTransfer_v4<AccDataType,
                                         AccDataType,
                                         remove_cvref_t<decltype(ds_wave_desc_[I0])>,
                                         remove_cvref_t<decltype(ds_thread_desc_[I0])>,
                                         Sequence<1, 1, NumBias>, // ToDo for NumScale
                                         Sequence<0, 1, 2>,
                                         2,
                                         1,
                                         1>;
    DsDataThreadLdsCopy ds_thread_copy_;
    float scale_;
    template <typename DsDesc, typename Index>
    static constexpr auto GetDsElementSpaceSize(const DsDesc&, Index i)
    {
        if constexpr(i < NumDTensor)
        {
            return DsDesc{}[i].GetElementSpaceSize();
        }
        else
        {
            return 0;
        }
    }
    struct SharedMemTrait
    {
        static constexpr auto max_lds_align = 8;

        static constexpr auto d0_block_space_size_aligned =
            DsEnableLds ? math::integer_least_multiple(GetDsElementSpaceSize(DsBlockDesc{}, I0),
                                                       max_lds_align)
                        : 0;

        static constexpr auto d1_block_space_size_aligned =
            DsEnableLds ? math::integer_least_multiple(GetDsElementSpaceSize(DsBlockDesc{}, I1),
                                                       max_lds_align)
                        : 0;

        static constexpr auto ds_block_space_size_aligned =
            make_tuple(d0_block_space_size_aligned, d1_block_space_size_aligned);

        // LDS allocation for Ds in LDS
        static constexpr auto d0_block_space_offset = 0;
        static constexpr auto d1_block_space_offset = d0_block_space_size_aligned;

        static constexpr auto ds_block_space_offset =
            make_tuple(d0_block_space_offset, d1_block_space_offset);

        static constexpr auto lds_size = d0_block_space_size_aligned * sizeof(AccDataType) +
                                         d1_block_space_size_aligned * sizeof(AccDataType);
    };

    struct LaneSharedMemTrait
    {
        static constexpr index_t max_lane_shared_align = 4;

        static constexpr index_t d0_block_space_size_aligned =
            EnableWaveGroup && (DsEnableLds == false)
                ? math::integer_least_multiple(GetDsElementSpaceSize(DsWaveDesc{}, I0) *
                                                   sizeof(AccDataType),
                                               max_lane_shared_align)
                : 0;
        static constexpr index_t d1_block_space_size_aligned =
            EnableWaveGroup && (DsEnableLds == false)
                ? math::integer_least_multiple(GetDsElementSpaceSize(DsWaveDesc{}, I1) *
                                                   sizeof(AccDataType),
                                               max_lane_shared_align)
                : 0;

        static constexpr index_t d0_block_space_offset = 0;
        static constexpr index_t d1_block_space_offset = d0_block_space_size_aligned;
        static constexpr auto ds_block_space_offset =
            make_tuple(d0_block_space_offset, d1_block_space_offset);
        static constexpr index_t lane_shared_size =
            d0_block_space_size_aligned + d1_block_space_size_aligned;
    };
};

template <typename ThisThreadBlock,
          typename AccDataType,
          index_t HPerBlock,
          index_t WPerBlock,
          index_t KPerBlock,
          index_t HPerWconv,
          index_t WPerWconv,
          index_t KPerWconv,
          index_t HRepeat,
          index_t WRepeat,
          index_t KRepeat,
          bool DsEnableLds,
          bool Aco,
          typename BlockwiseElementOp>
struct Blockwise_element_wconv_suba<ThisThreadBlock,
                                    AccDataType,
                                    Tuple<>,
                                    HPerBlock,
                                    WPerBlock,
                                    KPerBlock,
                                    HPerWconv,
                                    WPerWconv,
                                    KPerWconv,
                                    HRepeat,
                                    WRepeat,
                                    KRepeat,
                                    DsEnableLds,
                                    Aco,
                                    BlockwiseElementOp>
{
    static constexpr auto acc_sba = AccSba<AccDataType,
                                           HPerWconv,
                                           WPerWconv,
                                           BlockwiseElementOp::activeFunc,
                                           BlockwiseElementOp::packed,
                                           BlockwiseElementOp::uniform,
                                           Aco>{};

    template <typename DsGridDesc>
    __host__ __device__ static constexpr auto MakeDsGridBlockDescriptor(const DsGridDesc&)
    {
        return Tuple<>{};
    }

    template <typename DsBlockDesc_>
    __host__ __device__ static constexpr auto MakeDsWaveDescriptor(const DsBlockDesc_&)
    {
        return Tuple<>{};
    }

    __host__ __device__ static auto CalculateThreadOriginDataIndex() { return make_tuple(); }

    __host__ __device__ static constexpr auto GetDsWaveDescLength() { return make_tuple(); }

    __host__ __device__ Blockwise_element_wconv_suba(const BlockwiseElementOp& blockOp)
        : scale_(type_convert<AccDataType>(blockOp.scale_))
    {
    }

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wundefined-reinterpret-cast"
    template <typename DsBlockBuffer,
              typename AccDataVec,
              typename NumberH0,
              typename NumberW0,
              typename NumberK0>
    __device__ void
    Run(const DsBlockBuffer& ds_block_buf, AccDataVec& acc_vec, NumberH0, NumberW0, NumberK0) const
    {
        if constexpr(std::is_same<float, AccDataType>::value)
        {
            acc_sba.sba_instr.Run(acc_vec, scale_, 0, acc_vec);
        }
        else if constexpr(std::is_same<half_t, AccDataType>::value ||
                          std::is_same<bhalf_t, AccDataType>::value)
        {
            if constexpr((HPerWconv == 4) && (WPerWconv == 2))
            {
                using sba_out_vector_t = typename vector_type_maker_t<AccDataType, 2>::type;
                sba_out_vector_t& sba_uba_output0 =
                    acc_vec.template AsType<sba_out_vector_t>()(Number<0>{});
                sba_out_vector_t& sba_uba_output1 =
                    acc_vec.template AsType<sba_out_vector_t>()(Number<1>{});
                acc_sba.sba_instr.Run(acc_vec, scale_, 0, sba_uba_output0, sba_uba_output1);
            }
            else
            {
                acc_sba.sba_instr.Run(acc_vec, scale_, 0, acc_vec);
            }
        }
    }
#pragma clang diagnostic pop

    struct SharedMemTrait
    {
        static constexpr auto max_lds_align               = 8;
        static constexpr auto ds_block_space_size_aligned = make_tuple(0, 0);

        // LDS allocation for Ds in LDS
        static constexpr auto ds_block_space_offset = make_tuple(0, 0);
        static constexpr auto lds_size              = 0;
    };

    struct LaneSharedMemTrait
    {
        static constexpr auto max_lane_shared_align = 4;
        static constexpr auto ds_block_space_offset = make_tuple(0, 0);
        static constexpr auto lane_shared_size      = 0;
    };

    AccDataType scale_;
    static constexpr auto ds_wave_desc_ = make_tuple();
};

template <typename ThisThreadBlock,
          typename DsDataType,
          typename AccDataType,
          typename DsBlockDescs, // {Scale, Residual}
          index_t HPerBlock,
          index_t WPerBlock,
          index_t KPerBlock,
          index_t HPerWconv,
          index_t WPerWconv,
          index_t HRepeat,
          index_t WRepeat,
          index_t KRepeat,
          bool DsEnableLds,
          bool Aco,
          typename BlockwiseElementOp>
struct Blockwise_element_wconv_fma
{
    static constexpr auto I0              = Number<0>{};
    static constexpr auto I1              = Number<1>{};
    static constexpr auto I2              = Number<2>{};
    static constexpr auto I3              = Number<3>{};
    static constexpr auto I4              = Number<4>{};
    static constexpr index_t WaveSize     = 32;
    static constexpr bool EnableWaveGroup = ThisThreadBlock::InWaveGroup();
    static_assert((DsDataType::Size() == 2) ||
                  (DsDataType::Size() == 1 && BlockwiseElementOp::uniform));
    static constexpr bool UniformScale     = BlockwiseElementOp::uniform;
    static constexpr index_t ResidualIndex = UniformScale ? 0 : 1;
    static constexpr index_t ScaleIndex    = UniformScale ? -1 : 0;
    static constexpr auto ResidualId       = Number<ResidualIndex>{};
    static constexpr auto ScaleId          = Number<ScaleIndex>{};
    using ResidualDataType = remove_cvref_t<tuple_element_t<ResidualIndex, DsDataType>>;
    using WconvFma = WconvFmaFromTensor<ResidualDataType, AccDataType, HPerWconv, WPerWconv>;

    static constexpr WconvFma wconv_fma;
    static_assert(wconv_fma.GetAccumChannelOrder() == Aco);
    using WconvConv = WconvConv<ResidualDataType,
                                ResidualDataType,
                                AccDataType,
                                HPerWconv,
                                WPerWconv,
                                1,
                                1,
                                1,
                                1,
                                EnableWaveGroup,
                                Aco>;
    static constexpr WconvConv wconv_conv;
    using ScaleDataVec = typename WconvConv::AccDataVec;

    static constexpr index_t CPerWconv               = wconv_conv.GetNumInputChannels();
    static constexpr index_t KPerWconv               = wconv_conv.GetNumOutputChannels();
    static constexpr index_t NumSubTilePerImage      = wconv_conv.GetNumSubTilesPerImageTile();
    static constexpr index_t NumSubTilesPerWeightTap = wconv_conv.GetNumSubTilesPerWeightTap();
    static constexpr index_t NumWeightCompPerTile    = wconv_conv.GetNumWeightCompPerTile();
    static constexpr index_t NumWeightTap            = wconv_conv.GetNumWeightTap();
    static constexpr index_t HPerWave                = HRepeat * HPerWconv;
    static constexpr index_t WPerWave                = WRepeat * WPerWconv;
    static constexpr index_t KPerWave                = KPerBlock;
    static constexpr auto NumDataCompPerTile         = wconv_conv.GetNumDataCompPerTile();

    // Accum descriptor info
    static constexpr auto NumAccComp        = wconv_conv.GetNumAccumComponents();
    static constexpr auto NumAccCompPerTile = NumAccComp / NumSubTilePerImage;
    static constexpr auto NumAccSwizzleComp = Aco ? 2 : 4;
    static constexpr auto NumAccCompSubTile =
        NumAccCompPerTile > NumAccSwizzleComp ? NumAccCompPerTile / NumAccSwizzleComp : 1;

    template <typename DsGridDesc>
    __host__ __device__ static constexpr auto
    MakeDsGridBlockDescriptor(const DsGridDesc& ds_grid_desc)
    {
        if constexpr(DsEnableLds)
        {
            return ds_grid_desc;
        }
        else
        {
            // Residual: H x W x C -> W0 x C0 x H0 x H1 x H2 x W1 x C1
            const auto H0                 = ds_grid_desc[I0].GetLength(I0);
            const auto W0                 = ds_grid_desc[I0].GetLength(I1);
            const auto C0                 = ds_grid_desc[I0].GetLength(I2);
            auto residual_grid_block_desc = transform_tensor_descriptor(
                ds_grid_desc[ResidualId],
                make_tuple(
                    make_unmerge_transform(make_tuple(H0 / HPerWconv,
                                                      Number<NumSubTilePerImage>{},
                                                      Number<HPerWconv / NumSubTilePerImage>{})),
                    make_unmerge_transform(make_tuple(W0 / WPerWconv, Number<WPerWconv>{})),
                    make_unmerge_transform(make_tuple(C0 / CPerWconv, Number<CPerWconv>{}))),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                make_tuple(Sequence<2, 3, 4>{}, Sequence<0, 5>{}, Sequence<1, 6>{}));
            if constexpr(UniformScale)
            {
                return make_tuple(residual_grid_block_desc);
            }
            else
            {
                // Scale: H x W x K -> H0 x W0 x K0 x H1 x H2 x W1 x K1 x K2
                const auto H1              = ds_grid_desc[I1].GetLength(I0);
                const auto W1              = ds_grid_desc[I1].GetLength(I1);
                const auto C1              = ds_grid_desc[I1].GetLength(I2);
                auto scale_grid_block_desc = transform_tensor_descriptor(
                    ds_grid_desc[ScaleId],
                    make_tuple(
                        make_unmerge_transform(
                            make_tuple(H1 / HPerWconv,
                                       Number<NumSubTilePerImage>{},
                                       Number<HPerWconv / NumSubTilePerImage>{})),
                        make_unmerge_transform(make_tuple(W1 / WPerWconv, Number<WPerWconv>{})),
                        make_unmerge_transform(
                            make_tuple(C1 / KPerWconv,
                                       Number<NumAccCompSubTile>{},
                                       Number<KPerWconv / NumAccCompSubTile>{}))),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                    make_tuple(Sequence<0, 3, 4>{}, Sequence<1, 5>{}, Sequence<2, 6, 7>{}));
                return make_tuple(scale_grid_block_desc, residual_grid_block_desc);
            }
        }
    }

    // Describe how data read from (LDS/VGPR) buffer, used by Block level classes
    template <typename DsBlockDesc_>
    __host__ __device__ static constexpr auto MakeScaleWaveDescriptor(const DsBlockDesc_&)
    {
        // H x W x K -> H0 x W0 x K0 x H1 x H2 x W1 x K1 x K2
        if constexpr(DsEnableLds)
        {
            // NOTE: always use I0 to create scale wave desc
            // In uniform mode, this descriptor isn't really used.
            return transform_tensor_descriptor(
                DsBlockDesc_{}[I0],
                make_tuple(
                    make_unmerge_transform(make_tuple(Number<HPerBlock / HPerWconv>{},
                                                      Number<NumSubTilePerImage>{},
                                                      Number<HPerWconv / NumSubTilePerImage>{})),
                    make_unmerge_transform(
                        make_tuple(Number<WPerBlock / WPerWconv>{}, Number<WPerWconv>{})),
                    make_unmerge_transform(make_tuple(Number<KPerBlock / KPerWconv>{},
                                                      Number<NumAccCompSubTile>{},
                                                      Number<KPerWconv / NumAccCompSubTile>{}))),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                make_tuple(Sequence<0, 3, 4>{}, Sequence<1, 5>{}, Sequence<2, 6, 7>{}));
        }
        else
        {
            // TODO: adjust stride according to DsBlockDesc_{}[I1];
            return make_naive_tensor_descriptor_packed(
                make_tuple(Number<HPerWave / HPerWconv>{},
                           Number<WPerWave / WPerWconv>{},
                           Number<KPerWave / KPerWconv>{},
                           Number<NumSubTilePerImage>{},
                           I1,
                           I1,
                           Number<NumAccCompSubTile>{},
                           Number<NumAccCompPerTile / NumAccCompSubTile>{}));
        }
    }

    template <typename DsBlockDesc_>
    __host__ __device__ static constexpr auto MakeResidualWaveDescriptor(const DsBlockDesc_&)
    {
        if constexpr(DsEnableLds)
        {
            // H x W x C -> W0 x C0 x H0 x H1 x H2 x W1 x C1
            return transform_tensor_descriptor(
                DsBlockDesc_{}[ResidualId],
                make_tuple(
                    make_unmerge_transform(make_tuple(Number<HPerBlock / HPerWconv>{},
                                                      Number<NumSubTilePerImage>{},
                                                      Number<HPerWconv / NumSubTilePerImage>{})),
                    make_unmerge_transform(
                        make_tuple(Number<WPerBlock / WPerWconv>{}, Number<WPerWconv>{})),
                    make_unmerge_transform(
                        make_tuple(Number<KPerBlock / CPerWconv>{}, Number<CPerWconv>{}))),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                make_tuple(Sequence<2, 3, 4>{}, Sequence<0, 5>{}, Sequence<1, 6>{}));
        }
        else
        {
            // TODO: adjust stride according to DsBlockDesc_{}[I0];
            return make_naive_tensor_descriptor_packed(make_tuple(Number<WPerWave / WPerWconv>{},
                                                                  Number<KPerWave / CPerWconv>{},
                                                                  Number<HPerWave / HPerWconv>{},
                                                                  Number<NumSubTilePerImage>{},
                                                                  I1,
                                                                  I1,
                                                                  Number<NumDataCompPerTile>{}));
        }
    }

    template <typename DsBlockDesc_>
    __host__ __device__ static constexpr auto MakeDsWaveDescriptor(const DsBlockDesc_&)
    {
        if constexpr(UniformScale)
        {
            return make_tuple(MakeResidualWaveDescriptor(DsBlockDescs{}));
        }
        else
        {
            return make_tuple(MakeScaleWaveDescriptor(DsBlockDescs{}),
                              MakeResidualWaveDescriptor(DsBlockDescs{}));
        }
    }

    using ScaleWaveDesc    = decltype(MakeScaleWaveDescriptor(DsBlockDescs{}));
    using ResidualWaveDesc = decltype(MakeResidualWaveDescriptor(DsBlockDescs{}));

    static constexpr ScaleWaveDesc scale_wave_desc_;
    static constexpr ResidualWaveDesc residual_wave_desc_;
    static constexpr auto ds_wave_desc_ = MakeDsWaveDescriptor(DsBlockDescs{});

    static constexpr auto NumDataSubTiles = wconv_conv.GetNumSubTilesPerImageTile();
    static constexpr auto NumDataTiles    = wconv_conv.GetNumImageTilesInVertical();
    static constexpr auto residual_thread_desc_ =
        make_naive_tensor_descriptor_packed(make_tuple(I1,
                                                       I1,
                                                       Number<NumDataTiles>{},
                                                       Number<NumDataSubTiles>{},
                                                       I1,
                                                       I1,
                                                       Number<NumDataCompPerTile>{}));

    static constexpr auto scale_thread_desc_ = make_naive_tensor_descriptor_packed(
        make_tuple(I1,
                   I1,
                   I1,
                   Number<NumSubTilePerImage>{},
                   I1,
                   I1,
                   Number<NumAccCompSubTile>{},
                   Number<NumAccCompPerTile / NumAccCompSubTile>{}));

    using ResidualDataThreadLdsCopy = ThreadwiseTensorSliceTransfer_v4<
        ResidualDataType,
        ResidualDataType,
        ResidualWaveDesc,
        decltype(residual_thread_desc_),
        Sequence<1, 1, NumDataTiles, NumDataSubTiles, 1, 1, NumDataCompPerTile>,
        Sequence<0, 1, 2, 3, 4, 5, 6>,
        6,
        NumDataCompPerTile,
        NumDataCompPerTile>;
    ResidualDataThreadLdsCopy residual_thread_copy_;

    using ScaleDataThreadLdsCopy =
        ThreadwiseTensorSliceTransfer_v4<AccDataType,
                                         AccDataType,
                                         ScaleWaveDesc,
                                         decltype(scale_thread_desc_),
                                         Sequence<1,
                                                  1,
                                                  1,
                                                  NumSubTilePerImage,
                                                  1,
                                                  1,
                                                  NumAccCompSubTile,
                                                  NumAccCompPerTile / NumAccCompSubTile>,
                                         Sequence<0, 1, 2, 3, 4, 5, 6, 7>,
                                         7,
                                         1,
                                         1>;
    ScaleDataThreadLdsCopy scale_thread_copy_;

    __device__ static auto GetWaveIdx()
    {
        return GetWconvWaveIdx<ThisThreadBlock,
                               HPerBlock,
                               WPerBlock,
                               HRepeat,
                               WRepeat,
                               HPerWconv,
                               WPerWconv>();
    }

    __device__ static auto CalculateScaleThreadOriginDataIndex()
    {
#ifdef __HIP_DEVICE_COMPILE__
        const auto wave_idx     = GetWaveIdx();
        const auto waveId_h     = wave_idx[I0];
        const auto waveId_w     = wave_idx[I1];
        const auto waveId_k     = wave_idx[I2];
        const auto wconv_ds_idx = wconv_conv.CalculateAccThreadOriginDataIndex();

        return make_tuple(waveId_h * HRepeat,
                          waveId_w * WRepeat,
                          waveId_k * KRepeat,
                          wconv_ds_idx[I0],
                          wconv_ds_idx[I1],
                          wconv_ds_idx[I2],
                          wconv_ds_idx[I3],
                          wconv_ds_idx[I4]);
#else
        return make_tuple(0, 0, 0, 0, 0, 0, 0, 0);
#endif
    }
    __device__ static auto CalculateResidualThreadOriginDataIndex()
    {
#ifdef __HIP_DEVICE_COMPILE__
        const auto wave_idx          = GetWaveIdx();
        const auto waveId_h          = wave_idx[I0];
        const auto waveId_w          = wave_idx[I1];
        const auto wconv_in_data_idx = wconv_conv.CalculateInDataThreadOriginDataIndex();
        // W0 x C0 x H0 x H1 x H2 x W1 x C1
        return make_tuple(waveId_w * WRepeat,
                          0,
                          waveId_h * HRepeat,
                          0,
                          wconv_in_data_idx[I0],
                          wconv_in_data_idx[I1],
                          wconv_in_data_idx[I2]);
#else
        return make_tuple(0, 0, 0, 0, 0, 0, 0);
#endif
    }

    using TupleScaleData    = decltype(CalculateScaleThreadOriginDataIndex());
    using TupleResidualData = decltype(CalculateResidualThreadOriginDataIndex());
    __host__ __device__ Blockwise_element_wconv_fma(
        const BlockwiseElementOp& blockOp,
        TupleResidualData residualData_origin = CalculateResidualThreadOriginDataIndex(),
        TupleScaleData scaleData_origin       = CalculateScaleThreadOriginDataIndex())
        : residual_thread_copy_(residualData_origin), scale_thread_copy_(scaleData_origin)
    {
        if constexpr(UniformScale)
        {
            static_for<0, wconv_conv.GetNumAccumComponents(), 1>{}(
                [&](auto i) { scale_[i.value] = type_convert<AccDataType>(blockOp.scale_); });
        }
    }

    __host__ __device__ static auto CalculateThreadOriginDataIndex()
    {
        if constexpr(UniformScale)
        {
            return make_tuple(CalculateResidualThreadOriginDataIndex());
        }
        else
        {
            return make_tuple(CalculateScaleThreadOriginDataIndex(),
                              CalculateResidualThreadOriginDataIndex());
        }
    }

    __host__ __device__ static constexpr auto GetDsWaveDescLength()
    {
        using ScaleWaveDescLength = Sequence<HPerWave / HPerWconv,
                                             WPerWave / WPerWconv,
                                             KPerWave / KPerWconv,
                                             NumSubTilePerImage,
                                             1,
                                             1,
                                             NumAccCompSubTile,
                                             NumAccCompPerTile / NumAccCompSubTile>;
        using ResidualDescLength  = Sequence<WPerWave / WPerWconv,
                                            KPerWave / CPerWconv,
                                            HPerWave / HPerWconv,
                                            NumSubTilePerImage,
                                            1,
                                            1,
                                            NumDataCompPerTile>;
        if constexpr(UniformScale)
        {
            return make_tuple(ResidualDescLength{});
        }
        else
        {
            return make_tuple(ScaleWaveDescLength{}, ResidualDescLength{});
        }
    }
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wundefined-reinterpret-cast"
    template <typename DsBlockBuffer,
              typename AccDataVec,
              typename NumberH0,
              typename NumberW0,
              typename NumberK0>
    __device__ void
    Run(const DsBlockBuffer& ds_block_buf, AccDataVec& acc_vec, NumberH0, NumberW0, NumberK0) const
    {
        static_assert(std::is_same<AccDataVec, typename decltype(wconv_conv)::AccDataVec>::value);
        using ResidualDataVec = typename decltype(wconv_conv)::InDataVec;

        auto load_residual_data = [&](auto h0, auto w0, auto c0) {
            if constexpr(DsEnableLds)
            {
                // Load residual tensor data
                auto residual_thread_buf =
                    make_static_buffer<AddressSpaceEnum::Vgpr, ResidualDataType>(
                        residual_thread_desc_.GetElementSpaceSize());
                residual_thread_copy_.Run(residual_wave_desc_,
                                          make_tuple(w0, c0, h0, I0, I0, I0, I0),
                                          ds_block_buf[ResidualId],
                                          residual_thread_desc_,
                                          make_tuple(I0, I0, I0, I0, I0, I0, I0),
                                          residual_thread_buf);
                return bit_cast<typename ResidualDataVec::type>(residual_thread_buf);
            }
            else
            {
                constexpr index_t residual_offset =
                    residual_wave_desc_.CalculateOffset(make_tuple(w0, c0, h0, I0, I0, I0, I0));
                return *reinterpret_cast<const typename ResidualDataVec::type*>(
                    &ds_block_buf[ResidualId][Number<residual_offset>{}]);
            }
        };

        auto load_scale_data = [&](auto h0, auto w0, auto k0) {
            if constexpr(UniformScale)
            {
                return scale_;
            }
            else if constexpr(DsEnableLds)
            {
                auto scale_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, AccDataType>(
                    scale_thread_desc_.GetElementSpaceSize());

                scale_thread_copy_.Run(scale_wave_desc_,
                                       make_tuple(h0, w0, k0, I0, I0, I0, I0, I0),
                                       ds_block_buf[ScaleId],
                                       scale_thread_desc_,
                                       make_tuple(I0, I0, I0, I0, I0, I0, I0, I0),
                                       scale_thread_buf);
                return bit_cast<typename ScaleDataVec::type>(scale_thread_buf);
            }
            else
            {
                constexpr index_t scale_offset =
                    scale_wave_desc_.CalculateOffset(make_tuple(h0, w0, k0, I0, I0, I0, I0, I0));
                return *reinterpret_cast<const typename ScaleDataVec::type*>(
                    &ds_block_buf[ScaleId][Number<scale_offset>{}]);
            }
        };

        constexpr NumberH0 h0;
        constexpr NumberW0 w0;
        constexpr NumberK0 k0;

        auto scale = load_scale_data(h0, w0, k0);

        if constexpr(wconv_fma.GetNumResidual() == 4)
        {
            constexpr index_t c0 = k0 * 4;
            auto residual_data0  = load_residual_data(h0, w0, Number<c0>{});
            auto residual_data1  = load_residual_data(h0, w0, Number<c0 + 1>{});
            auto residual_data2  = load_residual_data(h0, w0, Number<c0 + 2>{});
            auto residual_data3  = load_residual_data(h0, w0, Number<c0 + 3>{});
            wconv_fma.fma_instr.Run(
                acc_vec.template AsType<typename AccDataVec::type>()(Number<0>{}),
                residual_data0,
                residual_data1,
                residual_data2,
                residual_data3,
                scale,
                acc_vec);
        }
        else if constexpr(wconv_fma.GetNumResidual() == 2)
        {
            constexpr index_t c0 = k0 * 2;
            auto residual_data0  = load_residual_data(h0, w0, Number<c0>{});
            auto residual_data1  = load_residual_data(h0, w0, Number<c0 + 1>{});
            wconv_fma.fma_instr.Run(
                acc_vec.template AsType<typename AccDataVec::type>()(Number<0>{}),
                residual_data0,
                residual_data1,
                scale,
                acc_vec);
        }
        else if constexpr(wconv_fma.GetNumResidual() == 1)
        {
            constexpr index_t numAccum = wconv_fma.GetNumAccum();
            constexpr index_t c0       = k0 / numAccum;
            auto residual_data0        = load_residual_data(h0, w0, Number<c0>{});
            if constexpr(k0 % numAccum == 0)
            {
                wconv_fma.fma_instr.Run(
                    acc_vec.template AsType<typename AccDataVec::type>()(Number<0>{}),
                    residual_data0,
                    scale,
                    acc_vec);
            }
            else
            {
                static_assert((numAccum == 2) && (k0 % numAccum == 1));
                constexpr index_t ChanOff = (HPerWconv == 8) && (WPerWconv == 4) ? 8 : 16;
                constexpr WconvFmaFromTensor<ResidualDataType,
                                             AccDataType,
                                             HPerWconv,
                                             WPerWconv,
                                             ChanOff>
                    wconvFma2;
                wconvFma2.fma_instr.Run(acc_vec.template AsType<AccDataVec::type>()(Number<0>{}),
                                        residual_data0,
                                        scale,
                                        acc_vec);
            }
        }
    }
#pragma clang diagnostic pop

    template <typename DsDescs>
    static constexpr auto GetScaleElementSpaceSize(const DsDescs&)
    {
        if constexpr(UniformScale)
        {
            return 0;
        }
        else
        {
            return DsDescs{}[ScaleId].GetElementSpaceSize();
        }
    }

    static constexpr auto GetDsLayoutTuple(index_t scale, index_t residual)
    {
        if constexpr(UniformScale)
        {
            return make_tuple(residual, 0);
        }
        else
        {
            return make_tuple(scale, residual);
        }
    }

    struct SharedMemTrait
    {
        static constexpr auto max_lds_align = 8;

        static constexpr auto scale_block_space_size_aligned =
            DsEnableLds ? math::integer_least_multiple(
                              DsBlockDescs{}[ResidualId].GetElementSpaceSize(), max_lds_align)
                        : 0;

        static constexpr auto residual_block_space_size_aligned =
            DsEnableLds ? math::integer_least_multiple(GetScaleElementSpaceSize(DsBlockDescs{}),
                                                       max_lds_align) *
                              WconvFma::template SizeOfBits<ResidualDataType>() /
                              WconvFma::template SizeOfBits<AccDataType>()
                        : 0;

        static constexpr auto ds_block_space_size_aligned =
            GetDsLayoutTuple(scale_block_space_size_aligned, residual_block_space_size_aligned);

        // LDS allocation for Ds in LDS
        static constexpr auto scale_block_space_offset = 0;
        static constexpr auto residual_block_space_offset =
            scale_block_space_offset + scale_block_space_size_aligned;

        static constexpr auto ds_block_space_offset =
            GetDsLayoutTuple(scale_block_space_offset, residual_block_space_offset);

        static constexpr auto lds_size = residual_block_space_size_aligned * sizeof(AccDataType) +
                                         scale_block_space_size_aligned * sizeof(AccDataType);
    };

    struct LaneSharedMemTrait
    {
        static constexpr index_t max_lane_shared_align = 4;

        static constexpr index_t residual_block_space_size_aligned =
            EnableWaveGroup && (DsEnableLds == false)
                ? math::integer_least_multiple(
                      ResidualWaveDesc{}.GetElementSpaceSize() *
                          WconvFma::template SizeOfBits<ResidualDataType>() / 8,
                      max_lane_shared_align)
                : 0;
        static constexpr index_t scale_block_space_size_aligned =
            EnableWaveGroup && (DsEnableLds == false)
                ? math::integer_least_multiple(GetScaleElementSpaceSize(ds_wave_desc_) *
                                                   sizeof(AccDataType),
                                               max_lane_shared_align)
                : 0;

        static constexpr index_t scale_block_space_offset = 0;
        static constexpr index_t residual_block_space_offset =
            scale_block_space_offset + scale_block_space_size_aligned;
        static constexpr auto ds_block_space_offset =
            GetDsLayoutTuple(scale_block_space_offset, residual_block_space_offset);
        static constexpr index_t lane_shared_size =
            residual_block_space_size_aligned + scale_block_space_size_aligned;
    };
    typename ScaleDataVec::type scale_;
};

template <typename ThisThreadBlock,
          typename DsDataType,
          typename AccDataType,
          typename DsBlockDescs,
          index_t HPerBlock,
          index_t WPerBlock,
          index_t KPerBlock,
          index_t HPerWconv,
          index_t WPerWconv,
          index_t KPerWconv,
          index_t HRepeat,
          index_t WRepeat,
          index_t KRepeat,
          bool DsEnableLds,
          bool Aco,
          typename BlockwiseElementOp>
struct Blockwise_element_wconv_suba_fma
{
    static constexpr index_t NumDTensor    = DsBlockDescs::Size();
    static constexpr index_t NumSbaDTensor = BlockwiseElementOp::Sba::GetNumDTensor();
    static constexpr index_t NumFmaDTensor = BlockwiseElementOp::Fma::GetNumDTensor();
    static_assert(NumDTensor == NumSbaDTensor + NumFmaDTensor);
    static_assert(NumDTensor == 2); // todo remove this limitation

    static constexpr auto GetSbaDsBlockDesc()
    {
        return generate_tuple([&](auto i) { return DsBlockDescs{}[i]; }, Number<NumSbaDTensor>{});
    }

    static constexpr auto GetFmaDsBlockDesc()
    {
        return generate_tuple([&](auto i) { return DsBlockDescs{}[Number<i + NumSbaDTensor>{}]; },
                              Number<NumFmaDTensor>{});
    }

    static constexpr auto GetFmaDsDataType()
    {
        return generate_tuple([&](auto i) { return DsDataType{}[Number<i + NumSbaDTensor>{}]; },
                              Number<NumFmaDTensor>{});
    }

    using BlockwiseSuba =
        Blockwise_element_wconv_suba<ThisThreadBlock,
                                     AccDataType,
                                     decltype(GetSbaDsBlockDesc()), // { Bias, Scale}
                                     HPerBlock,
                                     WPerBlock,
                                     KPerBlock,
                                     HPerWconv,
                                     WPerWconv,
                                     KPerWconv,
                                     HRepeat,
                                     WRepeat,
                                     KRepeat,
                                     DsEnableLds,
                                     Aco,
                                     typename BlockwiseElementOp::Sba>;
    using BlockwiseFma =
        Blockwise_element_wconv_fma<ThisThreadBlock,
                                    decltype(GetFmaDsDataType()),
                                    AccDataType,
                                    decltype(GetFmaDsBlockDesc()), // {Residual, Scale}
                                    HPerBlock,
                                    WPerBlock,
                                    KPerBlock,
                                    HPerWconv,
                                    WPerWconv,
                                    HRepeat,
                                    WRepeat,
                                    KRepeat,
                                    DsEnableLds,
                                    Aco,
                                    typename BlockwiseElementOp::Fma>;

    BlockwiseSuba blockwise_suba_;
    BlockwiseFma blockwise_fma_;

    static constexpr auto ds_wave_desc_ =
        container_concat(BlockwiseSuba::ds_wave_desc_, BlockwiseFma::ds_wave_desc_);

    __host__ __device__ Blockwise_element_wconv_suba_fma(const BlockwiseElementOp& blockOp)
        : blockwise_suba_(blockOp.sba_), blockwise_fma_(blockOp.fma_)

    {
    }

    template <typename DsGridDesc>
    __host__ __device__ static constexpr auto
    MakeDsGridBlockDescriptor(const DsGridDesc& ds_grid_desc)
    {
        auto sba_grid_desc =
            generate_tuple([&](auto i) { return ds_grid_desc[i]; }, Number<NumSbaDTensor>{});

        auto fma_grid_desc =
            generate_tuple([&](auto i) { return ds_grid_desc[Number<i + NumSbaDTensor>{}]; },
                           Number<NumFmaDTensor>{});
        return container_concat(BlockwiseSuba::MakeDsGridBlockDescriptor(sba_grid_desc),
                                BlockwiseFma::MakeDsGridBlockDescriptor(fma_grid_desc));
    }

    __host__ __device__ static auto CalculateThreadOriginDataIndex()
    {
        return container_concat(BlockwiseSuba::CalculateThreadOriginDataIndex(),
                                BlockwiseFma::CalculateThreadOriginDataIndex());
    }

    __host__ __device__ static constexpr auto GetDsWaveDescLength()
    {
        return container_concat(BlockwiseSuba::GetDsWaveDescLength(),
                                BlockwiseFma::GetDsWaveDescLength());
    }

    template <typename DsBlockBuffer,
              typename AccDataVec,
              typename NumberH0,
              typename NumberW0,
              typename NumberK0>
    __device__ void
    Run(const DsBlockBuffer& ds_block_buf, AccDataVec& acc_vec, NumberH0, NumberW0, NumberK0) const
    {
        auto sba_block_buf =
            generate_tuple([&](auto i) { return ds_block_buf[i]; }, Number<NumSbaDTensor>{});

        auto fma_block_buf =
            generate_tuple([&](auto i) { return ds_block_buf[Number<i + NumSbaDTensor>{}]; },
                           Number<NumFmaDTensor>{});

        blockwise_suba_.Run(sba_block_buf, acc_vec, NumberH0{}, NumberW0{}, NumberK0{});
        blockwise_fma_.Run(fma_block_buf, acc_vec, NumberH0{}, NumberW0{}, NumberK0{});
    }

    template <typename SbaMemTrait, typename FmaMemTrait>
    static constexpr auto
    MergeDsMemTraid(const SbaMemTrait& sba, const FmaMemTrait& fma, const index_t offset)
    {
        auto sba_ = generate_tuple([&](auto i) { return sba[i]; }, Number<NumSbaDTensor>{});

        auto fma_ =
            generate_tuple([&](auto i) { return fma[i] + offset; }, Number<NumFmaDTensor>{});
        return container_concat(sba_, fma_);
    }

    struct SharedMemTrait
    {
        static constexpr auto max_lds_align = 8;
        static constexpr auto ds_block_space_size_aligned =
            MergeDsMemTraid(BlockwiseSuba::SharedMemTrait::ds_block_space_size_aligned,
                            BlockwiseFma::SharedMemTrait::ds_block_space_size_aligned,
                            0);

        // LDS allocation for Ds in LDS
        static constexpr index_t fma_block_space_offset =
            BlockwiseSuba::SharedMemTrait::lds_size / sizeof(AccDataType);
        static constexpr auto ds_block_space_offset =
            MergeDsMemTraid(BlockwiseSuba::SharedMemTrait::ds_block_space_offset,
                            BlockwiseFma::SharedMemTrait::ds_block_space_offset,
                            fma_block_space_offset);
        static constexpr auto lds_size =
            BlockwiseSuba::SharedMemTrait::lds_size + BlockwiseFma::SharedMemTrait::lds_size;
    };

    struct LaneSharedMemTrait
    {
        static constexpr auto max_lane_shared_align = 4;
        static constexpr index_t fma_block_space_offset =
            BlockwiseSuba::LaneSharedMemTrait::lane_shared_size;
        static constexpr auto ds_block_space_offset =
            MergeDsMemTraid(BlockwiseSuba::LaneSharedMemTrait::ds_block_space_offset,
                            BlockwiseFma::LaneSharedMemTrait::ds_block_space_offset,
                            fma_block_space_offset);
        static constexpr auto lane_shared_size =
            BlockwiseSuba::LaneSharedMemTrait::lane_shared_size +
            BlockwiseFma::LaneSharedMemTrait::lane_shared_size;
    };
};

template <typename ThisThreadBlock,
          typename OutTensorDataType,
          typename AccDataType,
          typename OutTensorBlockDesc,
          typename AccBlockDesc,
          index_t HPerBlock,
          index_t WPerBlock,
          index_t KPerBlock,
          index_t HPerWconv,
          index_t WPerWconv,
          index_t KPerWconv,
          index_t CPerWconv,
          index_t HRepeat,
          index_t WRepeat,
          index_t KRepeat,
          bool Aco,
          typename BlockwiseNextElementOp>
struct Blockwise_element_wconv_cvtTensor
{
    static constexpr auto I0            = Number<0>{};
    static constexpr auto I1            = Number<1>{};
    static constexpr auto I2            = Number<2>{};
    static constexpr index_t WaveSize   = 32;
    static constexpr index_t NumDTensor = 2;

    __device__ static auto GetWaveIdx()
    {
        return GetWconvWaveIdx<ThisThreadBlock,
                               HPerBlock,
                               WPerBlock,
                               HRepeat,
                               WRepeat,
                               HPerWconv,
                               WPerWconv>();
    }

    static constexpr AccBlockDesc accum_thread_desc_;
    static constexpr OutTensorBlockDesc out_tensor_thread_desc_;

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wundefined-reinterpret-cast"
    template <typename AccDataVec,
              typename OutTensorThreadBuffer,
              typename NumberH0,
              typename NumberW0,
              typename NumberK0>
    __device__ void Run(const AccDataVec& acc_vec,
                        OutTensorThreadBuffer& out_thread_buf,
                        NumberH0,
                        NumberW0,
                        NumberK0) const
    {
        constexpr NumberH0 h0;
        constexpr NumberW0 w0;
        constexpr NumberK0 k0;

        constexpr auto accCvtInstance = ck::AccCvtTensor<OutTensorDataType,
                                                         AccDataType,
                                                         HPerWconv,
                                                         WPerWconv,
                                                         BlockwiseNextElementOp::activeFunc,
                                                         BlockwiseNextElementOp::clamp>();
        constexpr index_t indata_offset =
            out_tensor_thread_desc_.CalculateOffset(make_tuple(h0, w0, k0, I0));
        auto& outVec = out_thread_buf.GetVectorTypeReference(Number<indata_offset>{});
        constexpr index_t convert_scale = BlockwiseNextElementOp::scale;
        accCvtInstance.cvtTensor_instr.Run(acc_vec, convert_scale, outVec);
    }
#pragma clang diagnostic pop
};

template <typename AccBlockDesc>
struct Blockwise_element_wconv_out_passthrough
{
    template <typename AccDataVec,
              typename OutTensorThreadBuffer,
              typename NumberH0,
              typename NumberW0,
              typename NumberK0>
    __device__ void Run(const AccDataVec& acc_vec,
                        OutTensorThreadBuffer& out_thread_buf,
                        NumberH0,
                        NumberW0,
                        NumberK0) const
    {
        constexpr auto accum_offset = accum_thread_desc_.CalculateOffset(
            make_tuple(NumberH0{}, NumberW0{}, NumberK0{}, Number<0>{}));
        out_thread_buf.GetVectorTypeReference(Number<accum_offset>{}) = acc_vec;
    }
    static constexpr AccBlockDesc accum_thread_desc_;
};

struct Blockwise_element_passthrough
{
    static constexpr auto ds_wave_desc_ = make_tuple();
    __host__ __device__
    Blockwise_element_passthrough(const convolution ::BlockwiseElementOpPassThrough&)
    {
    }

    template <typename DsGridDesc>
    __host__ __device__ static constexpr auto
    MakeDsGridBlockDescriptor(const DsGridDesc& ds_grid_desc)
    {
        return ds_grid_desc;
    }

    __host__ __device__ static auto CalculateThreadOriginDataIndex() { return make_tuple(); }

    __host__ __device__ static constexpr auto GetDsWaveDescLength() { return make_tuple(); }

    template <typename DsBlockBuffer,
              typename AccDataVec,
              typename NumberH0,
              typename NumberW0,
              typename NumberK0>
    __device__ void Run(const DsBlockBuffer&, AccDataVec&, NumberH0, NumberW0, NumberK0) const
    {
    }

    struct SharedMemTrait
    {
        static constexpr auto max_lds_align               = 8;
        static constexpr auto ds_block_space_size_aligned = make_tuple(0, 0);

        // LDS allocation for Ds in LDS
        static constexpr auto ds_block_space_offset = make_tuple(0, 0);
        static constexpr auto lds_size              = 0;
    };

    struct LaneSharedMemTrait
    {
        static constexpr auto max_lane_shared_align = 4;
        static constexpr auto ds_block_space_offset = make_tuple(0, 0);
        static constexpr auto lane_shared_size      = 0;
    };
};

template <typename ThisThreadBlock,
          typename WeiDataType,
          typename InDataType,
          typename DsDataType,
          typename AccDataType,
          typename EDataType,
          typename AccBlockwiseOperation,
          typename AccBlockwiseNextOperation,
          typename WeiDataBlockDesc,
          typename InDataBlockDesc,
          typename DsBlockDesc,
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
          bool WeiDataEnableLds = false,
          bool InDataEnableLds  = false,
          bool DsEnableLds      = false,
          bool Transposed       = false>
/* Option: Read from LDS, big buffer hold all threads required data
 * Source
 * Weight: NumYX x KPerBlock x  CPerBlock (YXKC)
 * InData: HPerBlock x WPerBlock x CPerBlock (HWC)
 * Destination
 * Accum
 * thread level: HRepeat x WRepeat xKRepeat x AccVgprs
 * block  level: HWave x WWave x KWave x HRepeat x WRepeat x KRepeat x AccVgprs
 *
 * Option: Read from VGPR, small buffer hold each thread own required data (Skip LDS)
 * Source:
 * Weight(if skip LDS): KRepeat x CRepeat x WeightVgprs
 * InData(if skip LDS): WRepeat x CRepeat x HRepeat x InDataVgprs
 * Destination
 * Accum
 * block level: HWave x WWave x KWave x HRepeat x WRepeat x KRepeat x AccVgprs
 * NOTE: Assume stride is 2 if FilterSize = 2
 */
struct BlockwiseSubaConvWconv
{
    static constexpr auto I0            = Number<0>{};
    static constexpr auto I1            = Number<1>{};
    static constexpr auto I2            = Number<2>{};
    static constexpr auto I3            = Number<3>{};
    static constexpr auto I4            = Number<4>{};
    static constexpr auto I5            = Number<5>{};
    static constexpr auto I6            = Number<6>{};
    static constexpr index_t NumDTensor = DsBlockDesc::Size();

    // Hardcode of WaveSize, since GFX13 conv only support wave32 mode
    static constexpr index_t WaveSize = 32;

    static constexpr auto NumOfThread      = ThisThreadBlock::GetNumOfThread();
    static constexpr bool EnableWaveGroup  = ThisThreadBlock::InWaveGroup();
    static constexpr bool EnableWaveGroup4 = ThisThreadBlock::GetNumWavePerGroup() == 4;

    static constexpr index_t WaveFilterSize = (FilterSize == 2) ? 1 : FilterSize;
    static constexpr bool Aco               = [] {
        if constexpr(AccBlockwiseOperation::IsFma)
            return sizeof(InDataType) > 1 || sizeof(AccDataType) == 4;
        else
            return sizeof(InDataType) > 1;
    }();

    // Wave properties
    static constexpr index_t Iters   = GetFilterIters<WeiDataType,
                                                    InDataType,
                                                    AccDataType,
                                                    CPerBlock,
                                                    HPerWconv,
                                                    WPerWconv,
                                                    FilterSize,
                                                    Transposed>();
    static constexpr auto wconv_conv = WconvConv<WeiDataType,
                                                 InDataType,
                                                 AccDataType,
                                                 HPerWconv,
                                                 WPerWconv,
                                                 WaveFilterSize,
                                                 DilationX,
                                                 DilationY,
                                                 Iters,
                                                 ThisThreadBlock::InWaveGroup(),
                                                 Aco>{};

    static constexpr index_t CPerWconv = wconv_conv.GetNumInputChannels();
    static constexpr index_t KPerWconv = wconv_conv.GetNumOutputChannels();

    // Output Size Per Wave: (HRepeat * HPerWconv) * (WRepeat * WPerWconv) * (KRepeat * KPerWconv)
    static constexpr index_t HWaves = HPerBlock / (HRepeat * HPerWconv);
    static constexpr index_t WWaves = WPerBlock / (WRepeat * WPerWconv);
    static constexpr index_t KWaves = NumOfThread / WaveSize / HWaves / WWaves;

    static constexpr index_t KRepeat = KPerBlock / KWaves / KPerWconv;
    static constexpr index_t CRepeat = CPerBlock / CPerWconv;

    static constexpr index_t DataTileHeight = 4;
    static constexpr index_t H_Pad          = (FilterSize == 3) ? DataTileHeight : 0;
    static constexpr index_t W_Pad          = (FilterSize == 3) ? WPerWconv : 0;
    static constexpr index_t HPerBlockIn    = HPerBlock + H_Pad * 2;
    static constexpr index_t WPerBlockIn    = WPerBlock + W_Pad * 2;

    static constexpr index_t NumSubTilePerImage      = wconv_conv.GetNumSubTilesPerImageTile();
    static constexpr index_t NumSubTilesPerWeightTap = wconv_conv.GetNumSubTilesPerWeightTap();
    static constexpr index_t NumWeightCompPerTile    = wconv_conv.GetNumWeightCompPerTile();
    static constexpr index_t NumWeightTap            = wconv_conv.GetNumWeightTap();
    static constexpr index_t HPerWave                = HRepeat * HPerWconv;
    static constexpr index_t WPerWave                = WRepeat * WPerWconv;
    static constexpr index_t CPerWave                = CPerBlock;
    static constexpr index_t KPerWave                = KPerBlock;
    static constexpr index_t HPerWaveIn              = HPerWave + H_Pad * 2;
    static constexpr index_t WPerWaveIn              = WPerWave + W_Pad * 2;
    static constexpr index_t OutputScale2            = (FilterSize == 2) ? (Transposed ? 4 : 1) : 2;
    static constexpr index_t HRepeatOut              = HRepeat * OutputScale2 / 2;
    static constexpr index_t WRepeatOut              = WRepeat * OutputScale2 / 2;
    static constexpr index_t HPerBlockOut            = HPerBlock * OutputScale2 / 2;
    static constexpr index_t WPerBlockOut            = WPerBlock * OutputScale2 / 2;
    static constexpr index_t HPerWaveOut             = HPerWave * OutputScale2 / 2;
    static constexpr index_t WPerWaveOut             = WPerWave * OutputScale2 / 2;

    // Accum descriptor info
    static constexpr auto NumAccComp        = wconv_conv.GetNumAccumComponents();
    static constexpr auto NumAccCompPerTile = NumAccComp / NumSubTilePerImage;
    static constexpr auto NumAccSwizzleComp = Aco ? 2 : 4;
    static constexpr auto NumAccCompSubTile =
        NumAccCompPerTile > NumAccSwizzleComp ? NumAccCompPerTile / NumAccSwizzleComp : 1;
    static constexpr auto NumDataCompPerTile = wconv_conv.GetNumDataCompPerTile();

    template <bool HasMainLoop>
    __host__ __device__ constexpr auto MakeAccumThreadBuffer(AccDataType* data)
    {
        if constexpr(EnableWaveGroup4)
        {
            constexpr index_t AccRingVectorCount =
                LaneSharedMemTrait::acc_ring_block_space_aligned /
                wconv_conv.GetNumAccumComponents() / sizeof(AccDataType);
            constexpr auto VectorCount = HasMainLoop ? math::max(HRepeatOut * WRepeatOut * KRepeat,
                                                                 LaneSharedMemTrait::acc_ring_size)
                                                     : AccRingVectorCount;

            return StaticBufferTupleOfVector<
                AddressSpaceEnum::Vgpr,
                AccDataType,
                VectorCount,
                wconv_conv.GetNumAccumComponents(),
                true,
                StaticallyIndexedArray_v3<
                    vector_type<AccDataType, wconv_conv.GetNumAccumComponents()>,
                    VectorCount>>(data);
        }
        else
        {
            return StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr,
                                             AccDataType,
                                             HRepeatOut * WRepeatOut * KRepeat,
                                             wconv_conv.GetNumAccumComponents(),
                                             true>{};
        }
    }

    using KernelEDataType = decltype(wconv_conv.template GetKernelDataType<EDataType>());
    __host__ __device__ constexpr auto MakeOutThreadBuffer(KernelEDataType* data)
    {
        if constexpr(EnableWaveGroup4)
        {
            return StaticBufferTupleOfVector<
                AddressSpaceEnum::Vgpr,
                KernelEDataType,
                HRepeatOut * WRepeatOut * KRepeat,
                wconv_conv.GetNumAccumComponents(),
                true,
                StaticallyIndexedArray_v3<
                    vector_type<KernelEDataType, wconv_conv.GetNumAccumComponents()>,
                    HRepeatOut * WRepeatOut * KRepeat>>(data);
        }
        else
        {
            return StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr,
                                             KernelEDataType,
                                             HRepeatOut * WRepeatOut * KRepeat,
                                             wconv_conv.GetNumAccumComponents(),
                                             true>{};
        }
    }
    __device__ static auto GetWaveIdx()
    {
        return GetWconvWaveIdx<ThisThreadBlock,
                               HPerBlock,
                               WPerBlock,
                               HRepeat,
                               WRepeat,
                               HPerWconv,
                               WPerWconv>();
    }

    // Input data descriptor help functions

    // Describe how the InData in global memory is viewed in block level.
    template <typename InGridDesc>
    __host__ __device__ static constexpr auto
    MakeInGridBlockDescriptor(const InGridDesc& in_grid_pad_desc)
    {
        if constexpr(InDataEnableLds)
        {
            // H x W x C
            return in_grid_pad_desc;
        }
        else
        {
            // H x W x C -> W0 x C0 x H0 x H1 x H2 x W1 x C1
            const auto H = in_grid_pad_desc.GetLength(I0);
            const auto W = in_grid_pad_desc.GetLength(I1);
            const auto C = in_grid_pad_desc.GetLength(I2);
            return transform_tensor_descriptor(
                in_grid_pad_desc,
                make_tuple(
                    make_unmerge_transform(make_tuple(H / HPerWconv,
                                                      Number<NumSubTilePerImage>{},
                                                      Number<HPerWconv / NumSubTilePerImage>{})),
                    make_unmerge_transform(make_tuple(W / WPerWconv, Number<WPerWconv>{})),
                    make_unmerge_transform(make_tuple(C / CPerWconv, Number<CPerWconv>{}))),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                make_tuple(Sequence<2, 3, 4>{}, Sequence<0, 5>{}, Sequence<1, 6>{}));
        }
    }

    // Describe how data read from (LDS/VGPR) buffer, used by Block level classes.
    template <typename InBlockDesc_>
    __host__ __device__ static constexpr auto MakeInWaveDescriptor(const InBlockDesc_&)
    {
        constexpr auto in_wave_desc = [&]() {
            if constexpr(InDataEnableLds)
            {
                // H x W x C -> W0 x C0 x H0 x H1 x H2 x W1 x C1
                return transform_tensor_descriptor(
                    InBlockDesc_{},
                    make_tuple(make_unmerge_transform(
                                   make_tuple(Number<HPerBlockIn / HPerWconv>{},
                                              Number<NumSubTilePerImage>{},
                                              Number<HPerWconv / NumSubTilePerImage>{})),
                               make_unmerge_transform(make_tuple(Number<WPerBlockIn / WPerWconv>{},
                                                                 Number<WPerWconv>{})),
                               make_unmerge_transform(make_tuple(Number<CPerBlock / CPerWconv>{},
                                                                 Number<CPerWconv>{}))),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                    make_tuple(Sequence<2, 3, 4>{}, Sequence<0, 5>{}, Sequence<1, 6>{}));
            }
            else
            {
                return InBlockDesc_{};
            }
        }();

        return in_wave_desc;
    }

    __host__ __device__ static constexpr auto MakeInBlockSliceCopyStep()
    {
        constexpr auto in_block_copy_step = [&]() {
            if constexpr(InDataEnableLds)
            {
                return Sequence<I0, I0, Number<CPerBlock>{}>{};
            }
            else
            {
                return Sequence<I0, Number<CPerBlock / CPerWconv>{}, I0, I0, I0, I0, I0>{};
            }
        }();

        return in_block_copy_step;
    }

    __host__ __device__ static constexpr auto GetInWaveDescLength()
    {
        return Sequence<WPerWaveIn / WPerWconv,
                        CPerWave / CPerWconv,
                        HPerWaveIn / HPerWconv,
                        NumSubTilePerImage,
                        1,
                        1,
                        NumDataCompPerTile>{};
    }

    __device__ __host__ static auto CalculateInDataThreadOriginDataIndex()
    {
#ifdef __HIP_DEVICE_COMPILE__
        const auto wave_idx          = GetWaveIdx();
        const auto waveId_h          = wave_idx[I0];
        const auto waveId_w          = wave_idx[I1];
        const auto wconv_in_data_idx = wconv_conv.CalculateInDataThreadOriginDataIndex();
        // W0 x C0 x H0 x H1 x H2 x W1 x C1
        return make_tuple(waveId_w * WRepeat,
                          0,
                          waveId_h * HRepeat,
                          0,
                          wconv_in_data_idx[I0],
                          wconv_in_data_idx[I1],
                          wconv_in_data_idx[I2]);
#else
        return make_tuple(0, 0, 0, 0, 0, 0, 0);
#endif
    }

    // Wei data descriptor help functions

    // Describe how the WeiData in global memory is viewed in block level.
    template <typename WeiGridDesc>
    __host__ __device__ static constexpr auto
    MakeWeiGridBlockDescriptor(const WeiGridDesc& wei_grid_desc)
    {
        if constexpr(WeiDataEnableLds)
        {
            // K x YX x C
            return wei_grid_desc;
        }
        else
        {
            // K x YX x C -> K0 x C0 x YX x K1 x C1 x C2
            const auto K = wei_grid_desc.GetLength(I0);
            const auto C = wei_grid_desc.GetLength(I2);
            return transform_tensor_descriptor(
                wei_grid_desc,
                make_tuple(make_unmerge_transform(make_tuple(K / KPerWconv, Number<KPerWconv>{})),
                           make_pass_through_transform(Number<FilterSize * FilterSize>{}),
                           make_unmerge_transform(
                               make_tuple(C / CPerWconv,
                                          Number<NumSubTilesPerWeightTap>{},
                                          Number<CPerWconv / NumSubTilesPerWeightTap>{}))),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                make_tuple(Sequence<0, 3>{}, Sequence<2>{}, Sequence<1, 4, 5>{}));
        }
    }

    // Describe how wei data read from (LDS/VGPR) buffer, used by Block level classes.
    template <typename WeiBlockDesc_>
    __host__ __device__ static constexpr auto MakeWeiWaveDescriptor(const WeiBlockDesc_&)
    {
        constexpr auto wei_wave_desc = [&]() {
            if constexpr(WeiDataEnableLds)
            {
                // K x YX x C -> K0 x C0 x YX x K1 x C1 x C2
                return transform_tensor_descriptor(
                    WeiBlockDesc_{},
                    make_tuple(make_unmerge_transform(make_tuple(Number<KPerBlock / KPerWconv>{},
                                                                 Number<KPerWconv>{})),
                               make_pass_through_transform(Number<FilterSize * FilterSize>{}),
                               make_unmerge_transform(
                                   make_tuple(Number<CPerBlock / CPerWconv>{},
                                              Number<NumSubTilesPerWeightTap>{},
                                              Number<CPerWconv / NumSubTilesPerWeightTap>{}))),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                    make_tuple(Sequence<0, 3>{}, Sequence<2>{}, Sequence<1, 4, 5>{}));
            }
            else
            {
                return WeiBlockDesc_{};
            }
        }();

        return wei_wave_desc;
    }

    __host__ __device__ static constexpr auto MakeWeiBlockSliceCopyStep()
    {
        constexpr auto wei_block_copy_step = [&]() {
            if constexpr(WeiDataEnableLds)
            {
                return Sequence<I0, I0, Number<CPerBlock>{}>{};
            }
            else
            {
                return Sequence<I0, Number<CPerBlock / CPerWconv>{}, I0, I0, I0, I0>{};
            }
        }();

        return wei_block_copy_step;
    }

    __host__ __device__ static constexpr auto GetWeiWaveDescLength()
    {
        constexpr index_t NumTapPerCopy = (FilterSize == 3) ? 1 : FilterSize * FilterSize;
        return Sequence<KPerWave / KPerWconv,
                        CPerWave / CPerWconv,
                        NumTapPerCopy,
                        1,
                        NumSubTilesPerWeightTap,
                        NumWeightCompPerTile>{};
    }

    __device__ __host__ static auto CalculateWeiDataThreadOriginDataIndex()
    {
#ifdef __HIP_DEVICE_COMPILE__
        const auto wave_idx      = GetWaveIdx();
        const auto waveId_k      = wave_idx[I2];
        const auto wconv_wei_idx = wconv_conv.CalculateWeiDataThreadOriginDataIndex();

        if constexpr(FilterSize == 2)
        {
            return make_tuple(waveId_k * KRepeat,
                              0,
                              wconv_wei_idx[I0],
                              wconv_wei_idx[I1],
                              wconv_wei_idx[I2],
                              wconv_wei_idx[I3]);
        }
        else if constexpr(Iters > 1)
        {
            return make_tuple(waveId_k * KRepeat,
                              wconv_wei_idx[I0],
                              0,
                              wconv_wei_idx[I1],
                              wconv_wei_idx[I2],
                              wconv_wei_idx[I3]);
        }
        else
        {
            return make_tuple(waveId_k * KRepeat,
                              0,
                              wconv_wei_idx[I0],
                              wconv_wei_idx[I1],
                              wconv_wei_idx[I2],
                              wconv_wei_idx[I3]);
        }
#else
        return make_tuple(0, 0, 0, 0, 0, 0);
#endif
    }
    // Acc Data descriptor help functions

    // Describe how to read data from accum_thread_buf_ (VGPR)
    __host__ __device__ static constexpr auto GetAccThreadDescriptor()
    {
        static_assert(NumAccComp == 4 || NumAccComp == 8, "");
        static_assert(NumAccCompPerTile % NumAccCompSubTile == 0, "");

        // HRepeat x WRepeat x KRepeat x H1 x H2 x W1 x K1 x K2
        return make_naive_tensor_descriptor_packed(
            make_tuple(Number<HRepeatOut>{},
                       Number<WRepeatOut>{},
                       Number<KRepeat>{},
                       Number<NumSubTilePerImage>{},
                       I1,
                       I1,
                       Number<NumAccCompSubTile>{},
                       Number<NumAccCompPerTile / NumAccCompSubTile>{}));
    }

    __host__ __device__ static constexpr auto GetAccThreadDescLength()
    {
        return Sequence<HRepeatOut,
                        WRepeatOut,
                        KRepeat,
                        NumSubTilePerImage,
                        I1,
                        I1,
                        NumAccCompSubTile,
                        NumAccCompPerTile / NumAccCompSubTile>{};
    }

    // Describe how data store to LDS buffer
    __host__ __device__ static constexpr auto GetAccBlockDescriptor()
    {
        constexpr auto acc_block_desc = make_naive_tensor_descriptor_packed(
            make_tuple(Number<HPerBlockOut>{}, Number<WPerBlockOut>{}, Number<KPerBlock>{}));
        return acc_block_desc;
    }

    // Describe how to store accum data to LDS or Global memory.
    template <typename AccBlockDesc_>
    __host__ __device__ static constexpr auto GetAccWaveDescriptor(const AccBlockDesc_& accDesc)
    {
        // H x W x K -> H0 x W0 x K0 x H1 x H2 x W1 x K1 x K2
        return transform_tensor_descriptor(
            accDesc,
            make_tuple(make_unmerge_transform(make_tuple(Number<HPerBlockOut / HPerWconv>{},
                                                         Number<NumSubTilePerImage>{},
                                                         Number<HPerWconv / NumSubTilePerImage>{})),
                       make_unmerge_transform(
                           make_tuple(Number<WPerBlockOut / WPerWconv>{}, Number<WPerWconv>{})),
                       make_unmerge_transform(make_tuple(Number<KPerBlock / KPerWconv>{},
                                                         Number<NumAccCompSubTile>{},
                                                         Number<KPerWconv / NumAccCompSubTile>{}))),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
            make_tuple(Sequence<0, 3, 4>{}, Sequence<1, 5>{}, Sequence<2, 6, 7>{}));
    }

    __device__ __host__ static auto CalculateAccThreadOriginDataIndex()
    {
#ifdef __HIP_DEVICE_COMPILE__
        const auto wave_idx          = GetWaveIdx();
        const auto wconv_in_data_idx = wconv_conv.CalculateAccThreadOriginDataIndex();
        const auto waveId_h          = wave_idx[I0];
        const auto waveId_w          = wave_idx[I1];
        const auto waveId_k          = wave_idx[I2];

        return make_tuple(waveId_h * HRepeatOut,
                          waveId_w * WRepeatOut,
                          waveId_k * KRepeat,
                          wconv_in_data_idx[I0],
                          wconv_in_data_idx[I1],
                          wconv_in_data_idx[I2],
                          wconv_in_data_idx[I3],
                          wconv_in_data_idx[I4]);
#else
        return make_tuple(0, 0, 0, 0, 0, 0, 0, 0);
#endif
    }

    using TupleWeiData    = decltype(CalculateWeiDataThreadOriginDataIndex());
    using TupleInData     = decltype(CalculateInDataThreadOriginDataIndex());
    using InDataWaveDesc  = decltype(MakeInWaveDescriptor(InDataBlockDesc{}));
    using WeiDataWaveDesc = decltype(MakeWeiWaveDescriptor(WeiDataBlockDesc{}));

    template <bool HasMainLoop>
    static constexpr index_t GetLaneSharedMemCount()
    {
        return HasMainLoop ? 2 : 1;
    }

    __host__ __device__
    BlockwiseSubaConvWconv(const AccBlockwiseOperation& acc_blockwise_op,
                           const AccBlockwiseNextOperation& acc_blockwise_nextop,
                           TupleWeiData weight_origin = CalculateWeiDataThreadOriginDataIndex(),
                           TupleInData indata_origin  = CalculateInDataThreadOriginDataIndex())
        : weight_thread_copy_(weight_origin),
          indata_thread_copy_(indata_origin),
          element_op_(acc_blockwise_op)
    {
        static_assert(WeiDataBlockDesc::IsKnownAtCompileTime() &&
                          InDataBlockDesc::IsKnownAtCompileTime(),
                      "wrong! Desc should be known at compile-time");
        static_assert(WeiDataWaveDesc::IsKnownAtCompileTime() &&
                          InDataWaveDesc::IsKnownAtCompileTime(),
                      "wrong! Desc should be known at compile-time");

        static_assert(ThisThreadBlock::GetNumOfThread() == HWaves * WWaves * KWaves * WaveSize,
                      "ThisThreadBlock::GetNumOfThread() != MWaves * NWaves * WaveSize\n");

        static_assert(HPerBlock % (HPerWconv * HRepeat) == 0 &&
                          WPerBlock % (WPerWconv * WRepeat) == 0 &&
                          KPerBlock % (KPerWconv * KRepeat) == 0,
                      "wrong!");

        static_assert(!(Transposed && (FilterSize != 2)),
                      "Only support strided conv2x2 transpose conv");
        static_assert(!((Transposed == false) && (FilterSize == 2) &&
                        ((HRepeat % 2 != 0) || (WRepeat % 2 != 0))),
                      "Repeat must be even for strided conv 2x2 conv");
        static_assert(HPerBlock % (HRepeat * HPerWconv) == 0, "");
        static_assert(WPerBlock % (WRepeat * WPerWconv) == 0, "");
        static_assert(KWaves > 0, "");
        static_assert(NumOfThread == (WaveSize * HWaves * WWaves * KWaves), "");
        static_assert(KRepeat >= 1, "");
        static_assert(CRepeat >= 1, "");
        static_assert(HPerWaveIn % HPerWconv == 0, "");
        static_assert(WPerWaveIn % WPerWconv == 0, "");
    }

    __host__ __device__ static constexpr auto GetWeightRemapTable()
    {
        return wconv_conv.GetWeightRemapTable();
    }

    __host__ __device__ static constexpr auto GetWeightSecondTapMapTable()
    {
        return wconv_conv.GetWeightSecondTapMapTable();
    }
    // Describe how data allocated in thread copy src buffer
    static constexpr WeiDataWaveDesc weight_wave_desc_;
    static constexpr InDataWaveDesc indata_wave_desc_;

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wundefined-reinterpret-cast"
    template <typename VectorType, typename StaticBuffer>
    static constexpr void CopyVector(VectorType& x, const StaticBuffer& y)
    {
        using ScalarType = typename StaticBuffer::type;
        if constexpr(sizeof(StaticBuffer) == sizeof(VectorType))
        {
            x = bit_cast<VectorType>(y);
        }
        else
        {
            static_for<0, y.Size(), 1>{}(
                [&](auto i) { x.template AsType<ScalarType>()(i) = y[i]; });
        }
    }

    template <typename VectorType,
              typename StaticBuffer,
              typename NumDstOffset,
              typename NumSrcOffset,
              typename NumCount>
    static constexpr void
    CopySubVector(VectorType& x, const StaticBuffer& y, NumDstOffset, NumSrcOffset, NumCount)
    {
        using ScalarType = typename StaticBuffer::type;
#if 1
        static_assert(sizeof(ScalarType) * NumCount{} % sizeof(uint32_t) == 0);
        static_assert(sizeof(ScalarType) * NumDstOffset{} % sizeof(uint32_t) == 0);
        static_assert(sizeof(ScalarType) * NumSrcOffset{} % sizeof(uint32_t) == 0);
        constexpr index_t dwCount     = sizeof(ScalarType) * NumCount{} / sizeof(uint32_t);
        constexpr index_t dwSrcOffset = sizeof(ScalarType) * NumSrcOffset{} / sizeof(uint32_t);
        constexpr index_t dwDstOffset = sizeof(ScalarType) * NumDstOffset{} / sizeof(uint32_t);
        uint32_t* pDst                = reinterpret_cast<uint32_t*>(&x);
        const uint32_t* pSrc          = reinterpret_cast<const uint32_t*>(&y[I0]);
        static_for<0, dwCount, 1>{}([&](auto i) { pDst[i + dwDstOffset] = pSrc[i + dwSrcOffset]; });
#else
        // LLVM crash
        static_for<0, NumCount{}, 1>{}([&](auto i) {
            x.template AsType<ScalarType>()(i + NumDstOffset{}) = y[i + NumSrcOffset{}];
        });
#endif
    }
    template <typename WeightBlockBuffer,
              typename InDataBlockBuffer,
              typename DsBlockBuffer,
              typename AccumThreadBuffer,
              typename OutTensorThreadBuffer,
              typename WaveGroupSemaphores,
              typename HasMainLoop,
              typename IsLast>
    __device__ void RunEmulateConv2(const WeightBlockBuffer& weight_block_buf,
                                    const InDataBlockBuffer& indata_block_buf,
                                    const DsBlockBuffer& ds_block_buf,
                                    AccumThreadBuffer& accum_thread_buf,
                                    OutTensorThreadBuffer& out_thread_buf,
                                    WaveGroupSemaphores& semaAccums,
                                    HasMainLoop,
                                    IsLast) const
    {
        auto weight_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, WeiDataType>(
            weight_thread_desc_.GetElementSpaceSize());
        using WeiDataVec    = typename decltype(wconv_conv)::WeiDataVec;
        using WeiDataTapVec = typename decltype(wconv_conv)::WeiDataTapVec;
        using InDataVec     = typename decltype(wconv_conv)::InDataVec;
        using AccDataVec    = typename decltype(wconv_conv)::AccDataVec;

        const typename InDataVec::type* indata_thread_vec_ptr[4];
        static_assert(Iters <= 4 && FilterSize == 2, "");

        static constexpr auto NumYX          = FilterSize * FilterSize;
        static constexpr index_t x_offset[4] = {0, 1, 0, 1};
        static constexpr index_t y_offset[4] = {0, 0, 1, 1};
        static constexpr index_t CStep       = Transposed == false ? 1 : Iters;
        static constexpr index_t HStep       = Transposed == false ? 2 : 1;
        static constexpr index_t WStep       = Transposed == false ? 2 : 1;

        auto load_weight_data = [&](auto k0, auto c0, auto& tmp_buf, auto& weight_ptr) {
            constexpr auto TapPerIter = wconv_conv.GetNumWeightTapPerWave();
            if constexpr(WeiDataEnableLds)
            {
                static_for<0, NumYX, TapPerIter>{}([&](auto i) {
                    weight_thread_copy_.Run(weight_wave_desc_,
                                            make_tuple(k0, c0, i, I0, I0, I0),
                                            weight_block_buf,
                                            weight_thread_desc_,
                                            make_tuple(I0, I0, I0, I0, I0, I0),
                                            weight_thread_buf);
                    if constexpr(Transposed)
                    {
                        CopyVector(tmp_buf[i / TapPerIter], weight_thread_buf);
                        weight_ptr[i / TapPerIter] =
                            &tmp_buf[i / TapPerIter]
                                 .template AsType<typename WeiDataVec::type>()[I0];
                    }
                    else
                    {
                        static_assert(sizeof(WeiDataTapVec) % sizeof(WeiDataType) == 0, "");
                        constexpr auto CompCount  = sizeof(WeiDataTapVec) / sizeof(WeiDataType);
                        constexpr auto CompOffset = i * CompCount / TapPerIter;
                        CopySubVector(tmp_buf[0],
                                      weight_thread_buf,
                                      Number<CompOffset>{},
                                      I0,
                                      Number<CompCount>{});
                        weight_ptr[0] =
                            &tmp_buf[0].template AsType<typename WeiDataVec::type>()[I0];
                    }
                });
            }
            else
            {
                if constexpr(Transposed)
                {
                    static_for<0, NumYX, TapPerIter>{}([&](auto i) {
                        constexpr index_t wei_offset =
                            weight_wave_desc_.CalculateOffset(make_tuple(k0, c0, i, I0, I0, I0));
                        weight_ptr[i / TapPerIter] =
                            reinterpret_cast<const typename WeiDataVec::type*>(
                                &weight_block_buf[Number<wei_offset>{}]);
                    });
                }
                else
                {
                    constexpr index_t wei_offset =
                        weight_wave_desc_.CalculateOffset(make_tuple(k0, c0, I0, I0, I0, I0));
                    if constexpr(wconv_conv.GetNumWeightTapPerWave() == 2)
                    {
                        constexpr index_t wei_offset2 =
                            weight_wave_desc_.CalculateOffset(make_tuple(k0, c0, I2, I0, I0, I0));
                        static_assert(sizeof(WeiDataTapVec) % sizeof(WeiDataType) == 0, "");
                        constexpr auto CompCount = sizeof(WeiDataTapVec) / sizeof(WeiDataType);
                        CopySubVector(tmp_buf[0],
                                      weight_block_buf,
                                      I0,
                                      Number<wei_offset>{},
                                      Number<CompCount>{});
                        CopySubVector(tmp_buf[0],
                                      weight_block_buf,
                                      Number<CompCount>{},
                                      Number<wei_offset2>{},
                                      Number<CompCount>{});
                        weight_ptr[0] =
                            &tmp_buf[0].template AsType<typename WeiDataVec::type>()[I0];
                    }
                    else
                    {
                        weight_ptr[0] = reinterpret_cast<const typename WeiDataVec::type*>(
                            &weight_block_buf[Number<wei_offset>{}]);
                    }
                }
            }
        };

        auto load_in_data = [&](auto h0, auto w0, auto c0, auto& tmp_buf) {
            if constexpr(InDataEnableLds)
            {
                // Load input tensor data
                if constexpr(Transposed == false)
                {
                    static_assert(Iters == NumYX, "");
                    InDataVec indata_thread_vec_tmp[4];
                    static_for<0, NumYX, 1>{}([&](auto i) {
                        auto indata_thread_buf =
                            make_static_buffer_v3<AddressSpaceEnum::Vgpr, InDataType>(
                                indata_thread_desc_.GetElementSpaceSize(),
                                &indata_thread_vec_tmp[i].template AsType<InDataType>()(I0));
                        indata_thread_copy_.Run(indata_wave_desc_,
                                                make_tuple(Number<w0 + x_offset[i]>{},
                                                           c0,
                                                           Number<h0 + y_offset[i]>{},
                                                           I0,
                                                           I0,
                                                           I0,
                                                           I0),
                                                indata_block_buf,
                                                indata_thread_desc_,
                                                make_tuple(I0, I0, I0, I0, I0, I0, I0),
                                                indata_thread_buf);
                        indata_thread_vec_ptr[i] =
                            &tmp_buf[i].template AsType<typename InDataVec::type>()[I0];
                    });
                    wconv_conv.UnshuffleStridedConv2Data(indata_thread_vec_tmp, tmp_buf);
                }
                else
                {
                    auto indata_thread_buf =
                        make_static_buffer_v3<AddressSpaceEnum::Vgpr, InDataType>(
                            indata_thread_desc_.GetElementSpaceSize(),
                            &tmp_buf[I0].template AsType<InDataType>()(I0));
                    indata_thread_copy_.Run(indata_wave_desc_,
                                            make_tuple(w0, c0, h0, I0, I0, I0, I0),
                                            indata_block_buf,
                                            indata_thread_desc_,
                                            make_tuple(I0, I0, I0, I0, I0, I0, I0),
                                            indata_thread_buf);
                    indata_thread_vec_ptr[I0] =
                        &tmp_buf[I0].template AsType<typename InDataVec::type>()[I0];
                }
            }
            else
            {
                // Load input tensor data
                if constexpr(Transposed == false)
                {
                    InDataVec indata_thread_vec_tmp[4];
                    constexpr auto CompCount = sizeof(InDataVec) / sizeof(InDataType);
                    static_for<0, NumYX, 1>{}([&](auto i) {
                        constexpr index_t indata_offset = indata_wave_desc_.CalculateOffset(
                            make_tuple(w0 + x_offset[i], c0, h0 + y_offset[i], I0, I0, I0, I0));
                        CopySubVector(indata_thread_vec_tmp[i],
                                      indata_block_buf,
                                      I0,
                                      Number<indata_offset>{},
                                      Number<CompCount>{});
                        indata_thread_vec_ptr[i] =
                            &tmp_buf[i].template AsType<typename InDataVec::type>()[I0];
                    });
                    wconv_conv.UnshuffleStridedConv2Data(indata_thread_vec_tmp, tmp_buf);
                }
                else
                {
                    constexpr index_t indata_offset =
                        indata_wave_desc_.CalculateOffset(make_tuple(w0, c0, h0, I0, I0, I0, I0));
                    indata_thread_vec_ptr[I0] = reinterpret_cast<const typename InDataVec::type*>(
                        &indata_block_buf[Number<indata_offset>{}]);
                }
            }
        };

        constexpr auto get_accum_offset_transposed = [&](auto h0, auto w0, auto k0, auto i) {
            if constexpr(HasMainLoop{} || (EnableWaveGroup4 == false))
            {
                return accum_thread_desc_.CalculateOffset(make_tuple(
                    Number<h0 * 2 + y_offset[i]>{}, Number<w0 * 2 + x_offset[i]>{}, k0, I0));
            }
            else
            {
                constexpr index_t WStride = 1;
                constexpr index_t HStride = WRepeat / WStep;
                constexpr index_t Idx = (h0 / HStep * HStride + w0 / WStep * WStride) & 1 ? 1 : 0;
                return (NumYX * Idx + i) * NumAccComp;
            }
        };

        constexpr auto get_accum_offset = [&](auto h0, auto w0, auto k0) {
            if constexpr(HasMainLoop{} || (EnableWaveGroup4 == false))
            {
                return accum_thread_desc_.CalculateOffset(
                    make_tuple(Number<h0 / HStep>{}, Number<w0 / WStep>{}, k0, I0));
            }
            else
            {
                constexpr index_t WStride = 1;
                constexpr index_t HStride = WRepeat / WStep;
                constexpr index_t Idx = (h0 / HStep * HStride + w0 / WStep * WStride) & 1 ? 1 : 0;
                return Idx * NumAccComp;
            }
        };

        if constexpr(EnableWaveGroup4)
        {
            static_for<0, KRepeat, 1>{}([&](auto k0) {
                const typename WeiDataVec::type* weight_thread_vec_ptr[CRepeat / CStep][4];

                // Load weights
                WeiDataVec weight_thread_vec[CRepeat / CStep][4];
                auto semaAccumReady = semaAccums[I0];
                auto semaAccumFree  = semaAccums[I1];
                if(ThisThreadBlock::GetWaveIdInWaveGroup() == WaveIdRun)
                {
                    static_for<0, CRepeat, CStep>{}([&](auto c0) {
                        load_weight_data(k0,
                                         c0,
                                         weight_thread_vec[c0 / CStep],
                                         weight_thread_vec_ptr[c0 / CStep]);
                    });

                    InDataVec indata_thread_vec[4];
                    static_for<0, HRepeat, HStep>{}([&](auto h0) {
                        static_for<0, WRepeat, WStep>{}([&](auto w0) {
                            if constexpr(Transposed == false)
                            {
                                constexpr index_t accum_offset =
                                    get_accum_offset(Number<h0>{}, Number<w0>{}, Number<k0>{});
                                AccDataVec acc_vec = {};
                                if constexpr(HasMainLoop{})
                                {
                                    acc_vec = accum_thread_buf.GetVectorTypeReference(
                                        Number<accum_offset>{});
                                }

                                static_for<0, CRepeat, CStep>{}([&](auto c0) {
                                    load_in_data(h0, w0, c0, indata_thread_vec);
                                    wconv_conv.wconv_instr.Run(
                                        *weight_thread_vec_ptr[c0 / CStep][0],
                                        indata_thread_vec_ptr,
                                        acc_vec,
                                        I0);
                                });

                                if constexpr(IsLast{})
                                {
                                    semaAccumFree->template wait<0>();
                                    accum_thread_buf.GetVectorTypeReference(
                                        Number<accum_offset>{}) = acc_vec;
                                    semaAccumReady->template signal<0>();
                                }
                                else
                                {
                                    accum_thread_buf.GetVectorTypeReference(
                                        Number<accum_offset>{}) = acc_vec;
                                }
                            }
                            else
                            {
                                AccDataVec acc_vec[NumYX] = {};
                                if constexpr(HasMainLoop{})
                                {
                                    static_for<0, NumYX, 1>{}([&](auto i) {
                                        constexpr index_t accum_offset =
                                            get_accum_offset_transposed(Number<h0>{},
                                                                        Number<w0>{},
                                                                        Number<k0>{},
                                                                        Number<i>{});
                                        acc_vec[i] = accum_thread_buf.GetVectorTypeReference(
                                            Number<accum_offset>{});
                                    });
                                }

                                static_for<0, CRepeat, CStep>{}([&](auto c0) {
                                    load_in_data(h0, w0, c0, indata_thread_vec);
                                    static_for<0, NumYX, 1>{}([&](auto i) {
                                        constexpr index_t tapIdx =
                                            i / wconv_conv.GetNumWeightTapPerWave();
                                        constexpr bool isHigh =
                                            i % wconv_conv.GetNumWeightTapPerWave();
                                        constexpr auto Mod = []() {
                                            if constexpr(isHigh)
                                                return I1;
                                            else
                                                return I0;
                                        }();

                                        wconv_conv.wconv_instr.Run(
                                            *weight_thread_vec_ptr[c0 / CStep][tapIdx],
                                            indata_thread_vec_ptr,
                                            acc_vec[i],
                                            Mod);
                                    });
                                });

                                if constexpr(IsLast{})
                                {
                                    semaAccumFree->template wait<0>();
                                    static_for<0, NumYX, 1>{}([&](auto i) {
                                        constexpr index_t accum_offset =
                                            get_accum_offset_transposed(Number<h0>{},
                                                                        Number<w0>{},
                                                                        Number<k0>{},
                                                                        Number<i>{});
                                        accum_thread_buf.GetVectorTypeReference(
                                            Number<accum_offset>{}) = acc_vec[i];
                                    });
                                    semaAccumReady->template signal<0>();
                                }
                                else
                                {
                                    static_for<0, NumYX, 1>{}([&](auto i) {
                                        constexpr index_t accum_offset =
                                            get_accum_offset_transposed(Number<h0>{},
                                                                        Number<w0>{},
                                                                        Number<k0>{},
                                                                        Number<i>{});
                                        accum_thread_buf.GetVectorTypeReference(
                                            Number<accum_offset>{}) = acc_vec[i];
                                    });
                                }
                            }
                        });
                    });
                }
                else if(IsLast{} && ThisThreadBlock::GetWaveIdInWaveGroup() == WaveIdPostRun)
                {
                    static_for<0, HRepeat, HStep>{}([&](auto h0) {
                        static_for<0, WRepeat, WStep>{}([&](auto w0) {
                            if constexpr(Transposed == false)
                            {
                                constexpr index_t accum_offset =
                                    get_accum_offset(Number<h0>{}, Number<w0>{}, Number<k0>{});

                                semaAccumReady->template wait<0>();
                                AccDataVec acc_vec =
                                    accum_thread_buf.GetVectorTypeReference(Number<accum_offset>{});
                                element_op_.Run(ds_block_buf,
                                                acc_vec,
                                                Number<h0 / HStep>{},
                                                Number<w0 / WStep>{},
                                                k0);

                                element_next_op_.Run(acc_vec,
                                                     out_thread_buf,
                                                     Number<h0 / HStep>{},
                                                     Number<w0 / WStep>{},
                                                     k0);

                                semaAccumFree->template signal<0>();
                            }
                            else
                            {
                                semaAccumReady->template wait<0>();
                                AccDataVec acc_vec[NumYX] = {};
                                typename AccDataVec::type* accdata_thread_vec_ptr[NumYX];
                                static_for<0, NumYX, 1>{}([&](auto i) {
                                    constexpr index_t accum_offset = get_accum_offset_transposed(
                                        Number<h0>{}, Number<w0>{}, Number<k0>{}, Number<i>{});
                                    acc_vec[i] = accum_thread_buf.GetVectorTypeReference(
                                        Number<accum_offset>{});
                                    accdata_thread_vec_ptr[i] =
                                        &acc_vec[i].template AsType<typename AccDataVec::type>()(
                                            Number<0>{});
                                });
                                wconv_conv.ShuffleConv2TransposedData(accdata_thread_vec_ptr);
                                static_for<0, NumYX, 1>{}([&](auto i) {
                                    element_op_.Run(ds_block_buf,
                                                    acc_vec[i],
                                                    Number<h0 * 2 + y_offset[i]>{},
                                                    Number<w0 * 2 + x_offset[i]>{},
                                                    k0);
                                    element_next_op_.Run(acc_vec[i],
                                                         out_thread_buf,
                                                         Number<h0 * 2 + y_offset[i]>{},
                                                         Number<w0 * 2 + x_offset[i]>{},
                                                         k0);
                                });
                                semaAccumFree->template signal<0>();
                            }
                        });
                    });
                }
            });
        }
        else
        {
            static_for<0, KRepeat, 1>{}([&](auto k0) {
                const typename WeiDataVec::type* weight_thread_vec_ptr[CRepeat / CStep][4];

                // Load weights
                WeiDataVec weight_thread_vec[CRepeat / CStep][4];
                static_for<0, CRepeat, CStep>{}([&](auto c0) {
                    load_weight_data(
                        k0, c0, weight_thread_vec[c0 / CStep], weight_thread_vec_ptr[c0 / CStep]);
                });

                InDataVec indata_thread_vec[4];
                static_for<0, HRepeat, HStep>{}([&](auto h0) {
                    static_for<0, WRepeat, WStep>{}([&](auto w0) {
                        if constexpr(Transposed == false)
                        {
                            constexpr index_t accum_offset =
                                get_accum_offset(Number<h0>{}, Number<w0>{}, Number<k0>{});
                            AccDataVec acc_vec = {};
                            if constexpr(HasMainLoop{})
                            {
                                acc_vec =
                                    accum_thread_buf.GetVectorTypeReference(Number<accum_offset>{});
                            }
                            static_for<0, CRepeat, CStep>{}([&](auto c0) {
                                load_in_data(h0, w0, c0, indata_thread_vec);
                                wconv_conv.wconv_instr.Run(*weight_thread_vec_ptr[c0 / CStep][0],
                                                           indata_thread_vec_ptr,
                                                           acc_vec,
                                                           I0);
                            });
                            if constexpr(IsLast{})
                            {
                                element_op_.Run(ds_block_buf,
                                                acc_vec,
                                                Number<h0 / HStep>{},
                                                Number<w0 / WStep>{},
                                                k0);
                                element_next_op_.Run(acc_vec,
                                                     out_thread_buf,
                                                     Number<h0 / HStep>{},
                                                     Number<w0 / WStep>{},
                                                     k0);
                            }
                            else
                            {
                                accum_thread_buf.GetVectorTypeReference(Number<accum_offset>{}) =
                                    acc_vec;
                            }
                        }
                        else
                        {
                            AccDataVec acc_vec[NumYX] = {};
                            typename AccDataVec::type* accdata_thread_vec_ptr[NumYX];
                            if constexpr(HasMainLoop{})
                            {
                                static_for<0, NumYX, 1>{}([&](auto i) {
                                    constexpr index_t accum_offset = get_accum_offset_transposed(
                                        Number<h0>{}, Number<w0>{}, Number<k0>{}, Number<i>{});
                                    acc_vec[i] = accum_thread_buf.GetVectorTypeReference(
                                        Number<accum_offset>{});
                                });
                            }
                            static_for<0, CRepeat, CStep>{}([&](auto c0) {
                                load_in_data(h0, w0, c0, indata_thread_vec);
                                static_for<0, NumYX, 1>{}([&](auto i) {
                                    constexpr index_t tapIdx =
                                        i / wconv_conv.GetNumWeightTapPerWave();
                                    constexpr bool isHigh = i % wconv_conv.GetNumWeightTapPerWave();
                                    constexpr auto Mod    = []() {
                                        if constexpr(isHigh)
                                            return I1;
                                        else
                                            return I0;
                                    }();
                                    accdata_thread_vec_ptr[i] =
                                        &acc_vec[i].template AsType<typename AccDataVec::type>()(
                                            I0);

                                    wconv_conv.wconv_instr.Run(
                                        *weight_thread_vec_ptr[c0 / CStep][tapIdx],
                                        indata_thread_vec_ptr,
                                        acc_vec[i],
                                        Mod);
                                });
                            });

                            if constexpr(IsLast{})
                            {
                                wconv_conv.ShuffleConv2TransposedData(accdata_thread_vec_ptr);
                                static_for<0, NumYX, 1>{}([&](auto i) {
                                    element_op_.Run(ds_block_buf,
                                                    acc_vec[i],
                                                    Number<h0 * 2 + y_offset[i]>{},
                                                    Number<w0 * 2 + x_offset[i]>{},
                                                    k0);
                                    element_next_op_.Run(acc_vec[i],
                                                         out_thread_buf,
                                                         Number<h0 * 2 + y_offset[i]>{},
                                                         Number<w0 * 2 + x_offset[i]>{},
                                                         k0);
                                });
                            }
                            else
                            {
                                static_for<0, NumYX, 1>{}([&](auto i) {
                                    constexpr index_t accum_offset = get_accum_offset_transposed(
                                        Number<h0>{}, Number<w0>{}, Number<k0>{}, Number<i>{});
                                    accum_thread_buf.GetVectorTypeReference(
                                        Number<accum_offset>{}) = acc_vec[i];
                                });
                            }
                        }
                    });
                });
            });
        }
    }

    template <typename WeightBlockBuffer,
              typename InDataBlockBuffer,
              typename DsBlockBuffer,
              typename AccumThreadBuffer,
              typename OutTensorThreadBuffer,
              typename WaveGroupSemaphores,
              typename HasMainLoop,
              typename IsLast>
    __device__ void RunConv(const WeightBlockBuffer& weight_block_buf,
                            const InDataBlockBuffer& indata_block_buf,
                            const DsBlockBuffer& ds_block_buf,
                            AccumThreadBuffer& accum_thread_buf,
                            OutTensorThreadBuffer& out_thread_buf,
                            WaveGroupSemaphores& semaAccums,
                            HasMainLoop,
                            IsLast) const
    {
        auto weight_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, WeiDataType>(
            weight_thread_desc_.GetElementSpaceSize());

        using WeiDataVec    = typename decltype(wconv_conv)::WeiDataVec;
        using WeiDataTapVec = typename decltype(wconv_conv)::WeiDataTapVec;
        using InDataVec     = typename decltype(wconv_conv)::InDataVec;
        using AccDataVec    = typename decltype(wconv_conv)::AccDataVec;

        const typename InDataVec::type* indata_thread_vec_ptr[4];
        static_assert(Iters <= 4 && FilterSize < 4, "");
        constexpr index_t CStep = Iters;

        auto load_weight_data = [&](auto k0, auto c0, WeiDataVec& tmp_buf) {
            const typename WeiDataVec::type* weight_thread_vec_ptr;
            if constexpr(WeiDataEnableLds)
            {
                if constexpr(FilterSize == 1)
                {
                    constexpr auto TapPerIter = wconv_conv.GetNumWeightTapPerWave();
                    static_for<0, Iters, TapPerIter>{}([&](auto i) {
                        weight_thread_copy_.Run(weight_wave_desc_,
                                                make_tuple(k0, Number<c0 + i>{}, I0, I0, I0, I0),
                                                weight_block_buf,
                                                weight_thread_desc_,
                                                make_tuple(I0, I0, I0, I0, I0, I0),
                                                weight_thread_buf);
                        constexpr auto CompCount  = sizeof(WeiDataTapVec) / sizeof(WeiDataType);
                        constexpr auto CompOffset = i * CompCount / TapPerIter;
                        CopySubVector(tmp_buf,
                                      weight_thread_buf,
                                      Number<CompOffset>{},
                                      I0,
                                      Number<CompCount>{});
                    });
                    weight_thread_vec_ptr =
                        &tmp_buf.template AsType<typename WeiDataVec::type>()[I0];
                }
                else if constexpr(FilterSize == 3)
                {
                    static_assert(Iters == 1, "");
                    constexpr index_t WeightTapPerWave = wconv_conv.GetNumWeightTapPerWave();
                    static_for<0, wconv_conv.GetNumWeightTap(), 1>{}([&](auto tape_idx) {
                        weight_thread_copy_.Run(
                            weight_wave_desc_,
                            make_tuple(k0, c0, Number<WeightTapPerWave * tape_idx>{}, I0, I0, I0),
                            weight_block_buf,
                            weight_thread_desc_,
                            make_tuple(I0, I0, I0, I0, I0, I0),
                            weight_thread_buf);

                        constexpr auto CompCount  = sizeof(WeiDataTapVec) / sizeof(WeiDataType);
                        constexpr auto CompOffset = tape_idx * CompCount;
                        CopySubVector(tmp_buf,
                                      weight_thread_buf,
                                      Number<CompOffset>{},
                                      I0,
                                      Number<CompCount>{});
                    });
                    weight_thread_vec_ptr =
                        &tmp_buf.template AsType<typename WeiDataVec::type>()[I0];
                }
            }
            else
            {
                constexpr index_t wei_offset =
                    weight_wave_desc_.CalculateOffset(make_tuple(k0, c0, I0, I0, I0, I0));
                if constexpr((Iters == 4) && (wconv_conv.GetNumWeightTapPerWave() == 2))
                {
                    constexpr auto CompCount = sizeof(WeiDataTapVec) / sizeof(WeiDataType);
                    constexpr index_t wei_offset2 =
                        weight_wave_desc_.CalculateOffset(make_tuple(k0, c0 + I2, I0, I0, I0, I0));
                    CopySubVector(
                        tmp_buf, weight_block_buf, I0, Number<wei_offset>{}, Number<CompCount>{});
                    CopySubVector(tmp_buf,
                                  weight_block_buf,
                                  Number<CompCount>{},
                                  Number<wei_offset2>{},
                                  Number<CompCount>{});
                    weight_thread_vec_ptr =
                        &tmp_buf.template AsType<typename WeiDataVec::type>()[I0];
                }
                else
                {
                    weight_thread_vec_ptr = reinterpret_cast<const typename WeiDataVec::type*>(
                        &weight_block_buf[Number<wei_offset>{}]);
                }
            }
            return weight_thread_vec_ptr;
        };

        auto load_in_data = [&](auto h0, auto w0, auto c0, auto& tmp_buf) {
            if constexpr(InDataEnableLds)
            {
                // Load input tensor data
                if constexpr(FilterSize == 1)
                {
                    static_for<0, Iters, 1>{}([&](auto iter_idx) {
                        auto indata_thread_buf =
                            make_static_buffer_v3<AddressSpaceEnum::Vgpr, InDataType>(
                                indata_thread_desc_.GetElementSpaceSize(),
                                &tmp_buf[iter_idx].template AsType<InDataType>()(I0));
                        indata_thread_copy_.Run(indata_wave_desc_,
                                                make_tuple(w0, c0 + iter_idx, h0, I0, I0, I0, I0),
                                                indata_block_buf,
                                                indata_thread_desc_,
                                                make_tuple(I0, I0, I0, I0, I0, I0, I0),
                                                indata_thread_buf);
                        indata_thread_vec_ptr[iter_idx] =
                            &tmp_buf[iter_idx].template AsType<typename InDataVec::type>()[I0];
                    });
                }
                else if constexpr(FilterSize == 3)
                {
                    //  read tensor
                    static_for<0, FilterSize, 1>{}([&](auto array_idx) {
                        auto indata_thread_buf =
                            make_static_buffer_v3<AddressSpaceEnum::Vgpr, InDataType>(
                                indata_thread_desc_.GetElementSpaceSize(),
                                &tmp_buf[array_idx].template AsType<InDataType>()(I0));
                        indata_thread_copy_.Run(indata_wave_desc_,
                                                make_tuple(w0 + array_idx, c0, h0, I0, I0, I0, I0),
                                                indata_block_buf,
                                                indata_thread_desc_,
                                                make_tuple(I0, I0, I0, I0, I0, I0, I0),
                                                indata_thread_buf);
                        indata_thread_vec_ptr[array_idx] =
                            &tmp_buf[array_idx].template AsType<typename InDataVec::type>()[I0];
                    });
                }
            }
            else
            {
                // Load input tensor data
                if constexpr(FilterSize == 1)
                {
                    static_for<0, Iters, 1>{}([&](auto iter_idx) {
                        constexpr index_t indata_offset = indata_wave_desc_.CalculateOffset(
                            make_tuple(w0, c0 + iter_idx, h0, I0, I0, I0, I0));
                        indata_thread_vec_ptr[iter_idx] =
                            reinterpret_cast<const typename InDataVec::type*>(
                                &indata_block_buf[Number<indata_offset>{}]);
                    });
                }
                else if constexpr(FilterSize == 3)
                {
                    static_for<0, FilterSize, 1>{}([&](auto array_idx) {
                        constexpr index_t indata_offset = indata_wave_desc_.CalculateOffset(
                            make_tuple(w0 + array_idx, c0, h0, I0, I0, I0, I0));
                        indata_thread_vec_ptr[array_idx] =
                            reinterpret_cast<const typename InDataVec::type*>(
                                &indata_block_buf[Number<indata_offset>{}]);
                    });
                }
            }
        };

        constexpr auto get_accum_offset = [&](auto h0, auto w0, auto k0) {
            if constexpr(HasMainLoop{} || (EnableWaveGroup4 == false))
            {
                return accum_thread_desc_.CalculateOffset(make_tuple(h0, w0, k0, I0));
            }
            else
            {
                constexpr index_t Idx = (h0 * WRepeat + w0) & 1 ? 1 : 0;
                return Idx * NumAccComp;
            }
        };

        if constexpr(EnableWaveGroup4)
        {
            static_for<0, KRepeat, 1>{}([&](auto k0) {
                WeiDataVec weight_thread_vec[CRepeat / CStep];
                const typename WeiDataVec::type* weight_thread_vec_ptr[CRepeat / CStep];
                auto semaAccumReady = semaAccums[I0];
                auto semaAccumFree  = semaAccums[I1];
                if(ThisThreadBlock::GetWaveIdInWaveGroup() == WaveIdRun)
                {
                    static_for<0, CRepeat, CStep>{}([&](auto c0) {
                        // Load weights
                        weight_thread_vec_ptr[c0 / CStep] =
                            load_weight_data(k0, c0, weight_thread_vec[c0 / CStep]);
                    });

                    static_for<0, HRepeat, 1>{}([&](auto h0) {
                        static_for<0, WRepeat, 1>{}([&](auto w0) {
                            constexpr index_t accum_offset =
                                get_accum_offset(Number<h0>{}, Number<w0>{}, Number<k0>{});

                            InDataVec indata_thread_vec[4];
                            AccDataVec acc_vec = {};
                            if constexpr(HasMainLoop{})
                            {
                                acc_vec =
                                    accum_thread_buf.GetVectorTypeReference(Number<accum_offset>{});
                            }
                            static_for<0, CRepeat, CStep>{}([&](auto c0) {
                                load_in_data(h0, w0, c0, indata_thread_vec);
                                wconv_conv.wconv_instr.Run(*weight_thread_vec_ptr[c0 / CStep],
                                                           indata_thread_vec_ptr,
                                                           acc_vec,
                                                           I0);
                            });
                            if constexpr(IsLast{})
                            {
                                semaAccumFree->template wait<0>();
                                accum_thread_buf.GetVectorTypeReference(Number<accum_offset>{}) =
                                    acc_vec;
                                semaAccumReady->template signal<0>();
                            }
                            else
                            {
                                accum_thread_buf.GetVectorTypeReference(Number<accum_offset>{}) =
                                    acc_vec;
                            }
                        });
                    });
                }

                // run sba/uba
                if constexpr(IsLast{})
                {
                    if(ThisThreadBlock::GetWaveIdInWaveGroup() == WaveIdPostRun)
                    {
                        static_for<0, HRepeat, 1>{}([&](auto h0) {
                            static_for<0, WRepeat, 1>{}([&](auto w0) {
                                constexpr index_t accum_offset =
                                    get_accum_offset(Number<h0>{}, Number<w0>{}, Number<k0>{});

                                semaAccumReady->template wait<0>();
                                AccDataVec acc_vec =
                                    accum_thread_buf.GetVectorTypeReference(Number<accum_offset>{});
                                element_op_.Run(ds_block_buf, acc_vec, h0, w0, k0);
                                element_next_op_.Run(acc_vec, out_thread_buf, h0, w0, k0);
                                semaAccumFree->template signal<0>();
                            });
                        });
                    }
                }
            });
        }
        else
        {
            static_for<0, KRepeat, 1>{}([&](auto k0) {
                WeiDataVec weight_thread_vec[CRepeat / CStep];
                const typename WeiDataVec::type* weight_thread_vec_ptr[CRepeat / CStep];
                static_for<0, CRepeat, CStep>{}([&](auto c0) {
                    // Load weights
                    weight_thread_vec_ptr[c0 / CStep] =
                        load_weight_data(k0, c0, weight_thread_vec[c0 / CStep]);
                });

                static_for<0, HRepeat, 1>{}([&](auto h0) {
                    static_for<0, WRepeat, 1>{}([&](auto w0) {
                        constexpr index_t accum_offset =
                            get_accum_offset(Number<h0>{}, Number<w0>{}, Number<k0>{});

                        InDataVec indata_thread_vec[4];
                        AccDataVec acc_vec = {};
                        if constexpr(HasMainLoop{})
                        {
                            acc_vec =
                                accum_thread_buf.GetVectorTypeReference(Number<accum_offset>{});
                        }
                        static_for<0, CRepeat, CStep>{}([&](auto c0) {
                            load_in_data(h0, w0, c0, indata_thread_vec);
                            wconv_conv.wconv_instr.Run(*weight_thread_vec_ptr[c0 / CStep],
                                                       indata_thread_vec_ptr,
                                                       acc_vec,
                                                       I0);
                        });
                        // run sba/uba
                        if constexpr(IsLast{})
                        {
                            element_op_.Run(ds_block_buf, acc_vec, h0, w0, k0);
                            element_next_op_.Run(acc_vec, out_thread_buf, h0, w0, k0);
                        }
                        else
                        {
                            accum_thread_buf.GetVectorTypeReference(Number<accum_offset>{}) =
                                acc_vec;
                        }
                    });
                });
            });
        }
    };

#pragma clang diagnostic pop

    template <typename WeightBlockBuffer,
              typename InDataBlockBuffer,
              typename DsBlockBuffer,
              typename AccumThreadBuffer,
              typename OutTensorThreadBuffer,
              typename WaveGroupSemaphores,
              typename HasMainLoop,
              typename IsLast>
    __device__ void Run(const WeightBlockBuffer& weight_block_buf,
                        const InDataBlockBuffer& indata_block_buf,
                        const DsBlockBuffer& ds_block_buf,
                        AccumThreadBuffer& accum_thread_buf,
                        OutTensorThreadBuffer& out_thread_buf,
                        WaveGroupSemaphores& semaAccums,
                        HasMainLoop hasMainLoop,
                        IsLast isLast) const
    {
        if constexpr(FilterSize == 2)
        {
            RunEmulateConv2(weight_block_buf,
                            indata_block_buf,
                            ds_block_buf,
                            accum_thread_buf,
                            out_thread_buf,
                            semaAccums,
                            hasMainLoop,
                            isLast);
        }
        else
        {
            RunConv(weight_block_buf,
                    indata_block_buf,
                    ds_block_buf,
                    accum_thread_buf,
                    out_thread_buf,
                    semaAccums,
                    hasMainLoop,
                    isLast);
        }
    }

    protected:
    // Thread descriptor for weight and input data
    static constexpr auto weight_thread_desc_ = make_naive_tensor_descriptor_packed(make_tuple(
        I1, I1, I1, I1, Number<NumSubTilesPerWeightTap>{}, Number<NumWeightCompPerTile>{}));

    static constexpr auto NumDataSubTiles = wconv_conv.GetNumSubTilesPerImageTile();
    static constexpr auto NumDataTiles    = wconv_conv.GetNumImageTilesInVertical();
    static constexpr auto indata_thread_desc_ =
        make_naive_tensor_descriptor_packed(make_tuple(I1,
                                                       I1,
                                                       Number<NumDataTiles>{},
                                                       Number<NumDataSubTiles>{},
                                                       I1,
                                                       I1,
                                                       Number<NumDataCompPerTile>{}));

    // C[H, W, K, NumAccComp]
    static constexpr auto accum_thread_desc_ = make_naive_tensor_descriptor_packed(
        make_tuple(Number<HRepeatOut>{},
                   Number<WRepeatOut>{},
                   Number<KRepeat>{},
                   Number<wconv_conv.GetNumAccumComponents()>{}));

    static constexpr auto out_tensor_thread_desc_ = make_naive_tensor_descriptor_packed(
        make_tuple(Number<HRepeatOut>{},
                   Number<WRepeatOut>{},
                   Number<KRepeat>{},
                   Number<wconv_conv.GetNumOutTensorComponents()>{}));

    // Initialize thread copy classes
    using WeiDataThreadLdsCopy = ThreadwiseTensorSliceTransfer_v4<
        WeiDataType,
        WeiDataType,
        decltype(weight_wave_desc_),
        decltype(weight_thread_desc_),
        Sequence<1, 1, 1, 1, NumSubTilesPerWeightTap, NumWeightCompPerTile>,
        Sequence<0, 1, 2, 3, 4, 5>,
        5,
        NumWeightCompPerTile,
        NumWeightCompPerTile>;

    WeiDataThreadLdsCopy weight_thread_copy_;

    using InDataThreadLdsCopy = ThreadwiseTensorSliceTransfer_v4<
        InDataType,
        InDataType,
        decltype(indata_wave_desc_),
        decltype(indata_thread_desc_),
        Sequence<1, 1, NumDataTiles, NumDataSubTiles, 1, 1, NumDataCompPerTile>,
        Sequence<0, 1, 2, 3, 4, 5, 6>,
        6,
        NumDataCompPerTile,
        NumDataCompPerTile>;
    InDataThreadLdsCopy indata_thread_copy_;

    template <bool IsSuba, bool IsFma>
    struct BlockwiseElementSelect
    {
        using type = Blockwise_element_passthrough;
    };
    template <>
    struct BlockwiseElementSelect<true, false>
    {
        using type = Blockwise_element_wconv_suba<ThisThreadBlock,
                                                  AccDataType,
                                                  DsBlockDesc, // {Scale, Bias}
                                                  HPerBlock,
                                                  WPerBlock,
                                                  KPerBlock,
                                                  HPerWconv,
                                                  WPerWconv,
                                                  KPerWconv,
                                                  HRepeat,
                                                  WRepeat,
                                                  KRepeat,
                                                  DsEnableLds,
                                                  Aco,
                                                  AccBlockwiseOperation>;
    };

    template <>
    struct BlockwiseElementSelect<false, true>
    {
        using type = Blockwise_element_wconv_fma<ThisThreadBlock,
                                                 DsDataType,
                                                 AccDataType,
                                                 DsBlockDesc, // {Scale, Residual}
                                                 HPerBlock,
                                                 WPerBlock,
                                                 KPerBlock,
                                                 HPerWconv,
                                                 WPerWconv,
                                                 HRepeat,
                                                 WRepeat,
                                                 KRepeat,
                                                 DsEnableLds,
                                                 Aco,
                                                 AccBlockwiseOperation>;
    };

    template <>
    struct BlockwiseElementSelect<true, true>
    {
        using type = Blockwise_element_wconv_suba_fma<ThisThreadBlock,
                                                      DsDataType,
                                                      AccDataType,
                                                      DsBlockDesc,
                                                      HPerBlock,
                                                      WPerBlock,
                                                      KPerBlock,
                                                      HPerWconv,
                                                      WPerWconv,
                                                      KPerWconv,
                                                      HRepeat,
                                                      WRepeat,
                                                      KRepeat,
                                                      DsEnableLds,
                                                      Aco,
                                                      AccBlockwiseOperation>;
    };

    using BlockwiseElementOpType =
        typename BlockwiseElementSelect<AccBlockwiseOperation::IsSuba,
                                        AccBlockwiseOperation::IsFma>::type;
    BlockwiseElementOpType element_op_;

    template <bool shouldCvtToTensor>
    struct BlockwiseNextElementSelect
    {
        using type = Blockwise_element_wconv_out_passthrough<decltype(accum_thread_desc_)>;
    };
    template <>
    struct BlockwiseNextElementSelect<true>
    {
        using type = Blockwise_element_wconv_cvtTensor<ThisThreadBlock,
                                                       EDataType,
                                                       AccDataType,
                                                       decltype(out_tensor_thread_desc_),
                                                       decltype(accum_thread_desc_),
                                                       HPerBlock,
                                                       WPerBlock,
                                                       KPerBlock,
                                                       HPerWconv,
                                                       WPerWconv,
                                                       KPerWconv,
                                                       CPerWconv,
                                                       HRepeat,
                                                       WRepeat,
                                                       KRepeat,
                                                       Aco,
                                                       AccBlockwiseNextOperation>;
    };
    using BlockwiseElementNextOpType =
        typename BlockwiseNextElementSelect<AccBlockwiseNextOperation::cvt_to_tensor>::type;

    BlockwiseElementNextOpType element_next_op_;

    public:
    template <typename DsGridDesc>
    __host__ __device__ static constexpr auto
    MakeDsGridBlockDescriptor(const DsGridDesc& ds_grid_desc)
    {
        return BlockwiseElementOpType::MakeDsGridBlockDescriptor(ds_grid_desc);
    }

    __host__ __device__ static constexpr auto CalculateThreadOriginDataIndex()
    {
        return BlockwiseElementOpType::CalculateThreadOriginDataIndex();
    }

    __host__ __device__ static constexpr auto GetDsWaveDescLength()
    {
        return BlockwiseElementOpType::GetDsWaveDescLength();
    }

    __host__ __device__ static constexpr auto GetDsWaveDesc()
    {
        return BlockwiseElementOpType::ds_wave_desc_;
    }

    struct SharedMemTrait
    {
        // LDS allocation for A and B: be careful of alignment
        static constexpr index_t max_lds_align = 8;

        static constexpr index_t in_block_space_size_aligned =
            InDataEnableLds ? math::integer_least_multiple(InDataBlockDesc{}.GetElementSpaceSize(),
                                                           max_lds_align)
                            : 0;
        static constexpr index_t wei_block_space_size_aligned =
            WeiDataEnableLds ? math::integer_least_multiple(
                                   WeiDataBlockDesc{}.GetElementSpaceSize(), max_lds_align)
                             : 0;

        static constexpr auto ds_block_space_size_aligned =
            BlockwiseElementOpType::SharedMemTrait::ds_block_space_size_aligned;

        static constexpr index_t in_block_space_offset  = 0;
        static constexpr index_t wei_block_space_offset = in_block_space_size_aligned;

        // LDS allocation for Ds in LDS
        static constexpr index_t ds_base_offset =
            (wei_block_space_offset + wei_block_space_size_aligned) * sizeof(WeiDataType) /
            sizeof(AccDataType);

        static constexpr auto ds_block_space_offset = make_tuple(
            BlockwiseElementOpType::SharedMemTrait::ds_block_space_offset[I0] + ds_base_offset,
            BlockwiseElementOpType::SharedMemTrait::ds_block_space_offset[I1] + ds_base_offset);

        // LDS allocation for C shuffle in LDS
        static constexpr index_t acc_block_space_size =
            GetAccBlockDescriptor().GetElementSpaceSize();

        static constexpr index_t acc_block_space_offset = 0;

        static constexpr index_t lds_size =
            math::max(acc_block_space_size * sizeof(AccDataType),
                      in_block_space_size_aligned * sizeof(InDataType) +
                          wei_block_space_size_aligned * sizeof(WeiDataType) +
                          BlockwiseElementOpType::SharedMemTrait::lds_size);

        static constexpr index_t out_tensor_block_space_size =
            GetAccBlockDescriptor().GetElementSpaceSize();
        static constexpr index_t out_tensor_block_space_offset = 0;
    };

    struct LaneSharedMemTrait
    {
        static constexpr index_t max_lane_shared_align = 4;

        static constexpr index_t in_block_space_size_aligned =
            EnableWaveGroup && (InDataEnableLds == false)
                ? math::integer_least_multiple(InDataWaveDesc{}.GetElementSpaceSize() *
                                                   sizeof(InDataType),
                                               max_lane_shared_align)
                : 0;
        static constexpr index_t wei_block_space_size_aligned =
            EnableWaveGroup && (WeiDataEnableLds == false)
                ? math::integer_least_multiple(WeiDataWaveDesc{}.GetElementSpaceSize() *
                                                   sizeof(WeiDataType),
                                               max_lane_shared_align)
                : 0;
        static constexpr index_t ds_block_space_size_aligned =
            BlockwiseElementOpType::LaneSharedMemTrait::lane_shared_size;

        static constexpr auto ds_block_space_offset =
            BlockwiseElementOpType::LaneSharedMemTrait::ds_block_space_offset;

        static constexpr index_t out_block_space_aligned = math::integer_least_multiple(
            HRepeatOut * WRepeatOut * KRepeat * wconv_conv.GetNumAccumComponents() *
                wconv_conv.template SizeOfBits<EDataType>() / 8,
            max_lane_shared_align);

        static constexpr index_t acc_ring_size                = 2;
        static constexpr index_t acc_ring_block_space_aligned = math::integer_least_multiple(
            (FilterSize == 2 && Transposed)
                ? acc_ring_size * 4 * wconv_conv.GetNumAccumComponents() * sizeof(AccDataType)
                : acc_ring_size * wconv_conv.GetNumAccumComponents() * sizeof(AccDataType),
            max_lane_shared_align);

        static constexpr index_t acc_block_space_aligned = math::integer_least_multiple(
            HRepeatOut * WRepeatOut * KRepeat * wconv_conv.GetNumAccumComponents() *
                sizeof(AccDataType),
            max_lane_shared_align);
    };
};
} // namespace ck
