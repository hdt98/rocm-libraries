// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "common.hpp"

#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_weight_xdl_cshuffle_v3.hpp"

using InDataType = BF16;
using WeiDataType = BF16;
using OutDataType = BF16;
using AccDataType = F32;

using InElementOp  = PassThrough;
using WeiElementOp = PassThrough;
using OutElementOp = PassThrough;

template <ck::index_t NDimSpatial>
using DeviceConvBwdWeightInstance =
    // clang-format on
    ck::tensor_operation::device::DeviceGroupedConvBwdWeight_Xdl_CShuffleV3<
        NDimSpatial,
        ck::tensor_layout::convolution::NDHWGC,
        ck::tensor_layout::convolution::GKZYXC,
        ck::tensor_layout::convolution::NDHWGK,
        InDataType,           // InDataType
        WeiDataType,          // WeiDataType
        OutDataType,          // OutDataType
        AccDataType,          // AccDataType
        InElementOp,          // InElementwiseOperation
        WeiElementOp,         // WeiElementwiseOperation
        OutElementOp,         // OutElementwiseOperation
        ConvBwdWeightDefault, // ConvolutionBackwardWeightSpecialization
        64,                   // BlockSize
        128,                   
        32,     
        32,   
        8,   
        32,   
        32,    
        4,    
        1,  
        S<4, 16, 1>,  
        S<2, 0, 1>,  
        S<1, 0, 2>,                  
        1,              
        8,              
        8,      
        false,  
        S<4, 4,  1>,  
        S<2, 0, 1>,  
        S<1, 0, 2>,                
        1,              
        8,              
        8,      
        false,           
        1,           
        1,   
        S<1, 8, 1, 8>,                  
        2>; // CBlockTransferScalarPerVector_NWaveNPerXdl
                                          


template <ck::index_t NDimSpatial>
using HostConvBwdWeightInstance = ck::tensor_operation::host::ReferenceConvBwdWeight<NDimSpatial,
                                                                                     InDataType,
                                                                                     WeiDataType,
                                                                                     OutDataType,
                                                                                     InElementOp,
                                                                                     WeiElementOp,
                                                                                     OutElementOp>;

#include "run_grouped_conv_bwd_weight_example.inc"

int main(int argc, char* argv[])
{
    ExecutionConfig config;
    ck::utils::conv::ConvParam conv_param = DefaultConvParam;

    if(!parse_cmd_args(argc, argv, config, conv_param))
    {
        return 1;
    }

    // Only CPU verification, we want GPU code generated only for the conv kernel.
    run_grouped_conv_bwd_weight<3, true>(config, conv_param);

    return 1;
}
