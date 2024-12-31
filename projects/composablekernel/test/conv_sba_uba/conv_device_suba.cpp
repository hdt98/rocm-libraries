// SPDX-License-Identifier: MIT
// Copyriconv_device_impl.hght (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.
#include "conv_device_suba_cvt_tensor_impl.h"

// clang-format on
ExecutionConfig config;

template <typename SrcType,
          typename GPUAccType,
          typename CPUAccType,
          int LdsMode,
          bool ScaleBiasPacked,
          bool UniformScale,
          int32_t TestMask>
bool run_test_fmt()
{
    if((config.test_mask & TestMask) == 0)
    {
        return true;
    }
    bool pass = true;

#ifdef ENABLE_WAVEGROUP
    constexpr bool WaveGroup = true;
#else
    constexpr bool WaveGroup = false;
#endif

    // clang-format off
    //                                                           |ShapeType  |FilterType |Dilation |Lds |WaveGroup |activeFun | AccElementOp | scaleBiasPacked | uniformScale | TestMask
    if constexpr(std::is_same<GPUAccType, float>::value)
    { 
        pass &= run_test<SrcType, SrcType, GPUAccType, CPUAccType, SrcType, Shape_4X2, Filter_1X1, false, 0, WaveGroup, 0, OutElementNoneOp, ScaleBiasPacked, UniformScale, 0, TestMask | 0x100>(); 
        pass &= run_test<SrcType, SrcType, GPUAccType, CPUAccType, SrcType, Shape_4X2, Filter_1X1, false, 0, WaveGroup, 0, OutElementNoneOp, ScaleBiasPacked, UniformScale, 0, TestMask | 0x100>(); 
        pass &= run_test<SrcType, SrcType, GPUAccType, CPUAccType, SrcType, Shape_4X2, Filter_3X3, false, 0, WaveGroup, 0, OutElementNoneOp, ScaleBiasPacked, UniformScale, 0, TestMask | 0x100>();
        pass &= run_test<SrcType, SrcType, GPUAccType, CPUAccType, SrcType, Shape_4X2, Filter_3X3, true,  0, WaveGroup, 0, OutElementNoneOp, ScaleBiasPacked, UniformScale, 0, TestMask | 0x100>();      
        pass &= run_test<SrcType, SrcType, GPUAccType, CPUAccType, SrcType, Shape_4X2, Filter_3X3, false, LdsMode, WaveGroup, 0, OutElementNoneOp, ScaleBiasPacked, UniformScale, 0, TestMask | 0x100>();
        pass &= run_test<SrcType, SrcType, GPUAccType, CPUAccType, SrcType, Shape_4X2, Filter_3X3, false, 0, WaveGroup, 1, OutElementReluOp, ScaleBiasPacked, UniformScale, 0, TestMask | 0x200>();
        pass &= run_test<SrcType, SrcType, GPUAccType, CPUAccType, SrcType, Shape_4X2, Filter_3X3, true,  0, WaveGroup, 1, OutElementReluOp, ScaleBiasPacked, UniformScale, 0, TestMask | 0x200>();               
        pass &= run_test<SrcType, SrcType, GPUAccType, CPUAccType, SrcType, Shape_4X2, Filter_1X1, false, 0, WaveGroup, 1, OutElementReluOp, ScaleBiasPacked, UniformScale, 0, TestMask | 0x200>();
        pass &= run_test<SrcType, SrcType, GPUAccType, CPUAccType, SrcType, Shape_4X2, Filter_1X1, false, LdsMode, WaveGroup, 1, OutElementReluOp, ScaleBiasPacked, UniformScale, 0, TestMask | 0x200>();
        pass &= run_test<SrcType, SrcType, GPUAccType, CPUAccType, SrcType, Shape_4X2, Filter_3X3, false, LdsMode, WaveGroup, 1, OutElementReluOp, ScaleBiasPacked, UniformScale, 0, TestMask | 0x200>();
        //Tan (only for sba/uba)
        pass &= run_test<SrcType, SrcType, GPUAccType, CPUAccType, SrcType, Shape_4X2, Filter_1X1, false, 0, WaveGroup, 2, OutElementTanhOp, ScaleBiasPacked, UniformScale, 0, TestMask | 0x400>();
        pass &= run_test<SrcType, SrcType, GPUAccType, CPUAccType, SrcType, Shape_4X2, Filter_3X3, false, 0, WaveGroup, 2, OutElementTanhOp, ScaleBiasPacked, UniformScale, 0, TestMask | 0x400>();
        pass &= run_test<SrcType, SrcType, GPUAccType, CPUAccType, SrcType, Shape_4X2, Filter_3X3, true,  0, WaveGroup, 2, OutElementTanhOp, ScaleBiasPacked, UniformScale, 0, TestMask | 0x400>();     
        pass &= run_test<SrcType, SrcType, GPUAccType, CPUAccType, SrcType, Shape_4X2, Filter_1X1, false, LdsMode, WaveGroup, 2, OutElementTanhOp, ScaleBiasPacked, UniformScale, 0, TestMask | 0x400>();
        pass &= run_test<SrcType, SrcType, GPUAccType, CPUAccType, SrcType, Shape_4X2, Filter_3X3, false, LdsMode, WaveGroup, 2, OutElementTanhOp, ScaleBiasPacked, UniformScale, 0, TestMask | 0x400>();
    }
    else
    {  
        // 1st issue@llvm: https://ontrack-internal.amd.com/browse/LWPSCGFX13-478 for v_scale_bias_activate_f16 which will impact the all the accType=half case which will impact 4x4
        // 2nd issue@ffm: https://github.amd.com/GFX-Modeling/shader_complex_ffm/issues/960 impact on i8_f16 for 4X4 and 8x4   
// 4X4 issue      
        pass &= run_test<SrcType, SrcType, GPUAccType, CPUAccType, SrcType, Shape_4X4, Filter_1X1, false, 0, WaveGroup, 0, OutElementNoneOp, ScaleBiasPacked, UniformScale, 0, TestMask | 0x800>(); 
        pass &= run_test<SrcType, SrcType, GPUAccType, CPUAccType, SrcType, Shape_4X4, Filter_3X3, false, 0, WaveGroup, 0, OutElementNoneOp, ScaleBiasPacked, UniformScale, 0, TestMask | 0x800>();
        pass &= run_test<SrcType, SrcType, GPUAccType, CPUAccType, SrcType, Shape_4X4, Filter_3X3, true,  0, WaveGroup, 0, OutElementNoneOp, ScaleBiasPacked, UniformScale, 0, TestMask | 0x800>();
        pass &= run_test<SrcType, SrcType, GPUAccType, CPUAccType, SrcType, Shape_4X4, Filter_1X1, false, LdsMode, WaveGroup, 0, OutElementNoneOp, ScaleBiasPacked, UniformScale, 0, TestMask | 0x800>();
        pass &= run_test<SrcType, SrcType, GPUAccType, CPUAccType, SrcType, Shape_4X4, Filter_3X3, false,  LdsMode, WaveGroup, 0, OutElementNoneOp, ScaleBiasPacked, UniformScale, 0, TestMask | 0x800>();
        
        //ActivativeFun: 1
        pass &= run_test<SrcType, SrcType, GPUAccType, CPUAccType, SrcType, Shape_4X4, Filter_1X1, false, 0, WaveGroup, 1, OutElementReluOp, ScaleBiasPacked, UniformScale, 0, TestMask | 0x1000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, CPUAccType, SrcType, Shape_4X4, Filter_3X3, false, 0, WaveGroup, 1, OutElementReluOp, ScaleBiasPacked, UniformScale, 0, TestMask | 0x1000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, CPUAccType, SrcType, Shape_4X4, Filter_3X3, true,  0, WaveGroup, 1, OutElementReluOp, ScaleBiasPacked, UniformScale, 0, TestMask | 0x1000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, CPUAccType, SrcType, Shape_4X4, Filter_1X1, false,  LdsMode, WaveGroup, 1, OutElementReluOp, ScaleBiasPacked, UniformScale, 0, TestMask | 0x1000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, CPUAccType, SrcType, Shape_4X4, Filter_3X3, false,  LdsMode, WaveGroup, 1, OutElementReluOp, ScaleBiasPacked, UniformScale, 0, TestMask | 0x1000>();

        //ActivativeFun: 2
        pass &= run_test<SrcType, SrcType, GPUAccType, CPUAccType, SrcType, Shape_4X4, Filter_1X1, false, 0, WaveGroup, 2, OutElementTanhOp, ScaleBiasPacked, UniformScale, 0, TestMask | 0x2000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, CPUAccType, SrcType, Shape_4X4, Filter_3X3, false, 0, WaveGroup, 2, OutElementTanhOp, ScaleBiasPacked, UniformScale, 0, TestMask | 0x2000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, CPUAccType, SrcType, Shape_4X4, Filter_3X3, true,  0, WaveGroup, 2, OutElementTanhOp, ScaleBiasPacked, UniformScale, 0, TestMask | 0x2000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, CPUAccType, SrcType, Shape_4X4, Filter_1X1, false, LdsMode, WaveGroup, 2, OutElementTanhOp, ScaleBiasPacked, UniformScale, 0, TestMask | 0x2000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, CPUAccType, SrcType, Shape_4X4, Filter_3X3, false, LdsMode, WaveGroup, 2, OutElementTanhOp, ScaleBiasPacked, UniformScale, 0, TestMask | 0x2000>();
// 8x4
        //ActivativeFun: 0  
        pass &= run_test<SrcType, SrcType, GPUAccType, CPUAccType, SrcType, Shape_8X4, Filter_1X1, false, 0, WaveGroup, 0, OutElementNoneOp, ScaleBiasPacked, UniformScale, 0, TestMask | 0x4000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, CPUAccType, SrcType, Shape_8X4, Filter_3X3, false, 0, WaveGroup, 0, OutElementNoneOp, ScaleBiasPacked, UniformScale, 0, TestMask | 0x4000>();       
        pass &= run_test<SrcType, SrcType, GPUAccType, CPUAccType, SrcType, Shape_8X4, Filter_3X3, true,  0, WaveGroup, 0, OutElementNoneOp, ScaleBiasPacked, UniformScale, 0, TestMask | 0x4000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, CPUAccType, SrcType, Shape_8X4, Filter_1X1, false,  LdsMode, WaveGroup, 0, OutElementNoneOp, ScaleBiasPacked, UniformScale, 0, TestMask | 0x4000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, CPUAccType, SrcType, Shape_8X4, Filter_3X3, false,  LdsMode, WaveGroup, 0, OutElementNoneOp, ScaleBiasPacked, UniformScale, 0, TestMask | 0x4000>();
        
        //ActivativeFun: 1
        pass &= run_test<SrcType, SrcType, GPUAccType, CPUAccType, SrcType, Shape_8X4, Filter_1X1, false, 0, WaveGroup, 1, OutElementReluOp, ScaleBiasPacked, UniformScale, 0, TestMask | 0x8000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, CPUAccType, SrcType, Shape_8X4, Filter_3X3, false, 0, WaveGroup, 1, OutElementReluOp, ScaleBiasPacked, UniformScale, 0, TestMask | 0x8000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, CPUAccType, SrcType, Shape_8X4, Filter_3X3, true,  0, WaveGroup, 1, OutElementReluOp, ScaleBiasPacked, UniformScale, 0, TestMask | 0x8000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, CPUAccType, SrcType, Shape_8X4, Filter_1X1, false,  LdsMode, WaveGroup, 1, OutElementReluOp, ScaleBiasPacked, UniformScale, 0, TestMask | 0x8000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, CPUAccType, SrcType, Shape_8X4, Filter_3X3, false,  LdsMode, WaveGroup, 1, OutElementReluOp, ScaleBiasPacked, UniformScale, 0, TestMask | 0x8000>();

        //ActivativeFun: 2
        pass &= run_test<SrcType, SrcType, GPUAccType, CPUAccType, SrcType, Shape_8X4, Filter_1X1, false, 0, WaveGroup, 2, OutElementTanhOp, ScaleBiasPacked, UniformScale, 0, TestMask | 0x10000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, CPUAccType, SrcType, Shape_8X4, Filter_3X3, false, 0, WaveGroup, 2, OutElementTanhOp, ScaleBiasPacked, UniformScale, 0, TestMask | 0x10000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, CPUAccType, SrcType, Shape_8X4, Filter_3X3, true,  0, WaveGroup, 2, OutElementTanhOp, ScaleBiasPacked, UniformScale, 0, TestMask | 0x10000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, CPUAccType, SrcType, Shape_8X4, Filter_1X1, false,  LdsMode, WaveGroup, 2, OutElementTanhOp, ScaleBiasPacked, UniformScale, 0, TestMask | 0x10000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, CPUAccType, SrcType, Shape_8X4, Filter_3X3, false,  LdsMode, WaveGroup, 2, OutElementTanhOp, ScaleBiasPacked, UniformScale, 0, TestMask | 0x10000>();
// 4X2
        //NoneActFun
        pass &= run_test<SrcType, SrcType, GPUAccType, CPUAccType, SrcType, Shape_4X2, Filter_1X1, false, 0, WaveGroup, 0, OutElementNoneOp, ScaleBiasPacked, UniformScale, 0, TestMask | 0x20000>();   
        pass &= run_test<SrcType, SrcType, GPUAccType, CPUAccType, SrcType, Shape_4X2, Filter_3X3, false, 0, WaveGroup, 0, OutElementNoneOp, ScaleBiasPacked, UniformScale, 0, TestMask | 0x20000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, CPUAccType, SrcType, Shape_4X2, Filter_3X3, true,  0, WaveGroup, 0, OutElementNoneOp, ScaleBiasPacked, UniformScale, 0, TestMask | 0x20000>();
        //pass &= run_test<SrcType, SrcType, GPUAccType, CPUAccType, SrcType, Shape_4X2, Filter_2X2, false, 0, WaveGroup, 0, OutElementNoneOp, ScaleBiasPacked, UniformScale, 0, TestMask | 0x20000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, CPUAccType, SrcType, Shape_4X2, Filter_1X1, false, LdsMode, WaveGroup, 0, OutElementNoneOp, ScaleBiasPacked, UniformScale, 0, TestMask | 0x20000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, CPUAccType, SrcType, Shape_4X2, Filter_3X3, false,  LdsMode, WaveGroup, 0, OutElementNoneOp, ScaleBiasPacked, UniformScale, 0, TestMask | 0x20000>(); 
   
        //Relu
        pass &= run_test<SrcType, SrcType, GPUAccType, CPUAccType, SrcType, Shape_4X2, Filter_1X1, false, 0, WaveGroup, 1, OutElementReluOp, ScaleBiasPacked, UniformScale, 0, TestMask | 0x40000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, CPUAccType, SrcType, Shape_4X2, Filter_3X3, false, 0, WaveGroup, 1, OutElementReluOp, ScaleBiasPacked, UniformScale, 0, TestMask | 0x40000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, CPUAccType, SrcType, Shape_4X2, Filter_3X3, true,  0, WaveGroup, 1, OutElementReluOp, ScaleBiasPacked, UniformScale, 0, TestMask | 0x40000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, CPUAccType, SrcType, Shape_4X2, Filter_1X1, false, LdsMode, WaveGroup, 1, OutElementReluOp, ScaleBiasPacked, UniformScale, 0, TestMask | 0x40000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, CPUAccType, SrcType, Shape_4X2, Filter_3X3, false,  LdsMode, WaveGroup, 1, OutElementReluOp, ScaleBiasPacked, UniformScale, 0, TestMask | 0x40000>();

        //Tan (only for sba/uba)
        pass &= run_test<SrcType, SrcType, GPUAccType, CPUAccType, SrcType, Shape_4X2, Filter_1X1, false, 0, WaveGroup, 2, OutElementTanhOp, ScaleBiasPacked, UniformScale, 0, TestMask | 0x80000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, CPUAccType, SrcType, Shape_4X2, Filter_3X3, false, 0, WaveGroup, 2, OutElementTanhOp, ScaleBiasPacked, UniformScale, 0, TestMask | 0x80000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, CPUAccType, SrcType, Shape_4X2, Filter_3X3, true,  0, WaveGroup, 2, OutElementTanhOp, ScaleBiasPacked, UniformScale, 0, TestMask | 0x80000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, CPUAccType, SrcType, Shape_4X2, Filter_1X1, false, LdsMode, WaveGroup, 2, OutElementTanhOp, ScaleBiasPacked, UniformScale, 0, TestMask | 0x80000>();
        pass &= run_test<SrcType, SrcType, GPUAccType, CPUAccType, SrcType, Shape_4X2, Filter_3X3, false,  LdsMode, WaveGroup, 2, OutElementTanhOp, ScaleBiasPacked, UniformScale, 0, TestMask | 0x80000>();
    }
    // clang-format on
    return pass;
}

int main(int argc, char* argv[])
{
    // To refactor on only Acc=32bit supported on 4x2 and Acc=16bit supported on 4x2/4x4/8x4
    bool pass = true;
    if(parse_cmd_args(argc, argv, config) == false)
    {
        return -1;
    }

    // clang-format off
    // Ds keep same with acc currently
    //                |SrcType |GPUAccType |CPUAccType |LdsMode |scaleBiasPacked |uniformScale |convert_to_tensor |TestMask
    pass &= run_test_fmt<half_t,  float,   float,   0x17, 0, 0, 0x1>();
    pass &= run_test_fmt<half_t,  float,   float,   0x17, 1, 0, 0x1>();
    pass &= run_test_fmt<half_t,  float,   float,   0x17, 0, 1, 0x1>();
    pass &= run_test_fmt<half_t,  half_t,  half_t,  0x1f, 0, 0, 0x2>();
    pass &= run_test_fmt<half_t,  half_t,  half_t,  0x1f, 1, 0, 0x2>();
    pass &= run_test_fmt<half_t,  half_t,  half_t,  0x1f, 0, 1, 0x2>();
    pass &= run_test_fmt<bhalf_t, bhalf_t, half_t,  0x17, 0, 0, 0x4>();
    pass &= run_test_fmt<bhalf_t, bhalf_t, half_t,  0x17, 1, 0, 0x4>();
    pass &= run_test_fmt<bhalf_t, bhalf_t, half_t,  0x17, 0, 1, 0x4>();

    // clang-format on
    std::cout << "conv_device: ..... " << (pass ? "SUCCESS" : "FAILURE") << std::endl;
    return pass ? 0 : 1;
}
