// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "GpuConvBwdRefTestFixture.hpp"

#include <stdexcept>

// ============================================================================
// Shape Catalog — centralized convolution shapes for dgrad tests.
// Each function is kept separate for easy future splitting into tiers.
//
// Organization: Small (1D, 2D, 3D) → Medium (1D, 2D, 3D) → Large (1D, 2D, 3D)
// ============================================================================

namespace gpu_conv_bwd_ref_test
{

// Returns copies of the given cases with channel-last layout set.
// 3D (NLC) for 1D conv, 4D (NHWC) for 2D conv, 5D (NDHWC) for 3D conv.
inline std::vector<ConvBwdShapeCase> withChannelLastLayout(std::vector<ConvBwdShapeCase> cases)
{
    for(auto& tc : cases)
    {
        if(tc.xDims.size() == 5)
        {
            tc.layout = &TensorLayout::NDHWC;
        }
        else if(tc.xDims.size() == 4)
        {
            tc.layout = &TensorLayout::NHWC;
        }
        else if(tc.xDims.size() == 3)
        {
            tc.layout = &TensorLayout::NLC;
        }
        else
        {
            throw std::invalid_argument("Unsupported tensor rank for channel-last layout: "
                                        + std::to_string(tc.xDims.size()));
        }
    }
    return cases;
}

// ============================================================================
// Small shapes — fast binary (CI gate)
// ============================================================================

// Small 1D shapes: basic NCW convolution tests
inline std::vector<ConvBwdShapeCase> getSmall1dDgradCases()
{
    return {
        {{1, 1, 8}, {1, 1, 3}, {1}, {1}, {0}, 1, "Basic1d"},
        {{1, 1, 6}, {1, 1, 3}, {1}, {1}, {1}, 1, "Padded1d"},
        {{1, 1, 10}, {1, 1, 3}, {2}, {1}, {0}, 1, "Stride2x1d"},
        {{1, 1, 9}, {1, 1, 3}, {1}, {2}, {0}, 1, "Dilation2x1d"},
        {{1, 3, 8}, {2, 3, 3}, {1}, {1}, {0}, 1, "MultiChan1d"},
        {{2, 1, 8}, {1, 1, 3}, {1}, {1}, {0}, 1, "MultiBatch1d"},
        {{1, 4, 8}, {4, 2, 3}, {1}, {1}, {0}, 2, "Grouped2x1d"},
        {{1, 3, 8}, {2, 3, 1}, {1}, {1}, {0}, 1, "Pointwise1d"},
    };
}

// Small 2D shapes: output < 1K elements, suitable for all types
inline std::vector<ConvBwdShapeCase> getSmall2dDgradCases()
{
    return {
        {{1, 1, 8, 8}, {1, 1, 3, 3}, {1, 1}, {1, 1}, {0, 0}, 1, "Basic3x3"},
        {{1, 3, 8, 8}, {6, 3, 3, 3}, {1, 1}, {1, 1}, {1, 1}, 1, "MultiChanPad"},
        {{2, 4, 8, 8}, {4, 2, 3, 3}, {1, 1}, {1, 1}, {0, 0}, 2, "Grouped2Batch2"},
        {{1, 1, 8, 8}, {2, 1, 3, 3}, {2, 2}, {1, 1}, {0, 0}, 1, "Stride2"},
        {{1, 1, 12, 12}, {1, 1, 3, 3}, {1, 1}, {2, 2}, {0, 0}, 1, "Dilation2"},
        {{1, 3, 8, 8}, {3, 1, 3, 3}, {1, 1}, {1, 1}, {0, 0}, 3, "Depthwise3Chan"},
        {{1, 8, 4, 4}, {16, 8, 1, 1}, {1, 1}, {1, 1}, {0, 0}, 1, "Pointwise1x1"},
        {{1, 7, 8, 8}, {7, 1, 3, 3}, {1, 1}, {1, 1}, {1, 1}, 7, "DepthwiseOdd7"},
        // 5x5 kernel with padding (larger receptive field)
        {{1, 2, 10, 10}, {4, 2, 5, 5}, {1, 1}, {1, 1}, {2, 2}, 1, "Kernel5x5"},
        // Non-square spatial dimensions
        {{1, 2, 6, 10}, {4, 2, 3, 3}, {1, 1}, {1, 1}, {0, 0}, 1, "NonSquare6x10"},
        {{1, 1, 3, 3}, {1, 1, 3, 3}, {1, 1}, {1, 1}, {0, 0}, 1, "SingleElement"},
    };
}

// Small 3D shapes
inline std::vector<ConvBwdShapeCase> getSmall3dDgradCases()
{
    return {
        {{1, 1, 4, 4, 4}, {1, 1, 3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, 1, "Basic3d"},
        {{1, 1, 6, 6, 6}, {1, 1, 3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, 1, "Padded3d"},
        {{2, 4, 4, 4, 4}, {8, 2, 3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, 2, "Grouped2x3d"},
        {{1, 1, 5, 5, 5}, {1, 1, 3, 3, 3}, {2, 2, 2}, {1, 1, 1}, {0, 0, 0}, 1, "Stride2x3d"},
        {{1, 1, 7, 7, 7}, {1, 1, 3, 3, 3}, {1, 1, 1}, {2, 2, 2}, {0, 0, 0}, 1, "Dilation2x3d"},
        {{1, 3, 4, 4, 4}, {2, 3, 3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, 1, "MultiChan3d"},
    };
}

// ============================================================================
// Medium shapes — slow binary (nightly)
// ============================================================================

// Medium 1D shapes
inline std::vector<ConvBwdShapeCase> getMedium1dDgradCases()
{
    return {
        {{8, 64, 128}, {128, 64, 3}, {1}, {1}, {1}, 1, "WaveNet64Ch"},
        {{4, 32, 256}, {32, 32, 5}, {1}, {1}, {2}, 1, "Kernel5Pad2"},
        {{8, 128, 64}, {128, 16, 3}, {1}, {1}, {1}, 8, "Grouped8x1d"},
        {{4, 16, 512}, {16, 16, 7}, {2}, {1}, {3}, 1, "Stride2Kernel7"},
        {{8, 32, 128}, {32, 1, 3}, {1}, {1}, {1}, 32, "Depthwise32x1d"},
        {{4, 64, 64}, {128, 64, 1}, {1}, {1}, {0}, 1, "Pointwise64Ch"},
    };
}

// Medium 2D shapes: ResNet/ResNeXt-like, suitable for fp32 + fp16
inline std::vector<ConvBwdShapeCase> getMedium2dDgradCases()
{
    return {
        {{8, 64, 28, 28}, {128, 32, 3, 3}, {1, 1}, {1, 1}, {1, 1}, 2, "ResNeXt2Group"},
        {{8, 128, 14, 14}, {256, 4, 3, 3}, {1, 1}, {1, 1}, {1, 1}, 32, "ResNeXt32x4d"},
        {{4, 64, 56, 56}, {64, 64, 1, 1}, {1, 1}, {1, 1}, {0, 0}, 1, "ResNet1x1Reduce"},
        {{8, 3, 28, 28}, {64, 3, 7, 7}, {2, 2}, {1, 1}, {3, 3}, 1, "ResNetStem7x7"},
        {{8, 64, 14, 14}, {64, 8, 3, 3}, {1, 1}, {1, 1}, {1, 1}, 8, "Grouped8"},
        {{4, 16, 48, 48}, {16, 1, 3, 3}, {1, 1}, {1, 1}, {1, 1}, 16, "MobileNetDW16"},
        {{8, 3, 108, 108}, {63, 1, 3, 3}, {2, 2}, {1, 1}, {1, 1}, 3, "RGB3GroupStride2"},
        {{4, 32, 28, 28}, {32, 16, 5, 5}, {1, 1}, {1, 1}, {2, 2}, 2, "Grouped2Kernel5x5"},
        {{8, 128, 28, 28}, {128, 16, 3, 3}, {1, 1}, {1, 1}, {1, 1}, 8, "Grouped8MidRes"},
        {{2, 256, 14, 14}, {256, 256, 1, 1}, {1, 1}, {1, 1}, {0, 0}, 1, "Bottleneck1x1Expand"},
        {{4, 4, 48, 48}, {16, 1, 3, 3}, {1, 1}, {1, 1}, {1, 1}, 4, "Depthwise4Chan"},
        {{8, 7, 14, 14}, {63, 1, 3, 3}, {1, 1}, {1, 1}, {1, 1}, 7, "OddChanGrouped7"},
        // Dilation=2 at medium scale
        {{4, 32, 28, 28}, {32, 32, 3, 3}, {1, 1}, {2, 2}, {2, 2}, 1, "Dilation2MedScale"},
    };
}

// Medium 3D shapes
inline std::vector<ConvBwdShapeCase> getMedium3dDgradCases()
{
    return {
        {{2, 16, 8, 8, 8}, {32, 16, 3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, 1, "Standard16Ch3d"},
        {{1, 16, 4, 14, 14}, {16, 16, 3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, 1, "NonCube3d"},
        {{2, 16, 8, 8, 8}, {32, 16, 5, 5, 5}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, 1, "Kernel5x5x5"},
    };
}

// ============================================================================
// Large shapes — slow binary (nightly)
// ============================================================================

// Large 1D shapes: stress tests for audio/speech workloads
inline std::vector<ConvBwdShapeCase> getLarge1dDgradCases()
{
    return {
        // WaveNet-style high-channel long sequence
        {{16, 128, 1024}, {256, 128, 3}, {1}, {1}, {1}, 1, "WaveNetLarge"},
        // Large depthwise on long sequence
        {{16, 64, 2048}, {64, 1, 3}, {1}, {1}, {1}, 64, "Depthwise64Long"},
        // Stride-2 on long sequence with large kernel
        {{8, 32, 4096}, {64, 32, 7}, {2}, {1}, {3}, 1, "Stride2LargeKernel7"},
        // 8-group on medium-long sequence
        {{16, 64, 512}, {64, 8, 5}, {1}, {1}, {2}, 8, "Grouped8Kernel5"},
        // Odd input channels — vector alignment edge case (SWDEV-502833 pattern)
        {{8, 5, 512}, {32, 5, 3}, {1}, {1}, {1}, 1, "OddChan5x1d"},
        // Kernel = spatial → output length 1
        {{4, 32, 7}, {64, 32, 7}, {1}, {1}, {0}, 1, "Output1x1d"},
        // Large dilation on long sequence
        {{4, 16, 2048}, {32, 16, 3}, {1}, {4}, {4}, 1, "Dilation4Long1d"},
        // Prime output channels — non-power-of-2 K
        {{8, 32, 256}, {127, 32, 5}, {1}, {1}, {2}, 1, "PrimeK127x1d"},
    };
}

// Large 2D shapes: stress tests matching real workloads
inline std::vector<ConvBwdShapeCase> getLarge2dDgradCases()
{
    return {
        {{16, 128, 56, 56}, {256, 4, 3, 3}, {1, 1}, {1, 1}, {1, 1}, 32, "ResNeXt32x4dHiRes"},
        {{16, 512, 14, 14}, {1024, 16, 3, 3}, {1, 1}, {1, 1}, {1, 1}, 32, "ResNeXtDeep32Group"},
        {{16, 256, 28, 28}, {512, 8, 3, 3}, {2, 2}, {1, 1}, {1, 1}, 32, "ResNeXtStride2Down"},
        {{16, 3, 224, 224}, {63, 1, 7, 7}, {2, 2}, {1, 1}, {3, 3}, 3, "LargeStem7x7"},
        {{8, 128, 56, 56}, {128, 16, 3, 3}, {1, 1}, {1, 1}, {1, 1}, 8, "MidRes8Group56x56"},
        {{16, 192, 28, 28}, {32, 12, 5, 5}, {1, 1}, {1, 1}, {2, 2}, 16, "Inception5x5x16Group"},
        // DeepSpeech-like non-square spatial (161x700)
        {{4, 4, 161, 700}, {32, 1, 5, 20}, {2, 2}, {1, 1}, {0, 0}, 4, "DeepSpeechNonSquare"},
        // Non-square spatial with 2-group (79x341)
        {{8, 32, 79, 341}, {32, 16, 5, 10}, {2, 2}, {1, 1}, {0, 0}, 2, "NonSquareGrouped2"},
        // Odd input channels C=5 — vector alignment (SWDEV-502833)
        {{8, 5, 56, 56}, {32, 5, 3, 3}, {1, 1}, {1, 1}, {1, 1}, 1, "OddChanIn5"},
        // Asymmetric filter 5x10 (MIOpen #540)
        {{4, 32, 79, 141}, {64, 32, 5, 10}, {2, 2}, {1, 1}, {0, 0}, 1, "AsymFilter5x10"},
        // Stride-2 on tiny spatial → output 1x1 (ho=wo=1)
        {{16, 128, 2, 2}, {256, 128, 1, 1}, {2, 2}, {1, 1}, {0, 0}, 1, "Output1x1Stride2"},
        // Prime output channels K=127
        {{8, 64, 14, 14}, {127, 64, 3, 3}, {1, 1}, {1, 1}, {1, 1}, 1, "PrimeK127"},
        // High-channel 1x1 reduction (MIOpen #2012 / SWDEV-305815)
        {{8, 256, 14, 14}, {128, 256, 1, 1}, {1, 1}, {1, 1}, {0, 0}, 1, "HiChan1x1Reduce"},
        // Asymmetric stride (1,2) — CK edge case
        {{8, 32, 28, 56}, {64, 32, 3, 3}, {1, 2}, {1, 1}, {1, 1}, 1, "AsymStride1x2"},
    };
}

// Large 3D shapes: stress tests for volumetric/video workloads
inline std::vector<ConvBwdShapeCase> getLarge3dDgradCases()
{
    return {
        // 3D medical imaging style: 16ch, 32x32x32
        {{4, 16, 32, 32, 32},
         {32, 16, 3, 3, 3},
         {1, 1, 1},
         {1, 1, 1},
         {1, 1, 1},
         1,
         "MedImg32cube"},
        // 3D with stride-2 downsampling
        {{4, 32, 16, 16, 16},
         {64, 32, 3, 3, 3},
         {2, 2, 2},
         {1, 1, 1},
         {1, 1, 1},
         1,
         "Stride2x3dLarge"},
        // 3D 4-group on larger spatial
        {{2, 32, 16, 16, 16},
         {32, 8, 3, 3, 3},
         {1, 1, 1},
         {1, 1, 1},
         {1, 1, 1},
         4,
         "Grouped4x3dLarge"},
        // 3D non-cube spatial (8x32x32)
        {{4, 16, 8, 32, 32},
         {32, 16, 3, 3, 3},
         {1, 1, 1},
         {1, 1, 1},
         {1, 1, 1},
         1,
         "NonCube3dLarge"},
        // D=1 degeneracy — collapses depth to 2D-like behavior
        {{4, 16, 1, 32, 32},
         {32, 16, 1, 3, 3},
         {1, 1, 1},
         {1, 1, 1},
         {0, 1, 1},
         1,
         "Degenerate3dD1"},
    };
}

} // namespace gpu_conv_bwd_ref_test
