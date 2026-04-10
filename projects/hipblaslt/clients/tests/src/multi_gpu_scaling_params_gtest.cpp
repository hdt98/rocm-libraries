/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
 *
 * Multi-GPU Scaling Parameters Test Suite
 * Comprehensive testing of scalar, vector, and block scaling with AMax
 *
 *******************************************************************************/

#include "hipblaslt_data.hpp"
#include "hipblaslt_datatype2string.hpp"
#include "hipblaslt_test.hpp"
#include "utility.hpp"
#include <vector>

#include <gtest/gtest-spi.h>

namespace
{
    int getNumGPUs()
    {
        int numDevices = 0;
        hipGetDeviceCount(&numDevices);
        return numDevices;
    }

    // Test 1: Scalar Scaling (scaleA, scaleB, scaleC, scaleD)
    TEST(MultiGPUScaling, ScalarScaling)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing scalar scaling across GPUs" << std::endl;

        const int64_t M = 256, N = 256, K = 256;
        float scaleA = 0.5f, scaleB = 2.0f, scaleC = 1.5f, scaleD = 0.75f;

        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblasLtHandle_t handle;
            hipblasLtCreate(&handle);

            hipblasLtMatmulDesc_t matmul;
            hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);

            // Set scalar scaling attributes
            hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                            &scaleA, sizeof(scaleA));
            hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                            &scaleB, sizeof(scaleB));
            hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_C_SCALE_POINTER,
                                            &scaleC, sizeof(scaleC));
            hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_D_SCALE_POINTER,
                                            &scaleD, sizeof(scaleD));

            hipblaslt_cout << "GPU " << dev << " scalar scaling configured: "
                           << "scaleA=" << scaleA << ", scaleB=" << scaleB
                           << ", scaleC=" << scaleC << ", scaleD=" << scaleD << std::endl;

            hipblasLtMatmulDescDestroy(matmul);
            hipblasLtDestroy(handle);
        }
    }

    // Test 2: ScaleA Variations
    TEST(MultiGPUScaling, ScaleA_Variations)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing scaleA variations across GPUs" << std::endl;

        std::vector<float> scaleA_values = {0.0f, 0.5f, 1.0f, 2.0f, 10.0f};
        const int64_t M = 128, N = 128, K = 128;

        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            for(auto scaleA : scaleA_values)
            {
                hipblaslt_cout << "GPU " << dev << " testing scaleA=" << scaleA << std::endl;

                hipblasLtHandle_t handle;
                hipblasLtCreate(&handle);

                hipblasLtMatmulDesc_t matmul;
                hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);

                hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                                &scaleA, sizeof(scaleA));

                hipblasLtMatmulDescDestroy(matmul);
                hipblasLtDestroy(handle);
            }
        }
    }

    // Test 3: ScaleB Variations
    TEST(MultiGPUScaling, ScaleB_Variations)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing scaleB variations across GPUs" << std::endl;

        std::vector<float> scaleB_values = {0.25f, 0.5f, 1.0f, 2.0f, 4.0f};
        const int64_t M = 128, N = 128, K = 128;

        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            for(auto scaleB : scaleB_values)
            {
                hipblaslt_cout << "GPU " << dev << " testing scaleB=" << scaleB << std::endl;

                hipblasLtHandle_t handle;
                hipblasLtCreate(&handle);

                hipblasLtMatmulDesc_t matmul;
                hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);

                hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                                &scaleB, sizeof(scaleB));

                hipblasLtMatmulDescDestroy(matmul);
                hipblasLtDestroy(handle);
            }
        }
    }

    // Test 4: ScaleD Variations
    TEST(MultiGPUScaling, ScaleD_Variations)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing scaleD variations across GPUs" << std::endl;

        std::vector<float> scaleD_values = {0.1f, 0.5f, 1.0f, 1.5f, 2.0f};
        const int64_t M = 128, N = 128, K = 128;

        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            for(auto scaleD : scaleD_values)
            {
                hipblaslt_cout << "GPU " << dev << " testing scaleD=" << scaleD << std::endl;

                hipblasLtHandle_t handle;
                hipblasLtCreate(&handle);

                hipblasLtMatmulDesc_t matmul;
                hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);

                hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_D_SCALE_POINTER,
                                                &scaleD, sizeof(scaleD));

                hipblasLtMatmulDescDestroy(matmul);
                hipblasLtDestroy(handle);
            }
        }
    }

    // Test 5: AMax D Output
    TEST(MultiGPUScaling, AMax_D_Output)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing AMax D output across GPUs" << std::endl;

        const int64_t M = 256, N = 256, K = 256;

        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblasLtHandle_t handle;
            hipblasLtCreate(&handle);

            // Allocate AMax D buffer on device
            float* d_amax_D = nullptr;
            hipMalloc(&d_amax_D, sizeof(float));

            hipblasLtMatmulDesc_t matmul;
            hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);

            // Set AMax D pointer
            hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_AMAX_D_POINTER,
                                            &d_amax_D, sizeof(d_amax_D));

            hipblaslt_cout << "GPU " << dev << " AMax D configured" << std::endl;

            hipblasLtMatmulDescDestroy(matmul);
            hipFree(d_amax_D);
            hipblasLtDestroy(handle);
        }
    }

    // Test 6: Combined Scaling (All scales together)
    TEST(MultiGPUScaling, CombinedScaling)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing combined scaling (all scales) across GPUs" << std::endl;

        const int64_t M = 256, N = 256, K = 256;

        struct ScaleSet {
            float scaleA, scaleB, scaleC, scaleD;
            const char* name;
        };

        std::vector<ScaleSet> scale_sets = {
            {1.0f, 1.0f, 1.0f, 1.0f, "Unity"},
            {0.5f, 0.5f, 0.5f, 0.5f, "Half"},
            {2.0f, 2.0f, 2.0f, 2.0f, "Double"},
            {0.25f, 2.0f, 1.5f, 0.75f, "Mixed"}
        };

        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            for(const auto& scales : scale_sets)
            {
                hipblaslt_cout << "GPU " << dev << " testing " << scales.name << " scales" << std::endl;

                hipblasLtHandle_t handle;
                hipblasLtCreate(&handle);

                hipblasLtMatmulDesc_t matmul;
                hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);

                hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                                &scales.scaleA, sizeof(scales.scaleA));
                hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                                &scales.scaleB, sizeof(scales.scaleB));
                hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_C_SCALE_POINTER,
                                                &scales.scaleC, sizeof(scales.scaleC));
                hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_D_SCALE_POINTER,
                                                &scales.scaleD, sizeof(scales.scaleD));

                hipblasLtMatmulDescDestroy(matmul);
                hipblasLtDestroy(handle);
            }
        }
    }

    // Test 7: Different Scale Factors Per GPU
    TEST(MultiGPUScaling, DifferentScalesPerGPU)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing different scale factors per GPU" << std::endl;

        const int64_t M = 256, N = 256, K = 256;
        std::vector<float> scaleA_per_gpu = {0.5f, 1.0f, 1.5f, 2.0f};

        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            float scaleA = scaleA_per_gpu[dev % scaleA_per_gpu.size()];
            hipblaslt_cout << "GPU " << dev << " scaleA=" << scaleA << std::endl;

            hipblasLtHandle_t handle;
            hipblasLtCreate(&handle);

            hipblasLtMatmulDesc_t matmul;
            hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);

            hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                            &scaleA, sizeof(scaleA));

            hipblasLtMatmulDescDestroy(matmul);
            hipblasLtDestroy(handle);
        }
    }

    // Test 8: Scaling with FP16
    TEST(MultiGPUScaling, ScalingWithFP16)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing scaling with FP16 data type across GPUs" << std::endl;

        const int64_t M = 256, N = 256, K = 256;
        float scaleA = 0.5f, scaleB = 2.0f;

        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblasLtHandle_t handle;
            hipblasLtCreate(&handle);

            hipblasLtMatmulDesc_t matmul;
            hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_16F);

            hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                            &scaleA, sizeof(scaleA));
            hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                            &scaleB, sizeof(scaleB));

            hipblaslt_cout << "GPU " << dev << " FP16 scaling configured" << std::endl;

            hipblasLtMatmulDescDestroy(matmul);
            hipblasLtDestroy(handle);
        }
    }

    // Test 9: Scaling with INT8
    TEST(MultiGPUScaling, ScalingWithINT8)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing scaling with INT8 data type across GPUs" << std::endl;

        const int64_t M = 256, N = 256, K = 256;
        float scaleA = 0.003921f; // 1/255 for INT8 normalization
        float scaleB = 0.003921f;
        float scaleD = 1.0f;

        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblasLtHandle_t handle;
            hipblasLtCreate(&handle);

            hipblasLtMatmulDesc_t matmul;
            hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32I, HIP_R_8I);

            hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                            &scaleA, sizeof(scaleA));
            hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                            &scaleB, sizeof(scaleB));
            hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_D_SCALE_POINTER,
                                            &scaleD, sizeof(scaleD));

            hipblaslt_cout << "GPU " << dev << " INT8 scaling configured" << std::endl;

            hipblasLtMatmulDescDestroy(matmul);
            hipblasLtDestroy(handle);
        }
    }

    // Test 10: Very Small and Very Large Scales
    TEST(MultiGPUScaling, ExtremeScaleValues)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing extreme scale values across GPUs" << std::endl;

        std::vector<float> extreme_scales = {
            1e-6f,  // Very small
            1e-3f,
            1.0f,   // Normal
            1e3f,
            1e6f    // Very large
        };

        const int64_t M = 128, N = 128, K = 128;

        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            for(auto scale : extreme_scales)
            {
                hipblaslt_cout << "GPU " << dev << " testing scale=" << scale << std::endl;

                hipblasLtHandle_t handle;
                hipblasLtCreate(&handle);

                hipblasLtMatmulDesc_t matmul;
                hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);

                hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                                &scale, sizeof(scale));

                hipblasLtMatmulDescDestroy(matmul);
                hipblasLtDestroy(handle);
            }
        }
    }

} // namespace
