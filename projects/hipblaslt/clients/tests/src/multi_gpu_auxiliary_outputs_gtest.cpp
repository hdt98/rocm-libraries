/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
 *
 * Multi-GPU Auxiliary Outputs Test Suite
 * Testing bias pointers, AMax pointers, and auxiliary matrices
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

    // Test 1: Bias Pointer (NULL)
    TEST(MultiGPUAuxiliary, BiasPointer_NULL)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing NULL bias pointer across GPUs" << std::endl;

        const int64_t M = 256, N = 256, K = 256;

        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblasLtHandle_t handle;
            hipblasLtCreate(&handle);

            hipblasLtMatmulDesc_t matmul;
            hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);

            // Set bias pointer to NULL
            void* bias_ptr = nullptr;
            hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_BIAS_POINTER,
                                            &bias_ptr, sizeof(bias_ptr));

            hipblaslt_cout << "GPU " << dev << " NULL bias pointer configured" << std::endl;

            hipblasLtMatmulDescDestroy(matmul);
            hipblasLtDestroy(handle);
        }
    }

    // Test 2: Bias Pointer (Valid)
    TEST(MultiGPUAuxiliary, BiasPointer_Valid)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing valid bias pointer across GPUs" << std::endl;

        const int64_t M = 256, N = 256, K = 256;

        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblasLtHandle_t handle;
            hipblasLtCreate(&handle);

            // Allocate bias vector on device
            float* d_bias = nullptr;
            hipMalloc(&d_bias, M * sizeof(float));

            hipblasLtMatmulDesc_t matmul;
            hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);

            // Set valid bias pointer
            hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_BIAS_POINTER,
                                            &d_bias, sizeof(d_bias));

            hipblaslt_cout << "GPU " << dev << " valid bias pointer configured" << std::endl;

            hipblasLtMatmulDescDestroy(matmul);
            hipFree(d_bias);
            hipblasLtDestroy(handle);
        }
    }

    // Test 3: AMax D Pointer (NULL)
    TEST(MultiGPUAuxiliary, AMaxD_NULL)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing NULL AMax D pointer across GPUs" << std::endl;

        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblasLtHandle_t handle;
            hipblasLtCreate(&handle);

            hipblasLtMatmulDesc_t matmul;
            hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);

            // Set AMax D pointer to NULL
            void* amax_d_ptr = nullptr;
            hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_AMAX_D_POINTER,
                                            &amax_d_ptr, sizeof(amax_d_ptr));

            hipblaslt_cout << "GPU " << dev << " NULL AMax D pointer configured" << std::endl;

            hipblasLtMatmulDescDestroy(matmul);
            hipblasLtDestroy(handle);
        }
    }

    // Test 4: AMax D Pointer (Valid)
    TEST(MultiGPUAuxiliary, AMaxD_Valid)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing valid AMax D pointer across GPUs" << std::endl;

        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblasLtHandle_t handle;
            hipblasLtCreate(&handle);

            // Allocate AMax D on device
            float* d_amax_D = nullptr;
            hipMalloc(&d_amax_D, sizeof(float));

            hipblasLtMatmulDesc_t matmul;
            hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);

            // Set valid AMax D pointer
            hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_AMAX_D_POINTER,
                                            &d_amax_D, sizeof(d_amax_D));

            hipblaslt_cout << "GPU " << dev << " valid AMax D pointer configured" << std::endl;

            hipblasLtMatmulDescDestroy(matmul);
            hipFree(d_amax_D);
            hipblasLtDestroy(handle);
        }
    }

    // Test 5: Combined Bias and Epilogue
    TEST(MultiGPUAuxiliary, BiasWithEpilogue)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing bias with epilogue operations across GPUs" << std::endl;

        const int64_t M = 256, N = 256, K = 256;

        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblasLtHandle_t handle;
            hipblasLtCreate(&handle);

            // Allocate bias vector
            float* d_bias = nullptr;
            hipMalloc(&d_bias, M * sizeof(float));

            hipblasLtMatmulDesc_t matmul;
            hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);

            // Set bias pointer
            hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_BIAS_POINTER,
                                            &d_bias, sizeof(d_bias));

            // Set epilogue operation (RELU_BIAS)
            hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_RELU_BIAS;
            hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE,
                                            &epilogue, sizeof(epilogue));

            hipblaslt_cout << "GPU " << dev << " bias with RELU epilogue configured" << std::endl;

            hipblasLtMatmulDescDestroy(matmul);
            hipFree(d_bias);
            hipblasLtDestroy(handle);
        }
    }

    // Test 6: Multiple Auxiliary Outputs
    TEST(MultiGPUAuxiliary, MultipleAuxiliaryOutputs)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing multiple auxiliary outputs across GPUs" << std::endl;

        const int64_t M = 256, N = 256, K = 256;

        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblasLtHandle_t handle;
            hipblasLtCreate(&handle);

            // Allocate bias and AMax D
            float* d_bias = nullptr;
            float* d_amax_D = nullptr;
            hipMalloc(&d_bias, M * sizeof(float));
            hipMalloc(&d_amax_D, sizeof(float));

            hipblasLtMatmulDesc_t matmul;
            hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);

            // Set bias pointer
            hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_BIAS_POINTER,
                                            &d_bias, sizeof(d_bias));

            // Set AMax D pointer
            hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_AMAX_D_POINTER,
                                            &d_amax_D, sizeof(d_amax_D));

            hipblaslt_cout << "GPU " << dev << " multiple auxiliary outputs configured" << std::endl;

            hipblasLtMatmulDescDestroy(matmul);
            hipFree(d_bias);
            hipFree(d_amax_D);
            hipblasLtDestroy(handle);
        }
    }

    // Test 7: Different Auxiliary Outputs Per GPU
    TEST(MultiGPUAuxiliary, DifferentAuxiliaryPerGPU)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing different auxiliary outputs per GPU" << std::endl;

        const int64_t M = 256, N = 256, K = 256;

        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblasLtHandle_t handle;
            hipblasLtCreate(&handle);

            hipblasLtMatmulDesc_t matmul;
            hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);

            // Alternate between bias (even GPUs) and AMax D (odd GPUs)
            if(dev % 2 == 0)
            {
                // Even GPUs: Use bias only
                float* d_bias = nullptr;
                hipMalloc(&d_bias, M * sizeof(float));

                hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                &d_bias, sizeof(d_bias));

                hipblaslt_cout << "GPU " << dev << ": bias only" << std::endl;

                hipblasLtMatmulDescDestroy(matmul);
                hipFree(d_bias);
            }
            else
            {
                // Odd GPUs: Use AMax D only
                float* d_amax_D = nullptr;
                hipMalloc(&d_amax_D, sizeof(float));

                hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_AMAX_D_POINTER,
                                                &d_amax_D, sizeof(d_amax_D));

                hipblaslt_cout << "GPU " << dev << ": AMax D only" << std::endl;

                hipblasLtMatmulDescDestroy(matmul);
                hipFree(d_amax_D);
            }

            hipblasLtDestroy(handle);
        }
    }

    // Test 8: Bias with Different Data Types
    TEST(MultiGPUAuxiliary, BiasWithDifferentDataTypes)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing bias with different data types across GPUs" << std::endl;

        struct TypeConfig {
            hipDataType dataType;
            const char* name;
            size_t element_size;
        };

        std::vector<TypeConfig> types = {
            {HIP_R_32F, "FP32", sizeof(float)},
            {HIP_R_16F, "FP16", sizeof(uint16_t)},
            {HIP_R_16BF, "BF16", sizeof(uint16_t)}
        };

        const int64_t M = 256, N = 256, K = 256;

        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            for(const auto& type : types)
            {
                hipblasLtHandle_t handle;
                hipblasLtCreate(&handle);

                // Allocate bias vector for the data type
                void* d_bias = nullptr;
                hipMalloc(&d_bias, M * type.element_size);

                hipblasLtMatmulDesc_t matmul;
                hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, type.dataType);

                // Set bias pointer
                hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                &d_bias, sizeof(d_bias));

                // Set bias data type
                hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
                                                &type.dataType, sizeof(type.dataType));

                hipblaslt_cout << "GPU " << dev << " bias with " << type.name
                               << " data type configured" << std::endl;

                hipblasLtMatmulDescDestroy(matmul);
                hipFree(d_bias);
                hipblasLtDestroy(handle);
            }
        }
    }

    // Test 9: Bias Vector vs Broadcasted Bias
    TEST(MultiGPUAuxiliary, BiasVectorVsBroadcasted)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing vector bias vs broadcasted bias across GPUs" << std::endl;

        const int64_t M = 256, N = 256, K = 256;

        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblasLtHandle_t handle;
            hipblasLtCreate(&handle);

            // Test 1: Vector bias (length M)
            {
                float* d_bias = nullptr;
                hipMalloc(&d_bias, M * sizeof(float));

                hipblasLtMatmulDesc_t matmul;
                hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);

                hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                &d_bias, sizeof(d_bias));

                hipblaslt_cout << "GPU " << dev << " vector bias (length M) configured" << std::endl;

                hipblasLtMatmulDescDestroy(matmul);
                hipFree(d_bias);
            }

            // Test 2: Broadcasted bias (single value)
            {
                float* d_bias = nullptr;
                hipMalloc(&d_bias, sizeof(float));

                hipblasLtMatmulDesc_t matmul;
                hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);

                hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                &d_bias, sizeof(d_bias));

                hipblaslt_cout << "GPU " << dev << " broadcasted bias (single value) configured" << std::endl;

                hipblasLtMatmulDescDestroy(matmul);
                hipFree(d_bias);
            }

            hipblasLtDestroy(handle);
        }
    }

} // namespace
