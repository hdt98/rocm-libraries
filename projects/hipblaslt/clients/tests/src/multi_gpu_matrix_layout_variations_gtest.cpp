/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
 *
 * Multi-GPU Matrix Layout Variations Test Suite
 * Testing column-major, row-major, and swizzle patterns
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

    // Test 1: Column-Major Layout (Default)
    TEST(MultiGPULayout, ColumnMajorLayout)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing column-major layout across GPUs" << std::endl;

        const int64_t M = 256, N = 256, K = 256;

        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblasLtHandle_t handle;
            hipblasLtCreate(&handle);

            hipblasLtMatrixLayout_t matA, matB, matD;

            // Column-major is default (ld = M for A and D, ld = K for B)
            hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M, K, M);
            hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, K, N, K);
            hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M, N, M);

            // Explicitly set order to column-major
            hipblasLtOrder_t order = HIPBLASLT_ORDER_COL;
            hipblasLtMatrixLayoutSetAttribute(matA, HIPBLASLT_MATRIX_LAYOUT_ORDER,
                                              &order, sizeof(order));
            hipblasLtMatrixLayoutSetAttribute(matB, HIPBLASLT_MATRIX_LAYOUT_ORDER,
                                              &order, sizeof(order));
            hipblasLtMatrixLayoutSetAttribute(matD, HIPBLASLT_MATRIX_LAYOUT_ORDER,
                                              &order, sizeof(order));

            hipblaslt_cout << "GPU " << dev << " column-major layout configured" << std::endl;

            hipblasLtMatrixLayoutDestroy(matA);
            hipblasLtMatrixLayoutDestroy(matB);
            hipblasLtMatrixLayoutDestroy(matD);
            hipblasLtDestroy(handle);
        }
    }

    // Test 2: Row-Major Layout
    TEST(MultiGPULayout, RowMajorLayout)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing row-major layout across GPUs" << std::endl;

        const int64_t M = 256, N = 256, K = 256;

        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblasLtHandle_t handle;
            hipblasLtCreate(&handle);

            hipblasLtMatrixLayout_t matA, matB, matD;

            // Row-major (ld = K for A, ld = N for B, ld = N for D)
            hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M, K, K);
            hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, K, N, N);
            hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M, N, N);

            hipblasLtOrder_t order = HIPBLASLT_ORDER_ROW;
            hipblasLtMatrixLayoutSetAttribute(matA, HIPBLASLT_MATRIX_LAYOUT_ORDER,
                                              &order, sizeof(order));
            hipblasLtMatrixLayoutSetAttribute(matB, HIPBLASLT_MATRIX_LAYOUT_ORDER,
                                              &order, sizeof(order));
            hipblasLtMatrixLayoutSetAttribute(matD, HIPBLASLT_MATRIX_LAYOUT_ORDER,
                                              &order, sizeof(order));

            hipblaslt_cout << "GPU " << dev << " row-major layout configured" << std::endl;

            hipblasLtMatrixLayoutDestroy(matA);
            hipblasLtMatrixLayoutDestroy(matB);
            hipblasLtMatrixLayoutDestroy(matD);
            hipblasLtDestroy(handle);
        }
    }

    // Test 3: Mixed Layouts (A column-major, B row-major)
    TEST(MultiGPULayout, MixedLayouts)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing mixed layouts across GPUs" << std::endl;

        const int64_t M = 256, N = 256, K = 256;

        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblasLtHandle_t handle;
            hipblasLtCreate(&handle);

            hipblasLtMatrixLayout_t matA, matB, matD;

            // A: column-major, B: row-major, D: column-major
            hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M, K, M);
            hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, K, N, N);
            hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, M, N, M);

            hipblasLtOrder_t orderCol = HIPBLASLT_ORDER_COL;
            hipblasLtOrder_t orderRow = HIPBLASLT_ORDER_ROW;

            hipblasLtMatrixLayoutSetAttribute(matA, HIPBLASLT_MATRIX_LAYOUT_ORDER,
                                              &orderCol, sizeof(orderCol));
            hipblasLtMatrixLayoutSetAttribute(matB, HIPBLASLT_MATRIX_LAYOUT_ORDER,
                                              &orderRow, sizeof(orderRow));
            hipblasLtMatrixLayoutSetAttribute(matD, HIPBLASLT_MATRIX_LAYOUT_ORDER,
                                              &orderCol, sizeof(orderCol));

            hipblaslt_cout << "GPU " << dev << " mixed layouts configured" << std::endl;

            hipblasLtMatrixLayoutDestroy(matA);
            hipblasLtMatrixLayoutDestroy(matB);
            hipblasLtMatrixLayoutDestroy(matD);
            hipblasLtDestroy(handle);
        }
    }

    // Test 4: COL16_4R32 Layout (Tensor Core optimization)
    TEST(MultiGPULayout, COL16_4R32_Layout)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing COL16_4R32 layout across GPUs" << std::endl;

        // Must be multiples of 16 for COL16_4R32
        const int64_t M = 256, N = 256, K = 256;

        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblasLtHandle_t handle;
            hipblasLtCreate(&handle);

            hipblasLtMatrixLayout_t matA;
            hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M, K, M);

            // Try to set COL16_4R32 layout (may not be supported on all configs)
            hipblasLtOrder_t order = HIPBLASLT_ORDER_COL16_4R32;
            auto status = hipblasLtMatrixLayoutSetAttribute(matA, HIPBLASLT_MATRIX_LAYOUT_ORDER,
                                                            &order, sizeof(order));

            if(status == HIPBLAS_STATUS_SUCCESS)
            {
                hipblaslt_cout << "GPU " << dev << " COL16_4R32 layout configured" << std::endl;
            }
            else
            {
                hipblaslt_cout << "GPU " << dev << " COL16_4R32 layout not supported (skipped)" << std::endl;
            }

            hipblasLtMatrixLayoutDestroy(matA);
            hipblasLtDestroy(handle);
        }
    }

    // Test 5: COL16_4R16 Layout
    TEST(MultiGPULayout, COL16_4R16_Layout)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing COL16_4R16 layout across GPUs" << std::endl;

        const int64_t M = 256, N = 256, K = 256;

        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblasLtHandle_t handle;
            hipblasLtCreate(&handle);

            hipblasLtMatrixLayout_t matA;
            hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M, K, M);

            hipblasLtOrder_t order = HIPBLASLT_ORDER_COL16_4R16;
            auto status = hipblasLtMatrixLayoutSetAttribute(matA, HIPBLASLT_MATRIX_LAYOUT_ORDER,
                                                            &order, sizeof(order));

            if(status == HIPBLAS_STATUS_SUCCESS)
            {
                hipblaslt_cout << "GPU " << dev << " COL16_4R16 layout configured" << std::endl;
            }
            else
            {
                hipblaslt_cout << "GPU " << dev << " COL16_4R16 layout not supported (skipped)" << std::endl;
            }

            hipblasLtMatrixLayoutDestroy(matA);
            hipblasLtDestroy(handle);
        }
    }

    // Test 6: COL16_4R8 Layout
    TEST(MultiGPULayout, COL16_4R8_Layout)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing COL16_4R8 layout across GPUs" << std::endl;

        const int64_t M = 256, N = 256, K = 256;

        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblasLtHandle_t handle;
            hipblasLtCreate(&handle);

            hipblasLtMatrixLayout_t matA;
            hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M, K, M);

            hipblasLtOrder_t order = HIPBLASLT_ORDER_COL16_4R8;
            auto status = hipblasLtMatrixLayoutSetAttribute(matA, HIPBLASLT_MATRIX_LAYOUT_ORDER,
                                                            &order, sizeof(order));

            if(status == HIPBLAS_STATUS_SUCCESS)
            {
                hipblaslt_cout << "GPU " << dev << " COL16_4R8 layout configured" << std::endl;
            }
            else
            {
                hipblaslt_cout << "GPU " << dev << " COL16_4R8 layout not supported (skipped)" << std::endl;
            }

            hipblasLtMatrixLayoutDestroy(matA);
            hipblasLtDestroy(handle);
        }
    }

    // Test 7: Different Layouts Per GPU
    TEST(MultiGPULayout, DifferentLayoutsPerGPU)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing different layouts per GPU" << std::endl;

        std::vector<hipblasLtOrder_t> layouts = {
            HIPBLASLT_ORDER_COL,
            HIPBLASLT_ORDER_ROW,
            HIPBLASLT_ORDER_COL16_4R32,
            HIPBLASLT_ORDER_COL16_4R16
        };

        const char* layout_names[] = {
            "COL", "ROW", "COL16_4R32", "COL16_4R16"
        };

        const int64_t M = 256, N = 256, K = 256;

        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            hipblasLtHandle_t handle;
            hipblasLtCreate(&handle);

            hipblasLtMatrixLayout_t matA;
            hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, M, K, M);

            // Cycle through available layouts if we have more GPUs than layouts
            int layout_idx = dev % layouts.size();
            auto status = hipblasLtMatrixLayoutSetAttribute(matA, HIPBLASLT_MATRIX_LAYOUT_ORDER,
                                                            &layouts[layout_idx], sizeof(layouts[layout_idx]));

            if(status == HIPBLAS_STATUS_SUCCESS)
            {
                hipblaslt_cout << "GPU " << dev << " using " << layout_names[layout_idx] << " layout" << std::endl;
            }
            else
            {
                hipblaslt_cout << "GPU " << dev << " " << layout_names[layout_idx] << " layout not supported" << std::endl;
            }

            hipblasLtMatrixLayoutDestroy(matA);
            hipblasLtDestroy(handle);
        }
    }

    // Test 8: Layout with Different Data Types
    TEST(MultiGPULayout, LayoutWithDifferentDataTypes)
    {
        int numDevices = getNumGPUs();
        if(numDevices < 2) GTEST_SKIP() << "Requires 2+ GPUs";

        hipblaslt_cout << "Testing layouts with different data types across GPUs" << std::endl;

        struct TypeConfig {
            hipDataType dataType;
            const char* name;
        };

        std::vector<TypeConfig> types = {
            {HIP_R_32F, "FP32"},
            {HIP_R_16F, "FP16"},
            {HIP_R_16BF, "BF16"}
        };

        std::vector<hipblasLtOrder_t> layouts = {
            HIPBLASLT_ORDER_COL,
            HIPBLASLT_ORDER_ROW
        };

        const int64_t M = 256, N = 256, K = 256;

        for(int dev = 0; dev < numDevices; ++dev)
        {
            auto hipErr = hipSetDevice(dev);
            ASSERT_EQ(hipErr, hipSuccess);

            for(const auto& type : types)
            {
                for(auto layout : layouts)
                {
                    hipblasLtHandle_t handle;
                    hipblasLtCreate(&handle);

                    int64_t ld = (layout == HIPBLASLT_ORDER_COL) ? M : K;
                    hipblasLtMatrixLayout_t matA;
                    hipblasLtMatrixLayoutCreate(&matA, type.dataType, M, K, ld);

                    auto status = hipblasLtMatrixLayoutSetAttribute(matA, HIPBLASLT_MATRIX_LAYOUT_ORDER,
                                                                    &layout, sizeof(layout));

                    if(status == HIPBLAS_STATUS_SUCCESS)
                    {
                        hipblaslt_cout << "GPU " << dev << " " << type.name << " with layout "
                                       << (layout == HIPBLASLT_ORDER_COL ? "COL" : "ROW") << std::endl;
                    }

                    hipblasLtMatrixLayoutDestroy(matA);
                    hipblasLtDestroy(handle);
                }
            }
        }
    }

} // namespace
