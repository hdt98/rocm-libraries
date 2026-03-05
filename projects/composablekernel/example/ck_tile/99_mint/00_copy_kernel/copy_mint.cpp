// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Use CK_Tile for host-side utilities
#include "ck_tile/host.hpp"
#include <cstring>

// Include MINT kernel implementation
#include "copy_mint.hpp"

/**
 * @brief Create and parse command-line arguments
 *
 * Uses CK_Tile's ArgParser for convenience
 */
auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("m", "64", "m dimension (rows)")
        .insert("n", "128", "n dimension (cols)")
        .insert("v", "1", "cpu validation or not")
        .insert("warmup", "50", "cold iter")
        .insert("repeat", "100", "hot iter");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

/**
 * @brief Run the MINT copy kernel with CK_Tile host utilities
 *
 * This demonstrates the hybrid approach:
 * - MINT for device-side kernel implementation
 * - CK_Tile for host-side memory management, kernel launch, and validation
 *
 * @tparam DataType Currently only fp16 (ck_tile::half_t) is supported by MINT
 */
template <typename DataType>
bool run(const ck_tile::ArgParser& arg_parser)
{
    using XDataType = DataType;
    using YDataType = DataType;

    // Parse arguments
    ck_tile::index_t m = arg_parser.get_int("m");
    ck_tile::index_t n = arg_parser.get_int("n");
    int do_validation  = arg_parser.get_int("v");
    int warmup         = arg_parser.get_int("warmup");
    int repeat         = arg_parser.get_int("repeat");

    std::cout << "=== MINT Copy Kernel Tutorial ===" << std::endl;
    std::cout << "Matrix dimensions: " << m << " x " << n << std::endl;
    std::cout << "Data type: fp16" << std::endl;
    std::cout << "Warmup iterations: " << warmup << std::endl;
    std::cout << "Repeat iterations: " << repeat << std::endl;
    std::cout << "===================================" << std::endl;

    // Create host tensors using CK_Tile utilities
    ck_tile::HostTensor<XDataType> x_host({m, n});     // input matrix
    ck_tile::HostTensor<YDataType> y_host_ref({m, n}); // reference output matrix
    ck_tile::HostTensor<YDataType> y_host_dev({m, n}); // device output matrix

    // Initialize input data with sequential values for easy verification
    ck_tile::half_t value = 1;
    for(int i = 0; i < m; i++)
    {
        value = 1;
        for(int j = 0; j < n; j++)
        {
            x_host(i, j) = value++;
        }
    }

    // Allocate device memory using CK_Tile utilities
    ck_tile::DeviceMem x_buf(x_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem y_buf(y_host_dev.get_element_space_size_in_bytes());

    // Copy input data to device
    x_buf.ToDevice(x_host.data());

    // Define MINT kernel configuration
    // Each warp processes a 16x16 tile
    constexpr ck_tile::index_t RowPerTile = 16;
    constexpr ck_tile::index_t ColPerTile = 16;

    // Calculate grid size
    // In this simple example, each block contains one warp
    // Multiple blocks process different row ranges
    constexpr ck_tile::index_t kWaveSize = 64; // ROCm wave size
    constexpr ck_tile::index_t kBlockSize = kWaveSize; // One warp per block

    // Calculate number of blocks needed
    // Each block processes RowPerTile rows
    ck_tile::index_t kGridSize = ck_tile::integer_divide_ceil(m, RowPerTile);

    std::cout << "\n=== Kernel Configuration ===" << std::endl;
    std::cout << "Tile size: " << RowPerTile << " x " << ColPerTile << std::endl;
    std::cout << "Block size (threads per block): " << kBlockSize << std::endl;
    std::cout << "Grid size (number of blocks): " << kGridSize << std::endl;
    std::cout << "Wave size: " << kWaveSize << std::endl;
    std::cout << "=============================" << std::endl;

    // Launch MINT kernel using HIP kernel launch
    hipStream_t stream = nullptr;

    // Warmup iterations
    for(int i = 0; i < warmup; i++)
    {
        hipLaunchKernelGGL(
            (mint_tutorial::MintCopyKernel<RowPerTile, ColPerTile, XDataType>),
            dim3(kGridSize), dim3(kBlockSize), 0, stream,
            static_cast<YDataType*>(y_buf.GetDeviceBuffer()),
            static_cast<XDataType*>(x_buf.GetDeviceBuffer()),
            m, n);
    }
    hipDeviceSynchronize();

    // Timed iterations
    auto start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < repeat; i++)
    {
        hipLaunchKernelGGL(
            (mint_tutorial::MintCopyKernel<RowPerTile, ColPerTile, XDataType>),
            dim3(kGridSize), dim3(kBlockSize), 0, stream,
            static_cast<YDataType*>(y_buf.GetDeviceBuffer()),
            static_cast<XDataType*>(x_buf.GetDeviceBuffer()),
            m, n);
    }
    hipDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    float ave_time = std::chrono::duration<float, std::milli>(end - start).count() / repeat;

    // Calculate and print performance metrics
    std::size_t num_bytes = sizeof(XDataType) * m * n + sizeof(YDataType) * m * n;
    float gb_per_sec = num_bytes / 1.E6 / ave_time;

    std::cout << "\n=== Performance ===" << std::endl;
    std::cout << "Average time: " << ave_time << " ms" << std::endl;
    std::cout << "Bandwidth: " << gb_per_sec << " GB/s" << std::endl;
    std::cout << "===================" << std::endl;

    bool pass = true;

    if(do_validation)
    {
        std::cout << "\n=== Validation ===" << std::endl;

        // Copy results back to host
        y_buf.FromDevice(y_host_dev.mData.data());

        // Use CK_Tile's validation utility
        // For copy operations, we expect exact match (tolerance = 0)
        pass = ck_tile::check_err(
            y_host_dev,
            x_host,
            "Error: MINT copy operation failed!",
            0.0,  // rtol
            0.0); // atol

        std::cout << "Validation result: " << (pass ? "PASSED" : "FAILED") << std::endl;
        std::cout << "===================" << std::endl;
    }

    // Optional: Print sample data for debugging
    if(false)
    {
        std::cout << "\n=== Sample Data (first 5x5) ===" << std::endl;
        std::cout << "Input:" << std::endl;
        for(int i = 0; i < std::min(5, (int)m); i++)
        {
            for(int j = 0; j < std::min(5, (int)n); j++)
            {
                std::cout << static_cast<float>(x_host(i, j)) << " ";
            }
            std::cout << std::endl;
        }
        std::cout << "\nOutput:" << std::endl;
        for(int i = 0; i < std::min(5, (int)m); i++)
        {
            for(int j = 0; j < std::min(5, (int)n); j++)
            {
                std::cout << static_cast<float>(y_host_dev(i, j)) << " ";
            }
            std::cout << std::endl;
        }
        std::cout << "================================" << std::endl;
    }

    return pass;
}

int main(int argc, char* argv[])
{
    std::cout << "MINT Copy Kernel Tutorial" << std::endl;
    std::cout << "Using MINT for device kernels + CK_Tile for host utilities" << std::endl;
    std::cout << std::endl;

    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
    {
        std::cerr << "Failed to parse arguments" << std::endl;
        return -1;
    }

    // MINT currently only supports fp16
    std::cout << "Running with fp16 (MINT only supports fp16 currently)" << std::endl;
    return run<ck_tile::half_t>(arg_parser) ? 0 : -2;
}
