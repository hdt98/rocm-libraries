// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Use CK_Tile for host-side utilities
#include "ck_tile/host.hpp"
#include <cstring>
#include <chrono>

// Include MINT kernel implementation
#include "copy_mint_simple.hpp"

/**
 * @brief Create and parse command-line arguments
 */
auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("v", "1", "cpu validation or not")
        .insert("warmup", "50", "cold iter")
        .insert("repeat", "100", "hot iter");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

/**
 * @brief Run tiled MINT copy kernel
 */
template <typename DataType>
bool run_tiled(const ck_tile::ArgParser& arg_parser, ck_tile::index_t m, ck_tile::index_t n)
{
    using XDataType = DataType;
    using YDataType = DataType;

    int do_validation = arg_parser.get_int("v");
    int warmup        = arg_parser.get_int("warmup");
    int repeat        = arg_parser.get_int("repeat");

    std::cout << "=== MINT Tiled Copy Kernel ===" << std::endl;
    std::cout << "Matrix dimensions: " << m << " x " << n << std::endl;

    // Create host tensors
    ck_tile::HostTensor<XDataType> x_host({m, n});
    ck_tile::HostTensor<YDataType> y_host_dev({m, n});

    // Initialize input data
    for(int i = 0; i < m; i++)
    {
        for(int j = 0; j < n; j++)
        {
            x_host(i, j) = static_cast<ck_tile::half_t>((i * n + j + 1) / 1000.0f);
        }
    }

    // Allocate device memory
    ck_tile::DeviceMem x_buf(x_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem y_buf(y_host_dev.get_element_space_size_in_bytes());

    x_buf.ToDevice(x_host.data());

    // Kernel configuration
    constexpr ck_tile::index_t RowPerTile = 16;
    constexpr ck_tile::index_t ColPerTile = 16;
    constexpr ck_tile::index_t kBlockSize = 64; // One warp per block
    ck_tile::index_t kGridSize            = ck_tile::integer_divide_ceil(m, RowPerTile);

    std::cout << "Tile: " << RowPerTile << "x" << ColPerTile << ", Block size: " << kBlockSize
              << ", Grid size: " << kGridSize << std::endl;

    hipStream_t stream = nullptr;

    // Warmup
    for(int i = 0; i < warmup; i++)
    {
        hipLaunchKernelGGL((mint_tutorial::MintCopyKernelTiled<RowPerTile, ColPerTile, XDataType>),
                           dim3(kGridSize),
                           dim3(kBlockSize),
                           0,
                           stream,
                           static_cast<YDataType*>(y_buf.GetDeviceBuffer()),
                           static_cast<XDataType*>(x_buf.GetDeviceBuffer()),
                           m,
                           n);
    }
    hipDeviceSynchronize();

    // Timed iterations
    auto start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < repeat; i++)
    {
        hipLaunchKernelGGL((mint_tutorial::MintCopyKernelTiled<RowPerTile, ColPerTile, XDataType>),
                           dim3(kGridSize),
                           dim3(kBlockSize),
                           0,
                           stream,
                           static_cast<YDataType*>(y_buf.GetDeviceBuffer()),
                           static_cast<XDataType*>(x_buf.GetDeviceBuffer()),
                           m,
                           n);
    }
    hipDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    float ave_time = std::chrono::duration<float, std::milli>(end - start).count() / repeat;

    // Performance metrics
    std::size_t num_bytes = sizeof(XDataType) * m * n + sizeof(YDataType) * m * n;
    float gb_per_sec      = num_bytes / 1.E6 / ave_time;

    std::cout << "Average time: " << ave_time << " ms, Bandwidth: " << gb_per_sec << " GB/s"
              << std::endl;

    bool pass = true;
    if(do_validation)
    {
        y_buf.FromDevice(y_host_dev.mData.data());
        pass = ck_tile::check_err(y_host_dev, x_host, "Error: Tiled copy failed!", 0.0, 0.0);
        std::cout << "Validation: " << (pass ? "PASSED" : "FAILED") << std::endl;
    }

    return pass;
}

int main(int argc, char* argv[])
{
    std::cout << "====================================" << std::endl;
    std::cout << "MINT Copy Kernel Tutorial" << std::endl;
    std::cout << "Using MINT + CK_Tile host utilities" << std::endl;
    std::cout << "====================================" << std::endl << std::endl;

    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
    {
        std::cerr << "Failed to parse arguments" << std::endl;
        return -1;
    }

    bool pass = true;

    // Test tiled kernel with different sizes
    std::cout << "\n=== Testing Tiled Kernel ===" << std::endl;
    pass = pass && run_tiled<ck_tile::half_t>(arg_parser, 256, 512);
    pass = pass && run_tiled<ck_tile::half_t>(arg_parser, 1024, 2048);

    std::cout << "\n====================================" << std::endl;
    std::cout << (pass ? "All tests PASSED" : "Some tests FAILED") << std::endl;
    std::cout << "====================================" << std::endl;

    return pass ? 0 : -2;
}
