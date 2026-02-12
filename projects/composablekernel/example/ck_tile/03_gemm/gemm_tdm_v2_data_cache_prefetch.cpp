// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gemm_tdm_data_cache_prefetch_common.hpp"

// TDM V2 GEMM Configuration with Data Cache Prefetch control
template <typename PrecType, bool UseDataCachePrefetch_, bool DataCachePrefetchToL1_ = false>
struct GemmConfigTDMV2Prefetch : public GemmConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 128;
    static constexpr ck_tile::index_t K_Tile = 64;

    //  TDM V2 (requires 4 waves):  M_Warp * N_Warp * K_Warp == 4
    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile =
        ck_tile::get_k_warp_tile<PrecType, M_Warp_Tile>();

    static constexpr bool kPadM = true;
    static constexpr bool kPadN = true;
    static constexpr bool kPadK = true;

    static constexpr bool DoubleSmemBuffer          = true;
    static constexpr ck_tile::GemmPipeline Pipeline = ck_tile::GemmPipeline::COMPUTE_TDM_V2;
    static constexpr bool UseDataCachePrefetch      = UseDataCachePrefetch_;
    static constexpr bool DataCachePrefetchToL1     = DataCachePrefetchToL1_;
};

int run_gemm_example(ck_tile::ArgParser& arg_parser)
{
    return run_gemm_example_with_prefetch<GemmConfigTDMV2Prefetch>(arg_parser);
}

int main(int argc, char* argv[])
{
    auto arg_parser = create_args();
    arg_parser.insert(
        "compare",
        "0",
        "0: Run with data cache prefetch only, 1: Compare with/without data cache prefetch");
    arg_parser.insert("prefetch_l1", "0", "0: Prefetch to L2 cache, 1: Prefetch to L1 cache");
    auto result = arg_parser.parse(argc, argv);

    if(!result)
        return -1;

    try
    {
        return !run_gemm_example(arg_parser);
    }
    catch(std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return -1;
    }
}
