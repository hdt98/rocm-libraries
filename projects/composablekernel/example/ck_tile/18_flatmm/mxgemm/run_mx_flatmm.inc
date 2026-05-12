// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstring>
#include <set>
#include <type_traits>

template <typename DataType>
constexpr DataType float_to_data_element(float x)
{
    if constexpr(std::is_same_v<DataType, ck_tile::pk_fp4_t>)
        return ck_tile::type_convert<DataType>(ck_tile::fp32x2_t(x));
    else if constexpr(std::is_same_v<DataType, ck_tile::pk_fp6_t>)
        return ck_tile::pk_fp6_t(x);
    else
        return ck_tile::type_convert<DataType>(x);
}

// Initialization proxy: the legacy ck_tile::pk_fp6x16_t type lacks the
// converters that FillUniformDistribution / type_convert / set_element rely
// on, but is byte-equivalent to the modern ck_tile::pk_fp6_t (both are
// 16 fp6 e2m3 lanes packed in 96 bits with the same intra-word layout).
// We therefore initialize through a pk_fp6_t HostTensor and std::memcpy the
// raw bytes into the legacy tensor before the kernel and reference calls.
template <typename T>
using mx_flatmm_init_proxy_t =
    std::conditional_t<std::is_same_v<T, ck_tile::pk_fp6x16_t>, ck_tile::pk_fp6_t, T>;

// Convert a single host-tensor element of any supported A/B data type to float
// at unpacked-lane index `lane`. For ck_tile::pk_fp6x16_t (legacy fp6) we cannot
// bit_cast to ck_tile::pk_fp6_t because their sizes differ (12 vs 16 bytes: the
// modern pk_fp6_t uses uint32x3_t which clang aligns to 16 bytes, while the
// legacy pk_fp6x16_t stores int32_t[3] = 12 bytes). The on-disk packed layout
// (3 x 32-bit words holding 16 x fp6 values) is the same, so we unpack directly
// from the legacy storage and convert via pk_fp6_legacy_t::fp6_e2m3_to_float.
template <typename T>
inline float mx_flatmm_lane_to_float(const T& v, ck_tile::index_t lane)
{
    ck_tile::index_t lane_index = lane % ck_tile::numeric_traits<T>::PackedSize;

    if constexpr(std::is_same_v<T, ck_tile::pk_fp4_t>)
    {
        return v.to_fp32x2()[lane_index];
    }
    else if constexpr(std::is_same_v<T, ck_tile::pk_fp6_t> || std::is_same_v<T, ck_tile::pk_bf6_t>)
    {
        // Note: pk_fp6_t / pk_bf6_t expose to_fp32x16() only when CK_TILE_AVX512F_WA
        // is undefined. Use the always-available ck_tile::convert_to_float<T>(raw_bits)
        // free function (defined in mxfp_convert.hpp) plus get_element(idx) so this
        // compiles on hosts without AVX-512F too.
        return ck_tile::convert_to_float<T>(v.get_element(lane_index));
    }
    else if constexpr(std::is_same_v<T, ck_tile::pk_fp6x16_t>)
    {
        return ck_tile::pk_fp6x16_t::fp6_e2m3_to_float(v.unpack(lane_index));
    }
    else
    {
        return ck_tile::type_convert<float>(v);
    }
}

template <typename PrecActType,
          typename PrecWeightType,
          typename CDataType,
          typename MXFlatmmArchTraits,
          bool UsePersistentKernel = false,
          typename ALayout,
          typename BLayout,
          typename CLayout>
int run_mx_flatmm_with_layouts(const ck_tile::ArgParser& arg_parser,
                               const ALayout a_layout = ALayout{},
                               const BLayout b_layout = BLayout{},
                               const CLayout c_layout = CLayout{})
{
    using ADataType   = PrecActType;
    using BDataType   = PrecWeightType;
    using AccDataType = float;

    // Init-side proxy types: collapse to ADataType / BDataType for everything
    // except ck_tile::pk_fp6x16_t (legacy fp6), which is initialized through a
    // byte-equivalent ck_tile::pk_fp6_t buffer and memcpy'd into the real tensor.
    using AInitType = mx_flatmm_init_proxy_t<ADataType>;
    using BInitType = mx_flatmm_init_proxy_t<BDataType>;

    // The proxy and the real type may have different sizeof (e.g. pk_fp6_t is
    // 16 bytes due to uint32x3_t alignment, while pk_fp6x16_t is 12 bytes). We
    // therefore only require they share PackedSize and copy element-wise (see
    // the proxy->real transfer below). For all other A/B types the proxy is
    // identical to the real type.
    static_assert(ck_tile::numeric_traits<AInitType>::PackedSize ==
                      ck_tile::numeric_traits<ADataType>::PackedSize,
                  "init proxy must share PackedSize with ADataType");
    static_assert(ck_tile::numeric_traits<BInitType>::PackedSize ==
                      ck_tile::numeric_traits<BDataType>::PackedSize,
                  "init proxy must share PackedSize with BDataType");

    using ScaleType = ck_tile::e8m0_t;

    constexpr int ScaleGranularityM = 1;
    constexpr int ScaleGranularityN = 1;
    constexpr int ScaleGranularityK = 32;

    const bool verbose = arg_parser.get_int("verbose") == 1;

    ck_tile::index_t M = arg_parser.get_int("m");
    ck_tile::index_t N = arg_parser.get_int("n");
    ck_tile::index_t K = arg_parser.get_int("k");

    ck_tile::index_t stride_A = arg_parser.get_int("stride_a");
    ck_tile::index_t stride_B = arg_parser.get_int("stride_b");
    ck_tile::index_t stride_C = arg_parser.get_int("stride_c");

    ck_tile::index_t init_method = arg_parser.get_int("init");
    ck_tile::index_t n_warmup    = arg_parser.get_int("warmup");
    ck_tile::index_t n_repeat    = arg_parser.get_int("repeat");

    stride_A = ck_tile::get_default_stride(M, K, stride_A, is_row_major(a_layout));
    stride_B = ck_tile::get_default_stride(K, N, stride_B, is_row_major(b_layout));
    stride_C = ck_tile::get_default_stride(M, N, stride_C, is_row_major(c_layout));

    auto scale_stride_A = ck_tile::get_default_stride(
        M / ScaleGranularityM, K / ScaleGranularityK, 0, is_row_major(a_layout));
    auto scale_stride_B = ck_tile::get_default_stride(
        K / ScaleGranularityK, N / ScaleGranularityN, 0, is_row_major(b_layout));

    if(K % ScaleGranularityK != 0)
        throw std::runtime_error("wrong! K must be multiple of ScaleGranularityK.");
    if(K % ck_tile::numeric_traits<ADataType>::PackedSize != 0 ||
       K % ck_tile::numeric_traits<BDataType>::PackedSize != 0)
        throw std::runtime_error("wrong! K must be multiple of packed size.");

    ck_tile::HostTensor<ADataType> a_host(
        ck_tile::host_tensor_descriptor(M, K, stride_A, is_row_major(a_layout)));
    ck_tile::HostTensor<BDataType> b_origin_host(
        ck_tile::host_tensor_descriptor(K, N, stride_B, is_row_major(b_layout)));
    ck_tile::HostTensor<CDataType> c_rslt_host(
        ck_tile::host_tensor_descriptor(M, N, stride_C, is_row_major(CLayout{})));

    // Proxy tensors used for initialization. When the init proxy collapses to
    // the real type (i.e. ADataType / BDataType is not pk_fp6x16_t) these are
    // direct stand-ins; otherwise they are byte-equivalent pk_fp6_t buffers
    // that are memcpy'd into a_host / b_origin_host once init is done.
    ck_tile::HostTensor<AInitType> a_init_host(
        ck_tile::host_tensor_descriptor(M, K, stride_A, is_row_major(a_layout)));
    ck_tile::HostTensor<BInitType> b_init_host(
        ck_tile::host_tensor_descriptor(K, N, stride_B, is_row_major(b_layout)));

    ck_tile::HostTensor<ScaleType> scale_a(ck_tile::host_tensor_descriptor(
        M / ScaleGranularityM, K / ScaleGranularityK, scale_stride_A, is_row_major(a_layout)));
    ck_tile::HostTensor<ScaleType> scale_b(ck_tile::host_tensor_descriptor(
        K / ScaleGranularityK, N / ScaleGranularityN, scale_stride_B, is_row_major(b_layout)));

    if(verbose)
    {
        std::cout << "a_host: " << a_host.mDesc << std::endl;
        std::cout << "b_origin_host: " << b_origin_host.mDesc << std::endl;
        std::cout << "c_rslt_host: " << c_rslt_host.mDesc << std::endl;
        std::cout << "scale_a: " << scale_a.mDesc << std::endl;
        std::cout << "scale_b: " << scale_b.mDesc << std::endl;
    }

    if(init_method == 0)
    {
        // Random tensor and scale values
        ck_tile::FillUniformDistribution<>{0.0f, 1.0f}(a_init_host);
        ck_tile::FillUniformDistribution<>{-.5f, .5f}(b_init_host);
        ck_tile::FillUniformDistribution<>{-2.f, 2.f}(scale_a);
        ck_tile::FillUniformDistribution<>{-2.f, 2.f}(scale_b);
    }
    else if(init_method == 1)
    {
        // Constant tensor and scale values
        ck_tile::FillUniformDistribution<>{2.f, 2.f}(a_init_host);
        ck_tile::FillUniformDistribution<>{0.5f, 0.5f}(b_init_host);
        ck_tile::FillUniformDistribution<>{0.5f, 0.5f}(scale_a);
        ck_tile::FillUniformDistribution<>{2.f, 2.f}(scale_b);
    }
    else if(init_method == 13)
    {
        constexpr ck_tile::index_t PackedSizeA = ck_tile::numeric_traits<AInitType>::PackedSize;

        ck_tile::FillUniformDistribution<>{0.f, 0.f}(a_init_host);

        const auto a_row = 0;
#if 1
        for(ck_tile::index_t j = 0; j < K; j += PackedSizeA)
        {
            a_init_host(a_row, j) = (j % 3 == 0)   ? float_to_data_element<AInitType>(1.0f)
                                    : (j % 3 == 1) ? float_to_data_element<AInitType>(0.50f)
                                                   : float_to_data_element<AInitType>(-1.0f);
        }
#else
        for(ck_tile::index_t j = 0; j < K; j += PackedSizeA)
        {
            a_init_host(a_row, j) = float_to_data_element<AInitType>(1.0f);
        }
#endif

        std::cout << "Submatrix of a_init_host (16x32):" << std::endl;
        for(int i = 0; i < 16; ++i)
        {
#if 0
            for(int j = 0; j < 32; ++j)
            {
                float v_a = mx_flatmm_lane_to_float(a_init_host(i, j), j);
                std::cout << std::setw(11) << v_a;
            }
            std::cout << std::endl;
#endif
        }

        ck_tile::FillUniformDistribution<>{1.f, 1.f}(scale_a);
        ck_tile::FillUniformDistribution<>{0.f, 0.f}(b_init_host);
        ck_tile::FillUniformDistribution<>{1.f, 1.f}(scale_b);

#if 0
        std::set<int> col_ids = {0, 3, 32, 64, 69};
#else
        std::set<int> col_ids = {0,   10,  25,  42,  50,  70,   80,   90,   103, 120,
                                 150, 170, 180, 190, 200, 230,  245,  256,  300, 350,
                                 400, 450, 500, 550, 600, 1111, 2222, 3333, 4000};
#endif
        for(auto col_id : col_ids)
        {
            if(col_id >= N)
                continue;

#if 1
            scale_b(0, col_id)     = ck_tile::type_convert<ScaleType>(8.0f);
            scale_b(0, col_id + 1) = ck_tile::type_convert<ScaleType>(16.0f);
            scale_b(1, col_id)     = ck_tile::type_convert<ScaleType>(4.0f / 1);
            scale_b(3, col_id)     = ck_tile::type_convert<ScaleType>(2.0f / 1);
            scale_b(5, col_id)     = ck_tile::type_convert<ScaleType>(1.0f / 2);
            scale_b(10, col_id)    = ck_tile::type_convert<ScaleType>(1.0f / 4);
            scale_b(13, col_id)    = ck_tile::type_convert<ScaleType>(1.0f / 8);
            scale_b(K / ScaleGranularityK - 1, col_id) =
                ck_tile::type_convert<ScaleType>(1.0f / 64);
#elif 0
            scale_b(0, col_id)                         = ck_tile::type_convert<ScaleType>(4.0f);
            scale_b(0, col_id + 1)                     = ck_tile::type_convert<ScaleType>(8.0f);
            scale_b(1, col_id)                         = ck_tile::type_convert<ScaleType>(2.0f / 1);
            scale_b(5, col_id)                         = ck_tile::type_convert<ScaleType>(1.0f / 2);
            scale_b(10, col_id)                        = ck_tile::type_convert<ScaleType>(1.0f / 4);
            scale_b(13, col_id)                        = ck_tile::type_convert<ScaleType>(1.0f / 8);
            scale_b(K / ScaleGranularityK - 1, col_id) = ck_tile::type_convert<ScaleType>(1.0f / 4);
#endif

            constexpr ck_tile::index_t PackedSizeB = ck_tile::numeric_traits<BInitType>::PackedSize;
            if constexpr(PackedSizeB == 1)
            {
                b_init_host(K - 1, col_id) = float_to_data_element<BInitType>(-1.0f);

                for(int i = 0; i < K; i += 7)
                {
                    auto coeff = ((i / 7) % 2 == 0) ? 1.0f : -1.0f;
                    b_init_host(i, col_id) =
                        float_to_data_element<BInitType>(coeff / 10.0f * (i % 512));
                }
            }
            else if constexpr(PackedSizeB == 2)
            {
                b_init_host(K - 1, col_id) = BInitType::pack(0, BInitType(-1.0f));

                for(int i = 0; i < K; i += 7)
                {
                    auto coeff             = ((i / 7) % 2 == 0) ? 1.0f : -1.0f;
                    b_init_host(i, col_id) = BInitType::pack(BInitType(coeff / 2.0f * (i % 13)), 0);
                }
            }
            else if constexpr(PackedSizeB == 16)
            {
                // BInitType is pk_fp6_t (or pk_bf6_t); both expose set_element(lane,
                // raw_byte). Mirror the PackedSize == 2 spike pattern, writing into
                // lane 0 of each packed entity.
                auto float_to_pk6_bits = [](float v) -> uint8_t {
                    if constexpr(std::is_same_v<BInitType, ck_tile::pk_bf6_t>)
                        return ck_tile::float_to_bf6(v, 1.f);
                    else
                        return ck_tile::float_to_fp6(v, 1.f);
                };

                b_init_host(K - 1, col_id).set_element(15, float_to_pk6_bits(-1.0f));

                for(int i = 0; i < K; i += 7)
                {
                    auto coeff = ((i / 7) % 2 == 0) ? 1.0f : -1.0f;
                    b_init_host(i, col_id).set_element(i % PackedSizeB,
                                                       float_to_pk6_bits(coeff / 2.0f * (i % 16)));
                }
            }
        }
    }
    else if(init_method == 14)
    {
        constexpr ck_tile::index_t PackedSizeB = ck_tile::numeric_traits<BInitType>::PackedSize;

        ck_tile::FillUniformDistribution<>{0.f, 0.f}(a_init_host);
        ck_tile::FillUniformDistribution<>{1.f, 1.f}(scale_a);
        ck_tile::FillUniformDistribution<>{0.f, 0.f}(b_init_host);
        ck_tile::FillUniformDistribution<>{1.f, 1.f}(scale_b);

        const auto b_col = 0;
#if 1
        for(ck_tile::index_t j = 0; j < K; j += PackedSizeB)
        {
            b_init_host(j, b_col) = (j % 3 == 0)   ? float_to_data_element<BInitType>(1.0f)
                                    : (j % 3 == 1) ? float_to_data_element<BInitType>(0.50f)
                                                   : float_to_data_element<BInitType>(-1.0f);
        }
#else
        for(ck_tile::index_t j = 0; j < K; j += PackedSizeB)
        {
            b_init_host(j, b_col) = float_to_data_element<BInitType>(1.0f);
        }
#endif

#if 0
        std::set<int> row_ids = {0};
#else
        std::set<int> row_ids = {0,   10,  25,  42,  50,  70,   80,   90,   103, 120,
                                 150, 170, 180, 190, 200, 230,  245,  256,  300, 350,
                                 400, 450, 500, 550, 600, 1111, 2222, 3333, 4000};
#endif
        for(auto row_id : row_ids)
        {
            if(row_id >= M)
                continue;

#if 0
            scale_a(row_id, 0)     = ck_tile::type_convert<ScaleType>(8.0f);
            scale_a(row_id + 1, 0) = ck_tile::type_convert<ScaleType>(16.0f);
            scale_a(row_id, 1)     = ck_tile::type_convert<ScaleType>(4.0f / 1);
            scale_a(row_id, 3)     = ck_tile::type_convert<ScaleType>(2.0f / 1);
            scale_a(row_id, 5)     = ck_tile::type_convert<ScaleType>(1.0f / 2);
            scale_a(row_id, 10)    = ck_tile::type_convert<ScaleType>(1.0f / 4);
            scale_a(row_id, 13)    = ck_tile::type_convert<ScaleType>(1.0f / 8);
            scale_a(row_id, K / ScaleGranularityK - 1) =
                ck_tile::type_convert<ScaleType>(1.0f / 64);
#elif 1
            scale_a(row_id, 0)     = ck_tile::type_convert<ScaleType>(0.5f);
            scale_a(row_id + 1, 0) = ck_tile::type_convert<ScaleType>(0.25f);
            scale_a(row_id, 1)     = ck_tile::type_convert<ScaleType>(2.0f / 1);
            scale_a(row_id, 5)     = ck_tile::type_convert<ScaleType>(1.0f / 2);
            scale_a(row_id, 10)    = ck_tile::type_convert<ScaleType>(1.0f / 4);
            scale_a(row_id, 13)    = ck_tile::type_convert<ScaleType>(1.0f / 8);
            if(K / ScaleGranularityK >= 24)
            {
                scale_a(row_id, 15)     = ck_tile::type_convert<ScaleType>(1.0f / 64);
                scale_a(row_id, 16)     = ck_tile::type_convert<ScaleType>(0.5f);
                scale_a(row_id + 1, 17) = ck_tile::type_convert<ScaleType>(0.25f);
                scale_a(row_id, 20)     = ck_tile::type_convert<ScaleType>(2.0f / 1);
                scale_a(row_id, 22)     = ck_tile::type_convert<ScaleType>(1.0f / 2);
            }
            scale_a(row_id, K / ScaleGranularityK - 1) = ck_tile::type_convert<ScaleType>(1.0f / 4);
#endif
            constexpr ck_tile::index_t PackedSizeA = ck_tile::numeric_traits<AInitType>::PackedSize;
            constexpr ck_tile::index_t stop_at     = 5120;
            if constexpr(PackedSizeA == 1)
            {
                if(K <= stop_at)
                {
                    a_init_host(row_id, K - 1) = float_to_data_element<AInitType>(-1.0f);
                }

                for(int i = 0; i < (K <= stop_at ? K : stop_at); i += 7)
                {
                    auto coeff = ((i / 7) % 2 == 0) ? 1.0f : -1.0f;
                    a_init_host(row_id, i) =
                        float_to_data_element<AInitType>(coeff / 10.0f * (i % 512));
                }
            }
            else if constexpr(PackedSizeA == 2)
            {
                a_init_host(row_id, K - 1) = AInitType::pack(0, AInitType(-1.0f));

                for(int i = 0; i < K; i += 7)
                {
                    auto coeff             = ((i / 7) % 2 == 0) ? 1.0f : -1.0f;
                    a_init_host(row_id, i) = AInitType::pack(AInitType(coeff / 2.0f * (i % 13)), 0);
                }
            }
            else if constexpr(PackedSizeA == 16)
            {
                // AInitType is pk_fp6_t (or pk_bf6_t); both expose set_element(lane,
                // raw_byte). Mirror the PackedSize == 2 spike pattern, writing into
                // lane 0 of each packed entity.
                auto float_to_pk6_bits = [](float v) -> uint8_t {
                    if constexpr(std::is_same_v<AInitType, ck_tile::pk_bf6_t>)
                        return ck_tile::float_to_bf6(v, 1.f);
                    else
                        return ck_tile::float_to_fp6(v, 1.f);
                };

                a_init_host(row_id, K - 1).set_element(15, float_to_pk6_bits(-1.0f));

                for(int i = 0; i < K; i += 7)
                {
                    auto coeff = ((i / 7) % 2 == 0) ? 1.0f : -1.0f;
                    a_init_host(row_id, i).set_element(i % PackedSizeA,
                                                       float_to_pk6_bits(coeff / 2.0f * (i % 16)));
                }
            }
        }
    }
    else
    {
        throw std::runtime_error("wrong! Unexpected init_method");
    }

    // Transfer the proxy-initialized data into the real host tensors. When the
    // proxy collapsed to the real type (A/BInitType == A/BDataType) this is a
    // same-type move. Otherwise the proxy is pk_fp6_t (16 bytes incl. vector
    // padding) and the real type is pk_fp6x16_t (12 bytes); both store the same
    // packed layout (3 x 32-bit words holding 16 x fp6 values) so we copy the
    // first sizeof(real-type) bytes per element, dropping the trailing pad.
    auto copy_proxy_to_real = [](auto& dst, const auto& src) {
        using DstT = typename std::remove_reference_t<decltype(dst)>::Data::value_type;
        using SrcT = typename std::remove_reference_t<decltype(src)>::Data::value_type;

        const std::size_t n_elements = src.get_element_space_size();
        auto* dst_bytes              = reinterpret_cast<char*>(dst.data());
        const auto* src_bytes        = reinterpret_cast<const char*>(src.data());

        for(std::size_t i = 0; i < n_elements; ++i)
        {
            std::memcpy(dst_bytes + i * sizeof(DstT), src_bytes + i * sizeof(SrcT), sizeof(DstT));
        }
    };

    if constexpr(std::is_same_v<AInitType, ADataType>)
    {
        a_host = std::move(a_init_host);
    }
    else
    {
        static_assert(std::is_same_v<ADataType, ck_tile::pk_fp6x16_t>,
                      "init proxy mismatch is only expected for pk_fp6x16_t");
        copy_proxy_to_real(a_host, a_init_host);
    }
    if constexpr(std::is_same_v<BInitType, BDataType>)
    {
        b_origin_host = std::move(b_init_host);
    }
    else
    {
        static_assert(std::is_same_v<BDataType, ck_tile::pk_fp6x16_t>,
                      "init proxy mismatch is only expected for pk_fp6x16_t");
        copy_proxy_to_real(b_origin_host, b_init_host);
    }

    if(verbose)
        std::cout << "B pre-shuffle..." << std::endl;
    const auto b_shuffled_host = preShuffleWeight<MXFlatmmArchTraits::GetNLane()>(b_origin_host);
    if(verbose)
        std::cout << "B pre-shuffle done" << std::endl;
    if(verbose)
        std::cout << "A scale pre-shuffle..." << std::endl;
    const auto scale_a_shuffled = MXFlatmmArchTraits::template preShuffleScale<true>(scale_a);
    if(verbose)
        std::cout << "A scale pre-shuffle done" << std::endl;
    if(verbose)
        std::cout << "B scale pre-shuffle..." << std::endl;
    const auto scale_b_shuffled = MXFlatmmArchTraits::template preShuffleScale<false>(scale_b);
    if(verbose)
        std::cout << "B scale pre-shuffle done\n\n" << std::endl;

    if(verbose)
    {
        std::cout << "b_shuffled_host: " << b_shuffled_host.mDesc << std::endl;
        std::cout << "scale_a_shuffled: " << scale_a_shuffled.mDesc << std::endl;
        std::cout << "scale_b_shuffled: " << scale_b_shuffled.mDesc << std::endl;
    }

    if(verbose)
        std::cout << "\n\nDevice memory allocation...\n" << std::endl;
    ck_tile::DeviceMem a_dev_buf(a_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem b_shuffled_dev_buf(b_shuffled_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem c_dev_buf(c_rslt_host.get_element_space_size_in_bytes());

    ck_tile::DeviceMem scale_a_dev_buf(scale_a_shuffled.get_element_space_size_in_bytes());
    ck_tile::DeviceMem scale_b_dev_buf(scale_b_shuffled.get_element_space_size_in_bytes());

    if(verbose)
        std::cout << "\n\nUpload data to device...\n" << std::endl;
    a_dev_buf.ToDevice(a_host.data());
    b_shuffled_dev_buf.ToDevice(b_shuffled_host.data());
    c_rslt_host.SetZero();
    scale_a_dev_buf.ToDevice(scale_a_shuffled.data());
    scale_b_dev_buf.ToDevice(scale_b_shuffled.data());

    if(verbose)
        std::cout << "\n\nUpload data to device... Done.\n" << std::endl;

    auto scale_a_dev_ptr =
        ck_tile::FlatmmScalePointer<ScaleGranularityM, ScaleGranularityK, ScaleType>{
            static_cast<ScaleType*>(scale_a_dev_buf.GetDeviceBuffer()), M / ScaleGranularityM};
    auto scale_b_dev_ptr =
        ck_tile::FlatmmScalePointer<ScaleGranularityN, ScaleGranularityK, ScaleType>{
            static_cast<ScaleType*>(scale_b_dev_buf.GetDeviceBuffer()), N / ScaleGranularityN};

    if(verbose)
    {
        std::cout << "Computing GEMM on device..." << std::endl << std::endl;
    }

    invoke_mx_flatmm<MXFlatmmArchTraits,
                     ADataType,
                     BDataType,
                     ck_tile::tuple<>,
                     AccDataType,
                     CDataType,
                     ALayout,
                     BLayout,
                     ck_tile::tuple<>,
                     CLayout,
                     decltype(scale_a_dev_ptr),
                     decltype(scale_b_dev_ptr),
                     UsePersistentKernel>(a_dev_buf,
                                          b_shuffled_dev_buf,
                                          c_dev_buf,
                                          M,
                                          N,
                                          K,
                                          stride_A,
                                          stride_B,
                                          stride_C,
                                          scale_a_dev_ptr,
                                          scale_b_dev_ptr,
                                          n_warmup,
                                          n_repeat);

    c_dev_buf.FromDevice(c_rslt_host.data());
    if(verbose)
    {
        std::cout << "Computing GEMM on device... Done.\n" << std::endl;
    }

    bool pass = true; // validation of the results
    if(arg_parser.get_int("v") == 1)
    {
        ck_tile::HostTensor<CDataType> c_m_n_host_ref(
            ck_tile::host_tensor_descriptor(M, N, stride_C, is_row_major(CLayout{})));
        c_m_n_host_ref.SetZero();

        if(verbose)
        {
            std::cout << "Computing GEMM on host..." << std::endl;
        }
        ck_tile::
            reference_mx_gemm<ADataType, BDataType, ScaleType, ScaleType, AccDataType, CDataType>(
                a_host, b_origin_host, c_m_n_host_ref, scale_a, scale_b);

        if(verbose)
        {
            std::cout << "Computing GEMM on host... Done.\n" << std::endl;
            std::cout << "Comparing results..." << std::endl;
        }

        constexpr ck_tile::index_t PackedSizeA = ck_tile::numeric_traits<ADataType>::PackedSize;
        constexpr ck_tile::index_t PackedSizeB = ck_tile::numeric_traits<BDataType>::PackedSize;

        if(init_method == 1) // constant
        {
#if 1
#if 1
            std::cout << "Submatrix of a_host (16x32):" << std::endl;
            for(int i = 0; i < 16; ++i)
            {
                for(int j = 0; j < 32; ++j)
                {
                    float v_a = mx_flatmm_lane_to_float(a_host(i, j), j);
                    std::cout << std::setw(11) << v_a;
                }
                std::cout << std::endl;
            }
#endif

            if(K < 1000)
            {
                std::cout << std::endl;
                const auto row = 0;

                std::cout << "a_host(" << row << ",:):";
                for(int j = 0; j < K; ++j)
                {
                    if(j % (std::max(16, PackedSizeA)) == 0)
                    {
                        std::cout << std::endl << "k = " << j << " : ";
                    }

                    float v_a = mx_flatmm_lane_to_float(a_host(row, j), j);
                    std::cout << std::setw(11) << v_a;
                }
                std::cout << std::endl;
            }

#if 1
            std::cout << "Submatrix of b_origin_host (32x16):" << std::endl;
            for(int k = 0; k < 32; k++)
            {
                for(int n = 0; n < 16; n++)
                {
                    float v_b = mx_flatmm_lane_to_float(b_origin_host(k, n), k);
                    std::cout << std::setw(12) << v_b;
                }
                std::cout << std::endl;
            }
#endif

            if(K < 1000)
            {
                std::cout << std::endl;
                const auto col = 0;

                std::cout << "b_origin_host(:, " << col << "):";
                for(int i = 0; i < K; ++i)
                {
                    if(i % (std::max(16, PackedSizeB)) == 0)
                    {
                        std::cout << std::endl << "k = " << i << " : ";
                    }

                    float v_b = mx_flatmm_lane_to_float(b_origin_host(i, col), i);
                    std::cout << std::setw(12) << v_b;
                }
                std::cout << std::endl;
            }

#if 1
            std::cout << "Submatrix of b_shuffled_host (32x16):" << std::endl;
            // From preShuffleWeight
            constexpr ck_tile::index_t KPack = 16 * PackedSizeB;
            constexpr ck_tile::index_t NLane = MXFlatmmArchTraits::GetNLane();

            constexpr ck_tile::index_t num_blocks = 2; // print this number of MFMA/WMMA blocks

            for(ck_tile::index_t block_id = 0; block_id < num_blocks; ++block_id)
            {
                for(ck_tile::index_t k = 0; k < KPack; k += 1)
                {
                    for(ck_tile::index_t n = 0; n < NLane; ++n)
                    {
                        float v_b = mx_flatmm_lane_to_float(
                            b_shuffled_host(block_id * KPack * NLane + n * KPack + k), k);
                        std::cout << std::setw(12) << v_b;
                    }
                    std::cout << std::endl;
                }
                std::cout << std::endl;
            }
#endif

#if 1
            std::cout << "Submatrix of scale_a (16x16):" << std::endl;
            for(int i = 0; i < 16 && i < M; ++i)
            {
                for(int j = 0; j < 16 && j < K / ScaleGranularityK; ++j)
                {
                    std::cout << std::setw(11) << type_convert<float>(scale_a(i, j));
                }

                std::cout << std::endl;
            }

            std::cout << "Submatrix of scale_a_shuffled (16x16):" << std::endl;
            for(int i = 0; i < 64 && i < M * K / ScaleGranularityK / 4; ++i)
            {
                if(i > 0 && i % 16 == 0)
                    std::cout << std::endl;

                for(int j = 0; j < 4; ++j)
                {
                    std::cout << std::setw(11) << type_convert<float>(scale_a_shuffled(i * 4 + j));
                }

                std::cout << std::endl;
            }
#endif

#if 1
            std::cout << "Submatrix of scale_b (16x16):" << std::endl;
            for(int i = 0; i < 16 && i < K / ScaleGranularityK; ++i)
            {
                for(int j = 0; j < 16 && j < N; ++j)
                {
                    std::cout << std::setw(11) << type_convert<float>(scale_b(i, j));
                }

                std::cout << std::endl;
            }

            std::cout << "Submatrix of scale_b_shuffled (16x16):" << std::endl;
            for(int i = 0; i < 64 && i < N * K / ScaleGranularityK / 4; ++i)
            {
                if(i > 0 && i % 16 == 0)
                    std::cout << std::endl;

                for(int j = 0; j < 4; ++j)
                {
                    std::cout << std::setw(11) << type_convert<float>(scale_b_shuffled(i * 4 + j));
                }

                std::cout << std::endl;
            }

#endif

#if 1 // c_rslt_host (16x16)
            std::cout << "Submatrix of c_rslt_host (32x16):" << std::endl;
            for(int i = 0; i < 32; ++i)
            {
                for(int j = 0; j < 16; ++j)
                {
                    std::cout << std::setw(12) << ck_tile::type_convert<float>(c_rslt_host(i, j));
                }

                std::cout << std::endl;
            }
#endif
#endif

            auto expected = static_cast<float>(K);
            auto computed = ck_tile::type_convert<float>(c_rslt_host(3, 7));

            pass = pass && std::abs(expected - computed) <= 0.0f;
            std::cout << "\nExpected vs Computed: " << expected << " vs " << computed
                      << ((pass) ? " (PASSED!)" : " (FAILED!)") << std::endl
                      << std::endl;
        }
        else // random
        {
#if 1
            std::cout << "Submatrix of a_host (16x32):" << std::endl;
            for(int i = 0; i < 16; ++i)
            {
#if 1
                for(int j = 0; j < 32; ++j)
                {
                    float v_a = mx_flatmm_lane_to_float(a_host(i, j), j);
                    std::cout << std::setw(11) << v_a;
                }
                std::cout << std::endl;
#endif
            }

            if(K < 1000)
            {
                std::cout << std::endl;
                const auto row = 0;

                std::cout << "a_host(" << row << ",:):";
                for(int j = 0; j < K; ++j)
                {
                    if(j % (std::max(16, PackedSizeA)) == 0)
                    {
                        std::cout << std::endl << "k = " << j << " : ";
                    }

                    float v_a = mx_flatmm_lane_to_float(a_host(row, j), j);
                    std::cout << std::setw(11) << v_a;
                }
                std::cout << std::endl;
            }

#if 0
            std::cout << "Submatrix of b_origin_host (32x16):" << std::endl;
            for(int k = 0; k < 32; k++)
            {
                for(int n = 0; n < 16; n++)
                {
                    float v_b = mx_flatmm_lane_to_float(b_origin_host(k, n), k);
                    std::cout << std::setw(12) << v_b;
                }
                std::cout << std::endl;
            }
#endif

            if(K < 1000)
            {
#if 1
                std::cout << std::endl;
                const auto col = 0;

                std::cout << "b_origin_host(:, " << col << "):";
                for(int i = 0; i < K; ++i)
                {
                    if(i % (std::max(16, PackedSizeB)) == 0)
                    {
                        std::cout << std::endl << "k = " << i << " : ";
                    }

                    float v_b = mx_flatmm_lane_to_float(b_origin_host(i, col), i);
                    std::cout << std::setw(12) << v_b;
                }
                std::cout << std::endl;
#endif
            }

#if 0
            std::cout << "Submatrix of b_shuffled_host (32x16):" << std::endl;
            // From preShuffleWeight
            constexpr ck_tile::index_t KPack = 16 * PackedSizeB;
            constexpr ck_tile::index_t NLane = MXFlatmmArchTraits::GetNLane();

            constexpr ck_tile::index_t num_blocks = 2; // print this number of MFMA/WMMA blocks

            const auto b_shuffled_host_col = 0;
            for(ck_tile::index_t block_id = 0; block_id < num_blocks; ++block_id)
            {
                for(ck_tile::index_t k = 0; k < KPack; k += 1)
                {
                    for(ck_tile::index_t n = b_shuffled_host_col;
                        n < (NLane + b_shuffled_host_col) && n < N;
                        ++n)
                    {
                        float v_b = mx_flatmm_lane_to_float(
                            b_shuffled_host(block_id * KPack * NLane + n * KPack + k), k);
                        std::cout << std::setw(12) << v_b;
                    }
                    std::cout << std::endl;
                }
                std::cout << std::endl;
            }
#endif

#if 0
            std::cout << "Submatrix of scale_a (18x16):" << std::endl;
            const auto scale_a_row = 0;
            for(int i = scale_a_row; i < (18 + scale_a_row) && i < M; ++i)
            {
                for(int j = 0; j < 32 && j < K / ScaleGranularityK; ++j)
                {
                    std::cout << std::setw(11) << type_convert<float>(scale_a(i, j));
                }

                std::cout << std::endl;
            }

            std::cout << "Submatrix of scale_a_shuffled:" << std::endl;
            auto scale_a_do_print = false;
            for(int i = 0; i < M * K / ScaleGranularityK / 4; ++i)
            {
                if(i % 16 == 0)
                {
                    auto xdl_idx      = i / 16;
                    auto nXdlK        = K / 128; // number of xdl in k direction
                    auto MXdlPack     = 2;
                    auto mXdlKPack    = MXdlPack * nXdlK; // number of xdl in a pack
                    auto pack_row_idx = xdl_idx / mXdlKPack;
                    auto loc_xdl_idx  = xdl_idx % mXdlKPack;
                    // upper left corner of xdl data in tensor A
                    auto k = 128 * (loc_xdl_idx / MXdlPack);
                    auto m = pack_row_idx * MXdlPack * 16 + 16 * (loc_xdl_idx % MXdlPack);

                    if(i < 12 * 16)
                        scale_a_do_print = true;
                    else
                        scale_a_do_print = false;

                    if(scale_a_do_print)
                    {
                        auto offset_a       = m * K + k; // offset in tensor A
                        auto offset_scale   = i * 4;
                        auto offset_scale_4 = i;
                        std::cout << "idx = " << i / 16 << "; m = " << m << "; k = " << k
                                  << "; offset_a = " << offset_a
                                  << "; offset_scale = " << offset_scale
                                  << "; offset_scale_4 = " << offset_scale_4 << std::endl;
                    }
                }

                if(scale_a_do_print)
                {
                    for(int j = 0; j < 4; ++j)
                    {
                        std::cout << std::setw(11)
                                  << type_convert<float>(scale_a_shuffled(i * 4 + j));
                    }
                    std::cout << std::endl;
                }
            }
#endif

#if 0
            std::cout << "Submatrix of scale_b (16x18):" << std::endl;
            const auto scale_b_col = 0;
            for(int i = 0; i < 16 && i < K / ScaleGranularityK; ++i)
            {
                for(int j = scale_b_col; j < (18 + scale_b_col) && j < N; ++j)
                {
                    std::cout << std::setw(11) << type_convert<float>(scale_b(i, j));
                }

                std::cout << std::endl;
            }

            std::cout << "Submatrix of scale_b_shuffled:" << std::endl;
            auto do_print = false;
            for(int i = 0; i < N * K / ScaleGranularityK / 4; ++i)
            {
                if(i % 16 == 0)
                {
                    auto xdl_idx      = i / 16;
                    auto nXdlK        = K / 128; // number of xdl in k direction
                    auto NXdlPack     = 2;
                    auto nXdlKPack    = NXdlPack * nXdlK; // number of xdl in a pack
                    auto pack_col_idx = xdl_idx / nXdlKPack;
                    auto loc_xdl_idx  = xdl_idx % nXdlKPack;
                    // upper left corner of xdl data in tensor B
                    auto k = 128 * (loc_xdl_idx / NXdlPack);
                    auto n = pack_col_idx * NXdlPack * 16 + 16 * (loc_xdl_idx % NXdlPack);

                    if(n == 0)
                        do_print = true;
                    else
                        do_print = false;

                    if(do_print)
                    {
                        auto offset_b       = n * K + k; // offset in tensor B
                        auto offset_scale   = i * 4;
                        auto offset_scale_4 = i;
                        std::cout << "k = " << k << "; n = " << n << "; offset_b = " << offset_b
                                  << "; offset_scale = " << offset_scale
                                  << "; offset_scale_4 = " << offset_scale_4 << std::endl;
                    }
                }
                if(do_print)
                {
                    for(int j = 0; j < 4; ++j)
                    {
                        std::cout << std::setw(11)
                                  << type_convert<float>(scale_b_shuffled(i * 4 + j));
                    }
                    std::cout << std::endl;
                }
            }

#endif

#if 1 // c_rslt_host (16x16)
            std::cout << "Submatrix of c_rslt_host (32x16):" << std::endl;
            const auto c_rslt_host_row = 0;
            const auto c_rslt_host_col = 0;
            for(int i = c_rslt_host_row; i < (32 + c_rslt_host_row) && i < M; ++i)
            {
                for(int j = c_rslt_host_col; j < (17 + c_rslt_host_col) && j < N; ++j)
                {
                    std::cout << std::setw(12) << ck_tile::type_convert<float>(c_rslt_host(i, j));
                }

                std::cout << std::endl;
            }
#endif

#if 0 // Difference between c_rslt_host and c_m_n_host_ref
            std::cout << "C Ref vs C Device (32x16):" << std::endl;
            const auto c_diff_row = 0;
            const auto c_diff_col = 0;
            for(int i = c_diff_row; i < (32 + c_diff_row) && i < M; ++i)
            {
                for(int j = c_diff_col; j < (17 + c_diff_col) && j < N; ++j)
                {
                    std::cout << std::setw(5)
                              << ck_tile::type_convert<float>(c_rslt_host(i, j)) -
                                     ck_tile::type_convert<float>(c_m_n_host_ref(i, j));
                }

                std::cout << std::endl;
            }
#endif

#endif
        }

        const float rtol = std::is_same_v<ADataType, ck_tile::half_t> ? 1e-3 : 1e-2;
        const float atol = std::is_same_v<ADataType, ck_tile::half_t> ? 1e-3 : 1e-2;

        pass = ck_tile::check_err(
            c_rslt_host, c_m_n_host_ref, "\n\nError: Incorrect results!", rtol, atol);

        std::cout << "\n\nRelative error threshold: " << rtol
                  << " Absolute error threshold: " << atol << std::endl;
        std::cout << "The GPU veification result is: " << (pass ? "correct" : "fail") << std::endl
                  << std::endl
                  << std::endl
                  << std::endl;
    }

    return pass ? 0 : -1;
}
