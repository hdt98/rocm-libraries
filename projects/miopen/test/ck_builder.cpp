#include <iostream>

#include <ck_tile/builder/conv_builder.hpp>
#include <ck_tile/builder/reflect/conv_description.hpp>
#include <ck_tile/builder/reflect/instance_traits.hpp>

namespace ckb = ck_tile::builder;
using BaseOperator = ck::tensor_operation::device::BaseOperator;

constexpr auto types = std::array<ckb::DataType, 6> {
    ckb::DataType::FP32,
    ckb::DataType::FP16,
    ckb::DataType::BF16,
    ckb::DataType::FP8,
    ckb::DataType::I8,
};

using BaseOperator = ck::tensor_operation::device::BaseOperator;

template <typename T, std::size_t N, typename F>
constexpr auto map_array(const std::array<T,N>& input, F&& func) {
    using U = std::invoke_result_t<F, T>;
    std::array<U, N> result{};
    for (auto i = 0; i < N; i++) {
        result[i] = func(input[i]);
    }

    return result;
}

template <typename T1, std::size_t N1, typename T2, std::size_t N2, typename F>
constexpr auto multiplex_array(const std::array<T1, N1> &input1, const std::array<T2, N2> &input2, F &&func) {
    using U = std::invoke_result_t<F, T1, T2>;
    std::array<U, N1 * N2> result{};
    for (auto i1 = 0; i1 < N1; i1++) {
        auto arg1 = input1[i1];
        for (auto i2 = 0; i2 < N2; i2++) {
            auto arg2 = input2[i2];
            auto retval = func(arg1, arg2);
            result[i1 * N1 + i2] = retval;
        }
    }

    return result;
}

template <auto KernelDescriptor>
constexpr void InstantiateKernel(std::vector<std::unique_ptr<BaseOperator>> &kernels) {
    // Create a ConvBuilder instance with the signature and algorithm
    // This will instantiate the DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3 kernel
    using Builder = ckb::ConvBuilder<KernelDescriptor.Signature, KernelDescriptor.Algorithm>;

    // Verify that Builder is a class type
    static_assert(std::is_class_v<Builder>, "Builder should be a class type");

    // Verify that Builder::Instance exists and is the actual device kernel class
    static_assert(std::is_class_v<typename Builder::Instance>,
                  "Builder::Instance should be a class type");

    static_assert(ck_tile::reflect::HasInstanceTraits<typename Builder::Instance>);
    kernels.push_back(std::make_unique<typename Builder::Instance>());
}

//*
template <typename T, T... values>
constexpr void build_kernels_helper(std::vector<std::unique_ptr<BaseOperator>> &kernels) {
    std::array<std::unique_ptr<BaseOperator>, sizeof...(values)> result{};
    ((InstantiateKernel<values>(kernels)),...);
}


template <typename T, std::size_t N, std::array<T, N> arr, std::size_t... I>
constexpr void build_kernels_impl(std::vector<std::unique_ptr<BaseOperator>> &kernels, std::index_sequence<I...>) {
    build_kernels_helper<T, arr[I]...>(kernels);
}

template <typename T, std::size_t N, std::array<T, N> arr>
constexpr void build_kernels(std::vector<std::unique_ptr<BaseOperator>> &kernels) {
    build_kernels_impl<T, N, arr>(kernels, std::make_index_sequence<N>{});
}
// */

struct DefaultAlgorithm
{
    using ConvSpecial = ckb::ConvFwdSpecialization;
    using GemmSpecial = ckb::GemmSpecialization;
    using PipeVers    = ckb::PipelineVersion;
    using PipeSched   = ckb::PipelineScheduler;

    struct ThreadBlock
    {
        int block_size = 256;
        struct TileSize
        {
            int m = 256;
            int n = 256;
            int k = 32;
        } tile_size;
    } thread_block;

    struct GridwiseGemm
    {
        int ak1            = 8;
        int bk1            = 8;
        int m_per_xdl      = 16;
        int n_per_xdl      = 16;
        int m_xdl_per_wave = 4;
        int n_xdl_per_wave = 4;
    } gridwise_gemm;

    struct TransferABC
    {
        struct TransferAB
        {
            struct BlockTransfer
            {
                int k0  = 4;
                int m_n = 256;
                int k1  = 8;
            } block_transfer;
            struct LdsTransfer
            {
                int src_vector_dim            = 2;
                int src_scalar_per_vector     = 8;
                int lds_dst_scalar_per_vector = 8;
                bool is_direct_load           = true;
                bool lds_padding              = false;
            } lds_transfer;
            struct BlockTransferAccessOrder
            {
                std::array<size_t, 3> order{0, 1, 2};
            } block_transfer_access_order;
            struct SrcAccessOrder
            {
                std::array<size_t, 3> order{0, 1, 2};
            } src_access_order;
        };
        TransferAB a;
        TransferAB b;
        struct TransferC
        {
            struct ThreadClusterDims
            {
                int m_block        = 1;
                int m_wave_per_xdl = 32;
                int n_block        = 1;
                int n_wave_per_xdl = 8;
            } thread_cluster_dims;
            struct Epilogue
            {
                int m_per_wave_per_shuffle = 1;
                int n_per_wave_per_shuffle = 1;
                int scalar_per_vector      = 8;
            } epilogue;
        } c;
    } transfer;

    // TODO: Fix CK Builder schema to not require these defaults.
    ConvSpecial fwd_specialization  = ConvSpecial::DEFAULT;
    GemmSpecial gemm_specialization = GemmSpecial::Default;
    struct BlockGemm
    {
        PipeVers pipeline_version = PipeVers::V4;
        PipeSched scheduler       = PipeSched::INTRAWAVE;
    } block_gemm;
};

// Define a struct to specify the signature
struct Signature
{
    int spatial_dim = 2;
    // TODO: This direction should be OK as default, but the factory fails.
    ckb::ConvDirection direction = ckb::ConvDirection::FORWARD;
    ckb::GroupConvLayout layout  = ckb::GroupConvLayout2D::GNHWC_GKYXC_GNHWK;
    ckb::DataType data_type      = ckb::DataType::FP16;
};

struct KernelArguments {
    Signature Signature;
    DefaultAlgorithm Algorithm;
};

int main()
{
    // Verify that the signature structure conforms to the signature concept.
    static_assert(ckb::ConvSignatureDescriptor<Signature>);
    // Specify the signature in a constexpr value
    constexpr Signature kSignature{};
    // Verify the signature value is valid
    static_assert(ckb::ValidConvSignature<kSignature>);

    // Define a struct to specify the algorithm.
    // TODO improve CK Builder schema to reduce duplication and simplify.

    //     // Verify that the signature conforms to the expected descriptor
    // static_assert(ckb::ConvSignatureDescriptor<Signature>);
    // // Specify the signature in a constexpr value
    // constexpr Signature kSignature{};
    // // Verify the signature value is valid
    // static_assert(ckb::ValidConvSignature<kSignature>);
    // Verify that the algorithm conforms to the algorithm concept
    static_assert(ckb::ConvAlgorithmDescriptor<DefaultAlgorithm>);

    // Specify the algorithm in a constexpr value
    static constexpr const DefaultAlgorithm kAlgorithm;

    // TODO: Verify the algorithm value is valid.

    // Create a ConvBuilder instance with the signature and algorithm
    // This will instantiate the DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3 kernel
    using Builder = ckb::ConvBuilder<kSignature, kAlgorithm>;

    // Verify that Builder is a class type
    static_assert(std::is_class_v<Builder>, "Builder should be a class type");

    // Verify that Builder::Instance exists and is the actual device kernel class
    static_assert(std::is_class_v<typename Builder::Instance>,
                  "Builder::Instance should be a class type");

    static_assert(ck_tile::reflect::HasInstanceTraits<typename Builder::Instance>);

    auto instance = Builder::Instance{};
    auto instanceString = instance.GetInstanceString();
    std::cout << "Instance string: " << instanceString << std::endl;

    constexpr std::array<int, 5> sizes{256, 128, 64, 32};

    constexpr auto algorithms = map_array(sizes, [](int size) {
        DefaultAlgorithm algorithm;
        algorithm.thread_block.block_size = size;
        return algorithm;
    });

    struct BuilderParameters {
        DefaultAlgorithm Algorithm;
        Signature Signature;
    };

    static constexpr auto arr = std::array<Signature, 1>{kSignature};
    static constexpr auto typesAdded = multiplex_array(arr, types, [](Signature s, ckb::DataType dt) {
        s.data_type = dt;
        return s;
    });

    static constexpr auto parameters = multiplex_array(algorithms, typesAdded, [](DefaultAlgorithm algorithm, Signature signature){
        BuilderParameters params {
            .Algorithm = algorithm,
            .Signature = signature,
        };

        return params;
    });

    std::cout << std::endl;

    std::vector<std::unique_ptr<BaseOperator>> kernels{};
    build_kernels<BuilderParameters, parameters.size(), parameters>(kernels);

    std::cout << "Kernel count: " << kernels.size() << std::endl;

    for (auto &&kernel : kernels) {
        std::cout << "    - " << kernel->GetInstanceString() << std::endl;
    }

    return 0;
}
