// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "simple_tensor_descriptor_wrapper.hpp"
#include <iostream>

using namespace unified_wrapper;

// ============================================================================
// Example Device Kernel using Unified Wrapper
// ============================================================================

template <typename BackendTag, typename T>
__global__ void unified_gemm_setup_kernel(
    const T* a_ptr,
    const T* b_ptr,
    T* c_ptr,
    index_t M,
    index_t N,
    index_t K)
{
    // Define wrapper types
    using DescWrapper = TensorDescriptorWrapper<BackendTag, T, 4>;
    using ViewWrapper = TensorViewWrapper<BackendTag, T, 4>;

    // Step 1: Create descriptors
    auto a_desc = DescWrapper::create_a_descriptor(
        M, K,
        K,      // stride
        true    // row-major
    );

    auto b_desc = DescWrapper::create_b_descriptor(
        K, N,
        N,      // stride
        true    // row-major
    );

    auto c_desc = DescWrapper::create_c_descriptor(
        M, N,
        N,      // stride
        true    // row-major
    );

    // Step 2: Create tensor views from descriptors
    auto a_view = ViewWrapper::create_a_view(a_ptr, a_desc);
    auto b_view = ViewWrapper::create_b_view(b_ptr, b_desc);
    auto c_view = ViewWrapper::create_c_view(c_ptr, c_desc);

    // Now a_view, b_view, c_view can be used for tile operations!
    // Same kernel code works for both MINT and CK_Tile backends
}

// ============================================================================
// Host Code Examples
// ============================================================================

void example_ck_tile_workflow() {
    std::cout << "=== CK_Tile Workflow ===" << std::endl;

    constexpr index_t M = 128;
    constexpr index_t N = 256;
    constexpr index_t K = 64;

    using DescWrapper = CKTileDescriptorWrapper<float, 4>;
    using ViewWrapper = CKTileViewWrapper<float, 4>;

    std::cout << "1. User creates descriptors using wrapper:" << std::endl;
    std::cout << "   auto a_desc = DescWrapper::create_a_descriptor(M, K, K, true);" << std::endl;

    std::cout << "\n2. Internally, wrapper calls CK_Tile:" << std::endl;
    std::cout << "   make_naive_tensor_descriptor(" << std::endl;
    std::cout << "       make_tuple(M, K)," << std::endl;
    std::cout << "       make_tuple(stride, 1)," << std::endl;
    std::cout << "       number<VectorSize>{}," << std::endl;
    std::cout << "       number<1>{})" << std::endl;

    std::cout << "\n3. User creates tensor view:" << std::endl;
    std::cout << "   auto a_view = ViewWrapper::create_view(a_ptr, a_desc);" << std::endl;

    std::cout << "\n4. Internally, wrapper calls CK_Tile:" << std::endl;
    std::cout << "   make_tensor_view<address_space_enum::global>(a_ptr, a_desc)" << std::endl;
}

void example_mint_workflow() {
    std::cout << "\n=== MINT Workflow ===" << std::endl;

    constexpr index_t M = 128;
    constexpr index_t N = 256;
    constexpr index_t K = 64;

    using DescWrapper = MintDescriptorWrapper<float>;
    using ViewWrapper = MintViewWrapper<float>;

    std::cout << "1. User creates descriptors using wrapper (SAME API!):" << std::endl;
    std::cout << "   auto a_desc = DescWrapper::create_a_descriptor(M, K, K, true);" << std::endl;

    std::cout << "\n2. Internally, wrapper calls MINT:" << std::endl;
    std::cout << "   make_aliased_naive_packed_tensor_descriptor(" << std::endl;
    std::cout << "       aliases<\"M\", \"K\">{}," << std::endl;
    std::cout << "       alias<\"Offset\">{}," << std::endl;
    std::cout << "       {M, K})" << std::endl;

    std::cout << "\n3. User creates tensor view (SAME API!):" << std::endl;
    std::cout << "   auto a_view = ViewWrapper::create_view(a_ptr, a_desc);" << std::endl;

    std::cout << "\n4. Internally, wrapper calls MINT:" << std::endl;
    std::cout << "   make_tensor_view(a_desc, make_global_memory_view(a_ptr, size))" << std::endl;
}

void example_comparison() {
    std::cout << "\n=== API Comparison ===" << std::endl;
    std::cout << "\nCK_Tile Backend:" << std::endl;
    std::cout << "  Descriptor: make_naive_tensor_descriptor(dims, strides, ...)" << std::endl;
    std::cout << "  View:       make_tensor_view<address_space>(ptr, desc)" << std::endl;

    std::cout << "\nMINT Backend:" << std::endl;
    std::cout << "  Descriptor: make_aliased_naive_packed_tensor_descriptor(aliases, dims)" << std::endl;
    std::cout << "  View:       make_tensor_view(desc, memory_view)" << std::endl;

    std::cout << "\n✅ Unified Wrapper:" << std::endl;
    std::cout << "  Descriptor: DescWrapper::create_a_descriptor(M, K, stride, layout)" << std::endl;
    std::cout << "  View:       ViewWrapper::create_view(ptr, desc)" << std::endl;
    std::cout << "  → Same user code, different backends!" << std::endl;
}

// ============================================================================
// Generic Function Working with Both Backends
// ============================================================================

template <typename BackendTag>
void setup_gemm_descriptors_and_views(
    index_t M, index_t N, index_t K,
    const float* a_ptr,
    const float* b_ptr,
    float* c_ptr)
{
    using DescWrapper = TensorDescriptorWrapper<BackendTag, float, 4>;
    using ViewWrapper = TensorViewWrapper<BackendTag, float, 4>;

    // Create descriptors
    auto a_desc = DescWrapper::create_a_descriptor(M, K, K, true);
    auto b_desc = DescWrapper::create_b_descriptor(K, N, N, true);
    auto c_desc = DescWrapper::create_c_descriptor(M, N, N, true);

    // Create views
    // Note: This would be in device code, shown for illustration
    // auto a_view = ViewWrapper::create_view(a_ptr, a_desc);
    // auto b_view = ViewWrapper::create_view(b_ptr, b_desc);
    // auto c_view = ViewWrapper::create_view(c_ptr, c_desc);

    std::cout << "Created descriptors and views for "
              << M << "x" << N << "x" << K << " GEMM" << std::endl;
}

int main() {
    example_ck_tile_workflow();
    example_mint_workflow();
    example_comparison();

    std::cout << "\n=== Generic Function Demo ===" << std::endl;

    // Dummy pointers for demo
    float *a = nullptr, *b = nullptr, *c = nullptr;

    std::cout << "\nWith CK_Tile backend:" << std::endl;
    setup_gemm_descriptors_and_views<CKTileBackend>(512, 512, 512, a, b, c);

    std::cout << "\nWith MINT backend:" << std::endl;
    setup_gemm_descriptors_and_views<MintBackend>(512, 512, 512, a, b, c);

    return 0;
}
