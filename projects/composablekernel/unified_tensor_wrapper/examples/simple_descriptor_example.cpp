// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "simple_tensor_descriptor_wrapper.hpp"
#include <iostream>

using namespace unified_wrapper;

// ============================================================================
// Example: Using the wrapper with both backends
// ============================================================================

void example_ck_tile_backend() {
    std::cout << "=== CK_Tile Backend ===" << std::endl;

    // GEMM problem: C[M, N] = A[M, K] * B[K, N]
    constexpr index_t M = 128;
    constexpr index_t N = 256;
    constexpr index_t K = 64;

    // User just provides numbers - no need to know about aliases!
    using Wrapper = CKTileDescriptorWrapper<float, 4>; // fp32, vector size 4

    // Create descriptors
    auto a_desc = Wrapper::create_a_descriptor(
        M, K,
        K,      // stride (row-major: leading dim = K)
        true    // row-major
    );

    auto b_desc = Wrapper::create_b_descriptor(
        K, N,
        N,      // stride (row-major: leading dim = N)
        true    // row-major
    );

    auto c_desc = Wrapper::create_c_descriptor(
        M, N,
        N,      // stride (row-major: leading dim = N)
        true    // row-major
    );

    std::cout << "Created CK_Tile descriptors for:" << std::endl;
    std::cout << "  A: " << M << "x" << K << " (row-major)" << std::endl;
    std::cout << "  B: " << K << "x" << N << " (row-major)" << std::endl;
    std::cout << "  C: " << M << "x" << N << " (row-major)" << std::endl;
}

void example_mint_backend() {
    std::cout << "\n=== MINT Backend ===" << std::endl;

    // Same GEMM problem
    constexpr index_t M = 128;
    constexpr index_t N = 256;
    constexpr index_t K = 64;

    // Just change the wrapper type - same API!
    using Wrapper = MintDescriptorWrapper<float>;

    // Create descriptors - identical user code!
    auto a_desc = Wrapper::create_a_descriptor(
        M, K,
        K,      // stride (ignored for packed MINT descriptors)
        true    // row-major
    );

    auto b_desc = Wrapper::create_b_descriptor(
        K, N,
        N,      // stride (ignored for packed MINT descriptors)
        true    // row-major (but MINT B is actually transposed internally)
    );

    auto c_desc = Wrapper::create_c_descriptor(
        M, N,
        N,      // stride
        true    // row-major
    );

    std::cout << "Created MINT descriptors for:" << std::endl;
    std::cout << "  A: " << M << "x" << K << " (row-major) → aliases<\"M\", \"K\">" << std::endl;
    std::cout << "  B: " << K << "x" << N << " (row-major) → aliases<\"N\", \"K\">" << std::endl;
    std::cout << "  C: " << M << "x" << N << " (row-major) → aliases<\"M\", \"N\">" << std::endl;
}

// ============================================================================
// Generic function that works with both backends
// ============================================================================

template <typename BackendTag>
void generic_gemm_setup(index_t M, index_t N, index_t K, bool row_major) {
    using Wrapper = TensorDescriptorWrapper<BackendTag, float>;

    auto a_desc = Wrapper::create_a_descriptor(M, K, K, row_major);
    auto b_desc = Wrapper::create_b_descriptor(K, N, N, row_major);
    auto c_desc = Wrapper::create_c_descriptor(M, N, N, row_major);

    std::cout << "Generic setup completed for " << M << "x" << N << "x" << K << std::endl;
}

int main() {
    example_ck_tile_backend();
    example_mint_backend();

    std::cout << "\n=== Generic Function with Backend Tags ===" << std::endl;

    std::cout << "CK_Tile: ";
    generic_gemm_setup<CKTileBackend>(512, 512, 512, true);

    std::cout << "MINT: ";
    generic_gemm_setup<MintBackend>(512, 512, 512, true);

    return 0;
}
