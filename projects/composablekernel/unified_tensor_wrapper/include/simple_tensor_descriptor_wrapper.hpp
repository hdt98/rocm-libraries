// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include <mint/mint.h>

namespace unified_wrapper {

// ============================================================================
// Simple Tensor Descriptor Specification (Backend-agnostic)
// ============================================================================

struct TensorDescriptorSpec {
    index_t dim0;           // First logical dimension (M for A matrix)
    index_t dim1;           // Second logical dimension (K for A matrix)
    index_t stride;         // Leading dimension stride (row stride for row-major)
    bool is_row_major;      // Layout: true=RowMajor, false=ColumnMajor
};

// ============================================================================
// Backend Tags (for template dispatch)
// ============================================================================

struct CKTileBackend {};
struct MintBackend {};

// ============================================================================
// CK_Tile Backend Adapter
// ============================================================================

template <typename DataType, int VectorSize = 1>
class CKTileDescriptorAdapter {
public:
    // Create A matrix descriptor (M x K)
    static auto create_a_descriptor(const TensorDescriptorSpec& spec) {
        using namespace ck_tile;

        if (spec.is_row_major) {
            // Row-major: [M, K] with stride (stride, 1)
            return make_naive_tensor_descriptor(
                make_tuple(spec.dim0, spec.dim1),  // (M, K)
                make_tuple(spec.stride, 1),         // strides
                number<VectorSize>{},
                number<1>{});
        } else {
            // Column-major: [K, M] with stride (stride, 1)
            return make_naive_tensor_descriptor(
                make_tuple(spec.dim1, spec.dim0),  // (K, M)
                make_tuple(spec.stride, 1),         // strides
                number<VectorSize>{},
                number<1>{});
        }
    }

    // Create B matrix descriptor (K x N)
    static auto create_b_descriptor(const TensorDescriptorSpec& spec) {
        using namespace ck_tile;

        if (spec.is_row_major) {
            // Row-major: [K, N] with stride (stride, 1)
            return make_naive_tensor_descriptor(
                make_tuple(spec.dim0, spec.dim1),  // (K, N)
                make_tuple(spec.stride, 1),
                number<VectorSize>{},
                number<1>{});
        } else {
            // Column-major: [N, K] with stride (stride, 1)
            return make_naive_tensor_descriptor(
                make_tuple(spec.dim1, spec.dim0),  // (N, K)
                make_tuple(spec.stride, 1),
                number<VectorSize>{},
                number<1>{});
        }
    }

    // Create C matrix descriptor (M x N)
    static auto create_c_descriptor(const TensorDescriptorSpec& spec) {
        using namespace ck_tile;

        if (spec.is_row_major) {
            // Row-major: [M, N] with stride (stride, 1)
            return make_naive_tensor_descriptor(
                make_tuple(spec.dim0, spec.dim1),  // (M, N)
                make_tuple(spec.stride, 1),
                number<VectorSize>{},
                number<1>{});
        } else {
            // Column-major: [N, M] with stride (stride, 1)
            return make_naive_tensor_descriptor(
                make_tuple(spec.dim1, spec.dim0),  // (N, M)
                make_tuple(1, spec.stride),         // strides for col-major C
                number<1>{},
                number<1>{});
        }
    }
};

// ============================================================================
// MINT Backend Adapter
// ============================================================================

template <typename DataType>
class MintDescriptorAdapter {
public:
    // Create A matrix descriptor (M x K)
    static auto create_a_descriptor(const TensorDescriptorSpec& spec) {
        using namespace mint;
        using namespace mint::tensor;

        if (spec.is_row_major) {
            // Row-major A: [M, K]
            return make_aliased_naive_packed_tensor_descriptor(
                aliases<"M", "K">{},
                alias<"Offset">{},
                {spec.dim0, spec.dim1});
        } else {
            // Column-major A: [K, M] (transposed)
            return make_aliased_naive_packed_tensor_descriptor(
                aliases<"K", "M">{},
                alias<"Offset">{},
                {spec.dim1, spec.dim0});
        }
    }

    // Create B matrix descriptor (K x N)
    static auto create_b_descriptor(const TensorDescriptorSpec& spec) {
        using namespace mint;
        using namespace mint::tensor;

        if (spec.is_row_major) {
            // Row-major B: [K, N]
            // But MINT example uses [N, K] for B!
            // This is because B is column-major in their example
            return make_aliased_naive_packed_tensor_descriptor(
                aliases<"N", "K">{},
                alias<"Offset">{},
                {spec.dim1, spec.dim0});
        } else {
            // Column-major B: [N, K]
            return make_aliased_naive_packed_tensor_descriptor(
                aliases<"N", "K">{},
                alias<"Offset">{},
                {spec.dim0, spec.dim1});
        }
    }

    // Create C matrix descriptor (M x N)
    static auto create_c_descriptor(const TensorDescriptorSpec& spec) {
        using namespace mint;
        using namespace mint::tensor;

        if (spec.is_row_major) {
            // Row-major C: [M, N]
            return make_aliased_naive_packed_tensor_descriptor(
                aliases<"M", "N">{},
                alias<"Offset">{},
                {spec.dim0, spec.dim1});
        } else {
            // Column-major C: [N, M] (transposed)
            return make_aliased_naive_packed_tensor_descriptor(
                aliases<"N", "M">{},
                alias<"Offset">{},
                {spec.dim1, spec.dim0});
        }
    }
};

// ============================================================================
// Unified Wrapper (Template-based Backend Selection via Tag Dispatch)
// ============================================================================

template <typename BackendTag, typename DataType, int VectorSize = 1>
class TensorDescriptorWrapper {
private:
    // Select adapter based on backend tag
    using Adapter = std::conditional_t<
        std::is_same_v<BackendTag, CKTileBackend>,
        CKTileDescriptorAdapter<DataType, VectorSize>,
        MintDescriptorAdapter<DataType>
    >;

public:
    // Create A matrix descriptor (M x K)
    static auto create_a_descriptor(index_t M, index_t K, index_t stride_a, bool row_major) {
        TensorDescriptorSpec spec{M, K, stride_a, row_major};
        return Adapter::create_a_descriptor(spec);
    }

    // Create B matrix descriptor (K x N)
    static auto create_b_descriptor(index_t K, index_t N, index_t stride_b, bool row_major) {
        TensorDescriptorSpec spec{K, N, stride_b, row_major};
        return Adapter::create_b_descriptor(spec);
    }

    // Create C matrix descriptor (M x N)
    static auto create_c_descriptor(index_t M, index_t N, index_t stride_c, bool row_major) {
        TensorDescriptorSpec spec{M, N, stride_c, row_major};
        return Adapter::create_c_descriptor(spec);
    }
};

// ============================================================================
// Tensor View Adapters (Backend-specific implementations)
// ============================================================================

template <typename DataType, int VectorSize = 1>
class CKTileTensorViewAdapter {
public:
    // CK_Tile: make_tensor_view<address_space>(pointer, descriptor)
    template <typename DescriptorType>
    CK_TILE_DEVICE static auto create_view(const DataType* ptr, const DescriptorType& desc) {
        using namespace ck_tile;
        return make_tensor_view<address_space_enum::global>(ptr, desc);
    }

    template <typename DescriptorType>
    CK_TILE_DEVICE static auto create_view(DataType* ptr, const DescriptorType& desc) {
        using namespace ck_tile;
        return make_tensor_view<address_space_enum::global>(ptr, desc);
    }
};

template <typename DataType>
class MintTensorViewAdapter {
public:
    // MINT: make_tensor_view(descriptor, memory_view)
    template <typename DescriptorType>
    __device__ static auto create_view(const DataType* ptr, const DescriptorType& desc) {
        using namespace mint;
        using namespace mint::tensor;

        // MINT requires bottom_lengths to create memory view
        const auto bottom_size = desc.bottom_lengths()[0];
        auto mem_view = make_global_memory_view(ptr, bottom_size);
        return make_tensor_view(desc, mem_view);
    }

    template <typename DescriptorType>
    __device__ static auto create_view(DataType* ptr, const DescriptorType& desc) {
        using namespace mint;
        using namespace mint::tensor;

        const auto bottom_size = desc.bottom_lengths()[0];
        auto mem_view = make_global_memory_view(ptr, bottom_size);
        return make_tensor_view(desc, mem_view);
    }
};

// ============================================================================
// Unified Tensor View Wrapper
// ============================================================================

template <typename BackendTag, typename DataType, int VectorSize = 1>
class TensorViewWrapper {
private:
    using ViewAdapter = std::conditional_t<
        std::is_same_v<BackendTag, CKTileBackend>,
        CKTileTensorViewAdapter<DataType, VectorSize>,
        MintTensorViewAdapter<DataType>
    >;

public:
    // Generic: create view from pointer + descriptor
    template <typename DescriptorType>
    CK_TILE_DEVICE static auto create_view(const DataType* ptr, const DescriptorType& desc) {
        return ViewAdapter::create_view(ptr, desc);
    }

    template <typename DescriptorType>
    CK_TILE_DEVICE static auto create_view(DataType* ptr, const DescriptorType& desc) {
        return ViewAdapter::create_view(ptr, desc);
    }

    // Convenience: named methods for GEMM matrices
    template <typename DescriptorType>
    CK_TILE_DEVICE static auto create_a_view(const DataType* a_ptr, const DescriptorType& a_desc) {
        return ViewAdapter::create_view(a_ptr, a_desc);
    }

    template <typename DescriptorType>
    CK_TILE_DEVICE static auto create_b_view(const DataType* b_ptr, const DescriptorType& b_desc) {
        return ViewAdapter::create_view(b_ptr, b_desc);
    }

    template <typename DescriptorType>
    CK_TILE_DEVICE static auto create_c_view(DataType* c_ptr, const DescriptorType& c_desc) {
        return ViewAdapter::create_view(c_ptr, c_desc);
    }
};

// ============================================================================
// Convenience Type Aliases
// ============================================================================

// Descriptor wrappers
template <typename DataType, int VectorSize = 1>
using CKTileDescriptorWrapper = TensorDescriptorWrapper<CKTileBackend, DataType, VectorSize>;

template <typename DataType>
using MintDescriptorWrapper = TensorDescriptorWrapper<MintBackend, DataType>;

// View wrappers
template <typename DataType, int VectorSize = 1>
using CKTileViewWrapper = TensorViewWrapper<CKTileBackend, DataType, VectorSize>;

template <typename DataType>
using MintViewWrapper = TensorViewWrapper<MintBackend, DataType>;

} // namespace unified_wrapper
