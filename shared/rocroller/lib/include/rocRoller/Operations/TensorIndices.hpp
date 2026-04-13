// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * Tensor dimension index structures for tensor operations.
 */

#pragma once

#include <ostream>

namespace rocRoller
{
    namespace Operations
    {
        /**
         * FreeIndex - Represents a free (uncontracted) dimension in a tensor operation.
         *
         * ab: Index in the input tensor (A or B)
         * d:  Index in the destination tensor (D)
         */
        struct FreeIndex
        {
            size_t ab, d;
        };

        inline bool operator==(const FreeIndex& lhs, const FreeIndex& rhs)
        {
            return lhs.ab == rhs.ab && lhs.d == rhs.d;
        }

        inline std::ostream& operator<<(std::ostream& os, const FreeIndex& idx)
        {
            os << "FreeIndex(ab=" << idx.ab << ", d=" << idx.d << ")";
            return os;
        }

        /**
         * BoundIndex - Represents a bound (contracted) dimension in a tensor operation.
         *
         * a: Index in tensor A
         * b: Index in tensor B
         */
        struct BoundIndex
        {
            size_t a, b;
        };

        inline bool operator==(const BoundIndex& lhs, const BoundIndex& rhs)
        {
            return lhs.a == rhs.a && lhs.b == rhs.b;
        }

        inline std::ostream& operator<<(std::ostream& os, const BoundIndex& idx)
        {
            os << "BoundIndex(a=" << idx.a << ", b=" << idx.b << ")";
            return os;
        }

        /**
         * BatchIndex - Represents a batch dimension in a tensor contraction.
         *
         * a: Batch dimension index in tensor A (NoBatch if A is broadcast)
         * b: Batch dimension index in tensor B (NoBatch if B is broadcast)
         * d: Batch dimension index in tensor D (always present)
         */
        struct BatchIndex
        {
            static constexpr size_t NoBatch = std::numeric_limits<size_t>::max();

            size_t a = NoBatch;
            size_t b = NoBatch;
            size_t d;
        };

        inline bool operator==(const BatchIndex& lhs, const BatchIndex& rhs)
        {
            return lhs.a == rhs.a && lhs.b == rhs.b && lhs.d == rhs.d;
        }

        inline std::ostream& operator<<(std::ostream& os, const BatchIndex& idx)
        {
            os << "BatchIndex(a=";
            if(idx.a == BatchIndex::NoBatch)
                os << "NoBatch";
            else
                os << idx.a;
            os << ", b=";
            if(idx.b == BatchIndex::NoBatch)
                os << "NoBatch";
            else
                os << idx.b;
            os << ", d=" << idx.d << ")";
            return os;
        }

        /**
         * GemmIndices - All index information for a GEMM operation,
         * including optional batch dimensions.
         */
        struct GemmIndices
        {
            FreeIndex               freeDimsA;
            FreeIndex               freeDimsB;
            BoundIndex              boundDims;
            std::vector<BatchIndex> batchDims;
        };

        /**
         * Create GemmIndices for a standard (non-batched) GEMM: A[M,K] * B[K,N] -> D[M,N].
         *
         * Applies the transpose swap pattern for physical dimension ordering:
         * - transA swaps freeDimsA.ab with boundDims.a
         * - transB swaps freeDimsB.ab with boundDims.b
         *
         * batchDims is empty (non-batched).
         */
        inline GemmIndices MakeGemmIndices(bool transA, bool transB)
        {
            FreeIndex  freeDimsA{0, 0};
            FreeIndex  freeDimsB{1, 1};
            BoundIndex boundDims{1, 0};

            if(transA)
                std::swap(freeDimsA.ab, boundDims.a);
            if(transB)
                std::swap(freeDimsB.ab, boundDims.b);

            return {freeDimsA, freeDimsB, boundDims, {}};
        }

        /**
         * Create GemmIndices for a strided-batched GEMM.
         *
         * The batch dimension is always the last (slowest) dimension in each tensor.
         * For a 2D+batch tensor, the batch dim index is 2.
         *
         * batchA/batchB control whether A/B has a batch dimension.
         * A broadcast tensor stays 2D (no batch dim), represented by NoBatch.
         */
        inline GemmIndices
            MakeBatchedGemmIndices(bool transA, bool transB, bool batchA = true, bool batchB = true)
        {
            auto indices = MakeGemmIndices(transA, transB);

            BatchIndex batch;
            batch.a = batchA ? 2 : BatchIndex::NoBatch;
            batch.b = batchB ? 2 : BatchIndex::NoBatch;
            batch.d = 2;
            indices.batchDims.push_back(batch);

            return indices;
        }
    }
}
