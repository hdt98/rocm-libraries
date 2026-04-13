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

        inline std::ostream& operator<<(std::ostream& os, const BoundIndex& idx)
        {
            os << "BoundIndex(a=" << idx.a << ", b=" << idx.b << ")";
            return os;
        }

        /**
         * Create FreeIndex/BoundIndex for a standard GEMM: A[M,K] * B[K,N] -> D[M,N].
         *
         * Applies the transpose swap pattern for physical dimension ordering:
         * - transA swaps freeDimsA.ab with boundDims.a
         * - transB swaps freeDimsB.ab with boundDims.b
         */
        inline std::tuple<FreeIndex, FreeIndex, BoundIndex> MakeGemmIndices(bool transA,
                                                                            bool transB)
        {
            FreeIndex  freeDimsA{0, 0};
            FreeIndex  freeDimsB{1, 1};
            BoundIndex boundDims{1, 0};

            if(transA)
                std::swap(freeDimsA.ab, boundDims.a);
            if(transB)
                std::swap(freeDimsB.ab, boundDims.b);

            return {freeDimsA, freeDimsB, boundDims};
        }
    }
}
