// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <catch2/catch_test_macros.hpp>

#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>
#include <rocRoller/Operations/Command.hpp>
#include <rocRoller/Operations/TensorIndices.hpp>

namespace BatchedGEMMTest
{
    using namespace rocRoller;
    using namespace rocRoller::Operations;

    TEST_CASE("BatchIndex NoBatch sentinel", "[batched-gemm]")
    {
        BatchIndex idx;
        CHECK(idx.a == BatchIndex::NoBatch);
        CHECK(idx.b == BatchIndex::NoBatch);
    }

    TEST_CASE("BatchIndex equality", "[batched-gemm]")
    {
        BatchIndex a{2, 2, 2};
        BatchIndex b{2, 2, 2};
        BatchIndex c{2, BatchIndex::NoBatch, 2};

        CHECK(a == b);
        CHECK(!(a == c));
    }

    TEST_CASE("MakeGemmIndices returns GemmIndices", "[batched-gemm]")
    {
        auto indices = MakeGemmIndices(false, false);
        CHECK(indices.freeDimsA.ab == 0);
        CHECK(indices.freeDimsA.d == 0);
        CHECK(indices.freeDimsB.ab == 1);
        CHECK(indices.freeDimsB.d == 1);
        CHECK(indices.boundDims.a == 1);
        CHECK(indices.boundDims.b == 0);
        CHECK(indices.batchDims.empty());
    }

    TEST_CASE("MakeBatchedGemmIndices all batched", "[batched-gemm]")
    {
        auto indices = MakeBatchedGemmIndices(false, false);

        CHECK(indices.freeDimsA.ab == 0);
        CHECK(indices.freeDimsA.d == 0);
        CHECK(indices.freeDimsB.ab == 1);
        CHECK(indices.freeDimsB.d == 1);
        CHECK(indices.boundDims.a == 1);
        CHECK(indices.boundDims.b == 0);

        REQUIRE(indices.batchDims.size() == 1);
        CHECK(indices.batchDims[0].a == 2);
        CHECK(indices.batchDims[0].b == 2);
        CHECK(indices.batchDims[0].d == 2);
    }

    TEST_CASE("MakeBatchedGemmIndices transposed", "[batched-gemm]")
    {
        auto indices = MakeBatchedGemmIndices(true, true);

        // transA swaps freeDimsA.ab with boundDims.a
        CHECK(indices.freeDimsA.ab == 1);
        CHECK(indices.boundDims.a == 0);

        // transB swaps freeDimsB.ab with boundDims.b
        CHECK(indices.freeDimsB.ab == 0);
        CHECK(indices.boundDims.b == 1);

        // Batch dims unchanged by transpose
        REQUIRE(indices.batchDims.size() == 1);
        CHECK(indices.batchDims[0].a == 2);
        CHECK(indices.batchDims[0].b == 2);
        CHECK(indices.batchDims[0].d == 2);
    }

    TEST_CASE("MakeBatchedGemmIndices broadcast B", "[batched-gemm]")
    {
        auto indices = MakeBatchedGemmIndices(false, false, true, false);

        REQUIRE(indices.batchDims.size() == 1);
        CHECK(indices.batchDims[0].a == 2);
        CHECK(indices.batchDims[0].b == BatchIndex::NoBatch);
        CHECK(indices.batchDims[0].d == 2);
    }

    TEST_CASE("MakeBatchedGemmIndices broadcast A", "[batched-gemm]")
    {
        auto indices = MakeBatchedGemmIndices(false, false, false, true);

        REQUIRE(indices.batchDims.size() == 1);
        CHECK(indices.batchDims[0].a == BatchIndex::NoBatch);
        CHECK(indices.batchDims[0].b == 2);
        CHECK(indices.batchDims[0].d == 2);
    }

    TEST_CASE("Batched T_Mul construction", "[batched-gemm]")
    {
        auto command = std::make_shared<Command>();

        size_t M          = 64;
        size_t K          = 32;
        size_t N          = 48;
        size_t batchCount = 4;

        // A: batched 3D tensor {M, K, batchCount}
        auto tagTensorA
            = command->addOperation(Tensor(3, DataType::Float));
        auto tagLoadA = command->addOperation(T_Load_Tiled(tagTensorA));

        // B: batched 3D tensor {K, N, batchCount}
        auto tagTensorB
            = command->addOperation(Tensor(3, DataType::Float));
        auto tagLoadB = command->addOperation(T_Load_Tiled(tagTensorB));

        auto indices = MakeBatchedGemmIndices(false, false);

        auto tagD = command->addOperation(T_Mul(tagLoadA,
                                                tagLoadB,
                                                {indices.freeDimsA},
                                                {indices.freeDimsB},
                                                {indices.boundDims},
                                                DataType::Float,
                                                indices.batchDims));

        // D: batched 3D tensor {M, N, batchCount}
        auto tagTensorD
            = command->addOperation(Tensor(3, DataType::Float));
        command->addOperation(T_Store_Tiled(tagD, tagTensorD));

        // Verify the command was constructed successfully
        auto ops = command->operations();
        CHECK(ops.size() > 0);

        // Translate to kernel graph and verify batch dims propagated
        auto graph = KernelGraph::translate(command);

        // Find TensorContraction node
        bool foundContraction = false;
        for(auto nodeID : graph.control.getNodes())
        {
            auto node = graph.control.getNode(nodeID);
            if(std::holds_alternative<KernelGraph::ControlGraph::TensorContraction>(node))
            {
                auto contraction
                    = std::get<KernelGraph::ControlGraph::TensorContraction>(node);
                REQUIRE(contraction.batchDims.size() == 1);
                CHECK(contraction.batchDims[0].a == 2);
                CHECK(contraction.batchDims[0].b == 2);
                CHECK(contraction.batchDims[0].d == 2);
                foundContraction = true;
            }
        }
        CHECK(foundContraction);
    }

    TEST_CASE("Batched T_Mul with broadcast B", "[batched-gemm]")
    {
        auto command = std::make_shared<Command>();

        // A: batched 3D tensor
        auto tagTensorA
            = command->addOperation(Tensor(3, DataType::Float));
        auto tagLoadA = command->addOperation(T_Load_Tiled(tagTensorA));

        // B: non-batched 2D tensor (broadcast)
        auto tagTensorB
            = command->addOperation(Tensor(2, DataType::Float));
        auto tagLoadB = command->addOperation(T_Load_Tiled(tagTensorB));

        auto indices = MakeBatchedGemmIndices(false, false, true, false);

        auto tagD = command->addOperation(T_Mul(tagLoadA,
                                                tagLoadB,
                                                {indices.freeDimsA},
                                                {indices.freeDimsB},
                                                {indices.boundDims},
                                                DataType::Float,
                                                indices.batchDims));

        // D: batched 3D tensor
        auto tagTensorD
            = command->addOperation(Tensor(3, DataType::Float));
        command->addOperation(T_Store_Tiled(tagD, tagTensorD));

        auto graph = KernelGraph::translate(command);

        bool foundContraction = false;
        for(auto nodeID : graph.control.getNodes())
        {
            auto node = graph.control.getNode(nodeID);
            if(std::holds_alternative<KernelGraph::ControlGraph::TensorContraction>(node))
            {
                auto contraction
                    = std::get<KernelGraph::ControlGraph::TensorContraction>(node);
                REQUIRE(contraction.batchDims.size() == 1);
                CHECK(contraction.batchDims[0].a == 2);
                CHECK(contraction.batchDims[0].b == BatchIndex::NoBatch);
                CHECK(contraction.batchDims[0].d == 2);
                foundContraction = true;
            }
        }
        CHECK(foundContraction);
    }

    TEST_CASE("Non-batched T_Mul backward compatible", "[batched-gemm]")
    {
        auto command = std::make_shared<Command>();

        auto tagTensorA
            = command->addOperation(Tensor(2, DataType::Float));
        auto tagLoadA = command->addOperation(T_Load_Tiled(tagTensorA));

        auto tagTensorB
            = command->addOperation(Tensor(2, DataType::Float));
        auto tagLoadB = command->addOperation(T_Load_Tiled(tagTensorB));

        auto gemmIndices = MakeGemmIndices(false, false);

        // No batchDims argument -- uses default empty
        auto tagD = command->addOperation(T_Mul(tagLoadA,
                                                tagLoadB,
                                                {gemmIndices.freeDimsA},
                                                {gemmIndices.freeDimsB},
                                                {gemmIndices.boundDims}));

        auto tagTensorD
            = command->addOperation(Tensor(2, DataType::Float));
        command->addOperation(T_Store_Tiled(tagD, tagTensorD));

        auto graph = KernelGraph::translate(command);

        bool foundContraction = false;
        for(auto nodeID : graph.control.getNodes())
        {
            auto node = graph.control.getNode(nodeID);
            if(std::holds_alternative<KernelGraph::ControlGraph::TensorContraction>(node))
            {
                auto contraction
                    = std::get<KernelGraph::ControlGraph::TensorContraction>(node);
                CHECK(contraction.batchDims.empty());
                foundContraction = true;
            }
        }
        CHECK(foundContraction);
    }
}
