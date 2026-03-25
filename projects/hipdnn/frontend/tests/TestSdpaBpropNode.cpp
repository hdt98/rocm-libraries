// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_frontend/attributes/SdpaBackwardAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_frontend/node/SdpaBpropNode.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

namespace
{
// Helper: create a rank-4 tensor with given dims [batch, heads, seq, head_dim]
std::shared_ptr<TensorAttributes>
    makeTensor4D(int64_t batch, int64_t heads, int64_t seq, int64_t headDim)
{
    auto t = std::make_shared<TensorAttributes>();
    t->set_dim({batch, heads, seq, headDim});
    return t;
}

// Build a minimal SdpaBackwardAttributes with Q, K, V, O, dO, Stats, dQ, dK, dV set
SdpaBackwardAttributes makeMinimalAttrs(const std::shared_ptr<TensorAttributes>& q,
                                        const std::shared_ptr<TensorAttributes>& k,
                                        const std::shared_ptr<TensorAttributes>& v)
{
    SdpaBackwardAttributes attrs;
    attrs.set_q(q);
    attrs.set_k(k);
    attrs.set_v(v);
    attrs.set_o(std::make_shared<TensorAttributes>());
    attrs.set_do(std::make_shared<TensorAttributes>());
    attrs.set_stats(std::make_shared<TensorAttributes>());
    attrs.set_dq(std::make_shared<TensorAttributes>());
    attrs.set_dk(std::make_shared<TensorAttributes>());
    attrs.set_dv(std::make_shared<TensorAttributes>());
    return attrs;
}
} // namespace

//==============================================================================
// pre_validate_node tests
//==============================================================================

TEST(TestSdpaBpropNode, PreValidateSucceedsMinimal)
{
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(2, 8, 32, 64);
    auto v = makeTensor4D(2, 8, 32, 64);
    auto attrs = makeMinimalAttrs(q, k, v);

    GraphAttributes graphAttrs;
    SdpaBpropNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;
}

TEST(TestSdpaBpropNode, PreValidateSucceedsGQA)
{
    // num_heads=8, num_kv_heads=2 — GQA valid (8 % 2 == 0)
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(2, 2, 32, 64);
    auto v = makeTensor4D(2, 2, 32, 64);
    auto attrs = makeMinimalAttrs(q, k, v);

    GraphAttributes graphAttrs;
    SdpaBpropNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;
}

TEST(TestSdpaBpropNode, PreValidateSucceedsMQA)
{
    // MQA: num_kv_heads=1
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(2, 1, 32, 64);
    auto v = makeTensor4D(2, 1, 32, 64);
    auto attrs = makeMinimalAttrs(q, k, v);

    GraphAttributes graphAttrs;
    SdpaBpropNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;
}

TEST(TestSdpaBpropNode, PreValidateFailsMissingQ)
{
    SdpaBackwardAttributes attrs;
    attrs.set_k(makeTensor4D(2, 8, 32, 64));
    attrs.set_v(makeTensor4D(2, 8, 32, 64));
    attrs.set_o(std::make_shared<TensorAttributes>());
    attrs.set_do(std::make_shared<TensorAttributes>());
    attrs.set_stats(std::make_shared<TensorAttributes>());
    attrs.set_dq(std::make_shared<TensorAttributes>());
    attrs.set_dk(std::make_shared<TensorAttributes>());
    attrs.set_dv(std::make_shared<TensorAttributes>());

    GraphAttributes graphAttrs;
    SdpaBpropNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestSdpaBpropNode, PreValidateFailsMissingK)
{
    SdpaBackwardAttributes attrs;
    attrs.set_q(makeTensor4D(2, 8, 16, 64));
    attrs.set_v(makeTensor4D(2, 8, 32, 64));
    attrs.set_o(std::make_shared<TensorAttributes>());
    attrs.set_do(std::make_shared<TensorAttributes>());
    attrs.set_stats(std::make_shared<TensorAttributes>());
    attrs.set_dq(std::make_shared<TensorAttributes>());
    attrs.set_dk(std::make_shared<TensorAttributes>());
    attrs.set_dv(std::make_shared<TensorAttributes>());

    GraphAttributes graphAttrs;
    SdpaBpropNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestSdpaBpropNode, PreValidateFailsMissingV)
{
    SdpaBackwardAttributes attrs;
    attrs.set_q(makeTensor4D(2, 8, 16, 64));
    attrs.set_k(makeTensor4D(2, 8, 32, 64));
    attrs.set_o(std::make_shared<TensorAttributes>());
    attrs.set_do(std::make_shared<TensorAttributes>());
    attrs.set_stats(std::make_shared<TensorAttributes>());
    attrs.set_dq(std::make_shared<TensorAttributes>());
    attrs.set_dk(std::make_shared<TensorAttributes>());
    attrs.set_dv(std::make_shared<TensorAttributes>());

    GraphAttributes graphAttrs;
    SdpaBpropNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestSdpaBpropNode, PreValidateFailsMissingDo)
{
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(2, 8, 32, 64);
    auto v = makeTensor4D(2, 8, 32, 64);

    SdpaBackwardAttributes attrs;
    attrs.set_q(q).set_k(k).set_v(v);
    attrs.set_o(std::make_shared<TensorAttributes>());
    // dO intentionally not set
    attrs.set_stats(std::make_shared<TensorAttributes>());
    attrs.set_dq(std::make_shared<TensorAttributes>());
    attrs.set_dk(std::make_shared<TensorAttributes>());
    attrs.set_dv(std::make_shared<TensorAttributes>());

    GraphAttributes graphAttrs;
    SdpaBpropNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestSdpaBpropNode, PreValidateFailsMissingStats)
{
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(2, 8, 32, 64);
    auto v = makeTensor4D(2, 8, 32, 64);

    SdpaBackwardAttributes attrs;
    attrs.set_q(q).set_k(k).set_v(v);
    attrs.set_o(std::make_shared<TensorAttributes>());
    attrs.set_do(std::make_shared<TensorAttributes>());
    // stats intentionally not set
    attrs.set_dq(std::make_shared<TensorAttributes>());
    attrs.set_dk(std::make_shared<TensorAttributes>());
    attrs.set_dv(std::make_shared<TensorAttributes>());

    GraphAttributes graphAttrs;
    SdpaBpropNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestSdpaBpropNode, PreValidateFailsMissingDq)
{
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(2, 8, 32, 64);
    auto v = makeTensor4D(2, 8, 32, 64);

    SdpaBackwardAttributes attrs;
    attrs.set_q(q).set_k(k).set_v(v);
    attrs.set_o(std::make_shared<TensorAttributes>());
    attrs.set_do(std::make_shared<TensorAttributes>());
    attrs.set_stats(std::make_shared<TensorAttributes>());
    // dq intentionally not set
    attrs.set_dk(std::make_shared<TensorAttributes>());
    attrs.set_dv(std::make_shared<TensorAttributes>());

    GraphAttributes graphAttrs;
    SdpaBpropNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestSdpaBpropNode, PreValidateFailsRankLessThan4)
{
    // Q is rank-3
    auto q = std::make_shared<TensorAttributes>();
    q->set_dim({2, 8, 64}); // rank-3
    auto k = makeTensor4D(2, 8, 32, 64);
    auto v = makeTensor4D(2, 8, 32, 64);
    auto attrs = makeMinimalAttrs(q, k, v);

    GraphAttributes graphAttrs;
    SdpaBpropNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE);
}

TEST(TestSdpaBpropNode, PreValidateFailsRankGreaterThan4)
{
    // K is rank-5
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = std::make_shared<TensorAttributes>();
    k->set_dim({1, 2, 8, 32, 64}); // rank-5
    auto v = makeTensor4D(2, 8, 32, 64);
    auto attrs = makeMinimalAttrs(q, k, v);

    GraphAttributes graphAttrs;
    SdpaBpropNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE);
}

TEST(TestSdpaBpropNode, PreValidateFailsBatchMismatchQK)
{
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(4, 8, 32, 64); // batch=4 != 2
    auto v = makeTensor4D(2, 8, 32, 64);
    auto attrs = makeMinimalAttrs(q, k, v);

    GraphAttributes graphAttrs;
    SdpaBpropNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE);
}

TEST(TestSdpaBpropNode, PreValidateFailsBatchMismatchQV)
{
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(2, 8, 32, 64);
    auto v = makeTensor4D(4, 8, 32, 64); // batch=4 != 2
    auto attrs = makeMinimalAttrs(q, k, v);

    GraphAttributes graphAttrs;
    SdpaBpropNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE);
}

TEST(TestSdpaBpropNode, PreValidateFailsHeadDimMismatch)
{
    // K head_dim=32, Q head_dim=64 — must match
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(2, 8, 32, 32); // head_dim mismatch
    auto v = makeTensor4D(2, 8, 32, 64);
    auto attrs = makeMinimalAttrs(q, k, v);

    GraphAttributes graphAttrs;
    SdpaBpropNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE);
}

TEST(TestSdpaBpropNode, PreValidateFailsSeqKvMismatch)
{
    // K seq_kv=32, V seq_kv=16 — must match
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(2, 8, 32, 64);
    auto v = makeTensor4D(2, 8, 16, 64); // seq_kv mismatch
    auto attrs = makeMinimalAttrs(q, k, v);

    GraphAttributes graphAttrs;
    SdpaBpropNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE);
}

TEST(TestSdpaBpropNode, PreValidateFailsNumKvHeadsMismatchKV)
{
    // K num_kv_heads=2, V num_kv_heads=4 — must match
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(2, 2, 32, 64);
    auto v = makeTensor4D(2, 4, 32, 64); // num_kv_heads mismatch
    auto attrs = makeMinimalAttrs(q, k, v);

    GraphAttributes graphAttrs;
    SdpaBpropNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE);
}

TEST(TestSdpaBpropNode, PreValidateFailsInvalidGQA)
{
    // num_heads=8, num_kv_heads=3 — 8 % 3 != 0
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(2, 3, 32, 64);
    auto v = makeTensor4D(2, 3, 32, 64);
    auto attrs = makeMinimalAttrs(q, k, v);

    GraphAttributes graphAttrs;
    SdpaBpropNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE);
}

TEST(TestSdpaBpropNode, PreValidateSuccessDoMatchesOShape)
{
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(2, 8, 32, 64);
    auto v = makeTensor4D(2, 8, 32, 64);

    SdpaBackwardAttributes attrs;
    attrs.set_q(q).set_k(k).set_v(v);
    auto o = makeTensor4D(2, 8, 16, 64);
    attrs.set_o(o);
    auto dOut = makeTensor4D(2, 8, 16, 64); // matches O shape
    attrs.set_do(dOut);
    attrs.set_stats(std::make_shared<TensorAttributes>());
    attrs.set_dq(std::make_shared<TensorAttributes>());
    attrs.set_dk(std::make_shared<TensorAttributes>());
    attrs.set_dv(std::make_shared<TensorAttributes>());

    GraphAttributes graphAttrs;
    SdpaBpropNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;
}

TEST(TestSdpaBpropNode, PreValidateFailsDoShapeMismatchWithO)
{
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(2, 8, 32, 64);
    auto v = makeTensor4D(2, 8, 32, 64);

    SdpaBackwardAttributes attrs;
    attrs.set_q(q).set_k(k).set_v(v);
    auto o = makeTensor4D(2, 8, 16, 64);
    attrs.set_o(o);
    auto dOut = makeTensor4D(2, 8, 16, 99); // head_dim mismatch
    attrs.set_do(dOut);
    attrs.set_stats(std::make_shared<TensorAttributes>());
    attrs.set_dq(std::make_shared<TensorAttributes>());
    attrs.set_dk(std::make_shared<TensorAttributes>());
    attrs.set_dv(std::make_shared<TensorAttributes>());

    GraphAttributes graphAttrs;
    SdpaBpropNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE);
}

//==============================================================================
// infer_properties_node tests
//==============================================================================

TEST(TestSdpaBpropNode, InferPropertiesSetsDqShape)
{
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(2, 8, 32, 64);
    auto v = makeTensor4D(2, 8, 32, 64);
    auto attrs = makeMinimalAttrs(q, k, v);
    auto dq = attrs.get_dq();

    GraphAttributes graphAttrs;
    SdpaBpropNode node(std::move(attrs), graphAttrs);
    auto err = node.infer_properties_node();
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;

    // dQ should have same shape as Q: [2, 8, 16, 64]
    auto dims = dq->get_dim();
    ASSERT_EQ(dims.size(), 4u);
    EXPECT_EQ(dims[0], 2);
    EXPECT_EQ(dims[1], 8);
    EXPECT_EQ(dims[2], 16);
    EXPECT_EQ(dims[3], 64);
}

TEST(TestSdpaBpropNode, InferPropertiesSetsDkShape)
{
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(2, 4, 32, 64); // GQA: num_kv_heads=4
    auto v = makeTensor4D(2, 4, 32, 64);
    auto attrs = makeMinimalAttrs(q, k, v);
    auto dk = attrs.get_dk();

    GraphAttributes graphAttrs;
    SdpaBpropNode node(std::move(attrs), graphAttrs);
    auto err = node.infer_properties_node();
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;

    // dK should have same shape as K: [2, 4, 32, 64]
    auto dims = dk->get_dim();
    ASSERT_EQ(dims.size(), 4u);
    EXPECT_EQ(dims[0], 2);
    EXPECT_EQ(dims[1], 4);
    EXPECT_EQ(dims[2], 32);
    EXPECT_EQ(dims[3], 64);
}

TEST(TestSdpaBpropNode, InferPropertiesSetsDvShape)
{
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(2, 8, 32, 64);
    auto v = makeTensor4D(2, 8, 32, 32); // headDimV=32 differs from headDimQK=64
    auto attrs = makeMinimalAttrs(q, k, v);
    auto dv = attrs.get_dv();

    GraphAttributes graphAttrs;
    SdpaBpropNode node(std::move(attrs), graphAttrs);
    auto err = node.infer_properties_node();
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;

    // dV should have same shape as V: [2, 8, 32, 32]
    auto dims = dv->get_dim();
    ASSERT_EQ(dims.size(), 4u);
    EXPECT_EQ(dims[0], 2);
    EXPECT_EQ(dims[1], 8);
    EXPECT_EQ(dims[2], 32);
    EXPECT_EQ(dims[3], 32);
}

TEST(TestSdpaBpropNode, InferPropertiesSetsStrides)
{
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(2, 8, 32, 64);
    auto v = makeTensor4D(2, 8, 32, 64);
    auto attrs = makeMinimalAttrs(q, k, v);
    auto dq = attrs.get_dq();

    GraphAttributes graphAttrs;
    SdpaBpropNode node(std::move(attrs), graphAttrs);
    node.infer_properties_node();

    // Row-major strides for [2, 8, 16, 64]: [8*16*64, 16*64, 64, 1]
    auto strides = dq->get_stride();
    ASSERT_EQ(strides.size(), 4u);
    EXPECT_EQ(strides[3], 1);
    EXPECT_EQ(strides[2], 64);
    EXPECT_EQ(strides[1], 16 * 64);
    EXPECT_EQ(strides[0], 8 * 16 * 64);
}

TEST(TestSdpaBpropNode, InferPropertiesPreservesExplicitShape)
{
    // If dQ already has dims set, they should not be overwritten
    auto q = makeTensor4D(2, 8, 16, 64);
    auto k = makeTensor4D(2, 8, 32, 64);
    auto v = makeTensor4D(2, 8, 32, 64);

    SdpaBackwardAttributes attrs;
    attrs.set_q(q).set_k(k).set_v(v);
    attrs.set_o(std::make_shared<TensorAttributes>());
    attrs.set_do(std::make_shared<TensorAttributes>());
    attrs.set_stats(std::make_shared<TensorAttributes>());

    auto dq = std::make_shared<TensorAttributes>();
    dq->set_dim({2, 8, 16, 64});
    dq->set_stride({8192, 1024, 64, 1});
    attrs.set_dq(dq);
    attrs.set_dk(std::make_shared<TensorAttributes>());
    attrs.set_dv(std::make_shared<TensorAttributes>());

    GraphAttributes graphAttrs;
    SdpaBpropNode node(std::move(attrs), graphAttrs);
    node.infer_properties_node();

    EXPECT_EQ(dq->get_dim(), (std::vector<int64_t>{2, 8, 16, 64}));
    EXPECT_EQ(dq->get_stride(), (std::vector<int64_t>{8192, 1024, 64, 1}));
}
