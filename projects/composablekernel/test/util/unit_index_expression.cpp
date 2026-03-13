// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "ck/utility/index_expression.hpp"

using namespace ck;

TEST(IndexExpression, EvalLiteralAndIk)
{
    static_assert(index_expr::eval_v<Number<7>, 3> == 7);
    static_assert(index_expr::eval_v<index_expr::Ik, 3> == 3);
    EXPECT_EQ((index_expr::eval_v<Number<7>, 3>), 7);
    EXPECT_EQ((index_expr::eval_v<index_expr::Ik, 3>), 3);
}

TEST(IndexExpression, EvalAddMultDivMod)
{
    using ExprAdd  = index_expr::Add<index_expr::Ik, Number<5>>;
    using ExprMult = index_expr::Mult<ExprAdd, Number<2>>;
    using ExprDiv  = index_expr::Div<ExprMult, Number<4>>;
    using ExprMod  = index_expr::Mod<ExprMult, Number<3>>;

    static_assert(index_expr::eval_v<ExprAdd, 3> == 8);
    static_assert(index_expr::eval_v<ExprMult, 3> == 16);
    static_assert(index_expr::eval_v<ExprDiv, 3> == 4);
    static_assert(index_expr::eval_v<ExprMod, 3> == 1);

    EXPECT_EQ((index_expr::eval_v<ExprAdd, 3>), 8);
    EXPECT_EQ((index_expr::eval_v<ExprMult, 3>), 16);
    EXPECT_EQ((index_expr::eval_v<ExprDiv, 3>), 4);
    EXPECT_EQ((index_expr::eval_v<ExprMod, 3>), 1);
}

TEST(IndexExpression, EvalNestedExpression)
{
    using Expr =
        index_expr::Div<index_expr::Add<index_expr::Ik, index_expr::Mult<Number<2>, Number<5>>>,
                        Number<2>>;

    static_assert(index_expr::eval_v<Expr, 6> == 8);
    EXPECT_EQ((index_expr::eval_v<Expr, 6>), 8);
}
