// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <Tensile/Expression.hpp>

#include <exprtk.hpp>

namespace TensileLite
{
    using exprtk::symbol_table<expression_compute_t> symbol_table_t;
    using exprtk::expression<expression_compute_t>   expression_t;
    using exprtk::parser<expression_compute_t>       parser_t;

    expression_compute_t processExpression(
        std::string const&                                                exprStr,
        std::vector<std::tuple<std::string, expression_compute_t>> const& argValues)
    {
        symbol_table_t symbolTable;
        for(auto const& entry : argValues)
        {
            symbolTable.add_variable(entry.get<0>(), entry.get<1>());
        }

        expression_t expression;
        expression.register_symbol_table(symbolTable);

        parser_t parser;

        if(!parser.compile(exprStr, expression))
        {
            throw std::runtime_error(concatenate("Invalid expression string: ", exprStr));
        }

        return expression.value();
    }
}
