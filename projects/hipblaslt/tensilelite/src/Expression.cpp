// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <Tensile/Expression.hpp>

#include <exprtk.hpp>
#include <libdivide.h>

namespace TensileLite
{
    using expression_t   = exprtk::expression<expression_compute_t>;
    using function_t     = exprtk::ifunction<expression_compute_t>;
    using parser_t       = exprtk::parser<expression_compute_t>;
    using symbol_table_t = exprtk::symbol_table<expression_compute_t>;

    struct magic_shifts final : public function_t
    {
        magic_shifts()
            : function_t(1)
        {
            exprtk::disable_has_side_effects(*this);
        }

        inline expression_compute_t operator()(expression_compute_t const& arg) override
        {
            if(arg == 0)
                return 0;

            // When the divisor is 1, we set the MSB of MagicShift to 1 so we can detect this case
            // by checking the MSB.
            if(arg == 1)
                return 1 << 31;

            auto magic = libdivide::libdivide_u32_branchfree_gen(arg);

            return magic.more & libdivide::LIBDIVIDE_32_SHIFT_MASK;
        }

        std::string const name = "magic_shifts";
    };

    struct magic_multiple_uint32 final : public function_t
    {
        magic_multiple_uint32()
            : function_t(1)
        {
            exprtk::disable_has_side_effects(*this);
        }

        inline expression_compute_t operator()(expression_compute_t const& arg) override
        {
            if(arg == 0)
                return static_cast<expression_compute_t>(std::numeric_limits<uint32_t>::max() / 2);

            // When the divisor is 1, this constant has no use, so return 0.
            if(arg == 1)
                return 0;

            auto magic = libdivide::libdivide_u32_branchfree_gen(arg);

            return magic.magic;
        }

        std::string const name = "magic_multiple_uint32";
    };

    expression_compute_t processExpression(
        std::string const&                                                exprStr,
        std::vector<std::tuple<std::string, expression_compute_t>> const& argValues)
    {
        symbol_table_t symbolTable;
        for(auto const& entry : argValues)
        {
            symbolTable.add_constant(std::get<0>(entry), std::get<1>(entry));
        }

        magic_multiple_uint32 fn0;
        symbolTable.add_function(fn0.name, fn0);
        magic_shifts fn1;
        symbolTable.add_function(fn1.name, fn1);

        expression_t expression;
        expression.register_symbol_table(symbolTable);

        parser_t parser;

        if(!parser.compile(exprStr, expression))
        {
            throw std::runtime_error("Invalid expression string: " + exprStr);
        }

        return expression.value();
    }
}
