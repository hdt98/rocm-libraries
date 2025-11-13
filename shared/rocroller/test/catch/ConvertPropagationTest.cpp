/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <cmath>
#include <memory>

#include "CustomMatchers.hpp"
#include "CustomSections.hpp"
#include "TestContext.hpp"
#include "TestKernels.hpp"

#include <common/SourceMatcher.hpp>
#include <common/TestValues.hpp>

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/CodeGen/CopyGenerator.hpp>
#include <rocRoller/CodeGen/Instruction.hpp>
#include <rocRoller/CodeGen/MemoryInstructions.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/ExpressionTransformations.hpp>

#include <catch2/catch_test_macros.hpp>

#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

using namespace rocRoller;

namespace ExpressionTest
{
    struct ConvertExpressionKernel : public AssemblyTestKernel
    {
        using ExpressionFunc = std::function<Expression::ExpressionPtr(
            Expression::ExpressionPtr, Expression::ExpressionPtr, Expression::ExpressionPtr)>;
        ConvertExpressionKernel(ContextPtr     context,
                                ExpressionFunc func,
                                DataType       resultType,
                                DataType       aType,
                                DataType       bType,
                                DataType       cType)
            : AssemblyTestKernel(context)
            , m_func(func)
            , m_resultType(resultType)
            , m_aType(aType)
            , m_bType(bType)
            , m_cType(cType)

        {
        }

        void generate() override
        {
            auto k = m_context->kernel();

            k->addArgument(
                {"result", {m_resultType, PointerType::PointerGlobal}, DataDirection::WriteOnly});
            k->addArgument({"a", m_aType});
            k->addArgument({"b", m_bType});
            k->addArgument({"c", m_cType});

            m_context->schedule(k->preamble());
            m_context->schedule(k->prolog());

            auto kb = [&]() -> Generator<Instruction> {
                Register::ValuePtr s_result, s_a, s_b, s_c;
                co_yield m_context->argLoader()->getValue("result", s_result);
                co_yield m_context->argLoader()->getValue("a", s_a);
                co_yield m_context->argLoader()->getValue("b", s_b);
                co_yield m_context->argLoader()->getValue("c", s_c);

                auto v_result
                    = Register::Value::Placeholder(m_context,
                                                   Register::Type::Vector,
                                                   {m_resultType, PointerType::PointerGlobal},
                                                   1);

                auto v_a = s_a->placeholder(Register::Type::Vector, {});
                auto v_b = s_b->placeholder(Register::Type::Vector, {});
                auto v_c = s_c->placeholder(Register::Type::Vector, {});

                auto a    = v_a->expression();
                auto b    = v_b->expression();
                auto c    = v_c->expression();
                auto expr = m_func(a, b, c);
                Log::info("Original expr = {}", toString(expr));

                co_yield v_a->allocate();
                co_yield v_b->allocate();
                co_yield v_result->allocate();

                co_yield m_context->copier()->copy(v_result, s_result, "Move pointer");

                co_yield m_context->copier()->copy(v_a, s_a, "Move pointer");
                co_yield m_context->copier()->copy(v_b, s_b, "Move pointer");
                co_yield m_context->copier()->copy(v_c, s_c, "Move pointer");

                Register::ValuePtr v_d;
                co_yield Expression::generate(v_d, expr, m_context);

                co_yield m_context->mem()->storeGlobal(
                    v_result, v_d, 0, DataTypeInfo::Get(m_resultType).elementBytes);
            };

            m_context->schedule(kb());
            m_context->schedule(k->postamble());
            m_context->schedule(k->amdgpu_metadata());
        }

    protected:
        ExpressionFunc m_func;
        DataType       m_resultType, m_aType, m_bType, m_cType;
    };

    TEST_CASE("Convert propagation with Subtraction", "[b123][gpu][convert-propagation]")
    {
        auto context = TestContext::ForTestDevice();

        auto expr
            = [](Expression::ExpressionPtr a,
                 Expression::ExpressionPtr b,
                 Expression::ExpressionPtr c) { return convert(DataType::Int32, (a - b) + c); };

        ConvertExpressionKernel kernel(context.get(),
                                       expr,
                                       DataType::Int32,
                                       DataType::Int64,
                                       DataType::Int64,
                                       DataType::Int64);

        auto d_result = make_shared_device<int32_t>();

        // A borrow occurs when doing (a-b) in 64-bit
        int64_t a = 0x0000'0001'6fff'ffff;
        int64_t b = 0x0000'0000'7fff'ffff;
        int32_t c = 0;

        std::cout << a << std::endl;

        int32_t r = int32_t(int64_t(a - b) + c);

        kernel({}, d_result.get(), a, b, c);

        int32_t result;

        hipMemcpy(&result, d_result.get(), sizeof(int32_t), hipMemcpyDefault);

        std::cout << r << " / " << result << "\n";

        CHECK_THAT(d_result, HasDeviceScalarEqualTo(r));

        auto const assembly = NormalizedSource(context.output());

        // Should use a 32-bit subtraction instruction and then 32-bit addition
        CHECK_THAT(assembly,
                   Catch::Matchers::ContainsSubstring("v_sub_i32")
                       or Catch::Matchers::ContainsSubstring("v_sub_nc_i32"));
        CHECK_THAT(assembly,
                   Catch::Matchers::ContainsSubstring("v_add_i32")
                       or Catch::Matchers::ContainsSubstring("v_add_nc_i32"));
    }

    TEST_CASE("Convert propagation with ArithmeticShiftR", "[x123][gpu][convert-propagation]")
    {
        auto context = TestContext::ForTestDevice();

        auto expr
            = [](Expression::ExpressionPtr a,
                 Expression::ExpressionPtr b,
                 Expression::ExpressionPtr c) { return convert(DataType::Int32, (a >> b) + c); };

        ConvertExpressionKernel kernel(context.get(),
                                       expr,
                                       DataType::Int32,
                                       DataType::Int64,
                                       DataType::Int64,
                                       DataType::Int64);

        auto d_result = make_shared_device<int32_t>();

        int64_t a = int64_t(std::numeric_limits<int32_t>::max()) * int64_t(4);
        int64_t b = 6;
        int32_t c = 0;

        std::cout << a << std::endl;

        int32_t r = int32_t(int64_t(a >> b) + c);

        kernel({}, d_result.get(), a, b, c);

        int32_t result;

        hipMemcpy(&result, d_result.get(), sizeof(int32_t), hipMemcpyDefault);

        std::cout << r << " / " << result << "\n";

        CHECK_THAT(d_result, HasDeviceScalarEqualTo(r));

        auto const assembly = NormalizedSource(context.output());

        // Should use a 64-bit arithmetic shift instruction and then 32-bit addition
        CHECK_THAT(assembly, Catch::Matchers::ContainsSubstring("v_ashrrev_i64"));
        CHECK_THAT(assembly,
                   Catch::Matchers::ContainsSubstring("v_add_i32")
                       or Catch::Matchers::ContainsSubstring("v_add_nc_i32"));

        CHECK_THAT(assembly, not Catch::Matchers::ContainsSubstring("v_addc_co_u32"));
    }

    TEST_CASE("Convert propagation with LogicalShiftR", "[r123][gpu][convert-propagation]")
    {
        auto context = TestContext::ForTestDevice();

        auto expr = [](Expression::ExpressionPtr a,
                       Expression::ExpressionPtr b,
                       Expression::ExpressionPtr c) {
            return convert(DataType::UInt32, logicalShiftR(a, b) + c);
        };

        ConvertExpressionKernel kernel(context.get(),
                                       expr,
                                       DataType::UInt32,
                                       DataType::UInt64,
                                       DataType::UInt64,
                                       DataType::UInt64);

        auto d_result = make_shared_device<uint32_t>();

        uint64_t a = uint64_t(std::numeric_limits<uint32_t>::max()) + uint64_t(1);
        uint64_t b = 6;
        uint32_t c = 0;

        std::cout << a << std::endl;

        uint32_t r = uint32_t(uint64_t(a >> b) + c);

        kernel({}, d_result.get(), a, b, c);

        uint32_t result;

        hipMemcpy(&result, d_result.get(), sizeof(uint32_t), hipMemcpyDefault);

        std::cout << r << " / " << result << "\n";

        CHECK_THAT(d_result, HasDeviceScalarEqualTo(r));

        // Should use a 64-bit logical shift instruction and then 32-bit addition
        auto const assembly = NormalizedSource(context.output());
        CHECK_THAT(assembly, Catch::Matchers::ContainsSubstring("v_lshrrev_b64"));
        CHECK_THAT(assembly,
                   Catch::Matchers::ContainsSubstring("v_add_u32")
                       or Catch::Matchers::ContainsSubstring("v_add_nc_u32"));

        CHECK_THAT(assembly, not Catch::Matchers::ContainsSubstring("v_addc_co_u32"));
    }

    TEST_CASE("Convert propagation with Division", "[q123][gpu][convert-propagation]")
    {
        auto context = TestContext::ForTestDevice({{.enableFullDivision = true}});

        auto expr
            = [](Expression::ExpressionPtr a,
                 Expression::ExpressionPtr b,
                 Expression::ExpressionPtr c) { return convert(DataType::Int32, (a / b) + c); };

        ConvertExpressionKernel kernel(context.get(),
                                       expr,
                                       DataType::Int32,
                                       DataType::Int64,
                                       DataType::Int64,
                                       DataType::Int64);

        auto d_result = make_shared_device<int32_t>();

        int64_t a = (int64_t(std::numeric_limits<int32_t>::max()) << 2) + int64_t(0xffff);
        int64_t b = 10;
        int64_t c = 1;

        std::cout << a << std::endl;

        int32_t r = int32_t(int64_t(a / b) + c);

        kernel({}, d_result.get(), a, b, c);

        int32_t result;

        hipMemcpy(&result, d_result.get(), sizeof(int32_t), hipMemcpyDefault);

        std::cout << r << " / " << result << "\n";

        auto const assembly = NormalizedSource(context.output());

        CHECK_THAT(d_result, HasDeviceScalarEqualTo(r));

        // Should not do sign extension and should do 32-bit addition
        CHECK_THAT(assembly, Catch::Matchers::ContainsSubstring("v_ashrrev_i32"));
        CHECK_THAT(assembly, Catch::Matchers::ContainsSubstring("v_add_i32"));
    }

}
