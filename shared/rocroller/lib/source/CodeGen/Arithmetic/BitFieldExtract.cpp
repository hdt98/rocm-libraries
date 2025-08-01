// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/CodeGen/Arithmetic/ArithmeticGenerator.hpp>
#include <rocRoller/CodeGen/Arithmetic/BitFieldExtract.hpp>
#include <rocRoller/CodeGen/CopyGenerator.hpp>
#include <rocRoller/Utilities/Component.hpp>

namespace rocRoller
{
    RegisterComponentTemplateSpec(BitFieldExtractGenerator, DataType::Half);
    RegisterComponentTemplateSpec(BitFieldExtractGenerator, DataType::BFloat16);
    RegisterComponentTemplateSpec(BitFieldExtractGenerator, DataType::FP8);
    RegisterComponentTemplateSpec(BitFieldExtractGenerator, DataType::BF8);
    RegisterComponentTemplateSpec(BitFieldExtractGenerator, DataType::FP6);
    RegisterComponentTemplateSpec(BitFieldExtractGenerator, DataType::BF6);
    RegisterComponentTemplateSpec(BitFieldExtractGenerator, DataType::FP4);
    RegisterComponentTemplateSpec(BitFieldExtractGenerator, DataType::Int8);
    RegisterComponentTemplateSpec(BitFieldExtractGenerator, DataType::Int16);
    RegisterComponentTemplateSpec(BitFieldExtractGenerator, DataType::Int32);
    RegisterComponentTemplateSpec(BitFieldExtractGenerator, DataType::Int64);
    RegisterComponentTemplateSpec(BitFieldExtractGenerator, DataType::Raw32);
    RegisterComponentTemplateSpec(BitFieldExtractGenerator, DataType::UInt8);
    RegisterComponentTemplateSpec(BitFieldExtractGenerator, DataType::UInt16);
    RegisterComponentTemplateSpec(BitFieldExtractGenerator, DataType::UInt32);
    RegisterComponentTemplateSpec(BitFieldExtractGenerator, DataType::UInt64);
    RegisterComponentTemplateSpec(BitFieldExtractGenerator, DataType::E8M0);
    RegisterComponentTemplateSpec(BitFieldExtractGenerator, DataType::E5M3);
    RegisterComponentTemplateSpec(BitFieldExtractGenerator, DataType::E4M3);

    template <>
    std::shared_ptr<UnaryArithmeticGenerator<Expression::BitFieldExtract>> GetGenerator(
        Register::ValuePtr dst, Register::ValuePtr arg, Expression::BitFieldExtract const&)
    {
        return Component::Get<UnaryArithmeticGenerator<Expression::BitFieldExtract>>(
            getContextFromValues(dst, arg), dst->regType(), dst->variableType().dataType);
    }
}
