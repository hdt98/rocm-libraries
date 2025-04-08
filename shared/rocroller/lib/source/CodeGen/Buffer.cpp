// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/CodeGen/Arithmetic/ArithmeticGenerator.hpp>
#include <rocRoller/CodeGen/Buffer.hpp>
#include <rocRoller/CodeGen/CopyGenerator.hpp>
#include <rocRoller/Context.hpp>
#include <rocRoller/Expression.hpp>

namespace rocRoller
{
    namespace BufferDescriptor
    {
        using Expression::ExpressionPtr;

        ExpressionPtr GetDefaultOptions(ContextPtr ctx)
        {
            AssertFatal(ctx, "Context cannot be null.");

        Throw<FatalError>("Invalid DataFormatValue: " + ShowValue((int)(val)));
    }

    std::ostream& operator<<(std::ostream& stream, GFX9BufferDescriptorOptions::DataFormatValue val)
    {
        return stream << toString(val);
    }

    GFX9BufferDescriptorOptions::GFX9BufferDescriptorOptions()
    {
        setRawValue(0);
        data_format = DF32;
    }

    GFX9BufferDescriptorOptions::GFX9BufferDescriptorOptions(uint32_t raw)
    {
        setRawValue(raw);
    }

    void GFX9BufferDescriptorOptions::setRawValue(uint32_t val)
    {
        static_assert(sizeof(val) == sizeof(*this));

        memcpy(this, &val, sizeof(val));

        validate();
    }

    void GFX9BufferDescriptorOptions::validate() const
    {
        AssertFatal(_unusedA == 0 && _unusedB == 0, "Reserved bits must be set to 0\n", toString());
        AssertFatal(type == 0, "Resource type must be set to 0 for buffers\n", toString());
    }

    uint32_t GFX9BufferDescriptorOptions::rawValue() const
    {
        uint32_t rv;

        static_assert(sizeof(rv) == sizeof(*this));

        memcpy(&rv, this, sizeof(rv));
        return rv;
    }

    Register::ValuePtr GFX9BufferDescriptorOptions::literal() const
    {
        return Register::Value::Literal(rawValue());
    }

    std::string GFX9BufferDescriptorOptions::toString() const
    {
        std::ostringstream msg;

        auto flags = msg.flags();
        msg << "GFX9BufferDescriptorOptions: " << std::showbase << std::hex << std::internal
            << std::setfill('0') << std::setw(2 + 32 / 4) << rawValue() << std::endl;
        msg.flags(flags);

        msg << "    dst_sel_x: " << dst_sel_x << std::endl;
        msg << "    dst_sel_y: " << dst_sel_y << std::endl;
        msg << "    dst_sel_z: " << dst_sel_z << std::endl;
        msg << "    dst_sel_w: " << dst_sel_w << std::endl;
        msg << "    num_format: " << num_format << std::endl;
        msg << "    data_format: " << data_format << std::endl;
        msg << "    user_vm_enable: " << user_vm_enable << std::endl;
        msg << "    user_vm_mode: " << user_vm_mode << std::endl;
        msg << "    index_stride: " << index_stride << std::endl;
        msg << "    add_tid_enable: " << add_tid_enable << std::endl;
        msg << "    _unusedA: " << _unusedA << std::endl;
        msg << "    nv: " << nv << std::endl;
        msg << "    _unusedB: " << _unusedB << std::endl;
        msg << "    type: " << type << std::endl;

        return msg.str();
    }

    /*
     * Creates buffer descriptor object from existing SGPRs
     */

    BufferDescriptor::BufferDescriptor(Register::ValuePtr srd, ContextPtr context)
    {
        m_bufferResourceDescriptor = srd;
        m_context                  = context;
    }

    /*
     * Creates buffer descriptor object from context, no existing SGPRs
     * Requires the use of the BufferDescriptor::setup()
     */
    BufferDescriptor::BufferDescriptor(ContextPtr context)
    {
        VariableType bufferPointer{DataType::None, PointerType::Buffer};
        m_bufferResourceDescriptor
            = std::make_shared<Register::Value>(context, Register::Type::Scalar, bufferPointer, 1);
        m_context = context;
    }

    Generator<Instruction> BufferDescriptor::setup()
    {
        co_yield m_context->copier()->copy(
            m_bufferResourceDescriptor->subset({2}), Register::Value::Literal(2147483548), "");
        co_yield setDefaultOpts();
    }

    uint32_t BufferDescriptor::getDefaultOptionsValue(ContextPtr ctx)
    {
        if(ctx->targetArchitecture().HasCapability(GPUCapability::HasBufferOutOfBoundsCheckOption))
        {
            // Bits 29:28 are for Out-of-Bounds check.
            //   0 - index >= NumRecords || offset + payload > stride, used for structured buffers.
            //   1 - index >= NumRecords, used for raw buffers (RR default)
            //   2 - NumRecords == 0, empty buffers
            //
            // Bits 17:12 are for data format.
            //   5 - 8_UINT. Currently, everything is buffer-loaded in terms of bytes.
            // TODO: Add GFX12 buffer descriptor when other formats and/or features are needed.
            uint32_t outOfBoundsCheck = (1u << 28);
            uint32_t dataFormat       = (5u << 12);
            if(ctx->targetArchitecture().HasCapability(
                   GPUCapability::HasBufferFormatSpecInSOffsetField))
            {
                // 0 - index >= NumRecords, used for raw buffers (RR default)
                // 1 - index >= NumRecords || offset + payload > stride, used for structured buffers.
                // 2 - NumRecords == 0, empty buffers
                outOfBoundsCheck = 0;
                dataFormat       = 0;
            }
            return outOfBoundsCheck | dataFormat;
        }

        ExpressionPtr SetDefaults(ExpressionPtr bufferExpr, ContextPtr ctx)
        {
            AssertFatal(bufferExpr && ctx, "Buffer and context cannot be null.");
            auto exprVarType = resultVariableType(bufferExpr);
            AssertFatal(exprVarType.pointerType == PointerType::Buffer,
                        "Buffer expression must be of buffer pointer type. ",
                        ShowValue(exprVarType));

            bufferExpr = BufferDescriptor::SetSize(
                bufferExpr, Expression::literal(2147483548u, DataType::UInt32));
            bufferExpr = BufferDescriptor::SetOptions(bufferExpr, GetDefaultOptions(ctx));
            return bufferExpr;
        }

    Generator<Instruction> BufferDescriptor::setBasePointer(Register::ValuePtr value)
    {
        if(m_context->targetArchitecture().HasCapability(
               GPUCapability::HasBufferFormatSpecInSOffsetField))
        {
            // s1[24:0] s0[31:0] 57-bit Base byte address.
            auto s1s0 = m_bufferResourceDescriptor->subset({0, 1});
            if(s1s0->allocationState() == Register::AllocationState::Unallocated)
            {
                // if unallocated then no higher 7 bits from size to care about.
                s1s0->allocateNow();
                co_yield m_context->copier()->copy(
                    m_bufferResourceDescriptor->subset({0, 1}), value, "");
            }
            else
            {
                auto tmp = Register::Value::Placeholder(
                    m_context, Register::Type::Scalar, DataType::UInt64, 1);

                auto clearUpper7BitsMask = Register::Value::Literal((1ull << 57) - 1ull);
                clearUpper7BitsMask->setVariableType(DataType::UInt64);
                auto clearLower57BitsMask = Register::Value::Literal(0xFEull << 56);
                clearLower57BitsMask->setVariableType(DataType::UInt64);

                co_yield generateOp<Expression::BitwiseAnd>(tmp, value, clearUpper7BitsMask);
                co_yield generateOp<Expression::BitwiseAnd>(s1s0, s1s0, clearLower57BitsMask);
                co_yield generateOp<Expression::BitwiseOr>(s1s0, s1s0, tmp);
            }
        }
        else
        {
            co_yield m_context->copier()->copy(
                m_bufferResourceDescriptor->subset({0, 1}), value, "");
        }
    }

    Generator<Instruction> BufferDescriptor::setSize(Register::ValuePtr value)
    {
        AssertFatal(value->variableType().getElementSize() == 4,
                    "Sizes with more than 32 bits are not supported yet.");

        if(m_context->targetArchitecture().HasCapability(
               GPUCapability::HasBufferFormatSpecInSOffsetField))
        {
            // s3[5:0] s2[31:0] s1[31:25] 45-bit numRecords
            auto s1 = m_bufferResourceDescriptor->subset({1});
            auto s2 = m_bufferResourceDescriptor->subset({2});

            auto tmp = Register::Value::Placeholder(
                m_context, Register::Type::Scalar, DataType::UInt32, 1);

            // s1[32:25]
            co_yield m_context->copier()->copy(tmp, value, "");
            co_yield Expression::generate(tmp, tmp->bitfield(0, 7)->expression(), m_context);
            co_yield generateOp<Expression::ShiftL>(tmp, tmp, Register::Value::Literal(25));
            co_yield generateOp<Expression::BitwiseAnd>(
                s1, s1, Register::Value::Literal(0x01FFFFFF));
            co_yield generateOp<Expression::BitwiseOr>(s1, s1, tmp);

            // s2[24:0]
            co_yield m_context->copier()->copy(s2, value, "");
            co_yield Expression::generate(s2, s2->bitfield(7, 25)->expression(), m_context);
            co_yield generateOp<Expression::BitwiseAnd>(
                s2, s2, Register::Value::Literal(0x07FFFFFF));
        }
        else
        {
            co_yield m_context->copier()->copy(m_bufferResourceDescriptor->subset({2}), value, "");
        }
    }

            return bfc(addressExpr, bufferExpr, 0, 0, 64);
        }

        ExpressionPtr GetBasePointer(ExpressionPtr bufferExpr)
        {
            AssertFatal(bufferExpr, "Buffer expression cannot be null.");
            auto exprVarType = resultVariableType(bufferExpr);
            AssertFatal(exprVarType.pointerType == PointerType::Buffer,
                        "Buffer expression must be of buffer pointer type. ",
                        ShowValue(exprVarType));

            return bfe(DataType::UInt64, bufferExpr, 0, 64);
        }

        ExpressionPtr IncrementBasePointer(ExpressionPtr bufferExpr, ExpressionPtr valueExpr)
        {
            AssertFatal(bufferExpr && valueExpr, "Buffer and value expressions cannot be null.");
            auto exprVarType = resultVariableType(bufferExpr);
            AssertFatal(exprVarType.pointerType == PointerType::Buffer,
                        "Buffer expression must be of buffer pointer type. ",
                        ShowValue(exprVarType));

            auto basePointer = bfe(DataType::UInt64, bufferExpr, 0, 64);
            return bfc(basePointer + valueExpr, bufferExpr, 0, 0, 64);
        }

        ExpressionPtr SetSize(ExpressionPtr bufferExpr, ExpressionPtr sizeExpr)
        {
            AssertFatal(bufferExpr && sizeExpr, "Buffer and size expressions cannot be null.");
            auto exprVarType = resultVariableType(bufferExpr);
            AssertFatal(exprVarType.pointerType == PointerType::Buffer,
                        "Buffer expression must be of buffer pointer type. ",
                        ShowValue(exprVarType));

            return bfc(sizeExpr, bufferExpr, 0, 64, 32);
        }

        ExpressionPtr GetSize(ExpressionPtr bufferExpr)
        {
            AssertFatal(bufferExpr, "Buffer expression cannot be null.");
            auto exprVarType = resultVariableType(bufferExpr);
            AssertFatal(exprVarType.pointerType == PointerType::Buffer,
                        "Buffer expression must be of buffer pointer type. ",
                        ShowValue(exprVarType));

            return bfe(DataType::UInt32, bufferExpr, 64, 32);
        }

        ExpressionPtr SetOptions(ExpressionPtr bufferExpr, ExpressionPtr optsExpr)
        {
            AssertFatal(bufferExpr && optsExpr, "Buffer and options expressions cannot be null.");
            auto exprVarType = resultVariableType(bufferExpr);
            AssertFatal(exprVarType.pointerType == PointerType::Buffer,
                        "Buffer expression must be of buffer pointer type. ",
                        ShowValue(exprVarType));

            return bfc(optsExpr, bufferExpr, 0, 96, 32);
        }

        ExpressionPtr GetOptions(ExpressionPtr bufferExpr)
        {
            AssertFatal(bufferExpr, "Buffer expression cannot be null.");
            auto exprVarType = resultVariableType(bufferExpr);
            AssertFatal(exprVarType.pointerType == PointerType::Buffer,
                        "Buffer expression must be of buffer pointer type. ",
                        ShowValue(exprVarType));

            return bfe(DataType::UInt32, bufferExpr, 96, 32);
        }
    }
}
