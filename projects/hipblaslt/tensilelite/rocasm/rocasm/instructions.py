################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

"""Instruction functions that return PendingOp objects.

Each function corresponds to a single ISA instruction. The function captures
the source operands and returns a PendingOp; the destination is captured by
the __setitem__ on the register array.

Usage:
    Acc[0:4] = vmfma_f32_16x16x32_bf16(B[0:4], A[0:4], Acc[0:4])
"""

from __future__ import annotations

from rocisa.enum import InstType
from rocisa.instruction import (
    MFMAInstruction,
    DSLoadB128 as _DSLoadB128,
    DSStoreB128 as _DSStoreB128,
    DSStoreB32 as _DSStoreB32,
    BufferLoadB128 as _BufferLoadB128,
)

from rocasm.ops import Op, PendingOp
from rocasm.regs import RegisterSlice


def vmfma_f32_16x16x32_bf16(src_b: RegisterSlice, src_a: RegisterSlice,
                            acc2: RegisterSlice) -> PendingOp:
    """v_mfma_f32_16x16x32_bf16: BF16 matrix fused multiply-add.

    dst = A * B + acc2

    Produces 4 F32 accumulator registers from 4 BF16 A-source and 4 BF16 B-source
    registers, added to the acc2 accumulator input.

    Args:
        src_b: B matrix tile (4 VGPRs, BF16)
        src_a: A matrix tile (4 VGPRs, BF16)
        acc2: Accumulator input to add to (4 AccVGPRs, F32)

    Returns:
        PendingOp to be resolved by assignment to Acc[dst_start:dst_end]
    """
    # Both sources must have a block reference through their arrays
    block = src_b.array.block
    if block is None:
        block = src_a.array.block
    if block is None:
        block = acc2.array.block
    if block is None:
        raise RuntimeError(
            "vmfma_f32_16x16x32_bf16: source registers are not attached to a Block")

    def _build(dst: RegisterSlice, srcs: list[RegisterSlice]) -> MFMAInstruction:
        return MFMAInstruction(
            instType=InstType.INST_BF16,
            accType=InstType.INST_F32,
            variant=[16, 16, 32, 1],
            mfma1k=False,
            acc=dst.container(),
            a=srcs[0].container(),
            b=srcs[1].container(),
            acc2=srcs[2].container(),
        )

    return PendingOp(
        inst_name="v_mfma_f32_16x16x32_bf16",
        srcs=[src_b, src_a, acc2],
        build_fn=_build,
        block=block,
    )


def vmfma_f32_16x16x16bf16_1k(src_b: RegisterSlice, src_a: RegisterSlice,
                               acc2: RegisterSlice) -> PendingOp:
    """v_mfma_f32_16x16x16bf16_1k: BF16 matrix fused multiply-add (gfx942/CDNA3).

    dst = A * B + acc2

    Produces 4 F32 accumulator registers from 2 BF16 A-source and 2 BF16 B-source
    registers, added to the acc2 accumulator input.

    Args:
        src_b: B matrix tile (2 VGPRs, BF16)
        src_a: A matrix tile (2 VGPRs, BF16)
        acc2: Accumulator input to add to (4 AccVGPRs, F32)

    Returns:
        PendingOp to be resolved by assignment to Acc[dst_start:dst_end]
    """
    block = src_b.array.block
    if block is None:
        block = src_a.array.block
    if block is None:
        block = acc2.array.block
    if block is None:
        raise RuntimeError(
            "vmfma_f32_16x16x16bf16_1k: source registers are not attached to a Block")

    def _build(dst: RegisterSlice, srcs: list[RegisterSlice]) -> MFMAInstruction:
        return MFMAInstruction(
            instType=InstType.INST_BF16,
            accType=InstType.INST_F32,
            variant=[16, 16, 16, 1],
            mfma1k=True,
            acc=dst.container(),
            a=srcs[0].container(),
            b=srcs[1].container(),
            acc2=srcs[2].container(),
        )

    return PendingOp(
        inst_name="v_mfma_f32_16x16x16bf16_1k",
        srcs=[src_b, src_a, acc2],
        build_fn=_build,
        block=block,
    )


def ds_read_b128(src: RegisterSlice, ds=None, comment: str = "") -> PendingOp:
    """ds_read_b128 / ds_load_b128: Load 128 bits (4 dwords) from LDS.

    Args:
        src: Address VGPR (1 VGPR)
        ds: Optional DSModifiers (offset, etc.)
        comment: Optional comment string

    Returns:
        PendingOp to be resolved by assignment to a VGPR slice (4 VGPRs)
    """
    block = src.array.block
    if block is None:
        raise RuntimeError(
            "ds_read_b128: source register is not attached to a Block")

    def _build(dst: RegisterSlice, srcs: list[RegisterSlice]):
        return _DSLoadB128(
            dst=dst.container(),
            src=srcs[0].container(),
            ds=ds,
            comment=comment,
        )

    return PendingOp(
        inst_name="ds_read_b128",
        srcs=[src],
        build_fn=_build,
        block=block,
    )


def buffer_load_dwordx4(vaddr: RegisterSlice, saddr: RegisterSlice,
                        soffset, mubuf=None, comment: str = "") -> PendingOp:
    """buffer_load_dwordx4 / buffer_load_b128: Load 128 bits from global memory via buffer.

    Args:
        vaddr: Address VGPR (offset within buffer)
        saddr: SGPR resource descriptor (4 SGPRs)
        soffset: Scalar offset (SGPR container or immediate)
        mubuf: Optional MUBUFModifiers (offen, offset12, etc.)
        comment: Optional comment string

    Returns:
        PendingOp to be resolved by assignment to a VGPR slice (4 VGPRs)
    """
    block = vaddr.array.block
    if block is None:
        block = saddr.array.block
    if block is None:
        raise RuntimeError(
            "buffer_load_dwordx4: source registers are not attached to a Block")

    def _build(dst: RegisterSlice, srcs: list[RegisterSlice]):
        return _BufferLoadB128(
            dst=dst.container(),
            vaddr=srcs[0].container(),
            saddr=srcs[1].container(),
            soffset=soffset,
            mubuf=mubuf,
            comment=comment,
        )

    return PendingOp(
        inst_name="buffer_load_dwordx4",
        srcs=[vaddr, saddr],
        build_fn=_build,
        block=block,
    )


def ds_write_b128(addr: RegisterSlice, src: RegisterSlice,
                  ds=None, comment: str = ""):
    """ds_write_b128 / ds_store_b128: Store 128 bits (4 dwords) to LDS.

    Side-effect instruction — no destination register, appends directly to the block.

    Args:
        addr: Address VGPR (1 VGPR)
        src: Data VGPRs to store (4 VGPRs)
        ds: Optional DSModifiers (offset, etc.)
        comment: Optional comment string
    """
    block = addr.array.block
    if block is None:
        block = src.array.block
    if block is None:
        raise RuntimeError(
            "ds_write_b128: source registers are not attached to a Block")

    def _build():
        return _DSStoreB128(
            dstAddr=addr.container(), src=src.container(),
            ds=ds, comment=comment)

    rocisa_inst = None
    if not block.is_virtual:
        rocisa_inst = _build()
    op = Op(inst="ds_write_b128", dst=None, srcs=[addr, src],
            rocisa_inst=rocisa_inst, build_fn=_build)
    block._append_op(op)


def ds_write_b32(addr: RegisterSlice, src: RegisterSlice,
                 ds=None, comment: str = ""):
    """ds_write_b32 / ds_store_b32: Store 32 bits (1 dword) to LDS.

    Side-effect instruction — no destination register, appends directly to the block.

    Args:
        addr: Address VGPR (1 VGPR)
        src: Data VGPR to store (1 VGPR)
        ds: Optional DSModifiers (offset, etc.)
        comment: Optional comment string
    """
    block = addr.array.block
    if block is None:
        block = src.array.block
    if block is None:
        raise RuntimeError(
            "ds_write_b32: source registers are not attached to a Block")

    def _build():
        return _DSStoreB32(
            dstAddr=addr.container(), src=src.container(),
            ds=ds, comment=comment)

    rocisa_inst = None
    if not block.is_virtual:
        rocisa_inst = _build()
    op = Op(inst="ds_write_b32", dst=None, srcs=[addr, src],
            rocisa_inst=rocisa_inst, build_fn=_build)
    block._append_op(op)
