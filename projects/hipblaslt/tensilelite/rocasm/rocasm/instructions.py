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
from rocisa.instruction import MFMAInstruction

from rocasm.ops import PendingOp
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
            a=srcs[1].container(),
            b=srcs[0].container(),
            acc2=srcs[2].container(),
        )

    return PendingOp(
        inst_name="v_mfma_f32_16x16x32_bf16",
        srcs=[src_b, src_a, acc2],
        build_fn=_build,
        block=block,
    )
