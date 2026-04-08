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

"""Op data structure: records what an instruction does.

Each Python statement in a rocasm function produces an Op that captures
the instruction name, destination, sources, and underlying rocisa object.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rocasm.regs import RegisterSlice


@dataclass
class Op:
    """A single instruction record."""
    inst: str                          # e.g. "v_mfma_f32_16x16x32_bf16"
    dst: RegisterSlice | None          # what it writes
    srcs: list[RegisterSlice]          # what it reads
    rocisa_inst: object                # the underlying rocisa Instruction


class PendingOp:
    """Intermediate object returned by instruction functions before destination is known.

    Created by e.g. vmfma_f32_16x16x32_bf16(B[0:4], A[0:4]).
    The __setitem__ on the LHS register array calls resolve() to finalize.
    """

    def __init__(self, inst_name: str, srcs: list[RegisterSlice],
                 build_fn, block: object):
        self._inst_name = inst_name
        self._srcs = srcs
        self._build_fn = build_fn
        self._block = block

    def resolve(self, dst: RegisterSlice):
        """Called by RegisterArray.__setitem__ to finalize the Op."""
        rocisa_inst = self._build_fn(dst, self._srcs)
        op = Op(
            inst=self._inst_name,
            dst=dst,
            srcs=list(self._srcs),
            rocisa_inst=rocisa_inst,
        )
        self._block._append_op(op)
