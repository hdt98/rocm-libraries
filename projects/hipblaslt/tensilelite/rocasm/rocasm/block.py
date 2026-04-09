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

"""Block: the central data structure combining register context with an instruction sequence.

A Block represents an instruction sequence within a specific register environment.
It has two parts:
  - Context: which registers hold what, their types, their names (immutable)
  - Ops: the instruction sequence (mutable)
"""

from __future__ import annotations

from copy import copy
from functools import partial
from typing import Any

from rocisa.instruction import SWaitCnt, SBarrier, SNop

from rocasm.ops import Op
from rocasm.regs import VgprArray, AccArray, SgprArray, _RegArray

# Side-effect instructions have no destination register.
_SIDE_EFFECT_INSTRUCTIONS: dict[str, type] = {
    "s_waitcnt": SWaitCnt,
    "s_barrier": SBarrier,
    "s_nop": SNop,
}


class Block:
    """A register context plus an instruction sequence.

    Create a Block by passing named register arrays as keyword arguments:

        block = Block(
            Acc=AccArray("Acc", base=0, count=64),
            A=VgprArray("A", base=24, count=8),
            B=VgprArray("B", base=0, count=8),
        )

    The register arrays are bound to this Block so that instructions
    written via __setitem__ (e.g. Acc[0:4] = vmfma(...)) automatically
    append Ops to this Block's op list.
    """

    def __init__(self, **regs: _RegArray):
        self._ops: list[Op] = []
        self._regs: dict[str, _RegArray] = {}

        for name, arr in regs.items():
            if not isinstance(arr, _RegArray):
                raise TypeError(
                    f"Block argument '{name}' must be a register array, "
                    f"got {type(arr).__name__}")
            # Bind the array to this block
            bound = copy(arr)
            bound.name = name
            bound.block = self
            self._regs[name] = bound

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        regs = object.__getattribute__(self, "_regs")
        if name in regs:
            return regs[name]
        if name in _SIDE_EFFECT_INSTRUCTIONS:
            return partial(self._side_effect, name)
        raise AttributeError(
            f"Block has no register array or side-effect instruction '{name}'")

    @property
    def ops(self) -> list[Op]:
        """The current instruction sequence (read-only view)."""
        return list(self._ops)

    def _append_op(self, op: Op):
        """Append an Op. Called by PendingOp.resolve()."""
        self._ops.append(op)

    def _side_effect(self, inst_name: str, *args, **kwargs):
        """Create and append an Op for a side-effect instruction (no destination).

        Called via closures: block.s_waitcnt(dscnt=0) → self._side_effect("s_waitcnt", dscnt=0)
        """
        cls = _SIDE_EFFECT_INSTRUCTIONS[inst_name]
        rocisa_inst = cls(*args, **kwargs)
        op = Op(
            inst=inst_name,
            dst=None,
            srcs=[],
            rocisa_inst=rocisa_inst,
        )
        self._ops.append(op)

    def new_body(self) -> Block:
        """Create a new Block with the same register context but no ops.

        The new block has its own op list. The original block is unchanged.
        """
        new = Block.__new__(Block)
        new._ops = []
        new._regs = {}
        for name, arr in self._regs.items():
            bound = copy(arr)
            bound.block = new
            new._regs[name] = bound
        return new

    def copy_body(self) -> Block:
        """Create a new Block with the same register context and a copy of the ops.

        The new block has its own (copied) op list. The original block is unchanged.
        """
        new = self.new_body()
        new._ops = list(self._ops)
        return new

    def emit(self) -> str:
        """Emit the instruction sequence as assembly text."""
        lines = []
        for op in self._ops:
            lines.append(str(op.rocisa_inst))
        return "".join(lines)

    def __len__(self):
        return len(self._ops)

    def __repr__(self):
        reg_desc = ", ".join(
            f"{name}={type(arr).__name__}(base={arr.base}, count={arr.count})"
            for name, arr in self._regs.items()
        )
        return f"Block({reg_desc}, ops={len(self._ops)})"
