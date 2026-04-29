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

from rocisa.code import Label
from rocisa.instruction import (
    SWaitCnt, SBarrier, SNop,
    SMovB32, SAddU32, SAddCU32, SSubU32, SSubBU32,
    SCmpEQU32, SCmpEQI32, SCSelectB32, SXorB32,
    VXorB32, SCBranchSCC0,
)

from rocasm.ops import Op
from rocasm.regs import VgprArray, AccArray, SgprArray, _RegArray

# Side-effect instructions have no destination register.
# All args/kwargs are passed directly through to the rocisa constructor.
_SIDE_EFFECT_INSTRUCTIONS: dict[str, type] = {
    "s_waitcnt": SWaitCnt,
    "s_barrier": SBarrier,
    "s_nop": SNop,
    "s_mov_b32": SMovB32,
    "s_add_u32": SAddU32,
    "s_addc_u32": SAddCU32,
    "s_sub_u32": SSubU32,
    "s_subb_u32": SSubBU32,
    "s_cmp_eq_u32": SCmpEQU32,
    "s_cmp_eq_i32": SCmpEQI32,
    "s_cselect_b32": SCSelectB32,
    "s_xor_b32": SXorB32,
    "v_xor_b32": VXorB32,
    "s_cbranch_scc0": SCBranchSCC0,
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
        if name == "label":
            return self._label
        if name in _SIDE_EFFECT_INSTRUCTIONS:
            return partial(self._side_effect, name)
        raise AttributeError(
            f"Block has no register array or side-effect instruction '{name}'")

    @property
    def is_virtual(self) -> bool:
        """True if any register array in this block has no physical base."""
        return any(arr.is_virtual for arr in self._regs.values())

    @property
    def ops(self) -> list[Op]:
        """The current instruction sequence (read-only view)."""
        return list(self._ops)

    def _append_op(self, op: Op):
        """Append an Op. Called by PendingOp.resolve()."""
        self._ops.append(op)

    def _label(self, name: str):
        """Emit an assembly label.

        If name starts with 'label_', the prefix is stripped since rocisa's
        Label class adds it back automatically.
        """
        label_name = name[6:] if name.startswith("label_") else name
        rocisa_label = None
        if not self.is_virtual:
            rocisa_label = Label(label_name, "")
        op = Op(
            inst="label",
            dst=None,
            srcs=[],
            rocisa_inst=rocisa_label,
            build_fn=Label,
            build_args=(label_name, ""),
        )
        self._ops.append(op)

    def _side_effect(self, inst_name: str, *args, **kwargs):
        """Create and append an Op for a side-effect instruction (no destination).

        Called via closures: block.s_waitcnt(dscnt=0) → self._side_effect("s_waitcnt", dscnt=0)

        Always stores build_fn so the Op can be re-materialized later.
        When physical, also builds the rocisa instruction eagerly.
        """
        cls = _SIDE_EFFECT_INSTRUCTIONS[inst_name]
        rocisa_inst = None
        if not self.is_virtual:
            rocisa_inst = cls(*args, **kwargs)
        op = Op(
            inst=inst_name,
            dst=None,
            srcs=[],
            rocisa_inst=rocisa_inst,
            build_fn=cls,
            build_args=args,
            build_kwargs=kwargs,
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

    def emit(self, register_map: dict[str, int] | None = None) -> str:
        """Emit the instruction sequence as assembly text.

        Args:
            register_map: Optional mapping of register array names to physical
                bases (e.g. ``{'A0': 16, 'B0': 48, 'Acc': 0}``).  Required
                when the block contains virtual register arrays.

        Raises:
            ValueError: If the block is virtual and no register_map is provided.
        """
        if register_map:
            for name, base in register_map.items():
                if name in self._regs:
                    self._regs[name].base = base
        if self.is_virtual:
            unresolved = [n for n, a in self._regs.items() if a.is_virtual]
            raise ValueError(
                f"Cannot emit: virtual arrays {unresolved} have no physical base. "
                "Provide a register_map covering all virtual arrays.")
        lines = []
        for op in self._ops:
            if op.rocisa_inst is None:
                op.materialize()
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


def virtualize(block: Block) -> dict[str, int]:
    """Convert a physical block to virtual by stripping physical register bases.

    Returns the register map (name -> physical base) so the block can be
    re-materialized later via ``block.emit(register_map=...)``.

    After this call, ``block.is_virtual`` is True and all ops have their
    ``rocisa_inst`` cleared (they will be rebuilt at emit time).

    Example::

        reg_map = virtualize(block)
        # block is now virtual -- inspect ops, reorder, etc.
        asm = block.emit(register_map=reg_map)  # round-trips to same assembly

    Args:
        block: A Block with physical register arrays.

    Returns:
        Dict mapping register array names to their original physical bases.

    Raises:
        ValueError: If the block is already virtual (some arrays have no base).
    """
    if block.is_virtual:
        unresolved = [n for n, a in block._regs.items() if a.is_virtual]
        raise ValueError(
            f"Cannot virtualize: arrays {unresolved} are already virtual")
    reg_map = {}
    for name, arr in block._regs.items():
        reg_map[name] = arr.base
        arr.base = None
    for op in block._ops:
        op.rocisa_inst = None
    return reg_map
