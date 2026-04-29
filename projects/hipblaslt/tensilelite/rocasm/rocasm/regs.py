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

"""Typed register arrays with Python slicing semantics.

Translates Python's exclusive slicing convention (e.g. Acc[12:16] = 4 registers)
to rocisa's physical register model (accvgpr(12, 4)).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from rocisa.container import vgpr, sgpr, accvgpr

if TYPE_CHECKING:
    from rocasm.block import Block


@dataclass(frozen=True)
class RegisterSlice:
    """A contiguous slice of a register array: base + count + type info.

    Supports sub-slicing for composability:
        Acc[16:32][0:4]  →  physical registers 16-19

    Also supports __setitem__ for the assignment protocol:
        sub = Acc[16:32]
        sub[0:4] = vmfma(...)  →  writes to physical 16-19
    """
    array: _RegArray
    start: int  # offset within the array
    count: int

    @property
    def phys_base(self) -> int:
        if self.array.is_virtual:
            raise ValueError(
                f"Cannot get phys_base from virtual array {self.array.name!r}. "
                "Assign physical bases via Block.emit(register_map=...)")
        return self.array.base + self.start

    def container(self):
        """Return the rocisa RegisterContainer for this slice.

        Raises ValueError if the array is virtual (no physical base assigned).
        """
        if self.array.is_virtual:
            raise ValueError(
                f"Cannot materialize container from virtual array {self.array.name!r}. "
                "Assign physical bases via Block.emit(register_map=...)")
        return self.array._make_container(self.phys_base, self.count)

    def _parse_sub_key(self, key) -> tuple[int, int]:
        """Parse a slice key into (start, count) within this sub-slice."""
        if isinstance(key, slice):
            start = key.start if key.start is not None else 0
            stop = key.stop if key.stop is not None else self.count
            if key.step is not None:
                raise IndexError("Register slices do not support step")
            if start < 0 or stop < 0:
                raise IndexError("Negative indices not supported for register slices")
            if stop > self.count:
                raise IndexError(
                    f"{self!r}[{start}:{stop}] exceeds slice size {self.count}")
            if start >= stop:
                raise IndexError(
                    f"{self!r}[{start}:{stop}] is empty (start >= stop)")
            return start, stop - start
        raise TypeError(f"Register slices require slice indexing, got {type(key).__name__}")

    def __getitem__(self, key) -> RegisterSlice:
        """Sub-slice: Acc[16:32][0:4] → RegisterSlice(array, start=16, count=4)."""
        sub_start, sub_count = self._parse_sub_key(key)
        return RegisterSlice(
            array=self.array,
            start=self.start + sub_start,
            count=sub_count,
        )

    def __setitem__(self, key, value):
        """Assignment protocol on sub-slices: sub[0:4] = vmfma(...)."""
        from rocasm.ops import PendingOp
        sub_start, sub_count = self._parse_sub_key(key)
        dst = RegisterSlice(
            array=self.array,
            start=self.start + sub_start,
            count=sub_count,
        )
        if isinstance(value, PendingOp):
            value.resolve(dst)
        else:
            raise TypeError(
                f"Cannot assign {type(value).__name__} to register slice. "
                "Use an instruction function (e.g. vmfma_f32_16x16x32_bf16(...))")

    def __repr__(self):
        end = self.start + self.count
        return f"{self.array.name}[{self.start}:{end}]"


class _RegArray:
    """Base class for typed register arrays.

    When ``base`` is provided, the array maps to specific physical registers
    (e.g. ``VgprArray(base=16, count=16)`` → v[16:31]).

    When ``base`` is omitted (``None``), the array is *virtual* — it declares
    how many registers are needed but defers physical assignment to emit time.
    """

    def __init__(self, name: str = "", *, base: int | None = None, count: int,
                 block: Block | None = None):
        self.name = name
        self.base = base
        self.count = count
        self.block = block

    @property
    def is_virtual(self) -> bool:
        """True if this array has no physical register base assigned."""
        return self.base is None

    def _make_container(self, phys_base: int, count: int):
        raise NotImplementedError

    def _parse_key(self, key) -> tuple[int, int]:
        """Parse a slice key into (start, count) within this array."""
        if isinstance(key, slice):
            start = key.start if key.start is not None else 0
            stop = key.stop if key.stop is not None else self.count
            if key.step is not None:
                raise IndexError("Register slices do not support step")
            if start < 0 or stop < 0:
                raise IndexError("Negative indices not supported for register slices")
            if stop > self.count:
                raise IndexError(
                    f"{self.name}[{start}:{stop}] exceeds array size {self.count}")
            if start >= stop:
                raise IndexError(
                    f"{self.name}[{start}:{stop}] is empty (start >= stop)")
            return start, stop - start
        raise TypeError(f"Register arrays require slice indexing, got {type(key).__name__}")

    def __getitem__(self, key) -> RegisterSlice:
        start, count = self._parse_key(key)
        return RegisterSlice(array=self, start=start, count=count)

    def __setitem__(self, key, value):
        """Capture instruction destination: Acc[0:4] = vmfma(...)"""
        from rocasm.ops import PendingOp
        start, count = self._parse_key(key)
        dst = RegisterSlice(array=self, start=start, count=count)
        if isinstance(value, PendingOp):
            value.resolve(dst)
        else:
            raise TypeError(
                f"Cannot assign {type(value).__name__} to register slice. "
                "Use an instruction function (e.g. vmfma_f32_16x16x32_bf16(...))")

    def __repr__(self):
        if self.is_virtual:
            return f"{type(self).__name__}({self.name}, count={self.count})"
        return f"{type(self).__name__}({self.name}, base={self.base}, count={self.count})"


class VgprArray(_RegArray):
    """Array of vector general-purpose registers."""

    def _make_container(self, phys_base, count):
        return vgpr(phys_base, count)


class AccArray(_RegArray):
    """Array of accumulator registers (accvgpr on pre-gfx950, mgpr aliases on gfx950)."""

    def _make_container(self, phys_base, count):
        return accvgpr(phys_base, count)


class SgprArray(_RegArray):
    """Array of scalar general-purpose registers."""

    def _make_container(self, phys_base, count):
        return sgpr(phys_base, count)
