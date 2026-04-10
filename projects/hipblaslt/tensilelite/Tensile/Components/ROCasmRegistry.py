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

"""Registry for user-defined rocasm mainloop modules.

Users register their rocasm mainloop functions using the
``@RegisterROCasmMainloop(...)`` decorator.  At build time,
``lookup_rocasm_mainloop(kernel)`` finds a registered function that
matches the kernel's tile config, dtype, and layout.

This follows the same pattern as ``CustomSchedule.py``'s
``_SCHEDULE_REGISTRY`` / ``RegisterSchedule``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class ROCasmKernelCriteria:
    """Matching criteria for a registered rocasm mainloop."""
    macro_tile_0: int
    macro_tile_1: int
    depth_u: int
    matrix_inst: tuple[int, ...]
    transpose_a: bool
    transpose_b: bool

    def matches(self, kernel: dict) -> bool:
        """Check whether a Tensile kernel dict matches these criteria."""
        if kernel["MacroTile0"] != self.macro_tile_0:
            return False
        if kernel["MacroTile1"] != self.macro_tile_1:
            return False
        if kernel["DepthU"] != self.depth_u:
            return False
        if tuple(kernel["MatrixInstruction"][:4]) != self.matrix_inst:
            return False
        pt = kernel["ProblemType"]
        if pt["TransposeA"] != self.transpose_a:
            return False
        if pt["TransposeB"] != self.transpose_b:
            return False
        return True


# Global registry: list of (criteria, func) pairs.
_ROCASM_REGISTRY: list[tuple[ROCasmKernelCriteria, Callable]] = []


class RegisterROCasmMainloop:
    """Decorator that registers a rocasm mainloop function.

    Usage::

        @RegisterROCasmMainloop(
            macro_tile_0=192, macro_tile_1=256, depth_u=64,
            matrix_inst=[16, 16, 32, 1],
            transpose_a=True, transpose_b=False,
        )
        def my_bf16_mainloop():
            block = Block(...)
            ...
            return block

    The decorated function is called with no arguments and must return a
    ``Block`` whose ``emit()`` method produces the main loop assembly text.
    """

    def __init__(self, *, macro_tile_0: int, macro_tile_1: int, depth_u: int,
                 matrix_inst: list[int],
                 transpose_a: bool, transpose_b: bool):
        self.criteria = ROCasmKernelCriteria(
            macro_tile_0=macro_tile_0,
            macro_tile_1=macro_tile_1,
            depth_u=depth_u,
            matrix_inst=tuple(matrix_inst),
            transpose_a=transpose_a,
            transpose_b=transpose_b,
        )

    def __call__(self, func: Callable) -> Callable:
        _ROCASM_REGISTRY.append((self.criteria, func))
        return func


def lookup_rocasm_mainloop(kernel: dict) -> Callable | None:
    """Find a registered rocasm mainloop function matching the kernel.

    Iterates ``_ROCASM_REGISTRY`` and returns the first function whose
    criteria match the kernel.  Returns ``None`` if no match is found.
    """
    for criteria, func in _ROCASM_REGISTRY:
        if criteria.matches(kernel):
            return func
    return None


def clear_registry():
    """Remove all registered mainloops.  Useful for testing."""
    _ROCASM_REGISTRY.clear()
