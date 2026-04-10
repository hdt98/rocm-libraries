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

"""Tests for the rocasm mainloop registry."""

import pytest

from Tensile.Components.ROCasmRegistry import (
    RegisterROCasmMainloop,
    ROCasmKernelCriteria,
    lookup_rocasm_mainloop,
    clear_registry,
)


def _make_kernel(mt0=192, mt1=256, du=64, mi=None, trans_a=True, trans_b=False):
    """Build a minimal Tensile-style kernel dict for testing."""
    if mi is None:
        mi = [16, 16, 32, 1]
    return {
        "MacroTile0": mt0,
        "MacroTile1": mt1,
        "DepthU": du,
        "MatrixInstruction": mi,
        "ProblemType": {
            "TransposeA": trans_a,
            "TransposeB": trans_b,
        },
    }


@pytest.fixture(autouse=True)
def _clean_registry():
    """Clear the registry before and after each test."""
    clear_registry()
    yield
    clear_registry()


class TestROCasmKernelCriteria:

    def test_matches(self):
        c = ROCasmKernelCriteria(192, 256, 64, (16, 16, 32, 1), True, False)
        assert c.matches(_make_kernel())

    def test_no_match_mt0(self):
        c = ROCasmKernelCriteria(192, 256, 64, (16, 16, 32, 1), True, False)
        assert not c.matches(_make_kernel(mt0=128))

    def test_no_match_depth_u(self):
        c = ROCasmKernelCriteria(192, 256, 64, (16, 16, 32, 1), True, False)
        assert not c.matches(_make_kernel(du=32))

    def test_no_match_transpose(self):
        c = ROCasmKernelCriteria(192, 256, 64, (16, 16, 32, 1), True, False)
        assert not c.matches(_make_kernel(trans_a=False))

    def test_no_match_matrix_inst(self):
        c = ROCasmKernelCriteria(192, 256, 64, (16, 16, 32, 1), True, False)
        assert not c.matches(_make_kernel(mi=[32, 32, 16, 1]))


class TestRegisterROCasmMainloop:

    def test_registration(self):
        @RegisterROCasmMainloop(
            macro_tile_0=192, macro_tile_1=256, depth_u=64,
            matrix_inst=[16, 16, 32, 1],
            transpose_a=True, transpose_b=False,
        )
        def my_mainloop():
            return "block"

        result = lookup_rocasm_mainloop(_make_kernel())
        assert result is my_mainloop

    def test_no_match_returns_none(self):
        @RegisterROCasmMainloop(
            macro_tile_0=192, macro_tile_1=256, depth_u=64,
            matrix_inst=[16, 16, 32, 1],
            transpose_a=True, transpose_b=False,
        )
        def my_mainloop():
            return "block"

        # Different tile size — should not match
        assert lookup_rocasm_mainloop(_make_kernel(mt0=128)) is None

    def test_multiple_registrations(self):
        @RegisterROCasmMainloop(
            macro_tile_0=192, macro_tile_1=256, depth_u=64,
            matrix_inst=[16, 16, 32, 1],
            transpose_a=True, transpose_b=False,
        )
        def mainloop_192x256():
            return "192x256"

        @RegisterROCasmMainloop(
            macro_tile_0=128, macro_tile_1=128, depth_u=64,
            matrix_inst=[16, 16, 32, 1],
            transpose_a=True, transpose_b=False,
        )
        def mainloop_128x128():
            return "128x128"

        assert lookup_rocasm_mainloop(_make_kernel(mt0=192, mt1=256)) is mainloop_192x256
        assert lookup_rocasm_mainloop(_make_kernel(mt0=128, mt1=128)) is mainloop_128x128

    def test_first_match_wins(self):
        @RegisterROCasmMainloop(
            macro_tile_0=192, macro_tile_1=256, depth_u=64,
            matrix_inst=[16, 16, 32, 1],
            transpose_a=True, transpose_b=False,
        )
        def first():
            return "first"

        @RegisterROCasmMainloop(
            macro_tile_0=192, macro_tile_1=256, depth_u=64,
            matrix_inst=[16, 16, 32, 1],
            transpose_a=True, transpose_b=False,
        )
        def second():
            return "second"

        assert lookup_rocasm_mainloop(_make_kernel()) is first

    def test_clear_registry(self):
        @RegisterROCasmMainloop(
            macro_tile_0=192, macro_tile_1=256, depth_u=64,
            matrix_inst=[16, 16, 32, 1],
            transpose_a=True, transpose_b=False,
        )
        def my_mainloop():
            return "block"

        assert lookup_rocasm_mainloop(_make_kernel()) is not None
        clear_registry()
        assert lookup_rocasm_mainloop(_make_kernel()) is None

    def test_decorator_preserves_function(self):
        @RegisterROCasmMainloop(
            macro_tile_0=192, macro_tile_1=256, depth_u=64,
            matrix_inst=[16, 16, 32, 1],
            transpose_a=True, transpose_b=False,
        )
        def my_mainloop():
            return "hello"

        assert my_mainloop() == "hello"

    def test_empty_registry(self):
        assert lookup_rocasm_mainloop(_make_kernel()) is None
