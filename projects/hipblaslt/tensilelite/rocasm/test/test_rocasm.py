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

"""Unit tests for the rocasm package: Block, register arrays, Op, and vmfma instruction."""

import pytest

import rocisa
from rocisa import rocIsa


@pytest.fixture(scope="session", autouse=True)
def init_rocisa():
    """Initialize rocisa singleton for gfx950 before any tests run."""
    ti = rocIsa.getInstance()
    ti.init((9, 5, 0), "amdclang++")
    ti.setKernel((9, 5, 0), 64)


# ---------- Register arrays ----------

from rocasm.regs import VgprArray, AccArray, SgprArray, RegisterSlice


class TestRegisterArrays:

    def test_vgpr_getitem(self):
        A = VgprArray("A", base=24, count=8)
        s = A[0:4]
        assert isinstance(s, RegisterSlice)
        assert s.phys_base == 24
        assert s.count == 4

    def test_acc_getitem(self):
        Acc = AccArray("Acc", base=0, count=64)
        s = Acc[12:16]
        assert s.phys_base == 12
        assert s.count == 4

    def test_sgpr_getitem(self):
        S = SgprArray("S", base=8, count=4)
        s = S[0:4]
        assert s.phys_base == 8
        assert s.count == 4

    def test_slice_composability(self):
        """Slicing a sub-array composes: Acc[16:32][0:4] maps to phys 16-19."""
        Acc = AccArray("Acc", base=0, count=64)
        sub = Acc[16:32]
        # sub is a RegisterSlice, not an array, so we test the phys_base math
        assert sub.phys_base == 16
        assert sub.count == 16

    def test_out_of_bounds(self):
        A = VgprArray("A", base=0, count=8)
        with pytest.raises(IndexError, match="exceeds array size"):
            A[0:10]

    def test_empty_slice(self):
        A = VgprArray("A", base=0, count=8)
        with pytest.raises(IndexError, match="empty"):
            A[4:4]

    def test_negative_index(self):
        A = VgprArray("A", base=0, count=8)
        with pytest.raises(IndexError, match="Negative"):
            A[-1:4]

    def test_step_not_allowed(self):
        A = VgprArray("A", base=0, count=8)
        with pytest.raises(IndexError, match="step"):
            A[0:8:2]

    def test_container_vgpr(self):
        A = VgprArray("A", base=24, count=8)
        c = A[0:4].container()
        assert str(c) == "v[24:27]"

    def test_container_acc(self):
        Acc = AccArray("Acc", base=0, count=64)
        c = Acc[12:16].container()
        assert str(c) == "acc[12:15]"

    def test_repr(self):
        A = VgprArray("A", base=24, count=8)
        assert repr(A[0:4]) == "A[0:4]"
        assert repr(A[4:8]) == "A[4:8]"


# ---------- Block ----------

from rocasm.block import Block


class TestBlock:

    def test_create_block(self):
        block = Block(
            Acc=AccArray("Acc", base=0, count=64),
            A=VgprArray("A", base=24, count=8),
            B=VgprArray("B", base=0, count=8),
        )
        assert len(block) == 0
        assert block.Acc.base == 0
        assert block.A.base == 24
        assert block.B.base == 0

    def test_register_arrays_bound_to_block(self):
        block = Block(
            Acc=AccArray("Acc", base=0, count=16),
        )
        assert block.Acc.block is block

    def test_new_body_shares_context(self):
        block = Block(
            Acc=AccArray("Acc", base=0, count=16),
            A=VgprArray("A", base=24, count=8),
        )
        new = block.new_body()
        assert new.Acc.base == block.Acc.base
        assert new.Acc.count == block.Acc.count
        assert new.A.base == block.A.base
        assert len(new) == 0

    def test_new_body_independent_ops(self):
        block = Block(Acc=AccArray("Acc", base=0, count=16))
        new = block.new_body()
        # They should have separate op lists
        assert new._ops is not block._ops

    def test_new_body_bound_to_new_block(self):
        block = Block(Acc=AccArray("Acc", base=0, count=16))
        new = block.new_body()
        assert new.Acc.block is new
        assert block.Acc.block is block

    def test_invalid_reg_type(self):
        with pytest.raises(TypeError, match="register array"):
            Block(Acc="not a register array")

    def test_unknown_attr(self):
        block = Block(Acc=AccArray("Acc", base=0, count=16))
        with pytest.raises(AttributeError, match="no register array"):
            block.X

    def test_repr(self):
        block = Block(
            Acc=AccArray("Acc", base=0, count=16),
            A=VgprArray("A", base=24, count=8),
        )
        r = repr(block)
        assert "Acc=" in r
        assert "ops=0" in r


# ---------- vmfma instruction ----------

from rocasm.instructions import vmfma_f32_16x16x32_bf16
from rocasm.ops import Op


class TestVmfma:

    def _make_block(self):
        return Block(
            Acc=AccArray("Acc", base=0, count=64),
            A=VgprArray("A", base=24, count=8),
            B=VgprArray("B", base=0, count=8),
        )

    def test_single_mfma(self):
        block = self._make_block()
        block.Acc[12:16] = vmfma_f32_16x16x32_bf16(block.B[0:4], block.A[0:4], block.Acc[12:16])

        assert len(block) == 1
        op = block.ops[0]
        assert isinstance(op, Op)
        assert op.inst == "v_mfma_f32_16x16x32_bf16"
        assert op.dst.phys_base == 12
        assert op.dst.count == 4

    def test_mfma_emits_correct_assembly(self):
        block = self._make_block()
        block.Acc[12:16] = vmfma_f32_16x16x32_bf16(block.B[0:4], block.A[0:4], block.Acc[12:16])

        asm = block.emit()
        assert "v_mfma_f32_16x16x32_bf16" in asm
        assert "acc[12:15]" in asm
        assert "v[0:3]" in asm
        assert "v[24:27]" in asm

    def test_multiple_mfmas(self):
        block = self._make_block()
        block.Acc[0:4] = vmfma_f32_16x16x32_bf16(block.B[0:4], block.A[0:4], block.Acc[0:4])
        block.Acc[4:8] = vmfma_f32_16x16x32_bf16(block.B[0:4], block.A[4:8], block.Acc[4:8])
        block.Acc[8:12] = vmfma_f32_16x16x32_bf16(block.B[0:4], block.A[0:4], block.Acc[8:12])
        block.Acc[12:16] = vmfma_f32_16x16x32_bf16(block.B[0:4], block.A[4:8], block.Acc[12:16])

        assert len(block) == 4
        asm = block.emit()
        assert asm.count("v_mfma_f32_16x16x32_bf16") == 4

    def test_mfma_sources_tracked(self):
        block = self._make_block()
        block.Acc[0:4] = vmfma_f32_16x16x32_bf16(block.B[0:4], block.A[4:8], block.Acc[0:4])

        op = block.ops[0]
        assert len(op.srcs) == 3
        # src_b is first, src_a is second, acc2 is third
        assert op.srcs[0].phys_base == 0   # B[0:4] -> v[0:3]
        assert op.srcs[1].phys_base == 28  # A[4:8] -> v[28:31]
        assert op.srcs[2].phys_base == 0   # Acc[0:4] -> acc[0:3]

    def test_unattached_registers_raise(self):
        A = VgprArray("A", base=24, count=8)
        B = VgprArray("B", base=0, count=8)
        Acc = AccArray("Acc", base=0, count=16)
        with pytest.raises(RuntimeError, match="not attached to a Block"):
            vmfma_f32_16x16x32_bf16(B[0:4], A[0:4], Acc[0:4])

    def test_bad_assignment(self):
        block = Block(Acc=AccArray("Acc", base=0, count=16))
        with pytest.raises(TypeError, match="Cannot assign"):
            block.Acc[0:4] = 42


# ---------- Block.emit round-trip ----------

class TestEmit:

    def test_emit_order_preserved(self):
        block = Block(
            Acc=AccArray("Acc", base=0, count=16),
            A=VgprArray("A", base=24, count=8),
            B=VgprArray("B", base=0, count=8),
        )
        block.Acc[0:4] = vmfma_f32_16x16x32_bf16(block.B[0:4], block.A[0:4], block.Acc[0:4])
        block.Acc[4:8] = vmfma_f32_16x16x32_bf16(block.B[4:8], block.A[0:4], block.Acc[4:8])

        asm = block.emit()
        lines = [l for l in asm.split("\n") if l.strip()]
        assert len(lines) == 2
        assert "acc[0:3]" in lines[0]
        assert "acc[4:7]" in lines[1]

    def test_empty_block_emits_empty(self):
        block = Block(Acc=AccArray("Acc", base=0, count=16))
        assert block.emit() == ""


# ---------- new_body / copy_body ----------

class TestBodyCloning:

    def test_copy_body_preserves_ops(self):
        block = Block(
            Acc=AccArray("Acc", base=0, count=16),
            A=VgprArray("A", base=24, count=8),
            B=VgprArray("B", base=0, count=8),
        )
        block.Acc[0:4] = vmfma_f32_16x16x32_bf16(block.B[0:4], block.A[0:4], block.Acc[0:4])

        cloned = block.copy_body()
        assert len(cloned) == 1
        assert cloned.ops[0].inst == "v_mfma_f32_16x16x32_bf16"

    def test_copy_body_is_independent(self):
        block = Block(
            Acc=AccArray("Acc", base=0, count=16),
            A=VgprArray("A", base=24, count=8),
            B=VgprArray("B", base=0, count=8),
        )
        block.Acc[0:4] = vmfma_f32_16x16x32_bf16(block.B[0:4], block.A[0:4], block.Acc[0:4])

        cloned = block.copy_body()
        cloned.Acc[4:8] = vmfma_f32_16x16x32_bf16(cloned.B[0:4], cloned.A[0:4], cloned.Acc[4:8])

        assert len(block) == 1
        assert len(cloned) == 2

    def test_new_body_then_write(self):
        block = Block(
            Acc=AccArray("Acc", base=0, count=16),
            A=VgprArray("A", base=24, count=8),
            B=VgprArray("B", base=0, count=8),
        )
        block.Acc[0:4] = vmfma_f32_16x16x32_bf16(block.B[0:4], block.A[0:4], block.Acc[0:4])

        fresh = block.new_body()
        assert len(fresh) == 0
        fresh.Acc[4:8] = vmfma_f32_16x16x32_bf16(fresh.B[0:4], fresh.A[4:8], fresh.Acc[4:8])
        assert len(fresh) == 1
        assert len(block) == 1  # original unchanged
