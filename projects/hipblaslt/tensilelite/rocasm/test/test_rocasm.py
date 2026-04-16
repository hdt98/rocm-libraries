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
        assert sub.phys_base == 16
        assert sub.count == 16

    def test_sub_slice(self):
        """RegisterSlice supports further slicing: Acc[16:32][0:4] → phys 16-19."""
        Acc = AccArray("Acc", base=0, count=64)
        sub = Acc[16:32]
        sub_sub = sub[0:4]
        assert sub_sub.phys_base == 16
        assert sub_sub.count == 4

    def test_sub_slice_offset(self):
        """Sub-slicing with offset: Acc[16:32][8:12] → phys 24-27."""
        Acc = AccArray("Acc", base=0, count=64)
        sub = Acc[16:32][8:12]
        assert sub.phys_base == 24
        assert sub.count == 4

    def test_sub_slice_with_base(self):
        """Sub-slicing with non-zero array base: A(base=24)[0:8][4:8] → phys 28-31."""
        A = VgprArray("A", base=24, count=8)
        sub = A[0:8][4:8]
        assert sub.phys_base == 28
        assert sub.count == 4

    def test_sub_slice_container(self):
        """Sub-sliced RegisterSlice produces correct rocisa container."""
        Acc = AccArray("Acc", base=0, count=64)
        c = Acc[16:32][0:4].container()
        assert str(c) == "acc[16:19]"

    def test_sub_slice_out_of_bounds(self):
        Acc = AccArray("Acc", base=0, count=64)
        sub = Acc[16:32]
        with pytest.raises(IndexError, match="exceeds slice size"):
            sub[0:20]

    def test_sub_slice_empty(self):
        Acc = AccArray("Acc", base=0, count=64)
        sub = Acc[16:32]
        with pytest.raises(IndexError, match="empty"):
            sub[4:4]

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
        with pytest.raises(AttributeError, match="no register array or side-effect"):
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

from rocasm.instructions import (
    vmfma_f32_16x16x32_bf16, vmfma_f32_16x16x16bf16_1k,
    ds_read_b128, buffer_load_dwordx4,
    ds_write_b128, ds_write_b32,
)
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


# ---------- vmfma_f32_16x16x16bf16_1k instruction (gfx942) ----------


class TestVmfma16x16x16bf16_1k:

    def _make_block(self):
        return Block(
            Acc=AccArray("Acc", base=0, count=64),
            A=VgprArray("A", base=18, count=24),
            B=VgprArray("B", base=66, count=32),
        )

    def test_single_mfma_1k(self):
        block = self._make_block()
        block.Acc[0:4] = vmfma_f32_16x16x16bf16_1k(block.B[0:2], block.A[0:2], block.Acc[0:4])

        assert len(block) == 1
        op = block.ops[0]
        assert isinstance(op, Op)
        assert op.inst == "v_mfma_f32_16x16x16bf16_1k"
        assert op.dst.phys_base == 0
        assert op.dst.count == 4

    def test_mfma_1k_emits_assembly(self):
        block = self._make_block()
        block.Acc[0:4] = vmfma_f32_16x16x16bf16_1k(block.B[0:2], block.A[0:2], block.Acc[0:4])

        asm = block.emit()
        # rocisa may emit v_mfma_ or v_wmma_ depending on asmCaps
        assert "16x16x16" in asm
        assert "acc[0:3]" in asm

    def test_mfma_1k_sources_tracked(self):
        block = self._make_block()
        block.Acc[4:8] = vmfma_f32_16x16x16bf16_1k(block.B[0:2], block.A[4:6], block.Acc[4:8])

        op = block.ops[0]
        assert len(op.srcs) == 3
        assert op.srcs[0].phys_base == 66   # B[0:2] -> v[66:67]
        assert op.srcs[0].count == 2
        assert op.srcs[1].phys_base == 22   # A[4:6] -> v[22:23]
        assert op.srcs[1].count == 2
        assert op.srcs[2].phys_base == 4    # Acc[4:8] -> acc[4:7]

    def test_mfma_1k_multiple(self):
        block = self._make_block()
        block.Acc[0:4] = vmfma_f32_16x16x16bf16_1k(block.B[0:2], block.A[0:2], block.Acc[0:4])
        block.Acc[4:8] = vmfma_f32_16x16x16bf16_1k(block.B[0:2], block.A[4:6], block.Acc[4:8])

        assert len(block) == 2

    def test_mfma_1k_unattached_raises(self):
        A = VgprArray("A", base=18, count=24)
        B = VgprArray("B", base=66, count=32)
        Acc = AccArray("Acc", base=0, count=16)
        with pytest.raises(RuntimeError, match="not attached to a Block"):
            vmfma_f32_16x16x16bf16_1k(B[0:2], A[0:2], Acc[0:4])


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


# ---------- Side-effect instructions ----------

class TestSideEffectInstructions:

    def _make_block(self):
        return Block(
            Acc=AccArray("Acc", base=0, count=64),
            A=VgprArray("A", base=24, count=8),
            B=VgprArray("B", base=0, count=8),
        )

    def test_s_waitcnt_dscnt(self):
        block = self._make_block()
        block.s_waitcnt(dscnt=0)
        assert len(block) == 1
        op = block.ops[0]
        assert op.inst == "s_waitcnt"
        assert op.dst is None
        assert op.srcs == []
        assert "lgkmcnt(0)" in str(op.rocisa_inst)

    def test_s_waitcnt_vlcnt(self):
        block = self._make_block()
        block.s_waitcnt(vlcnt=0)
        assert "vmcnt(0)" in block.emit()

    def test_s_barrier(self):
        block = self._make_block()
        block.s_barrier()
        assert len(block) == 1
        assert "s_barrier" in block.emit()

    def test_s_nop(self):
        block = self._make_block()
        block.s_nop(0)
        assert "s_nop 0" in block.emit()

    def test_s_nop_with_count(self):
        block = self._make_block()
        block.s_nop(3)
        assert "s_nop 3" in block.emit()

    def test_side_effect_closure_unpacked(self):
        """Side-effect closures can be unpacked at module scope (LEGB pattern)."""
        block = self._make_block()
        s_waitcnt = block.s_waitcnt
        s_barrier = block.s_barrier

        s_waitcnt(dscnt=0)
        s_barrier()

        assert len(block) == 2
        asm = block.emit()
        assert "lgkmcnt(0)" in asm
        assert "s_barrier" in asm

    def test_side_effect_on_new_body(self):
        block = self._make_block()
        block.s_barrier()

        new = block.new_body()
        new.s_waitcnt(dscnt=0)

        assert len(block) == 1
        assert len(new) == 1
        assert "s_barrier" in block.emit()
        assert "lgkmcnt(0)" in new.emit()


# ---------- Sub-slice assignment protocol ----------

class TestSubSliceAssignment:

    def _make_block(self):
        return Block(
            Acc=AccArray("Acc", base=0, count=64),
            A=VgprArray("A", base=24, count=8),
            B=VgprArray("B", base=0, count=8),
        )

    def test_setitem_on_sub_slice(self):
        """Assignment through a sub-slice: sub[0:4] = vmfma(...)."""
        block = self._make_block()
        sub_acc = block.Acc[16:32]
        sub_acc[0:4] = vmfma_f32_16x16x32_bf16(block.B[0:4], block.A[0:4], sub_acc[0:4])

        assert len(block) == 1
        op = block.ops[0]
        assert op.dst.phys_base == 16  # 16 + 0
        assert "acc[16:19]" in block.emit()

    def test_composable_helper_function(self):
        """Prove the matmul_16x16 pattern from the design doc works."""
        block = self._make_block()

        def matmul_16x16(Acc, A, B):
            Acc[0:4] = vmfma_f32_16x16x32_bf16(B[0:4], A[0:4], Acc[0:4])
            Acc[4:8] = vmfma_f32_16x16x32_bf16(B[0:4], A[4:8], Acc[4:8])
            Acc[8:12] = vmfma_f32_16x16x32_bf16(B[0:4], A[0:4], Acc[8:12])
            Acc[12:16] = vmfma_f32_16x16x32_bf16(B[0:4], A[4:8], Acc[12:16])

        # Call with sub-slices: first 16 accumulators
        matmul_16x16(block.Acc[0:16], block.A[0:8], block.B[0:8])

        assert len(block) == 4
        asm = block.emit()
        # First MFMA should write to acc[0:3]
        assert "acc[0:3]" in asm
        # Last MFMA should write to acc[12:15]
        assert "acc[12:15]" in asm

    def test_composable_helper_offset(self):
        """Helper called with offset sub-slices writes to correct physical registers."""
        block = self._make_block()

        def matmul_4(Acc, A, B):
            Acc[0:4] = vmfma_f32_16x16x32_bf16(B[0:4], A[0:4], Acc[0:4])

        # Call targeting acc[48:52] via Acc[48:64][0:4]
        matmul_4(block.Acc[48:64], block.A[0:8], block.B[0:8])

        assert len(block) == 1
        assert "acc[48:51]" in block.emit()

    def test_bad_assignment_on_sub_slice(self):
        block = self._make_block()
        sub = block.Acc[0:16]
        with pytest.raises(TypeError, match="Cannot assign"):
            sub[0:4] = 42


# ---------- Combined calling conventions ----------

class TestCombinedCallingConventions:

    def test_mfma_and_side_effects_interleaved(self):
        """Both calling conventions work together in one function body."""
        block = Block(
            Acc=AccArray("Acc", base=0, count=16),
            A=VgprArray("A", base=24, count=8),
            B=VgprArray("B", base=0, count=8),
        )

        # Unpack closures (LEGB pattern from design doc)
        s_waitcnt = block.s_waitcnt
        s_barrier = block.s_barrier

        # Simulate a main loop fragment
        block.Acc[0:4] = vmfma_f32_16x16x32_bf16(block.B[0:4], block.A[0:4], block.Acc[0:4])
        block.Acc[4:8] = vmfma_f32_16x16x32_bf16(block.B[0:4], block.A[4:8], block.Acc[4:8])
        s_waitcnt(dscnt=0)
        block.Acc[8:12] = vmfma_f32_16x16x32_bf16(block.B[0:4], block.A[0:4], block.Acc[8:12])
        s_barrier()
        block.Acc[12:16] = vmfma_f32_16x16x32_bf16(block.B[0:4], block.A[4:8], block.Acc[12:16])

        assert len(block) == 6
        asm = block.emit()
        lines = [l for l in asm.split("\n") if l.strip()]
        assert len(lines) == 6
        assert "v_mfma" in lines[0]
        assert "v_mfma" in lines[1]
        assert "lgkmcnt(0)" in lines[2]
        assert "v_mfma" in lines[3]
        assert "s_barrier" in lines[4]
        assert "v_mfma" in lines[5]

    def test_function_with_register_args_and_closures(self):
        """The full design doc pattern: register args + module-scope closures."""
        block = Block(
            Acc=AccArray("Acc", base=0, count=64),
            A=VgprArray("A", base=24, count=8),
            B=VgprArray("B", base=0, count=8),
        )

        # Module scope: unpack closures
        s_waitcnt = block.s_waitcnt

        # Function takes register arrays as arguments
        def label_LoopBeginL(Acc, A, B):
            Acc[0:4] = vmfma_f32_16x16x32_bf16(B[0:4], A[0:4], Acc[0:4])
            Acc[4:8] = vmfma_f32_16x16x32_bf16(B[0:4], A[4:8], Acc[4:8])
            s_waitcnt(dscnt=0)

        label_LoopBeginL(block.Acc, block.A, block.B)

        assert len(block) == 3
        asm = block.emit()
        assert asm.count("v_mfma") == 2
        assert "lgkmcnt(0)" in asm


# ---------- Scalar side-effect instructions ----------

from rocisa.container import sgpr, vgpr


class TestScalarSideEffects:

    def _make_block(self):
        return Block(
            Acc=AccArray("Acc", base=0, count=64),
            A=VgprArray("A", base=24, count=8),
            B=VgprArray("B", base=0, count=8),
            S=SgprArray("S", base=0, count=16),
        )

    def test_s_mov_b32(self):
        block = self._make_block()
        block.s_mov_b32(dst=sgpr(0), src=0x10)
        assert len(block) == 1
        assert "s_mov_b32" in block.emit()

    def test_s_add_u32(self):
        block = self._make_block()
        block.s_add_u32(dst=sgpr(0), src0=sgpr(0), src1=sgpr(1))
        assert len(block) == 1
        assert "s_add_u32" in block.emit()

    def test_s_addc_u32(self):
        block = self._make_block()
        block.s_addc_u32(dst=sgpr(1), src0=sgpr(1), src1=0)
        assert len(block) == 1
        assert "s_addc_u32" in block.emit()

    def test_s_sub_u32(self):
        block = self._make_block()
        block.s_sub_u32(dst=sgpr(2), src0=sgpr(2), src1=1)
        assert len(block) == 1
        assert "s_sub_u32" in block.emit()

    def test_s_subb_u32(self):
        block = self._make_block()
        block.s_subb_u32(dst=sgpr(3), src0=sgpr(3), src1=0)
        assert len(block) == 1
        assert "s_subb_u32" in block.emit()

    def test_s_cmp_eq_u32(self):
        block = self._make_block()
        block.s_cmp_eq_u32(src0=sgpr(0), src1=0)
        assert len(block) == 1
        assert "s_cmp_eq_u32" in block.emit()

    def test_s_cmp_eq_i32(self):
        block = self._make_block()
        block.s_cmp_eq_i32(src0=sgpr(0), src1=1)
        assert len(block) == 1
        assert "s_cmp_eq_i32" in block.emit()

    def test_s_cselect_b32(self):
        block = self._make_block()
        block.s_cselect_b32(dst=sgpr(4), src0=sgpr(5), src1=sgpr(6))
        assert len(block) == 1
        assert "s_cselect_b32" in block.emit()

    def test_s_xor_b32(self):
        block = self._make_block()
        block.s_xor_b32(dst=sgpr(0), src0=sgpr(0), src1=1)
        assert len(block) == 1
        assert "s_xor_b32" in block.emit()

    def test_v_xor_b32(self):
        block = self._make_block()
        block.v_xor_b32(dst=vgpr(0), src0=vgpr(0), src1=vgpr(1))
        assert len(block) == 1
        assert "v_xor_b32" in block.emit()

    def test_s_cbranch_scc0(self):
        block = self._make_block()
        block.s_cbranch_scc0(labelName="label_LoopBeginL")
        assert len(block) == 1
        asm = block.emit()
        assert "s_cbranch_scc0" in asm
        assert "label_LoopBeginL" in asm


# ---------- Scalar side-effect ordering ----------

class TestScalarSideEffectOrdering:

    def test_scalar_ops_interleaved_with_mfma(self):
        """Scalar arithmetic and comparisons interleave correctly with compute."""
        block = Block(
            Acc=AccArray("Acc", base=0, count=16),
            A=VgprArray("A", base=24, count=8),
            B=VgprArray("B", base=0, count=8),
            S=SgprArray("S", base=0, count=8),
        )

        block.Acc[0:4] = vmfma_f32_16x16x32_bf16(block.B[0:4], block.A[0:4], block.Acc[0:4])
        block.s_add_u32(dst=sgpr(0), src0=sgpr(0), src1=sgpr(1))
        block.s_addc_u32(dst=sgpr(1), src0=sgpr(1), src1=0)
        block.Acc[4:8] = vmfma_f32_16x16x32_bf16(block.B[0:4], block.A[4:8], block.Acc[4:8])
        block.s_cmp_eq_u32(src0=sgpr(0), src1=0)
        block.s_cbranch_scc0(labelName="label_LoopBeginL")

        assert len(block) == 6
        asm = block.emit()
        lines = [l for l in asm.split("\n") if l.strip()]
        assert "v_mfma" in lines[0]
        assert "s_add_u32" in lines[1]
        assert "s_addc_u32" in lines[2]
        assert "v_mfma" in lines[3]
        assert "s_cmp_eq_u32" in lines[4]
        assert "s_cbranch_scc0" in lines[5]


# ---------- ds_read_b128 instruction ----------

class TestDsReadB128:

    def _make_block(self):
        return Block(
            A=VgprArray("A", base=24, count=8),
            LocalReadAddr=VgprArray("LocalReadAddr", base=128, count=2),
        )

    def test_basic_ds_read(self):
        block = self._make_block()
        block.A[0:4] = ds_read_b128(block.LocalReadAddr[0:1])
        assert len(block) == 1
        op = block.ops[0]
        assert op.inst == "ds_read_b128"
        assert op.dst.phys_base == 24
        assert op.dst.count == 4

    def test_ds_read_emits_assembly(self):
        block = self._make_block()
        block.A[0:4] = ds_read_b128(block.LocalReadAddr[0:1])
        asm = block.emit()
        assert "ds_load_b128" in asm or "ds_read_b128" in asm

    def test_ds_read_with_offset(self):
        from rocisa.container import DSModifiers
        block = self._make_block()
        block.A[4:8] = ds_read_b128(block.LocalReadAddr[0:1], ds=DSModifiers(offset=0x100))
        assert len(block) == 1
        assert "0x100" in block.emit() or "256" in block.emit() or "offset" in block.emit().lower()

    def test_ds_read_unattached_raises(self):
        addr = VgprArray("Addr", base=0, count=2)
        with pytest.raises(RuntimeError, match="not attached"):
            ds_read_b128(addr[0:1])


# ---------- buffer_load_dwordx4 instruction ----------

class TestBufferLoadDwordx4:

    def _make_block(self):
        return Block(
            A=VgprArray("A", base=24, count=8),
            GlobalReadAddr=VgprArray("GlobalReadAddr", base=130, count=2),
            SrdA=SgprArray("SrdA", base=24, count=4),
            Soffset=SgprArray("Soffset", base=28, count=1),
        )

    def test_basic_buffer_load(self):
        block = self._make_block()
        block.A[0:4] = buffer_load_dwordx4(
            block.GlobalReadAddr[0:1],
            block.SrdA[0:4],
            block.Soffset[0:1].container(),
        )
        assert len(block) == 1
        op = block.ops[0]
        assert op.inst == "buffer_load_dwordx4"
        assert op.dst.phys_base == 24
        assert op.dst.count == 4

    def test_buffer_load_emits_assembly(self):
        block = self._make_block()
        block.A[0:4] = buffer_load_dwordx4(
            block.GlobalReadAddr[0:1],
            block.SrdA[0:4],
            block.Soffset[0:1].container(),
        )
        asm = block.emit()
        assert "buffer_load" in asm

    def test_buffer_load_with_mubuf(self):
        from rocisa.container import MUBUFModifiers
        block = self._make_block()
        block.A[0:4] = buffer_load_dwordx4(
            block.GlobalReadAddr[0:1],
            block.SrdA[0:4],
            block.Soffset[0:1].container(),
            mubuf=MUBUFModifiers(offen=True, offset12=0),
        )
        assert len(block) == 1
        asm = block.emit()
        assert "buffer_load" in asm

    def test_buffer_load_unattached_raises(self):
        vaddr = VgprArray("V", base=0, count=2)
        saddr = SgprArray("S", base=0, count=4)
        with pytest.raises(RuntimeError, match="not attached"):
            buffer_load_dwordx4(vaddr[0:1], saddr[0:4], 0)


# ---------- ds_write_b128 instruction ----------


class TestDsWriteB128:

    def _make_block(self):
        return Block(
            G2LA=VgprArray("G2LA", base=130, count=24),
            LocalWriteAddr=VgprArray("LocalWriteAddr", base=186, count=1),
        )

    def test_basic_ds_write_b128(self):
        block = self._make_block()
        ds_write_b128(block.LocalWriteAddr[0:1], block.G2LA[0:4])
        assert len(block) == 1
        op = block.ops[0]
        assert op.inst == "ds_write_b128"
        assert op.dst is None
        assert len(op.srcs) == 2

    def test_ds_write_b128_emits_assembly(self):
        block = self._make_block()
        ds_write_b128(block.LocalWriteAddr[0:1], block.G2LA[0:4])
        asm = block.emit()
        assert "ds_store_b128" in asm or "ds_write_b128" in asm

    def test_ds_write_b128_with_offset(self):
        from rocisa.container import DSModifiers
        block = self._make_block()
        ds_write_b128(block.LocalWriteAddr[0:1], block.G2LA[4:8],
                      ds=DSModifiers(offset=4608))
        assert len(block) == 1
        asm = block.emit()
        assert "4608" in asm or "offset" in asm.lower()

    def test_ds_write_b128_unattached_raises(self):
        addr = VgprArray("Addr", base=0, count=1)
        data = VgprArray("Data", base=4, count=4)
        with pytest.raises(RuntimeError, match="not attached"):
            ds_write_b128(addr[0:1], data[0:4])


# ---------- ds_write_b32 instruction ----------


class TestDsWriteB32:

    def _make_block(self):
        return Block(
            G2LB=VgprArray("G2LB", base=154, count=32),
            LocalWriteAddr=VgprArray("LocalWriteAddr", base=186, count=1),
        )

    def test_basic_ds_write_b32(self):
        block = self._make_block()
        ds_write_b32(block.LocalWriteAddr[0:1], block.G2LB[24:25])
        assert len(block) == 1
        op = block.ops[0]
        assert op.inst == "ds_write_b32"
        assert op.dst is None

    def test_ds_write_b32_emits_assembly(self):
        block = self._make_block()
        ds_write_b32(block.LocalWriteAddr[0:1], block.G2LB[0:1])
        asm = block.emit()
        assert "ds_store_b32" in asm or "ds_write_b32" in asm

    def test_ds_write_b32_with_offset(self):
        from rocisa.container import DSModifiers
        block = self._make_block()
        ds_write_b32(block.LocalWriteAddr[0:1], block.G2LB[24:25],
                     ds=DSModifiers(offset=25344))
        asm = block.emit()
        assert "25344" in asm or "offset" in asm.lower()

    def test_ds_write_b32_unattached_raises(self):
        addr = VgprArray("Addr", base=0, count=1)
        data = VgprArray("Data", base=4, count=1)
        with pytest.raises(RuntimeError, match="not attached"):
            ds_write_b32(addr[0:1], data[0:1])
