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

"""Tests for tile analysis: MFMA tile detection and tiled mainloop codegen."""

import pytest
from textwrap import dedent
from pathlib import Path

from rocasm.tile_analysis import (
    analyze_tile,
    generate_tiled_mainloop,
    _parse_mfma,
    _parse_ds_read,
    TileInfo,
    HalfSchedule,
)


# ─── Unit tests for line parsers ─────────────────────────────────────────────


class TestParseMfma:

    def test_basic(self):
        op = _parse_mfma("Acc[0:4] = vmfma_f32_16x16x32_bf16(B0[0:4], A0[0:4], Acc[0:4])")
        assert op is not None
        assert op.acc_start == 0
        assert op.acc_end == 4
        assert op.b_name == "B0"
        assert op.b_start == 0
        assert op.b_end == 4
        assert op.a_name == "A0"
        assert op.a_start == 0
        assert op.a_end == 4

    def test_offset(self):
        op = _parse_mfma("Acc[24:28] = vmfma_f32_16x16x32_bf16(B0[4:8], A0[0:4], Acc[24:28])")
        assert op.acc_start == 24
        assert op.b_start == 4
        assert op.b_end == 8

    def test_non_mfma_returns_none(self):
        assert _parse_mfma("s_barrier()") is None
        assert _parse_mfma("A1[0:4] = ds_read_b128(...)") is None


class TestParseDsRead:

    def test_with_offset(self):
        op = _parse_ds_read(
            "A1[0:4] = ds_read_b128(LocalReadAddrA[0:1], ds=DSModifiers(offset=64))")
        assert op is not None
        assert op.dst_name == "A1"
        assert op.dst_start == 0
        assert op.dst_end == 4
        assert op.offset == 64

    def test_no_offset(self):
        op = _parse_ds_read("A0[0:4] = ds_read_b128(LocalReadAddrA[0:1])")
        assert op is not None
        assert op.dst_name == "A0"
        assert op.offset == 0

    def test_non_ds_read_returns_none(self):
        assert _parse_ds_read("s_barrier()") is None


# ─── Integration tests with real generated mainloop ──────────────────────────


def _make_flat_mainloop_body() -> str:
    """Build a representative flat MFMA mainloop body for testing.

    This is a minimal 2-half, 6A x 8B tile with representative interleaving
    extracted from a real kernel. Uses the same register names and structure
    as the full generated mainloop.
    """
    lines = []
    lines.append('label("label_LoopBeginL")')

    M, N, W = 6, 8, 4
    # Half 0: B0/A0
    lines.append("s_waitcnt(dscnt=7)")
    for b in range(N):
        for a in range(M):
            idx = b * M + a
            acc_s = idx * W
            lines.append(
                f"Acc[{acc_s}:{acc_s+W}] = vmfma_f32_16x16x32_bf16("
                f"B0[{b*W}:{(b+1)*W}], A0[{a*W}:{(a+1)*W}], Acc[{acc_s}:{acc_s+W}])")
            # Interleave: A1 LDS reads during b=0
            if b == 0:
                a1_offsets = [64, 192, 8768, 8896, 17472, 17600]
                lines.append(
                    f"A1[{a*W}:{(a+1)*W}] = ds_read_b128("
                    f"LocalReadAddrA[0:1], ds=DSModifiers(offset={a1_offsets[a]}))")
            # Interleave: B1 LDS reads during b=1
            if b == 1:
                b1_offsets = [64, 192, 320, 448, 576, 704]
                lines.append(
                    f"B1[{a*W}:{(a+1)*W}] = ds_read_b128("
                    f"LocalReadAddrB[0:1], ds=DSModifiers(offset={b1_offsets[a]}))")
            # Global loads for A
            if idx == 20:
                lines.append("s_mov_b32(dst=m0, src=sgpr(53))")
                lines.append("s_waitcnt(dscnt=0)")
                lines.append("s_barrier()")
                lines.append("buffer_load_lds(GlobalReadOffsetA[0:1], SrdA[0:4], 0, mubuf=MUBUFModifiers(offen=True, lds=True))")
            if idx in (24, 28, 33, 37, 41):
                lines.append("s_add_u32(dst=m0, src0=m0, src1=4352)")
                lines.append("buffer_load_lds(GlobalReadOffsetA[1:2], SrdA[0:4], 0, mubuf=MUBUFModifiers(offen=True, lds=True))")

    # Half 1: B1/A1
    for b in range(N):
        for a in range(M):
            idx = b * M + a
            acc_s = idx * W
            lines.append(
                f"Acc[{acc_s}:{acc_s+W}] = vmfma_f32_16x16x32_bf16("
                f"B1[{b*W}:{(b+1)*W}], A1[{a*W}:{(a+1)*W}], Acc[{acc_s}:{acc_s+W}])")
            # Global loads for B
            if idx in (2, 6, 12, 16, 20, 24, 28):
                lines.append("s_add_u32(dst=m0, src0=m0, src1=4224)")
                lines.append("buffer_load_lds(GlobalReadOffsetB[0:1], SrdB[0:4], 0, mubuf=MUBUFModifiers(offen=True, lds=True))")
            # Barrier
            if idx == 28:
                lines.append("s_waitcnt(vlcnt=14)")
                lines.append("s_barrier()")
            # A0 LDS reads after barrier
            if 29 <= idx <= 34:
                a0_offsets = [0, 128, 8704, 8832, 17408, 17536]
                chunk = idx - 29
                offset = a0_offsets[chunk]
                if offset:
                    lines.append(
                        f"A0[{chunk*W}:{(chunk+1)*W}] = ds_read_b128("
                        f"LocalReadAddrA[0:1], ds=DSModifiers(offset={offset}))")
                else:
                    lines.append(
                        f"A0[{chunk*W}:{(chunk+1)*W}] = ds_read_b128(LocalReadAddrA[0:1])")
            # B0 LDS reads
            if 35 <= idx <= 42:
                b0_offsets = [0, 128, 256, 384, 512, 640, 768, 896]
                chunk = idx - 35
                if chunk < N:
                    offset = b0_offsets[chunk]
                    if offset:
                        lines.append(
                            f"B0[{chunk*W}:{(chunk+1)*W}] = ds_read_b128("
                            f"LocalReadAddrB[0:1], ds=DSModifiers(offset={offset}))")
                    else:
                        lines.append(
                            f"B0[{chunk*W}:{(chunk+1)*W}] = ds_read_b128(LocalReadAddrB[0:1])")

    # Loop back
    lines.append("s_sub_u32(dst=sgpr(8), src0=sgpr(8), src1=1)")
    lines.append('s_cmp_eq_i32(src0=sgpr(8), src1=hex(2))')
    lines.append('s_cbranch_scc0(labelName="label_LoopBeginL")')

    return "\n".join(lines)


class TestAnalyzeTileReal:

    @pytest.fixture
    def tile_info(self):
        body = _make_flat_mainloop_body()
        info = analyze_tile(body)
        assert info is not None, "analyze_tile returned None for real mainloop"
        return info

    def test_tile_dimensions(self, tile_info):
        assert tile_info.m_tiles == 6
        assert tile_info.n_tiles == 8
        assert tile_info.w == 4

    def test_two_halves(self, tile_info):
        assert tile_info.num_halves == 2

    def test_register_names(self, tile_info):
        assert tile_info.a_names == ["A0", "A1"]
        assert tile_info.b_names == ["B0", "B1"]

    def test_lds_offsets(self, tile_info):
        assert tile_info.a_lds_offsets == [0, 128, 8704, 8832, 17408, 17536]
        assert tile_info.b_lds_offsets == [0, 128, 256, 384, 512, 640, 768, 896]

    def test_lds_half_offset(self, tile_info):
        assert tile_info.lds_half_offset == 64

    def test_global_load_strides(self, tile_info):
        assert tile_info.gr_stride_a == 4352
        assert tile_info.gr_stride_b == 4224

    def test_global_load_counts(self, tile_info):
        assert tile_info.n_global_a == 6
        assert tile_info.n_global_b == 7

    def test_half_0_mfma_count(self, tile_info):
        assert tile_info.halves[0].mfma_count == 48
        assert tile_info.halves[0].b_name == "B0"
        assert tile_info.halves[0].a_name == "A0"

    def test_half_1_mfma_count(self, tile_info):
        assert tile_info.halves[1].mfma_count == 48
        assert tile_info.halves[1].b_name == "B1"
        assert tile_info.halves[1].a_name == "A1"

    def test_interleave_has_s_waitcnt(self, tile_info):
        """The first half should have s_waitcnt(dscnt=7) before the first MFMA."""
        pre_ops = tile_info.halves[0].interleave.get(-1, [])
        assert any("s_waitcnt" in line for line in pre_ops)


class TestGenerateTiledMainloop:

    @pytest.fixture
    def tiled_output(self):
        body = _make_flat_mainloop_body()
        info = analyze_tile(body)
        assert info is not None
        named_regs = {
            "A0": ("v", 16, 24), "A1": ("v", 40, 24),
            "B0": ("v", 64, 32), "B1": ("v", 96, 32),
            "GlobalReadOffsetA": ("v", 0, 6), "GlobalReadOffsetB": ("v", 6, 8),
            "LocalReadAddrA": ("v", 14, 1), "LocalReadAddrB": ("v", 15, 1),
            "SrdA": ("s", 64, 4), "SrdB": ("s", 68, 4),
        }
        return generate_tiled_mainloop(info, named_regs,
                                       {"vgpr": 256, "accvgpr": 192, "sgpr": 88})

    def test_has_imports(self, tiled_output):
        assert "from rocasm.tile_helpers import" in tiled_output

    def test_has_tile_constants(self, tiled_output):
        assert "M_TILES = 6" in tiled_output
        assert "N_TILES = 8" in tiled_output
        assert "W = 4" in tiled_output

    def test_has_lds_offsets(self, tiled_output):
        assert "A_LDS_OFFSETS = [0, 128, 8704, 8832, 17408, 17536]" in tiled_output

    def test_has_block_init(self, tiled_output):
        assert "block = Block(" in tiled_output
        assert "A0=VgprArray(base=16, count=24)" in tiled_output

    def test_has_for_loops(self, tiled_output):
        assert "for b in range(N_TILES):" in tiled_output
        assert "for a in range(M_TILES):" in tiled_output

    def test_has_run_half_calls(self, tiled_output):
        assert "run_half(B0, A0, half0_interleave)" in tiled_output
        assert "run_half(B1, A1, half1_interleave)" in tiled_output

    def test_has_mfma_at_in_run_half(self, tiled_output):
        assert "mfma_at(Acc, B, A, b, a, M_TILES, W, vmfma_f32_16x16x32_bf16)" in tiled_output

    def test_has_loop_back(self, tiled_output):
        assert 's_cbranch_scc0(labelName="label_LoopBeginL")' in tiled_output

    def test_has_return_block(self, tiled_output):
        assert "return block" in tiled_output

    def test_shorter_than_flat(self, tiled_output):
        """The tiled output should be shorter than the ~239 line flat version."""
        assert len(tiled_output.splitlines()) < 220


# ─── Edge case tests ─────────────────────────────────────────────────────────


class TestAnalyzeTileEdgeCases:

    def test_no_mfma_returns_none(self):
        code = dedent("""\
            s_barrier()
            s_waitcnt(dscnt=0)
        """)
        assert analyze_tile(code) is None

    def test_single_mfma(self):
        code = "Acc[0:4] = vmfma_f32_16x16x32_bf16(B0[0:4], A0[0:4], Acc[0:4])"
        info = analyze_tile(code)
        assert info is not None
        assert info.m_tiles == 1
        assert info.n_tiles == 1
        assert info.w == 4

    def test_small_2x2_tile(self):
        code = dedent("""\
            Acc[0:4] = vmfma_f32_16x16x32_bf16(B0[0:4], A0[0:4], Acc[0:4])
            Acc[4:8] = vmfma_f32_16x16x32_bf16(B0[0:4], A0[4:8], Acc[4:8])
            Acc[8:12] = vmfma_f32_16x16x32_bf16(B0[4:8], A0[0:4], Acc[8:12])
            Acc[12:16] = vmfma_f32_16x16x32_bf16(B0[4:8], A0[4:8], Acc[12:16])
        """)
        info = analyze_tile(code)
        assert info is not None
        assert info.m_tiles == 2
        assert info.n_tiles == 2
        assert info.num_halves == 1
