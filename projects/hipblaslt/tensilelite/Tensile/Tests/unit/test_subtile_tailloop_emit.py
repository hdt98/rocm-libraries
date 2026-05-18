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
# SPDX-License-Identifier: MIT
################################################################################
"""Tail-loop emit assertions (Phase 1.4).

These tests pin the *content* of the subtile tail-loop emit body, complementing
the structural assertions in `test_SubtileBasedLogicalScheduler.py`'s
`TestEmitAllLoopsTail_PGR0` (which cover the *placement* and *labels*).

Phase 3 will add to the tail body:
  - Step 2-1-2: SrdA / SrdB / SrdMXSA / SrdMXSB rewind (s_sub_u32 chain).
  - Step 2-2:   per-lane K-position vgpr (kReg_first via v_and_b32 #63,
                v_lshrrev_b32, v_lshlrev_b32 sequence).
  - Step 2-3:   per-vgpr-pair lane mask (v_cmp_ge_i32 + v_cndmask_b32 to 0).

When ASEM>=32 (the Phase-2-enabled minimum), step 2-4 (the legacy byte-shift
mask `s_lshlrev_b64` for ASEM<32 / odd-K) is unnecessary and must be skipped.

These tests invoke `LogicalScheduler.emitAllLoops` (the same emit path that
`KernelWriter.kernelBodySubtile` uses to materialise the tail body) and check
the asm string for the expected instruction patterns. They use the same
fixtures as `test_SubtileBasedLogicalScheduler.py`; helpers are duplicated
here to avoid cross-test-file imports.
"""
import re

import pytest

from types import SimpleNamespace
from unittest.mock import MagicMock

from Tensile.Components.Subtile.LogicalScheduler import (
    LogicalScheduler, SchedulerConfig, ReadGranularity,
)
from Tensile.Components.Subtile.Kernel import (
    TileInfo, AB_B16, AB_B4, MXSA_B4, MXSB_B4, CD_F32,
)


def makeTileInfo(tc, kernel):
    """Local TileInfo factory (same shape as test_SubtileBasedLogicalScheduler)."""
    fp4 = kernel["ProblemType"].get("MXBlockA", 0) > 0
    _geo = {
        "A": AB_B4 if fp4 else AB_B16,
        "B": AB_B4 if fp4 else AB_B16,
        "MXSA": MXSA_B4,
        "MXSB": MXSB_B4,
        "D": CD_F32,
    }
    return TileInfo(_geo[tc], tc, None, kernel)


# ── Mock kernel + writer (duplicated from test_SubtileBasedLogicalScheduler) ──

def _mock_dtype(num_bytes=2):
    """Create a mock DataType with numBytes() returning the given size."""
    mock = MagicMock()
    mock.numBytes.return_value = num_bytes
    mock.numRegisters.return_value = num_bytes / 4
    mock.isFloat4.return_value = num_bytes == 0.5
    mock.is6bitFloat.return_value = False
    mock.is8bitFloat.return_value = num_bytes == 1
    mock.isHalf.return_value = num_bytes == 2
    mock.isBFloat16.return_value = num_bytes == 2
    mock.isSingle.return_value = num_bytes == 4
    return mock


def _create_kernel(MT0=256, MT1=256, *, fp4=False, depthU=None, no_tail_loop=False):
    """Minimal kernel dict driving tail-loop emit logic."""
    mxblock = 32 if fp4 else 0
    bpe = 0.5 if fp4 else 2
    matrixInstK = 128 if fp4 else 32
    if depthU is None:
        depthU = 256 if fp4 else 64

    dtype = _mock_dtype(bpe)
    problemType = {
        "DataTypeA": dtype,
        "DataTypeB": dtype,
        "ComputeDataType": _mock_dtype(4),
    }
    if fp4:
        problemType["MXBlockA"] = mxblock
        problemType["MXBlockB"] = mxblock

    kernel = {
        "DepthU": depthU,
        "_DepthUA": depthU,
        "_DepthUB": depthU,
        "MacroTileA": MT0,
        "MacroTileB": MT1,
        "MacroTile0": MT0,
        "MacroTile1": MT1,
        "MatrixInstM": 16,
        "MatrixInstN": 16,
        "MatrixInstK": matrixInstK,
        "MIWaveGroup": [2, 2],
        "WavefrontSize": 64,
        "SourceSwap": False,
        "MIArchVgpr": False,
        "NonTemporalA": 0,
        "NonTemporalB": 0,
        "NonTemporalMXSA": 0,
        "NonTemporalMXSB": 0,
        "ProblemType": problemType,
        "NoTailLoop": no_tail_loop,
        "AssertSummationElementMultiple": 32,
    }
    if fp4:
        kernel["_DepthUMXSA"] = depthU // mxblock
        kernel["_DepthUMXSB"] = depthU // mxblock
    return kernel


def _make_writer_and_tileinfos(kernel, fp4=False):
    """Build a writer + TileInfos sufficient for emitAllLoops to run."""
    from rocisa import rocIsa
    from rocisa.register import RegisterPool
    from rocisa.enum import RegisterType

    ri = rocIsa.getInstance()
    if not ri.isInit():
        import shutil
        asmpath = shutil.which("amdclang++") or "/usr/bin/amdclang++"
        ri.init((9, 5, 0), asmpath)
    ri.setKernel((9, 5, 0), 64)

    tiA = makeTileInfo("A", kernel)
    tiB = makeTileInfo("B", kernel)
    scaleTiA = makeTileInfo("MXSA", kernel) if fp4 else None
    scaleTiB = makeTileInfo("MXSB", kernel) if fp4 else None

    writer = SimpleNamespace()
    writer.vgprPool = RegisterPool(0, RegisterType.Vgpr, False)
    writer.agprPool = RegisterPool(0, RegisterType.Accvgpr, False)
    writer.sgprPool = RegisterPool(0, RegisterType.Sgpr, False)
    writer.states = SimpleNamespace(
        regCaps={"MaxSgpr": 106, "MaxVgpr": 256, "PhysicalMaxVgpr": 512},
    )
    dTileInfo = makeTileInfo("D", kernel)
    dTileInfo.allocVgprTileRegisters_legacy(writer, kernel)
    writer.states.d = SimpleNamespace(tileInfo=dTileInfo)
    writer.states.a = SimpleNamespace(tileInfo=tiA)
    writer.states.b = SimpleNamespace(tileInfo=tiB)
    tiA.allocOffsetRegisters(writer, kernel)
    tiB.allocOffsetRegisters(writer, kernel)
    if scaleTiA and scaleTiB:
        writer.states.mxsa = SimpleNamespace(tileInfo=scaleTiA)
        writer.states.mxsb = SimpleNamespace(tileInfo=scaleTiB)
        scaleTiA.allocOffsetRegisters(writer, kernel)
        scaleTiB.allocOffsetRegisters(writer, kernel)

    return writer, tiA, tiB, scaleTiA, scaleTiB, dTileInfo


def _make_cfg_256x256_fp4(pgr=2):
    kernel = _create_kernel(256, 256, fp4=True, depthU=256)
    tiA = makeTileInfo("A", kernel)
    tiB = makeTileInfo("B", kernel)
    scaleTiA = makeTileInfo("MXSA", kernel)
    scaleTiB = makeTileInfo("MXSB", kernel)
    return SchedulerConfig(
        numMFMATilesM=tiA.localMMATileGrid[0],
        numMFMATilesN=tiB.localMMATileGrid[0],
        numSubIterK=tiA.localMMATileGrid[1],
        lrA=ReadGranularity(mn=1, k=1),
        lrB=ReadGranularity(mn=1, k=1),
        grA=ReadGranularity(mn=1, k=2),
        grB=ReadGranularity(mn=1, k=2),
        lrSA=ReadGranularity(mn=2, k=2),
        lrSB=ReadGranularity(mn=2, k=2),
        grSA=ReadGranularity(mn=scaleTiA.localMMATileGrid[0],
                            k=scaleTiA.localMMATileGrid[1]),
        grSB=ReadGranularity(mn=scaleTiB.localMMATileGrid[0],
                            k=scaleTiB.localMMATileGrid[1]),
        numPartitionsM=1,
        numPartitionsN=1,
        pgr=pgr,
    )


def _make_cfg_bf16(pgr=2, depthU=64):
    kernel = _create_kernel(256, 256, fp4=False, depthU=depthU)
    tiA = makeTileInfo("A", kernel)
    tiB = makeTileInfo("B", kernel)
    return SchedulerConfig(
        numMFMATilesM=tiA.localMMATileGrid[0],
        numMFMATilesN=tiB.localMMATileGrid[0],
        numSubIterK=tiA.localMMATileGrid[1],
        lrA=ReadGranularity(mn=1, k=1),
        lrB=ReadGranularity(mn=1, k=1),
        grA=ReadGranularity(mn=1, k=2),
        grB=ReadGranularity(mn=1, k=2),
        pgr=pgr,
    )


def _emit_all_loops_asm(*, fp4: bool, no_tail_loop: bool, pgr: int):
    """End-to-end emit: build scheduler, populate, emit, return asm string."""
    if fp4:
        kernel = _create_kernel(256, 256, fp4=True, depthU=256,
                                no_tail_loop=no_tail_loop)
        cfg = _make_cfg_256x256_fp4(pgr=pgr)
    else:
        kernel = _create_kernel(256, 256, fp4=False, depthU=64,
                                no_tail_loop=no_tail_loop)
        cfg = _make_cfg_bf16(pgr=pgr)

    writer, tiA, tiB, scaleTiA, scaleTiB, dTileInfo = \
        _make_writer_and_tileinfos(kernel, fp4=fp4)

    sched = LogicalScheduler(cfg)
    sched.build()
    sched.allocVgprTiles(writer, tiA, tiB,
                         scaleTileInfoA=scaleTiA, scaleTileInfoB=scaleTiB)
    try:
        sched.populate_instructions(
            writer, kernel,
            tileInfoA=tiA, tileInfoB=tiB,
            dtileInfo=dTileInfo,
            scaleTileInfoA=scaleTiA, scaleTileInfoB=scaleTiB,
        )
        module = sched.emitAllLoops(writer, kernel)
        return str(module)
    finally:
        sched.deallocVgprTiles(writer)


def _extract_tail_section(asm: str) -> str:
    """Extract the tail-loop section of the emitted asm.

    The tail body lives between the `TAILLOOP` comment / label region and the
    post-tail terminator (`SkipTailLoopL` after Phase 3.1, or `TailLoopEnd` in
    the imported template). Returns "" if no tail block is present.
    """
    tail_start = asm.find("TAILLOOP")
    if tail_start < 0:
        return ""
    # End at whichever post-tail label appears next.
    candidates = [
        asm.find("SkipTailLoopL:", tail_start),
        asm.find("TailLoopEnd:", tail_start),
        asm.find("SkipToEnd:", tail_start),
    ]
    candidates = [c for c in candidates if c > tail_start]
    tail_end = min(candidates) if candidates else len(asm)
    return asm[tail_start:tail_end]


# ── Tests: PGR=0 ─────────────────────────────────────────────────────────────

class TestTailEmitContent_PGR0:
    """Tail-body content assertions for PGR=0 kernels."""

    @pytest.fixture
    def fp4_pgr0_asm(self):
        return _emit_all_loops_asm(fp4=True, no_tail_loop=False, pgr=0)

    @pytest.fixture
    def bf16_pgr0_asm(self):
        return _emit_all_loops_asm(fp4=False, no_tail_loop=False, pgr=0)

    def test_emits_srd_rewind_A(self, fp4_pgr0_asm):
        """Tail body must rewind SrdA by the unread bytes (DepthU - K_rem) * bpe.

        Step 2-1-2 of the tail-loop init. Today's imported template doesn't
        emit this — only Phase 3.1 will.
        """
        tail = _extract_tail_section(fp4_pgr0_asm)
        assert tail, "No tail block emitted; cannot test SRD rewind"
        assert re.search(r"s_sub_u32.*SrdA", tail), (
            "Tail must rewind SrdA via s_sub_u32. "
            "Tail body excerpt:\n" + tail[:1500]
        )

    def test_emits_srd_rewind_B(self, fp4_pgr0_asm):
        """Same SRD-rewind requirement for SrdB."""
        tail = _extract_tail_section(fp4_pgr0_asm)
        assert tail
        assert re.search(r"s_sub_u32.*SrdB", tail), (
            "Tail must rewind SrdB via s_sub_u32"
        )

    def test_emits_srd_rewind_MXSA(self, fp4_pgr0_asm):
        """MX kernels must rewind the scale tensor SRDs too (32-element block)."""
        tail = _extract_tail_section(fp4_pgr0_asm)
        assert tail
        assert re.search(r"s_sub_u32.*SrdMXSA", tail), (
            "FP4 tail must rewind SrdMXSA"
        )

    def test_emits_srd_rewind_MXSB(self, fp4_pgr0_asm):
        tail = _extract_tail_section(fp4_pgr0_asm)
        assert tail
        assert re.search(r"s_sub_u32.*SrdMXSB", tail), (
            "FP4 tail must rewind SrdMXSB"
        )

    def test_emits_kReg_first_init(self, fp4_pgr0_asm):
        """Step 2-2: per-lane K position must be materialized.

        The expected sequence is `v_and_b32 v[?], 63, vgprSerial` followed by
        a shift chain to convert lane-id into a K-position. Phase 3.2 adds
        this.
        """
        tail = _extract_tail_section(fp4_pgr0_asm)
        assert tail
        assert re.search(r"v_and_b32.*\b63\b", tail) or \
               re.search(r"v_and_b32.*0x3f", tail), (
            "Tail must compute kReg_first via v_and_b32 with mask 63 "
            "(lane id within wave). Tail excerpt:\n" + tail[:1500]
        )

    def test_emits_lane_mask_for_valuA(self, fp4_pgr0_asm):
        """Step 2-3: each MFMA-input vgpr-pair gets a lane mask zeroing
        lanes whose K-position >= LoopCounterL.

        Pattern: `v_cmp_ge_i32 ..., LoopCounterL` followed by `v_cndmask_b32
        v[ValuA_*], v[ValuA_*], 0, ...`. Phase 3.3 adds this.
        """
        tail = _extract_tail_section(fp4_pgr0_asm)
        assert tail
        assert re.search(r"v_cmp_ge_i32.*LoopCounterL", tail), (
            "Tail must compare per-lane K-pos to LoopCounterL via v_cmp_ge_i32"
        )
        assert re.search(r"v_cndmask_b32.*ValuA", tail) or \
               re.search(r"v_cndmask_b32.*valuA", tail), (
            "Tail must zero out-of-range valuA vgprs via v_cndmask_b32"
        )

    def test_emits_lane_mask_for_valuB(self, fp4_pgr0_asm):
        tail = _extract_tail_section(fp4_pgr0_asm)
        assert tail
        assert re.search(r"v_cndmask_b32.*ValuB", tail) or \
               re.search(r"v_cndmask_b32.*valuB", tail), (
            "Tail must zero out-of-range valuB vgprs via v_cndmask_b32"
        )

    def test_emits_lane_mask_for_valuMXSA_MXSB(self, fp4_pgr0_asm):
        """MX kernels need the same lane mask for the scale tensor vgprs."""
        tail = _extract_tail_section(fp4_pgr0_asm)
        assert tail
        has_mxsa_mask = (re.search(r"v_cndmask_b32.*ValuMXSA", tail) or
                         re.search(r"v_cndmask_b32.*valuMXSA", tail))
        has_mxsb_mask = (re.search(r"v_cndmask_b32.*ValuMXSB", tail) or
                         re.search(r"v_cndmask_b32.*valuMXSB", tail))
        assert has_mxsa_mask, "Tail must lane-mask valuMXSA"
        assert has_mxsb_mask, "Tail must lane-mask valuMXSB"

    def test_omits_byte_shift_mask_when_asem_32(self, fp4_pgr0_asm):
        """Step 2-4 of the legacy non-subtile tail (ASEM<32 byte-shift mask
        via `s_lshlrev_b64`) is unnecessary when ASEM>=32. It must not appear.

        This passes today (the imported template doesn't emit step 2-4 either)
        but is a regression guard against accidental porting from the legacy
        emitter.
        """
        tail = _extract_tail_section(fp4_pgr0_asm)
        if not tail:
            pytest.skip("Tail block not emitted; nothing to assert")
        assert "s_lshlrev_b64" not in tail.lower(), (
            "ASEM=32 path must not emit the legacy byte-shift mask "
            "(s_lshlrev_b64 sequence)"
        )

    def test_NoTailLoop_true_omits_all_tail_emit(self):
        """Aligned-K kernels (NoTailLoop=True) must not emit any tail content.

        Control test — passes today.
        """
        asm = _emit_all_loops_asm(fp4=True, no_tail_loop=True, pgr=0)
        assert "TAILLOOP" not in asm, "NoTailLoop=True must skip emit"
        assert "SkipTailLoopL" not in asm
        assert "TailLoopEnd" not in asm
        assert not re.search(r"s_sub_u32.*SrdA.*K_rem", asm)


# ── Tests: PGR=2 ─────────────────────────────────────────────────────────────

class TestTailEmitContent_PGR2:
    """PGR=2 (scheduler-managed prefetch) tail-emit assertions."""

    @pytest.fixture
    def fp4_pgr2_asm(self):
        return _emit_all_loops_asm(fp4=True, no_tail_loop=False, pgr=2)

    def test_emits_SkipTailLoopL_label(self, fp4_pgr2_asm):
        """The PGR=2 path must use the same SkipTailLoopL label as PGR=0."""
        assert "SkipTailLoopL:" in fp4_pgr2_asm, (
            "PGR=2 tail emit must declare a SkipTailLoopL label"
        )

    def test_emits_srd_rewind_A_pgr2(self, fp4_pgr2_asm):
        tail = _extract_tail_section(fp4_pgr2_asm)
        assert tail
        assert re.search(r"s_sub_u32.*SrdA", tail), (
            "PGR=2 tail must rewind SrdA"
        )

    def test_emits_lane_mask_for_valuA_pgr2(self, fp4_pgr2_asm):
        tail = _extract_tail_section(fp4_pgr2_asm)
        assert tail
        assert re.search(r"v_cndmask_b32.*ValuA", tail) or \
               re.search(r"v_cndmask_b32.*valuA", tail), (
            "PGR=2 tail must lane-mask valuA"
        )

    def test_emits_kReg_first_init_pgr2(self, fp4_pgr2_asm):
        tail = _extract_tail_section(fp4_pgr2_asm)
        assert tail
        assert re.search(r"v_and_b32.*\b63\b", tail) or \
               re.search(r"v_and_b32.*0x3f", tail), (
            "PGR=2 tail must compute kReg_first"
        )
