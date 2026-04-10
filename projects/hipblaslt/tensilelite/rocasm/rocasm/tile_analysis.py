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

"""Tile analysis for generated rocasm mainloop code.

Parses the flat rocasm Python text produced by asm_to_rocasm, detects the
MFMA outer-product tile structure, and extracts a structured description
that can be used to generate loop-based code.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# ─── Data structures ─────────────────────────────────────────────────────────


@dataclass
class MfmaOp:
    """A single parsed MFMA instruction."""
    acc_start: int
    acc_end: int
    b_name: str      # e.g. "B0"
    b_start: int
    b_end: int
    a_name: str      # e.g. "A0"
    a_start: int
    a_end: int
    line: str         # original text line


@dataclass
class DsReadOp:
    """A parsed ds_read_b128 instruction."""
    dst_name: str     # e.g. "A1", "B0"
    dst_start: int
    dst_end: int
    offset: int       # LDS offset (0 if no DSModifiers)
    line: str


@dataclass
class HalfSchedule:
    """Schedule for one half of the double-buffered loop."""
    b_name: str         # e.g. "B0"
    a_name: str         # e.g. "A0"
    mfma_count: int
    interleave: dict[int, list[str]]  # MFMA index -> list of non-MFMA lines after it
    # index -1 = before first MFMA


@dataclass
class TileInfo:
    """Structured description of an MFMA outer-product tile."""
    m_tiles: int        # number of A chunks (inner loop dimension)
    n_tiles: int        # number of B chunks (outer loop dimension)
    w: int              # registers per MFMA operand (typically 4)
    num_halves: int     # typically 2 (double-buffered)
    a_names: list[str]  # e.g. ["A0", "A1"]
    b_names: list[str]  # e.g. ["B0", "B1"]
    a_lds_offsets: list[int] = field(default_factory=list)
    b_lds_offsets: list[int] = field(default_factory=list)
    lds_half_offset: int = 0
    gr_stride_a: int = 0
    gr_stride_b: int = 0
    n_global_a: int = 0
    n_global_b: int = 0
    halves: list[HalfSchedule] = field(default_factory=list)


# ─── Line classification ─────────────────────────────────────────────────────

# Regex for MFMA:  Acc[0:4] = vmfma_f32_16x16x32_bf16(B0[0:4], A0[0:4], Acc[0:4])
_MFMA_RE = re.compile(
    r'^Acc\[(\d+):(\d+)\]\s*=\s*vmfma_\w+\('
    r'(\w+)\[(\d+):(\d+)\],\s*'
    r'(\w+)\[(\d+):(\d+)\],\s*'
    r'Acc\[\d+:\d+\]\)$'
)

# Regex for ds_read:  A1[0:4] = ds_read_b128(LocalReadAddrA[0:1], ds=DSModifiers(offset=64))
_DS_READ_RE = re.compile(
    r'^(\w+)\[(\d+):(\d+)\]\s*=\s*ds_read_b128\('
    r'(\w+)\[.*\]'
    r'(?:,\s*ds=DSModifiers\(offset=(\d+)\))?'
    r'\)$'
)

# Regex for buffer_load_lds:  buffer_load_lds(GlobalReadOffsetA[0:1], SrdA[0:4], ...)
_BUFFER_LOAD_LDS_RE = re.compile(
    r'^buffer_load_lds\((\w+)\[.*\],\s*(\w+)\[.*\]'
)

# Regex for s_add_u32 with m0 and immediate (for global load stride detection)
_M0_ADD_RE = re.compile(
    r'^s_add_u32\(dst=m0,\s*src0=m0,\s*src1=(\d+)\)$'
)


def _parse_mfma(line: str) -> MfmaOp | None:
    """Try to parse a line as an MFMA instruction."""
    m = _MFMA_RE.match(line)
    if not m:
        return None
    return MfmaOp(
        acc_start=int(m.group(1)), acc_end=int(m.group(2)),
        b_name=m.group(3), b_start=int(m.group(4)), b_end=int(m.group(5)),
        a_name=m.group(6), a_start=int(m.group(7)), a_end=int(m.group(8)),
        line=line,
    )


def _parse_ds_read(line: str) -> DsReadOp | None:
    """Try to parse a line as a ds_read_b128 instruction."""
    m = _DS_READ_RE.match(line)
    if not m:
        return None
    return DsReadOp(
        dst_name=m.group(1), dst_start=int(m.group(2)), dst_end=int(m.group(3)),
        offset=int(m.group(5)) if m.group(5) else 0,
        line=line,
    )


# ─── Tile detection ──────────────────────────────────────────────────────────


def analyze_tile(rocasm_code: str) -> TileInfo | None:
    """Analyze flat rocasm code and extract the MFMA tile structure.

    Returns a TileInfo describing the tile dimensions, register names,
    LDS offsets, global load parameters, and the interleave schedule
    for each half.  Returns None if the code doesn't match the expected
    outer-product pattern.
    """
    lines = [l.strip() for l in rocasm_code.splitlines() if l.strip()]

    # Skip preamble (alias lines like "A0 = block.A0")
    body_start = 0
    for i, line in enumerate(lines):
        if "= block." in line:
            body_start = i + 1
        else:
            break
    # Actually, preamble lines all contain "= block." — find first non-preamble
    body_start = 0
    for i, line in enumerate(lines):
        if "= block." not in line:
            body_start = i
            break

    body = lines[body_start:]

    # Parse all MFMAs in order
    mfmas: list[MfmaOp] = []
    mfma_line_indices: list[int] = []
    for i, line in enumerate(body):
        op = _parse_mfma(line)
        if op:
            mfmas.append(op)
            mfma_line_indices.append(i)

    if not mfmas:
        return None

    # Detect halves: group consecutive MFMAs by (b_name, a_name)
    halves_raw: list[list[MfmaOp]] = []
    current_key = (mfmas[0].b_name, mfmas[0].a_name)
    current_group: list[MfmaOp] = [mfmas[0]]

    for op in mfmas[1:]:
        key = (op.b_name, op.a_name)
        if key != current_key:
            halves_raw.append(current_group)
            current_group = [op]
            current_key = key
        else:
            current_group.append(op)
    halves_raw.append(current_group)

    if len(halves_raw) < 1:
        return None

    # Detect tile dimensions from first half
    first_half = halves_raw[0]
    w = first_half[0].b_end - first_half[0].b_start  # MFMA width

    # A chunks: distinct A slices used. The inner loop cycles through them.
    a_slices = []
    seen_a = set()
    for op in first_half:
        key = (op.a_start, op.a_end)
        if key not in seen_a:
            seen_a.add(key)
            a_slices.append(key)

    # B chunks: distinct B slices used. The outer loop holds B constant.
    b_slices = []
    seen_b = set()
    for op in first_half:
        key = (op.b_start, op.b_end)
        if key not in seen_b:
            seen_b.add(key)
            b_slices.append(key)

    m_tiles = len(a_slices)
    n_tiles = len(b_slices)

    # Verify the tile is a complete outer product
    expected = m_tiles * n_tiles
    for half_ops in halves_raw:
        if len(half_ops) != expected:
            return None

    # Collect register names
    a_names = sorted({op.a_name for op in mfmas})
    b_names = sorted({op.b_name for op in mfmas})

    # Build interleave schedules — map non-MFMA lines to MFMA indices
    half_schedules = []
    mfma_idx = 0
    for half_ops in halves_raw:
        half_start = mfma_line_indices[mfma_idx]
        half_end = (mfma_line_indices[mfma_idx + len(half_ops) - 1]
                    if mfma_idx + len(half_ops) - 1 < len(mfma_line_indices)
                    else len(body) - 1)

        # Find the range of body lines for this half
        # Start: from previous half's last MFMA + 1 (or body start)
        if mfma_idx == 0:
            region_start = 0
        else:
            region_start = mfma_line_indices[mfma_idx - 1] + 1

        # End: next half's first MFMA (or end of body)
        next_half_start = (mfma_line_indices[mfma_idx + len(half_ops)]
                           if mfma_idx + len(half_ops) < len(mfma_line_indices)
                           else len(body))

        interleave: dict[int, list[str]] = {}
        local_mfma = 0
        current_mfma = -1  # -1 means "before first MFMA"

        for li in range(region_start, next_half_start):
            line = body[li]
            if li in mfma_line_indices and mfma_line_indices.index(li) >= mfma_idx:
                current_mfma = local_mfma
                local_mfma += 1
            else:
                interleave.setdefault(current_mfma, []).append(line)

        half_schedules.append(HalfSchedule(
            b_name=half_ops[0].b_name,
            a_name=half_ops[0].a_name,
            mfma_count=len(half_ops),
            interleave=interleave,
        ))
        mfma_idx += len(half_ops)

    # Extract LDS offsets from ds_read lines
    ds_reads: list[DsReadOp] = []
    for line in body:
        op = _parse_ds_read(line)
        if op:
            ds_reads.append(op)

    # Build offset tables: use the reads that target the "next iteration" registers
    # (A0 reads during half 1, which are the reads without the half offset)
    a_lds_offsets = _extract_lds_offsets(ds_reads, a_names[0], m_tiles, w)
    b_lds_offsets = _extract_lds_offsets(ds_reads, b_names[0], n_tiles, w)

    # Detect LDS half offset: difference between reads of A1 and A0 for the same chunk
    lds_half_offset = _detect_lds_half_offset(ds_reads, a_names, w)

    # Extract global load strides from s_add_u32(dst=m0, ...) patterns
    gr_stride_a, n_global_a, gr_stride_b, n_global_b = _extract_global_load_params(body)

    return TileInfo(
        m_tiles=m_tiles,
        n_tiles=n_tiles,
        w=w,
        num_halves=len(halves_raw),
        a_names=a_names,
        b_names=b_names,
        a_lds_offsets=a_lds_offsets,
        b_lds_offsets=b_lds_offsets,
        lds_half_offset=lds_half_offset,
        gr_stride_a=gr_stride_a,
        gr_stride_b=gr_stride_b,
        n_global_a=n_global_a,
        n_global_b=n_global_b,
        halves=half_schedules,
    )


def _extract_lds_offsets(ds_reads: list[DsReadOp], reg_name: str,
                         n_chunks: int, w: int) -> list[int]:
    """Extract the LDS offset table for a register from ds_read ops.

    Looks for reads targeting `reg_name` and builds the offset list
    indexed by chunk number (derived from dst_start / w).
    """
    offsets: dict[int, int] = {}
    for op in ds_reads:
        if op.dst_name == reg_name:
            chunk = op.dst_start // w
            if chunk not in offsets:
                offsets[chunk] = op.offset
    return [offsets.get(i, 0) for i in range(n_chunks)]


def _detect_lds_half_offset(ds_reads: list[DsReadOp],
                            a_names: list[str], w: int) -> int:
    """Detect the LDS double-buffer half offset.

    Compares offsets of chunk-0 reads for A0 vs A1. The difference is the
    half offset used for double-buffering.
    """
    if len(a_names) < 2:
        return 0

    chunk0_offsets: dict[str, int] = {}
    for op in ds_reads:
        if op.dst_name in a_names and op.dst_start == 0:
            if op.dst_name not in chunk0_offsets:
                chunk0_offsets[op.dst_name] = op.offset

    if len(chunk0_offsets) >= 2:
        vals = list(chunk0_offsets.values())
        diff = abs(vals[0] - vals[1])
        if diff > 0:
            return diff
    return 0


def _extract_global_load_params(body: list[str]) -> tuple[int, int, int, int]:
    """Extract global load stride and count for A and B.

    Scans for s_add_u32(dst=m0, src0=m0, src1=N) patterns and
    buffer_load_lds calls to determine strides and counts.

    Returns (gr_stride_a, n_global_a, gr_stride_b, n_global_b).
    """
    # Find all buffer_load_lds calls and the m0 strides preceding them
    loads_a: list[int] = []  # strides for A loads
    loads_b: list[int] = []  # strides for B loads
    pending_stride = 0

    for line in body:
        m = _M0_ADD_RE.match(line)
        if m:
            pending_stride = int(m.group(1))
            continue

        m = _BUFFER_LOAD_LDS_RE.match(line)
        if m:
            reg_name = m.group(1)  # e.g. "GlobalReadOffsetA"
            if "A" in reg_name:
                loads_a.append(pending_stride)
            elif "B" in reg_name:
                loads_b.append(pending_stride)
            pending_stride = 0

    gr_stride_a = max(loads_a) if loads_a else 0
    gr_stride_b = max(loads_b) if loads_b else 0
    return gr_stride_a, len(loads_a), gr_stride_b, len(loads_b)


# ─── Tiled mainloop code generation ──────────────────────────────────────────


def generate_tiled_mainloop(tile: TileInfo,
                            named_regs: dict[str, tuple[str, int, int]],
                            reg_counts: dict[str, int]) -> str:
    """Generate loop-based mainloop Python source from a TileInfo.

    Produces a complete Python module with:
    - Imports (rocasm, tile_helpers)
    - Tile parameter constants
    - ``rocasm_main_loop()`` function with Block init, helpers, and loops

    Args:
        tile: TileInfo from analyze_tile().
        named_regs: Alias -> (reg_type, phys_base, count) from asm_to_rocasm.
        reg_counts: {"vgpr": N, "accvgpr": N, "sgpr": N} from the kernel header.

    Returns:
        Complete Python source code string.
    """
    num_accvgprs = reg_counts.get("accvgpr", 192)
    _ARRAY_TYPE = {"v": "VgprArray", "s": "SgprArray", "acc": "AccArray"}

    out = []

    # ─── Imports ──────────────────────────────────────────────────────────
    out.append("from rocasm.block import Block")
    out.append("from rocasm.regs import VgprArray, AccArray, SgprArray")
    out.append("from rocasm.instructions import vmfma_f32_16x16x32_bf16, ds_read_b128, buffer_load_dwordx4")
    out.append("from rocasm.tile_helpers import mfma_at, lds_read_chunk, global_load")
    out.append("")
    out.append("")

    # ─── Tile constants ───────────────────────────────────────────────────
    out.append(f"M_TILES = {tile.m_tiles}   # A chunks (inner)")
    out.append(f"N_TILES = {tile.n_tiles}   # B chunks (outer)")
    out.append(f"W = {tile.w}         # registers per MFMA operand")
    out.append("")
    out.append(f"A_LDS_OFFSETS = {tile.a_lds_offsets}")
    out.append(f"B_LDS_OFFSETS = {tile.b_lds_offsets}")
    out.append(f"LDS_HALF = {tile.lds_half_offset}")
    out.append("")
    out.append(f"GR_STRIDE_A = {tile.gr_stride_a}")
    out.append(f"GR_STRIDE_B = {tile.gr_stride_b}")
    out.append("")
    out.append("")

    # ─── Function body ────────────────────────────────────────────────────
    out.append("def rocasm_main_loop():")
    out.append('    """ROCasm tiled translation of the main loop.')
    out.append("")
    out.append("    Auto-generated from assembly via asm_to_rocasm + tile_analysis.")
    out.append("    The MFMA outer product is expressed as loops; non-MFMA ops are")
    out.append("    interleaved at specific MFMA indices to hide latency.")
    out.append('    """')

    # Block init
    out.append("    block = Block(")
    for alias in sorted(named_regs):
        reg_type, phys_base, count = named_regs[alias]
        arr_type = _ARRAY_TYPE[reg_type]
        out.append(f"        {alias}={arr_type}(base={phys_base}, count={count}),")
    out.append(f"        Acc=AccArray(base=0, count={num_accvgprs}),")
    out.append("    )")
    out.append("")

    # Local aliases — collect all names used in the interleave schedules
    all_interleave_lines = []
    for h in tile.halves:
        for lines in h.interleave.values():
            all_interleave_lines.extend(lines)

    # Named reg aliases
    for alias in sorted(named_regs):
        out.append(f"    {alias} = block.{alias}")
    out.append("    Acc = block.Acc")

    # Detect block methods used in interleave lines
    method_names = set()
    for line in all_interleave_lines:
        # Extract function name from lines like "s_waitcnt(...)" or "s_add_u32(...)"
        m = re.match(r'^(\w+)\(', line)
        if m:
            method_names.add(m.group(1))

    for method in sorted(method_names):
        out.append(f"    {method} = block.{method}")
    out.append("")

    # ─── run_half: shared MFMA tile loop ────────────────────────────────
    out.append("    def run_half(B, A, interleave):")
    out.append('        """Run one half of the double-buffered MFMA tile.')
    out.append("")
    out.append("        The 8x6 outer product is identical in both halves;")
    out.append("        only the interleaved memory/scalar ops differ.")
    out.append('        """')
    out.append("        interleave(-1)")
    out.append("        for b in range(N_TILES):")
    out.append("            for a in range(M_TILES):")
    out.append("                idx = b * M_TILES + a")
    out.append("                mfma_at(Acc, B, A, b, a, M_TILES, W, vmfma_f32_16x16x32_bf16)")
    out.append("                interleave(idx)")
    out.append("")

    # ─── Per-half interleave callbacks ────────────────────────────────────
    for half_idx, half in enumerate(tile.halves):
        b_name = half.b_name
        a_name = half.a_name
        func_name = f"half{half_idx}_interleave"

        out.append(f"    def {func_name}(idx):")
        out.append(f'        """Interleaved ops for half {half_idx} ({b_name}/{a_name})."""')

        has_any = False

        # Pre-MFMA ops (idx == -1)
        if -1 in half.interleave:
            pre_ops = [l for l in half.interleave[-1] if not l.startswith('label(')]
            if pre_ops:
                has_any = True
                out.append("        if idx == -1:")
                for line in pre_ops:
                    out.append(f"            {line}")

        # Post-MFMA ops per index
        indices_with_ops = sorted(k for k in half.interleave if k >= 0)
        for idx_val in indices_with_ops:
            ops = [l for l in half.interleave[idx_val]
                   if not l.startswith("return ")]
            if not ops:
                continue
            has_any = True
            if len(ops) == 1:
                out.append(f"        if idx == {idx_val}: {ops[0]}")
            else:
                out.append(f"        if idx == {idx_val}:")
                for line in ops:
                    out.append(f"            {line}")

        if not has_any:
            out.append("        pass")

        out.append("")

    # ─── Drive the loop ──────────────────────────────────────────────────
    out.append('    label("label_LoopBeginL")')
    for half_idx, half in enumerate(tile.halves):
        out.append(f"    run_half({half.b_name}, {half.a_name}, half{half_idx}_interleave)")
    out.append("")
    out.append("    return block")

    return "\n".join(out) + "\n"
