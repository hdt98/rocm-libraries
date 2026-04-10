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

"""Tile helper functions for generated rocasm mainloop modules.

These are parameterized helpers used by the loop-based (tiled) mainloop code.
Each function takes register arrays and instruction callables as arguments so
it can be used from any generated module without closing over block state.

The helpers are designed to be *called at code-generation time* — they invoke
rocasm instruction functions which record operations into a Block.
"""

from __future__ import annotations

from rocisa.container import DSModifiers, MUBUFModifiers


def mfma_at(Acc, B_reg, A_reg, b, a, m_tiles, w, mfma_func):
    """Execute one MFMA at outer-product tile position (b, a).

    In an MxN tile where B is the outer (N) dimension and A is the inner (M),
    the accumulator index is ``(b * m_tiles + a) * w``.
    """
    i = (b * m_tiles + a) * w
    Acc[i:i+w] = mfma_func(
        B_reg[b*w:(b+1)*w], A_reg[a*w:(a+1)*w], Acc[i:i+w])


def lds_read_chunk(dst, chunk, w, addr_reg, offsets, ds_read_func, extra=0):
    """Read one chunk from LDS using a table of offsets.

    Args:
        dst: Destination register array (e.g. A1 or B0).
        chunk: Chunk index into the offset table.
        w: Registers per chunk.
        addr_reg: LDS address register slice (e.g. LocalReadAddrA[0:1]).
        offsets: List of base LDS offsets per chunk.
        ds_read_func: The ds_read_b128 instruction callable.
        extra: Additional offset (e.g. LDS double-buffer half offset).
    """
    offset = offsets[chunk] + extra
    start = chunk * w
    if offset:
        dst[start:start+w] = ds_read_func(addr_reg, ds=DSModifiers(offset=offset))
    else:
        dst[start:start+w] = ds_read_func(addr_reg)


def global_load(offset_reg, idx, srd, stride, s_add_u32_func,
                buffer_load_lds_func):
    """Issue one global-to-LDS load, advancing m0 if not the first.

    Args:
        offset_reg: The GlobalReadOffset register array.
        idx: Load index (0-based). If > 0, m0 is advanced by stride first.
        srd: SRD register slice (e.g. SrdA[0:4]).
        stride: Bytes between consecutive loads (m0 advance amount).
        s_add_u32_func: The s_add_u32 instruction callable.
        buffer_load_lds_func: The buffer_load_lds callable.
    """
    if idx > 0:
        s_add_u32_func(dst="m0", src0="m0", src1=stride)
    buffer_load_lds_func(
        offset_reg[idx:idx+1], srd, 0,
        mubuf=MUBUFModifiers(offen=True, lds=True))
