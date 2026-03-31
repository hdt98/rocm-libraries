################################################################################
#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################

import pytest
from copy import deepcopy

import rocisa
from rocisa import rocIsa
from rocisa.enum import InstType
from rocisa.container import vgpr, sgpr
from rocisa.instruction import MXMFMAInstruction


def init_rocisa(version=(9, 5, 0), wavefront_size=64):
    ti = rocIsa.getInstance()
    ti.init(version, "amdclang++")
    ti.setKernel(version, wavefront_size)


def make_mxmfma(instType=InstType.INST_F8, mxScaleAType=InstType.INST_F32,
                mxScaleBType=InstType.INST_F32, variant=None, block=0, comment=""):
    if variant is None:
        variant = [16, 16, 128]
    return MXMFMAInstruction(
        instType=instType,
        accType=InstType.INST_F32,
        mxScaleAType=mxScaleAType,
        mxScaleBType=mxScaleBType,
        variant=variant,
        acc=vgpr(0, 4),
        a=vgpr(4, 4),
        b=vgpr(8, 4),
        acc2=vgpr(12, 4),
        mxsa=sgpr(0, 4),
        mxsb=sgpr(4, 4),
        block=block,
        comment=comment,
    )


def _uses_mfma_path():
    """Runtime detection: does the current code+ISA produce MFMA-style output?"""
    inst = make_mxmfma()
    return "v_mfma_scale_" in str(inst)


# ---------------------------------------------------------------------------
# Basic construction and output
# ---------------------------------------------------------------------------

def test_mxmfma_basic_toString():
    init_rocisa()
    inst = make_mxmfma()
    result = str(inst)
    if _uses_mfma_path():
        assert result.startswith("v_mfma_scale_f32_16x16x128_f8f6f4 ")
    else:
        assert result.startswith("v_wmma_scale_f32_16x16x128_f8f6f4 ")
    assert "v[0:3]" in result
    assert "v[4:7]" in result
    assert "v[8:11]" in result
    assert "v[12:15]" in result
    assert "s[0:3]" in result
    assert "s[4:7]" in result
    assert result.endswith("\n")


def test_mxmfma_variant_in_name():
    init_rocisa()
    inst = make_mxmfma(variant=[32, 16, 64])
    result = str(inst)
    assert "32x16x64" in result


# ---------------------------------------------------------------------------
# WMMA path tests (1250-only branch always uses this; unified branch uses
# this when HasMFMA=false)
# ---------------------------------------------------------------------------

def test_mxmfma_wmma_prestr():
    init_rocisa((12, 5, 0))
    if _uses_mfma_path():
        pytest.skip("MFMA path active, WMMA tests not applicable")
    inst = make_mxmfma()
    result = str(inst)
    assert "v_wmma_scale_f32_16x16x128_f8f6f4" in result


def test_mxmfma_wmma_block16():
    init_rocisa((12, 5, 0))
    if _uses_mfma_path():
        pytest.skip("MFMA path active, WMMA tests not applicable")
    inst = make_mxmfma(block=16)
    result = str(inst)
    assert "v_wmma_scale16_f32_16x16x128_f8f6f4" in result


def test_mxmfma_wmma_block0():
    init_rocisa((12, 5, 0))
    if _uses_mfma_path():
        pytest.skip("MFMA path active, WMMA tests not applicable")
    inst = make_mxmfma(block=0)
    result = str(inst)
    assert "v_wmma_scale_f32_" in result
    assert "v_wmma_scale16_" not in result


def test_mxmfma_wmma_typeconvert_f8f6f4():
    init_rocisa((12, 5, 0))
    if _uses_mfma_path():
        pytest.skip("MFMA path active, WMMA tests not applicable")
    inst = make_mxmfma(variant=[16, 16, 128])
    result = str(inst)
    assert "_f8f6f4" in result


def test_mxmfma_wmma_typeconvert_f4():
    init_rocisa((12, 5, 0))
    if _uses_mfma_path():
        pytest.skip("MFMA path active, WMMA tests not applicable")
    inst = make_mxmfma(instType=InstType.INST_F4, variant=[32, 32, 128])
    result = str(inst)
    assert "v_wmma_scale_f32_32x32x128_f4" in result


@pytest.mark.parametrize("instType,expected_a,expected_b", [
    (InstType.INST_F8,     "MATRIX_FMT_FP8",  "MATRIX_FMT_FP8"),
    (InstType.INST_BF8,    "MATRIX_FMT_BF8",  "MATRIX_FMT_BF8"),
    (InstType.INST_F8_BF8, "MATRIX_FMT_FP8",  "MATRIX_FMT_BF8"),
    (InstType.INST_BF8_F8, "MATRIX_FMT_BF8",  "MATRIX_FMT_FP8"),
    (InstType.INST_F6,     "MATRIX_FMT_FP6",  "MATRIX_FMT_FP6"),
    (InstType.INST_BF6,    "MATRIX_FMT_BF6",  "MATRIX_FMT_BF6"),
    (InstType.INST_F6_B6,  "MATRIX_FMT_FP6",  "MATRIX_FMT_BF6"),
    (InstType.INST_B6_F6,  "MATRIX_FMT_BF6",  "MATRIX_FMT_FP6"),
    (InstType.INST_F8_F4,  "MATRIX_FMT_FP8",  "MATRIX_FMT_FP4"),
    (InstType.INST_F4_F8,  "MATRIX_FMT_FP4",  "MATRIX_FMT_FP8"),
    (InstType.INST_F6_F4,  "MATRIX_FMT_FP6",  "MATRIX_FMT_FP4"),
    (InstType.INST_F4_F6,  "MATRIX_FMT_FP4",  "MATRIX_FMT_FP6"),
    (InstType.INST_F8_F6,  "MATRIX_FMT_FP8",  "MATRIX_FMT_FP6"),
    (InstType.INST_F6_F8,  "MATRIX_FMT_FP6",  "MATRIX_FMT_FP8"),
    (InstType.INST_F8_B6,  "MATRIX_FMT_FP8",  "MATRIX_FMT_BF6"),
    (InstType.INST_B6_F8,  "MATRIX_FMT_BF6",  "MATRIX_FMT_FP8"),
    (InstType.INST_B8_F4,  "MATRIX_FMT_BF8",  "MATRIX_FMT_FP4"),
    (InstType.INST_F4_B8,  "MATRIX_FMT_FP4",  "MATRIX_FMT_BF8"),
    (InstType.INST_B6_F4,  "MATRIX_FMT_BF6",  "MATRIX_FMT_FP4"),
    (InstType.INST_F4_B6,  "MATRIX_FMT_FP4",  "MATRIX_FMT_BF6"),
    (InstType.INST_B8_F6,  "MATRIX_FMT_BF8",  "MATRIX_FMT_FP6"),
    (InstType.INST_F6_B8,  "MATRIX_FMT_FP6",  "MATRIX_FMT_BF8"),
    (InstType.INST_B8_B6,  "MATRIX_FMT_BF8",  "MATRIX_FMT_BF6"),
    (InstType.INST_B6_B8,  "MATRIX_FMT_BF6",  "MATRIX_FMT_BF8"),
])
def test_mxmfma_wmma_input_permute(instType, expected_a, expected_b):
    init_rocisa((12, 5, 0))
    if _uses_mfma_path():
        pytest.skip("MFMA path active, WMMA tests not applicable")
    inst = make_mxmfma(instType=instType, variant=[16, 16, 128])
    result = str(inst)
    assert f"matrix_a_fmt:{expected_a}" in result
    assert f"matrix_b_fmt:{expected_b}" in result


def test_mxmfma_wmma_no_permute_small_k():
    init_rocisa((12, 5, 0))
    if _uses_mfma_path():
        pytest.skip("MFMA path active, WMMA tests not applicable")
    inst = make_mxmfma(instType=InstType.INST_F8, variant=[16, 16, 64])
    result = str(inst)
    assert "matrix_a_fmt" not in result
    assert "matrix_b_fmt" not in result


def test_mxmfma_wmma_f4_permute_small_variant():
    init_rocisa((12, 5, 0))
    if _uses_mfma_path():
        pytest.skip("MFMA path active, WMMA tests not applicable")
    inst = make_mxmfma(instType=InstType.INST_F4, variant=[16, 16, 128])
    result = str(inst)
    assert "matrix_a_fmt:MATRIX_FMT_FP4" in result
    assert "matrix_b_fmt:MATRIX_FMT_FP4" in result


def test_mxmfma_wmma_f4_no_permute_large_variant():
    init_rocisa((12, 5, 0))
    if _uses_mfma_path():
        pytest.skip("MFMA path active, WMMA tests not applicable")
    inst = make_mxmfma(instType=InstType.INST_F4, variant=[32, 32, 128])
    result = str(inst)
    assert "matrix_a_fmt" not in result


def test_mxmfma_wmma_scale_fmt_e5m3():
    init_rocisa((12, 5, 0))
    if _uses_mfma_path():
        pytest.skip("MFMA path active, WMMA tests not applicable")
    inst = make_mxmfma(mxScaleAType=InstType.INST_E5M3, mxScaleBType=InstType.INST_E5M3)
    result = str(inst)
    assert "matrix_a_scale_fmt:1" in result
    assert "matrix_b_scale_fmt:1" in result


def test_mxmfma_wmma_scale_fmt_f8():
    init_rocisa((12, 5, 0))
    if _uses_mfma_path():
        pytest.skip("MFMA path active, WMMA tests not applicable")
    inst = make_mxmfma(mxScaleAType=InstType.INST_F8, mxScaleBType=InstType.INST_F8)
    result = str(inst)
    assert "matrix_a_scale_fmt:2" in result
    assert "matrix_b_scale_fmt:2" in result


def test_mxmfma_wmma_scale_fmt_mixed():
    init_rocisa((12, 5, 0))
    if _uses_mfma_path():
        pytest.skip("MFMA path active, WMMA tests not applicable")
    inst = make_mxmfma(mxScaleAType=InstType.INST_E5M3, mxScaleBType=InstType.INST_F8)
    result = str(inst)
    assert "matrix_a_scale_fmt:1" in result
    assert "matrix_b_scale_fmt:2" in result


def test_mxmfma_wmma_scale_fmt_default():
    init_rocisa((12, 5, 0))
    if _uses_mfma_path():
        pytest.skip("MFMA path active, WMMA tests not applicable")
    inst = make_mxmfma(mxScaleAType=InstType.INST_F32, mxScaleBType=InstType.INST_F32)
    result = str(inst)
    assert "matrix_a_scale_fmt" not in result
    assert "matrix_b_scale_fmt" not in result


def test_mxmfma_wmma_getparams():
    init_rocisa((12, 5, 0))
    if _uses_mfma_path():
        pytest.skip("MFMA path active, WMMA tests not applicable")
    inst = make_mxmfma()
    params = inst.getParams()
    assert len(params) == 7


# ---------------------------------------------------------------------------
# MFMA path tests (unified branch on gfx950 only)
# ---------------------------------------------------------------------------

def test_mxmfma_mfma_prestr():
    init_rocisa((9, 5, 0))
    if not _uses_mfma_path():
        pytest.skip("Requires MFMA path (unified branch on gfx950)")
    inst = make_mxmfma()
    result = str(inst)
    assert "v_mfma_scale_f32_16x16x128_f8f6f4" in result


def test_mxmfma_mfma_block_ignored():
    init_rocisa((9, 5, 0))
    if not _uses_mfma_path():
        pytest.skip("Requires MFMA path (unified branch on gfx950)")
    inst0  = make_mxmfma(block=0)
    inst16 = make_mxmfma(block=16)
    assert "v_mfma_scale_f32_" in str(inst0)
    assert "v_mfma_scale_f32_" in str(inst16)


@pytest.mark.parametrize("instType,cbsz,blgp", [
    (InstType.INST_F8,     0, 0),
    (InstType.INST_BF8,    1, 1),
    (InstType.INST_F8_BF8, 0, 1),
    (InstType.INST_BF8_F8, 1, 0),
    (InstType.INST_F6,     2, 2),
    (InstType.INST_BF6,    3, 3),
    (InstType.INST_F4,     4, 4),
    (InstType.INST_F8_F6,  0, 2),
    (InstType.INST_F6_F8,  2, 0),
    (InstType.INST_F8_F4,  0, 4),
    (InstType.INST_F4_F8,  4, 0),
    (InstType.INST_F6_B6,  2, 3),
    (InstType.INST_B6_F6,  3, 2),
    (InstType.INST_F6_F4,  2, 4),
    (InstType.INST_F4_F6,  4, 2),
    (InstType.INST_B6_F4,  3, 4),
    (InstType.INST_F4_B6,  4, 3),
])
def test_mxmfma_mfma_input_permute(instType, cbsz, blgp):
    init_rocisa((9, 5, 0))
    if not _uses_mfma_path():
        pytest.skip("Requires MFMA path (unified branch on gfx950)")
    inst = make_mxmfma(instType=instType, variant=[16, 16, 128])
    result = str(inst)
    assert f"cbsz:{cbsz}" in result
    assert f"blgp:{blgp}" in result


def test_mxmfma_mfma_no_permute_small_k():
    init_rocisa((9, 5, 0))
    if not _uses_mfma_path():
        pytest.skip("Requires MFMA path (unified branch on gfx950)")
    inst = make_mxmfma(variant=[16, 16, 32])
    result = str(inst)
    assert "cbsz:" not in result
    assert "blgp:" not in result


def test_mxmfma_mfma_getparams():
    init_rocisa((9, 5, 0))
    if not _uses_mfma_path():
        pytest.skip("Requires MFMA path (unified branch on gfx950)")
    inst = make_mxmfma()
    params = inst.getParams()
    assert len(params) == 6


def test_mxmfma_mfma_issue_latency():
    init_rocisa((9, 5, 0))
    if not _uses_mfma_path():
        pytest.skip("Requires MFMA path (unified branch on gfx950)")
    inst = make_mxmfma()
    if not hasattr(inst, 'getIssueLatency'):
        pytest.skip("getIssueLatency not available on this branch")
    latency = inst.getIssueLatency()
    assert isinstance(latency, int)
    assert latency > 0


# ---------------------------------------------------------------------------
# Common tests (work on any branch / any path)
# ---------------------------------------------------------------------------

def test_mxmfma_member_access():
    init_rocisa()
    inst = make_mxmfma()
    assert str(inst.a) == "v[4:7]"
    assert str(inst.b) == "v[8:11]"
    assert str(inst.acc) == "v[0:3]"
    assert str(inst.acc2) == "v[12:15]"
    assert str(inst.mxsa) == "s[0:3]"
    assert str(inst.mxsb) == "s[4:7]"


def test_mxmfma_member_write():
    init_rocisa()
    inst = make_mxmfma()
    new_a = vgpr(20, 4)
    inst.a = new_a
    assert str(inst.a) == "v[20:23]"
    assert "v[20:23]" in str(inst)


def test_mxmfma_deepcopy():
    init_rocisa()
    inst = make_mxmfma()
    inst2 = deepcopy(inst)
    assert str(inst) == str(inst2)
    inst2.a = vgpr(20, 4)
    assert "v[4:7]" in str(inst)
    assert "v[20:23]" in str(inst2)


def test_mxmfma_comment():
    init_rocisa()
    inst = make_mxmfma(comment="test mxmfma comment")
    result = str(inst)
    assert "// test mxmfma comment" in result


def test_mxmfma_no_comment():
    init_rocisa()
    inst = make_mxmfma(comment="")
    result = str(inst)
    assert "//" not in result
