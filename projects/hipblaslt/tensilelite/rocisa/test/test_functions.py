################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

from rocisa.functions import (
    vectorMultiplyBpe,
    vectorMultiply64Bpe,
    vectorAddMultiplyBpe,
    scalarMultiplyBpe,
    scalarMultiply64Bpe,
)

# --- vectorMultiplyBpe: scales a VGPR by bpe ---
def test_vectorMultiplyBpe_half_int():
    # bpe=0.5 -> v_lshrrev_b32 (right-shift by 1)
    module = vectorMultiplyBpe(5, 3, 0.5, "test")
    asm = str(module)
    print("vectorMultiplyBpe(5, 3, 0.5):", asm)
    assert "v_lshrrev_b32" in asm
    assert "v5" in asm
    assert "v3" in asm

def test_vectorMultiplyBpe_half_string():
    # bpe=0.5 with named registers
    module = vectorMultiplyBpe("ValuA_X0_I0", "ValuB_X0_I0", 0.5, "test")
    asm = str(module)
    print("vectorMultiplyBpe(str, str, 0.5):", asm)
    assert "v_lshrrev_b32" in asm
    assert "ValuA_X0_I0" in asm
    assert "ValuB_X0_I0" in asm

def test_vectorMultiplyBpe_three_quarters():
    # bpe=0.75 -> v_mul_lo_u32 by 6, then v_lshrrev_b32 by 3
    module = vectorMultiplyBpe(5, 3, 0.75, "test")
    asm = str(module)
    print("vectorMultiplyBpe(5, 3, 0.75):", asm)
    assert "v_mul_lo_u32" in asm
    assert "v_lshrrev_b32" in asm

def test_vectorMultiplyBpe_one_same():
    # bpe=1.0, dst==src -> no-op (comment only, no shift/mul instructions)
    module = vectorMultiplyBpe(5, 5, 1.0, "test")
    asm = str(module)
    print("vectorMultiplyBpe(5, 5, 1.0):", asm)
    assert "v_lshlrev_b32" not in asm
    assert "v_lshrrev_b32" not in asm

def test_vectorMultiplyBpe_one_diff():
    # bpe=1.0, dst!=src -> v_lshlrev_b32 with shift=0
    module = vectorMultiplyBpe(5, 3, 1.0, "test")
    asm = str(module)
    print("vectorMultiplyBpe(5, 3, 1.0):", asm)
    assert "v_lshlrev_b32" in asm

def test_vectorMultiplyBpe_two():
    # bpe=2.0 -> v_lshlrev_b32 by 1
    module = vectorMultiplyBpe(5, 3, 2.0, "test")
    asm = str(module)
    print("vectorMultiplyBpe(5, 3, 2.0):", asm)
    assert "v_lshlrev_b32" in asm

def test_vectorMultiplyBpe_four():
    # bpe=4.0 -> v_lshlrev_b32 by 2
    module = vectorMultiplyBpe(5, 3, 4.0, "test")
    asm = str(module)
    print("vectorMultiplyBpe(5, 3, 4.0):", asm)
    assert "v_lshlrev_b32" in asm

# --- vectorAddMultiplyBpe: adds two VGPRs then scales by bpe ---
def test_vectorAddMultiplyBpe_half():
    # bpe=0.5 -> v_add_u32 then v_lshrrev_b32
    module = vectorAddMultiplyBpe(5, 3, 4, 0.5, "test")
    asm = str(module)
    print("vectorAddMultiplyBpe(5, 3, 4, 0.5):", asm)
    assert "v_add_u32" in asm
    assert "v_lshrrev_b32" in asm

def test_vectorAddMultiplyBpe_three_quarters():
    # bpe=0.75 -> v_add_u32, v_mul_lo_u32 by 6, v_lshrrev_b32 by 3
    module = vectorAddMultiplyBpe(5, 3, 4, 0.75, "test")
    asm = str(module)
    print("vectorAddMultiplyBpe(5, 3, 4, 0.75):", asm)
    assert "v_add_u32" in asm
    assert "v_mul_lo_u32" in asm
    assert "v_lshrrev_b32" in asm

def test_vectorAddMultiplyBpe_one():
    # bpe=1.0 -> v_add_u32 only (no shift, bpe_log2==0)
    module = vectorAddMultiplyBpe(5, 3, 4, 1.0, "test")
    asm = str(module)
    print("vectorAddMultiplyBpe(5, 3, 4, 1.0):", asm)
    assert "v_add_u32" in asm
    assert "v_add_lshl_u32" not in asm

def test_vectorAddMultiplyBpe_two():
    # bpe=2.0 -> v_add_lshl_u32 (fused add + left-shift by 1)
    module = vectorAddMultiplyBpe(5, 3, 4, 2.0, "test")
    asm = str(module)
    print("vectorAddMultiplyBpe(5, 3, 4, 2.0):", asm)
    assert "v_add_lshl_u32" in asm

def test_vectorAddMultiplyBpe_four():
    # bpe=4.0 -> v_add_lshl_u32 with shift=2
    module = vectorAddMultiplyBpe(5, 3, 4, 4.0, "test")
    asm = str(module)
    print("vectorAddMultiplyBpe(5, 3, 4, 4.0):", asm)
    assert "v_add_lshl_u32" in asm

# --- vectorMultiply64Bpe: 64-bit VGPR scale by bpe ---
def test_vectorMultiply64Bpe_half():
    # bpe=0.5 -> v_lshrrev_b64
    module = vectorMultiply64Bpe(10, 20, 0.5, 30, "test")
    asm = str(module)
    print("vectorMultiply64Bpe(10, 20, 0.5):", asm)
    assert "v_lshrrev_b64" in asm

def test_vectorMultiply64Bpe_three_quarters():
    # bpe=0.75 -> v_mul_hi_u32, v_mul_lo_u32, v_add_u32, v_lshrrev_b64
    module = vectorMultiply64Bpe(10, 20, 0.75, 30, "test")
    asm = str(module)
    print("vectorMultiply64Bpe(10, 20, 0.75):", asm)
    assert "v_mul_hi_u32" in asm
    assert "v_mul_lo_u32" in asm
    assert "v_lshrrev_b64" in asm

def test_vectorMultiply64Bpe_one_same():
    # bpe=1.0, dst==src -> no-op
    module = vectorMultiply64Bpe(10, 10, 1.0, 30, "test")
    asm = str(module)
    print("vectorMultiply64Bpe(10, 10, 1.0):", asm)
    assert "v_lshlrev_b64" not in asm

def test_vectorMultiply64Bpe_two():
    # bpe=2.0 -> v_lshlrev_b64
    module = vectorMultiply64Bpe(10, 20, 2.0, 30, "test")
    asm = str(module)
    print("vectorMultiply64Bpe(10, 20, 2.0):", asm)
    assert "v_lshlrev_b64" in asm

# --- scalarMultiplyBpe: scales an SGPR by bpe ---
def test_scalarMultiplyBpe_half_int():
    # bpe=0.5 -> s_lshr_b32
    module = scalarMultiplyBpe(5, 3, 0.5, "test")
    asm = str(module)
    print("scalarMultiplyBpe(5, 3, 0.5):", asm)
    assert "s_lshr_b32" in asm

def test_scalarMultiplyBpe_half_string():
    # bpe=0.5 with named registers
    module = scalarMultiplyBpe("SizesFree+0", "SizesFree+1", 0.5, "test")
    asm = str(module)
    print("scalarMultiplyBpe(str, str, 0.5):", asm)
    assert "s_lshr_b32" in asm

def test_scalarMultiplyBpe_half_mixed():
    # <int, string> overload
    module = scalarMultiplyBpe(5, "SizesFree+0", 0.5, "test")
    asm = str(module)
    print("scalarMultiplyBpe(int, str, 0.5):", asm)
    assert "s_lshr_b32" in asm

def test_scalarMultiplyBpe_three_quarters():
    # bpe=0.75 -> s_mul_i32 by 6, s_lshr_b32 by 3
    module = scalarMultiplyBpe(5, 3, 0.75, "test")
    asm = str(module)
    print("scalarMultiplyBpe(5, 3, 0.75):", asm)
    assert "s_mul_i32" in asm
    assert "s_lshr_b32" in asm

def test_scalarMultiplyBpe_one_same():
    # bpe=1.0, dst==src -> no-op
    module = scalarMultiplyBpe(5, 5, 1.0, "test")
    asm = str(module)
    print("scalarMultiplyBpe(5, 5, 1.0):", asm)
    assert "s_lshl_b32" not in asm

def test_scalarMultiplyBpe_two():
    # bpe=2.0 -> s_lshl_b32
    module = scalarMultiplyBpe(5, 3, 2.0, "test")
    asm = str(module)
    print("scalarMultiplyBpe(5, 3, 2.0):", asm)
    assert "s_lshl_b32" in asm

# --- scalarMultiply64Bpe: 64-bit SGPR scale by bpe ---
def test_scalarMultiply64Bpe_half_int():
    # bpe=0.5 -> s_lshr_b64
    module = scalarMultiply64Bpe(10, 20, 0.5, 30, "test")
    asm = str(module)
    print("scalarMultiply64Bpe(10, 20, 0.5):", asm)
    assert "s_lshr_b64" in asm

def test_scalarMultiply64Bpe_half_string():
    # <string, string, int> overload
    module = scalarMultiply64Bpe("SrdA+0", "SrdB+0", 0.5, 30, "test")
    asm = str(module)
    print("scalarMultiply64Bpe(str, str, 0.5):", asm)
    assert "s_lshr_b64" in asm

def test_scalarMultiply64Bpe_three_quarters():
    # bpe=0.75 -> s_mul_hi_u32, s_mul_i32, s_add_u32, s_lshr_b64
    module = scalarMultiply64Bpe(10, 20, 0.75, 30, "test")
    asm = str(module)
    print("scalarMultiply64Bpe(10, 20, 0.75):", asm)
    assert "s_mul_hi_u32" in asm
    assert "s_mul_i32" in asm
    assert "s_lshr_b64" in asm

def test_scalarMultiply64Bpe_one_same():
    # bpe=1.0, dst==src -> no-op
    module = scalarMultiply64Bpe(10, 10, 1.0, 30, "test")
    asm = str(module)
    print("scalarMultiply64Bpe(10, 10, 1.0):", asm)
    assert "s_lshl_b64" not in asm

def test_scalarMultiply64Bpe_two():
    # bpe=2.0 -> s_lshl_b64
    module = scalarMultiply64Bpe(10, 20, 2.0, 30, "test")
    asm = str(module)
    print("scalarMultiply64Bpe(10, 20, 2.0):", asm)
    assert "s_lshl_b64" in asm


# Ensure `test_base.py` runs first before these tests — it initializes the rocIsa singleton at module level

test_vectorMultiplyBpe_half_int()
test_vectorMultiplyBpe_half_string()
test_vectorMultiplyBpe_three_quarters()
test_vectorMultiplyBpe_one_same()
test_vectorMultiplyBpe_one_diff()
test_vectorMultiplyBpe_two()
test_vectorMultiplyBpe_four()
test_vectorAddMultiplyBpe_half()
test_vectorAddMultiplyBpe_three_quarters()
test_vectorAddMultiplyBpe_one()
test_vectorAddMultiplyBpe_two()
test_vectorAddMultiplyBpe_four()
test_vectorMultiply64Bpe_half()
test_vectorMultiply64Bpe_three_quarters()
test_vectorMultiply64Bpe_one_same()
test_vectorMultiply64Bpe_two()
test_scalarMultiplyBpe_half_int()
test_scalarMultiplyBpe_half_string()
test_scalarMultiplyBpe_half_mixed()
test_scalarMultiplyBpe_three_quarters()
test_scalarMultiplyBpe_one_same()
test_scalarMultiplyBpe_two()
test_scalarMultiply64Bpe_half_int()
test_scalarMultiply64Bpe_half_string()
test_scalarMultiply64Bpe_three_quarters()
test_scalarMultiply64Bpe_one_same()
test_scalarMultiply64Bpe_two()

