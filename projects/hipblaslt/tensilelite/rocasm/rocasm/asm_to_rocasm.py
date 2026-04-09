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

"""Convert assembly text to rocasm Python source code.

Takes raw assembly instructions (as emitted by Tensile's code generator)
and produces equivalent rocasm Python code that, when executed, reconstructs
the same assembly.
"""

from __future__ import annotations

import re

from rocasm.setparse import parse_set_directives


def _resolve_expr(expr: str, symbols: dict[str, int]) -> int:
    """Resolve an assembler expression like 'vgprValuA_X0_I0+4+0+0' to an int."""
    total = 0
    for part in re.split(r'(?=[+-])', expr):
        part = part.strip()
        if not part:
            continue
        try:
            total += int(part, 0)
        except ValueError:
            if part in symbols:
                total += symbols[part]
            else:
                raise ValueError(f"Cannot resolve expression part '{part}' in '{expr}'")
    return total


def _parse_reg_operand(operand: str, symbols: dict[str, int]) -> tuple[str, int, int] | None:
    """Parse a register operand like 'v[vgprValuA+4:vgprValuA+4+3]' or 'acc[0:3]'.

    Returns (reg_type, phys_start, count) or None if not a register operand.
    reg_type is 'v', 'acc', or 's'.
    """
    m = re.match(r'^(v|acc|s)\[(.+):(.+)\]$', operand)
    if m:
        reg_type = m.group(1)
        start = _resolve_expr(m.group(2), symbols)
        end = _resolve_expr(m.group(3), symbols)
        return reg_type, start, end - start + 1  # inclusive to exclusive

    m = re.match(r'^(v|acc|s)\[(.+)\]$', operand)
    if m:
        reg_type = m.group(1)
        start = _resolve_expr(m.group(2), symbols)
        return reg_type, start, 1

    return None


def _parse_scalar_operand(operand: str, symbols: dict[str, int]) -> str:
    """Parse a scalar operand and return a rocasm Python expression.

    Handles:
    - Register: s[N] or s[N:M] -> sgpr(N) or sgpr(N, count)
    - Immediate: 0x10, 42 -> literal
    - Symbol: BufferLimit -> resolved int
    """
    # s[expr:expr] range
    m = re.match(r'^s\[(.+):(.+)\]$', operand)
    if m:
        start = _resolve_expr(m.group(1), symbols)
        end = _resolve_expr(m.group(2), symbols)
        count = end - start + 1
        return f"sgpr({start}, {count})"

    # s[expr] single
    m = re.match(r'^s\[(.+)\]$', operand)
    if m:
        val = _resolve_expr(m.group(1), symbols)
        return f"sgpr({val})"

    # sN bare scalar (e.g. s60)
    m = re.match(r'^s(\d+)$', operand)
    if m:
        return f"sgpr({m.group(1)})"

    # v[expr] single
    m = re.match(r'^v\[(.+)\]$', operand)
    if m:
        val = _resolve_expr(m.group(1), symbols)
        return f"vgpr({val})"

    # Special registers (m0, etc.)
    if operand == "m0":
        return "m0"

    # Try as integer literal
    try:
        val = int(operand, 0)
        return f"hex({val})" if operand.startswith("0x") else str(val)
    except ValueError:
        pass

    # Try as symbol reference
    if operand in symbols:
        return str(symbols[operand])

    return operand


def _reg_to_block(reg_type: str, start: int, count: int) -> str:
    """Convert a physical register reference to a block array slice expression."""
    if reg_type == "acc":
        return f"block.Acc[{start}:{start + count}]"
    elif reg_type == "v":
        return f"block.V[{start}:{start + count}]"
    elif reg_type == "s":
        return f"block.S[{start}:{start + count}]"
    raise ValueError(f"Unknown register type: {reg_type}")


def _split_operands(operand_str: str) -> list[str]:
    """Split comma-separated operands, respecting brackets."""
    operands = []
    depth = 0
    current = []
    for ch in operand_str:
        if ch == '[':
            depth += 1
        elif ch == ']':
            depth -= 1
        if ch == ',' and depth == 0:
            operands.append(''.join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        operands.append(''.join(current).strip())
    return operands


def _strip_comment(line: str) -> str:
    """Strip trailing // comment from an instruction line."""
    idx = line.find("//")
    return line[:idx].rstrip() if idx != -1 else line.rstrip()


def asm_to_rocasm(asm_text: str, symbols: dict[str, int] | None = None) -> str:
    """Convert assembly text to rocasm Python source code.

    Args:
        asm_text: Assembly source text (e.g. the main loop body).
        symbols: Optional pre-resolved symbol table from parse_set_directives.
                 If None, any .set directives in asm_text are parsed.

    Returns:
        Python source code string that uses rocasm to reconstruct the assembly.
    """
    if symbols is None:
        symbols = parse_set_directives(asm_text)

    lines = []
    for raw_line in asm_text.splitlines():
        stripped = raw_line.strip()

        # Skip empty lines
        if not stripped:
            continue

        # Skip block comments
        if stripped.startswith("/*") or stripped.startswith("*"):
            continue

        # Skip line comments
        if stripped.startswith("//"):
            continue

        # Skip .set directives
        if stripped.startswith(".set"):
            continue

        # Labels → emit as raw text
        if stripped.endswith(":"):
            lines.append(f'block.label("{stripped[:-1]}")')
            continue

        # Strip trailing comment for instruction parsing
        inst_line = _strip_comment(stripped)
        if not inst_line:
            continue

        code = _convert_instruction(inst_line, symbols)
        if code is not None:
            lines.append(code)
        else:
            lines.append(f"# UNHANDLED: {stripped}")

    return "\n".join(lines)


def _convert_instruction(inst_line: str, symbols: dict[str, int]) -> str | None:
    """Convert a single assembly instruction line to rocasm Python code."""
    # Split into mnemonic and operands
    parts = inst_line.split(None, 1)
    mnemonic = parts[0]
    operand_str = parts[1] if len(parts) > 1 else ""

    # --- MFMA ---
    if mnemonic == "v_mfma_f32_16x16x32_bf16":
        operands = _split_operands(operand_str)
        if len(operands) != 4:
            return None
        dst = _parse_reg_operand(operands[0], symbols)
        src_b = _parse_reg_operand(operands[1], symbols)
        src_a = _parse_reg_operand(operands[2], symbols)
        acc2 = _parse_reg_operand(operands[3], symbols)
        if not all([dst, src_b, src_a, acc2]):
            return None
        return (f"{_reg_to_block(*dst)} = vmfma_f32_16x16x32_bf16("
                f"{_reg_to_block(*src_b)}, {_reg_to_block(*src_a)}, {_reg_to_block(*acc2)})")

    # --- ds_read_b128 / ds_load_b128 ---
    if mnemonic in ("ds_read_b128", "ds_load_b128"):
        # Extract offset modifier if present
        offset = None
        clean = operand_str
        m = re.search(r'\boffset:(\S+)', operand_str)
        if m:
            offset = int(m.group(1), 0)
            clean = operand_str[:m.start()].rstrip().rstrip(',')

        operands = _split_operands(clean)
        if len(operands) < 2:
            return None
        dst = _parse_reg_operand(operands[0], symbols)
        src = _parse_reg_operand(operands[1], symbols)
        if not dst or not src:
            return None

        ds_arg = ""
        if offset is not None and offset != 0:
            ds_arg = f", ds=DSModifiers(offset={offset})"
        return (f"{_reg_to_block(*dst)} = ds_read_b128("
                f"{_reg_to_block(*src)}{ds_arg})")

    # --- buffer_load_dwordx4 / buffer_load_b128 ---
    if mnemonic in ("buffer_load_dwordx4", "buffer_load_b128"):
        # Extract modifiers: offen, offset:N, lds
        offen = "offen" in operand_str
        lds = " lds" in operand_str
        offset = 0
        m = re.search(r'\boffset:(\S+)', operand_str)
        if m:
            offset_str = m.group(1)
            offset = int(offset_str, 0)

        # Remove modifiers from operand string to parse registers
        clean = re.sub(r'\s+offen\b', '', operand_str)
        clean = re.sub(r'\s+lds\b', '', clean)
        clean = re.sub(r'\s+offset:\S+', '', clean)
        clean = clean.rstrip().rstrip(',')

        operands = _split_operands(clean)

        mubuf_parts = []
        if offen:
            mubuf_parts.append("offen=True")
        if offset != 0:
            mubuf_parts.append(f"offset12={offset}")
        if lds:
            mubuf_parts.append("lds=True")

        mubuf_arg = ""
        if mubuf_parts:
            mubuf_arg = f", mubuf=MUBUFModifiers({', '.join(mubuf_parts)})"

        if lds:
            # LDS direct load: only 3 operands (vaddr, saddr, soffset) — no dst
            if len(operands) < 3:
                return None
            vaddr = _parse_reg_operand(operands[0], symbols)
            saddr = _parse_reg_operand(operands[1], symbols)
            soffset_str = _parse_scalar_operand(operands[2], symbols)
            if not vaddr or not saddr:
                return None
            return (f"block.buffer_load_lds("
                    f"{_reg_to_block(*vaddr)}, {_reg_to_block(*saddr)}, "
                    f"{soffset_str}{mubuf_arg})")
        else:
            # Normal load: 4 operands (dst, vaddr, saddr, soffset)
            if len(operands) < 4:
                return None
            dst = _parse_reg_operand(operands[0], symbols)
            vaddr = _parse_reg_operand(operands[1], symbols)
            saddr = _parse_reg_operand(operands[2], symbols)
            soffset_str = _parse_scalar_operand(operands[3], symbols)
            if not dst or not vaddr or not saddr:
                return None
            return (f"{_reg_to_block(*dst)} = buffer_load_dwordx4("
                    f"{_reg_to_block(*vaddr)}, {_reg_to_block(*saddr)}, "
                    f"{soffset_str}{mubuf_arg})")

    # --- s_waitcnt ---
    if mnemonic == "s_waitcnt":
        # Parse waitcnt arguments like lgkmcnt(0), vmcnt(0)
        waits = re.findall(r'(\w+)\((\d+)\)', operand_str)
        # Map to rocisa keyword args
        kwmap = {
            "lgkmcnt": "dscnt",
            "vmcnt": "vlcnt",
        }
        kwargs = []
        for name, val in waits:
            kw = kwmap.get(name, name)
            kwargs.append(f"{kw}={val}")
        return f"block.s_waitcnt({', '.join(kwargs)})"

    # --- s_barrier ---
    if mnemonic == "s_barrier":
        return "block.s_barrier()"

    # --- s_nop ---
    if mnemonic == "s_nop":
        return f"block.s_nop({operand_str.strip()})"

    # --- Scalar 3-operand: s_add_u32, s_addc_u32, s_sub_u32, s_subb_u32,
    #     s_cselect_b32, s_xor_b32 ---
    scalar_3op = {
        "s_add_u32", "s_addc_u32", "s_sub_u32", "s_subb_u32",
        "s_cselect_b32", "s_xor_b32",
    }
    if mnemonic in scalar_3op:
        operands = _split_operands(operand_str)
        if len(operands) < 3:
            return None
        dst = _parse_scalar_operand(operands[0], symbols)
        src0 = _parse_scalar_operand(operands[1], symbols)
        src1 = _parse_scalar_operand(operands[2], symbols)
        return f"block.{mnemonic}(dst={dst}, src0={src0}, src1={src1})"

    # --- s_mov_b32 ---
    if mnemonic == "s_mov_b32":
        operands = _split_operands(operand_str)
        if len(operands) < 2:
            return None
        dst = _parse_scalar_operand(operands[0], symbols)
        src = _parse_scalar_operand(operands[1], symbols)
        return f"block.s_mov_b32(dst={dst}, src={src})"

    # --- s_cmp_* (2-operand scalar comparisons) ---
    scalar_cmp = {
        "s_cmp_eq_u32", "s_cmp_eq_i32",
        "s_cmp_le_u32", "s_cmp_lt_u32", "s_cmp_ge_u32", "s_cmp_gt_u32",
        "s_cmp_le_i32", "s_cmp_lt_i32", "s_cmp_ge_i32", "s_cmp_gt_i32",
    }
    if mnemonic in scalar_cmp:
        operands = _split_operands(operand_str)
        if len(operands) < 2:
            return None
        src0 = _parse_scalar_operand(operands[0], symbols)
        src1 = _parse_scalar_operand(operands[1], symbols)
        return f"block.{mnemonic}(src0={src0}, src1={src1})"

    # --- s_cbranch_scc0 / s_cbranch_scc1 ---
    if mnemonic in ("s_cbranch_scc0", "s_cbranch_scc1"):
        label = operand_str.strip()
        return f'block.{mnemonic}(labelName="{label}")'

    # --- v_xor_b32 ---
    if mnemonic == "v_xor_b32":
        operands = _split_operands(operand_str)
        if len(operands) < 3:
            return None
        dst = _parse_scalar_operand(operands[0], symbols)
        src0 = _parse_scalar_operand(operands[1], symbols)
        src1 = _parse_scalar_operand(operands[2], symbols)
        return f"block.v_xor_b32(dst={dst}, src0={src0}, src1={src1})"

    return None
