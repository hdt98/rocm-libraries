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

"""Parser for .set assembly directives.

Evaluates .set directives from generated GEMM kernel .s files,
resolving symbolic names to integer register values.
"""

from __future__ import annotations

import re

_SET_RE = re.compile(r"^\s*\.set\s+(\w+)\s*,\s*(.+?)\s*$")
_NAME_PLUS_INT_RE = re.compile(r"^(\w+)\+(\d+)$")
_NAME_MINUS_INT_RE = re.compile(r"^(\w+)-(\d+)$")


def _try_int(s: str) -> int | None:
    """Try to parse a string as a decimal or hex integer."""
    try:
        return int(s, 0)
    except ValueError:
        return None


def parse_set_directives(text: str) -> dict[str, int]:
    """Parse .set directives from assembly text and resolve all symbols.

    Processes lines of the form:
        .set name, value

    where value can be:
        - An integer literal (decimal or hex)
        - A symbol name (resolved from previously defined symbols)
        - A symbol+offset or symbol-offset expression
        - UNDEF (removes the symbol from the table)

    Args:
        text: Assembly source text containing .set directives.

    Returns:
        Dict mapping symbol names to their resolved integer values.
        Symbols set to UNDEF are excluded.

    Raises:
        ValueError: If a symbol references an undefined name.
    """
    table: dict[str, int] = {}

    for line in text.splitlines():
        m = _SET_RE.match(line)
        if not m:
            continue

        name = m.group(1)
        value_str = m.group(2)

        if value_str == "UNDEF":
            table.pop(name, None)
            continue

        # Try literal integer
        lit = _try_int(value_str)
        if lit is not None:
            table[name] = lit
            continue

        # Try name+offset
        m2 = _NAME_PLUS_INT_RE.match(value_str)
        if m2:
            ref_name, offset = m2.group(1), int(m2.group(2))
            if ref_name not in table:
                raise ValueError(
                    f".set {name}: undefined symbol '{ref_name}'")
            table[name] = table[ref_name] + offset
            continue

        # Try name-offset
        m3 = _NAME_MINUS_INT_RE.match(value_str)
        if m3:
            ref_name, offset = m3.group(1), int(m3.group(2))
            if ref_name not in table:
                raise ValueError(
                    f".set {name}: undefined symbol '{ref_name}'")
            table[name] = table[ref_name] - offset
            continue

        # Try bare symbol reference
        if re.match(r"^\w+$", value_str):
            if value_str not in table:
                raise ValueError(
                    f".set {name}: undefined symbol '{value_str}'")
            table[name] = table[value_str]
            continue

        raise ValueError(
            f".set {name}: cannot parse value '{value_str}'")

    return table
