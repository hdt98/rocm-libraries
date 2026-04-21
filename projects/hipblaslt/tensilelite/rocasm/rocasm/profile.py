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

"""Combine ATT profiling data with source maps to produce per-Python-line profiling.

Joins the positional ATT mainloop trace (from ``att.py``) with the assembly-to-Python
source map (from ``source_map.py``) to annotate each Python rocasm line with
performance data (hitcount, latency, stall, idle).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rocasm.att import ATTProfile
from rocasm.source_map import MainloopSourceMap


@dataclass
class ProfiledLine:
    """A Python rocasm line annotated with performance data."""
    python_index: int       # 0-based index in Python body (includes label)
    python_text: str        # the Python source line
    asm_text: str           # the original assembly instruction
    hitcount: int
    latency: int            # total latency across all hits
    stall: int              # total stall cycles
    idle: int

    @property
    def avg_latency(self) -> float:
        """Average latency per execution (latency / hitcount)."""
        return self.latency / self.hitcount if self.hitcount else 0.0


# Instruction type classification patterns
_INST_TYPE_PATTERNS = [
    ("mfma", re.compile(r"vmfma_|v_mfma_")),
    ("ds_read", re.compile(r"ds_read_|ds_load_")),
    ("ds_write", re.compile(r"ds_write_|ds_store_")),
    ("buffer_load", re.compile(r"buffer_load")),
    ("s_waitcnt", re.compile(r"s_waitcnt")),
    ("s_barrier", re.compile(r"s_barrier")),
    ("branch", re.compile(r"s_cbranch_|s_branch")),
    ("scalar", re.compile(r"s_add_|s_sub_|s_addc_|s_subb_|s_cmp_|s_cselect_|s_mov_|s_xor_")),
    ("label", re.compile(r"^label\(")),
]


def _classify_instruction(text: str) -> str:
    """Classify an instruction line into a type category."""
    for name, pattern in _INST_TYPE_PATTERNS:
        if pattern.search(text):
            return name
    return "other"


@dataclass
class ProfiledMainloop:
    """A complete mainloop with per-line profiling data."""
    lines: list[ProfiledLine]

    def hottest(self, n: int = 10, key: str = "stall") -> list[ProfiledLine]:
        """Return the N lines with highest stall/latency/idle, sorted descending."""
        return sorted(self.lines, key=lambda l: getattr(l, key), reverse=True)[:n]

    def by_instruction_type(self) -> dict[str, list[ProfiledLine]]:
        """Group profiled lines by instruction type."""
        groups: dict[str, list[ProfiledLine]] = {}
        for line in self.lines:
            # Classify using both Python and asm text
            itype = _classify_instruction(line.python_text)
            if itype == "other":
                itype = _classify_instruction(line.asm_text)
            groups.setdefault(itype, []).append(line)
        return groups


def profile_mainloop(att_profile: ATTProfile,
                     source_map: MainloopSourceMap) -> ProfiledMainloop:
    """Join ATT perf data with source map to produce per-Python-line profiling.

    The ATT mainloop instructions and the source map entries (excluding the
    label line at index 0) are joined positionally — instruction i in the
    ATT trace corresponds to source_map entry i+1 (offset by 1 for the label).

    The label line gets zero perf data since ATT doesn't trace labels.
    """
    ml_insts = att_profile.mainloop_instructions()
    # Source map includes the label at index 0; ATT trace does not.
    # So source_map[0] = label, source_map[1..] = instructions matching ATT.
    profiled: list[ProfiledLine] = []

    # Label line (no perf data)
    if source_map.entries and source_map.entries[0].asm_text.endswith(":"):
        label_entry = source_map.entries[0]
        profiled.append(ProfiledLine(
            python_index=label_entry.python_index,
            python_text=label_entry.python_text,
            asm_text=label_entry.asm_text,
            hitcount=0, latency=0, stall=0, idle=0,
        ))
        sm_offset = 1
    else:
        sm_offset = 0

    # Join instruction entries
    for i, perf in enumerate(ml_insts):
        sm_idx = i + sm_offset
        if sm_idx >= len(source_map.entries):
            break
        sm_entry = source_map.entries[sm_idx]
        profiled.append(ProfiledLine(
            python_index=sm_entry.python_index,
            python_text=sm_entry.python_text,
            asm_text=sm_entry.asm_text,
            hitcount=perf.hitcount,
            latency=perf.latency,
            stall=perf.stall,
            idle=perf.idle,
        ))

    return ProfiledMainloop(lines=profiled)
