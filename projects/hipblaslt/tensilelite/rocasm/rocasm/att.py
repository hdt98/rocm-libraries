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

"""Parse ATT (Advanced Thread Tracing) profiling data from rocprofv3.

Reads the ``code.json`` file produced by ``rocprofv3 --att`` and provides
per-instruction performance data (hitcount, latency, stall, idle).
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass
class InstructionPerf:
    """Per-instruction performance record from an ATT trace."""
    isa: str           # disassembled instruction text
    line: int          # line number in code.json (1-based)
    vaddr: int         # virtual address (PC)
    hitcount: int      # number of times executed
    latency: int       # total latency (sum across hits)
    stall: int         # total stall cycles
    idle: int          # total idle cycles


@dataclass
class ATTProfile:
    """Parsed ATT profile from a code.json file."""
    all_instructions: list[InstructionPerf]

    def mainloop_instructions(self) -> list[InstructionPerf]:
        """Return only steady-state mainloop instructions.

        The mainloop runs many more iterations than the prologue/epilogue,
        so mainloop instructions have the highest hitcount.  This method
        finds the maximum hitcount and returns all instructions with that
        hitcount — those are the steady-state loop body.
        """
        if not self.all_instructions:
            return []
        max_hc = max(inst.hitcount for inst in self.all_instructions)
        if max_hc == 0:
            return []
        return [inst for inst in self.all_instructions
                if inst.hitcount == max_hc]

    @classmethod
    def from_code_json(cls, path: str | Path) -> ATTProfile:
        """Parse a code.json file into an ATTProfile.

        The code.json format (version 3.0.0) has::

            {"code": [[isa, _, line, source, codeobj, vaddr, hit, lat, stall, idle], ...],
             "header": "ISA, _, LineNumber, Source, Codeobj, Vaddr, Hit, Latency, Stall, Idle",
             "version": "3.0.0"}
        """
        with open(path) as f:
            data = json.load(f)
        instructions = []
        for entry in data["code"]:
            isa, _, line, _source, _codeobj, vaddr, hit, lat, stall, idle = entry
            instructions.append(InstructionPerf(
                isa=isa, line=line, vaddr=vaddr,
                hitcount=hit, latency=lat, stall=stall, idle=idle,
            ))
        return cls(all_instructions=instructions)

    @classmethod
    def from_att_directory(cls, att_dir: str | Path) -> ATTProfile:
        """Find and parse code.json inside an ATT output directory.

        Handles the ``ui_output_agent_*`` nesting produced by rocprofv3.
        """
        att_dir = Path(att_dir)
        # Look for ui_output_agent_* directories
        ui_dirs = list(att_dir.glob("ui_output_agent_*"))
        if not ui_dirs:
            # Maybe att_dir IS the ui_output directory
            code_json = att_dir / "code.json"
            if code_json.exists():
                return cls.from_code_json(code_json)
            raise FileNotFoundError(
                f"No ui_output_agent_* directory or code.json found in {att_dir}")
        # Use the first (usually only) ui_output directory
        code_json = ui_dirs[0] / "code.json"
        if not code_json.exists():
            raise FileNotFoundError(f"code.json not found in {ui_dirs[0]}")
        return cls.from_code_json(code_json)
