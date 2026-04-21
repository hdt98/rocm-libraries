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

"""Source map between assembly instructions and generated Python rocasm lines.

Built during ``asm_to_rocasm`` conversion. Each assembly instruction in the
mainloop maps 1:1 to a Python instruction line in the generated ``_mainloop.py``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class SourceMapping:
    """A single mapping between an assembly instruction and its Python translation."""
    asm_index: int         # 0-based index into the assembly instruction list
    asm_text: str          # the original assembly instruction text (comment-stripped)
    python_index: int      # 0-based index into the Python body lines
    python_text: str       # the generated Python instruction text


@dataclass
class MainloopSourceMap:
    """Bidirectional map between assembly and Python instruction lines."""
    entries: list[SourceMapping]

    def asm_to_python(self, asm_index: int) -> SourceMapping:
        """Look up the mapping for assembly instruction at the given index."""
        return self.entries[asm_index]

    def python_to_asm(self, python_index: int) -> SourceMapping:
        """Look up the mapping for Python body line at the given index."""
        return self.entries[python_index]

    def to_json(self, path: str | Path):
        """Serialize the source map to a JSON file."""
        data = [asdict(e) for e in self.entries]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def from_json(cls, path: str | Path) -> MainloopSourceMap:
        """Deserialize a source map from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        entries = [SourceMapping(**e) for e in data]
        return cls(entries=entries)

    def __len__(self):
        return len(self.entries)
