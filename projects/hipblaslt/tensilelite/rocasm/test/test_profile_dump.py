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

"""Tests for the profile dump script."""

from pathlib import Path

import pytest

from rocasm.att import ATTProfile
from rocasm.asm_to_rocasm import asm_to_rocasm
from rocasm.setparse import parse_set_directives
from rocasm.source_map import MainloopSourceMap
from rocasm.profile import profile_mainloop
from rocasm.profile_dump import format_profile, main


FIXTURE_DIR = Path(__file__).parent / "fixtures"
CODE_JSON = FIXTURE_DIR / "att_code_128x256x64.json"
MAINLOOP_ASM = FIXTURE_DIR / "mainloop_128x256x64.s"
SET_DIRECTIVES = FIXTURE_DIR / "set_directives_128x256x64.s"

pytestmark = pytest.mark.skipif(
    not all(f.exists() for f in [CODE_JSON, MAINLOOP_ASM, SET_DIRECTIVES]),
    reason="Profile dump test fixtures not found",
)


@pytest.fixture
def profiled():
    att = ATTProfile.from_code_json(CODE_JSON)
    symbols = parse_set_directives(SET_DIRECTIVES.read_text())
    _, _, _, sm = asm_to_rocasm(MAINLOOP_ASM.read_text(), symbols)
    return profile_mainloop(att, MainloopSourceMap(entries=sm))


class TestFormatProfile:

    def test_header_line(self, profiled):
        text = format_profile(profiled)
        assert text.startswith("# TYPE")

    def test_line_count(self, profiled):
        text = format_profile(profiled)
        lines = [l for l in text.strip().splitlines() if not l.startswith("#")]
        assert len(lines) == 216  # 1 label + 215 instructions

    def test_pipe_delimited(self, profiled):
        text = format_profile(profiled)
        for line in text.strip().splitlines():
            if not line.startswith("#"):
                parts = line.split("|")
                assert len(parts) == 4, f"Expected 4 pipe-delimited fields: {line}"

    def test_instruction_types_present(self, profiled):
        text = format_profile(profiled)
        assert "mfma" in text
        assert "ds_read" in text
        assert "buffer_load" in text
        assert "ds_write" in text
        assert "s_waitcnt" in text

    def test_high_stall_buffer_load(self, profiled):
        """The last B-matrix buffer_load should have the highest stall."""
        text = format_profile(profiled)
        buffer_load_lines = [l for l in text.splitlines()
                             if l.startswith("buffer_load")]
        assert len(buffer_load_lines) == 12
        # Last buffer_load has highest stall
        last_bl = buffer_load_lines[-1]
        stall = int(last_bl.split("|")[2].strip())
        assert stall > 2000


class TestMainCLI:

    def test_output_to_file(self, tmp_path):
        """CLI writes output file."""
        out = tmp_path / "profile.txt"
        # Build a source map JSON fixture
        symbols = parse_set_directives(SET_DIRECTIVES.read_text())
        _, _, _, sm = asm_to_rocasm(MAINLOOP_ASM.read_text(), symbols)
        map_path = tmp_path / "test.map.json"
        MainloopSourceMap(entries=sm).to_json(map_path)
        # Set map mtime older than code.json to avoid staleness error
        import os
        code_mtime = os.path.getmtime(CODE_JSON)
        os.utime(map_path, (code_mtime - 1, code_mtime - 1))

        main(["--att-json", str(CODE_JSON),
              "--map", str(map_path),
              "-o", str(out)])
        assert out.exists()
        content = out.read_text()
        assert "mfma" in content
        lines = [l for l in content.strip().splitlines()
                 if not l.startswith("#")]
        assert len(lines) == 216

    def test_staleness_error(self, tmp_path):
        """CLI errors if source map is newer than ATT trace."""
        import os
        # Build a source map JSON fixture
        symbols = parse_set_directives(SET_DIRECTIVES.read_text())
        _, _, _, sm = asm_to_rocasm(MAINLOOP_ASM.read_text(), symbols)
        map_path = tmp_path / "test.map.json"
        MainloopSourceMap(entries=sm).to_json(map_path)
        # map is freshly written, so it's newer than CODE_JSON fixture → error
        with pytest.raises(SystemExit) as exc_info:
            main(["--att-json", str(CODE_JSON),
                  "--map", str(map_path),
                  "-o", str(tmp_path / "out.txt")])
        assert exc_info.value.code == 1
