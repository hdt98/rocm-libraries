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

"""Dump per-line profiling data as a flat two-column file.

Combines an ATT trace with a source map to produce a line-by-line annotated
listing of the mainloop.  Each line shows the instruction type (color category),
average latency, total stall cycles, and the Python rocasm instruction.

Output format (pipe-delimited, machine-parseable)::

    # TYPE       | AVG_LAT | STALL | PYTHON
    waitcnt      |    19.7 |  4896 | s_waitcnt(dscnt=7)
    mfma         |     4.0 |     0 | Acc[0:4] = vmfma_f32_16x16x16bf16_1k(B0[0:2], A0[0:2], Acc[0:4])
    ds_read      |     4.0 |     4 | A2[0:4] = ds_read_b128(LocalReadAddrA[0:1], ds=DSModifiers(offset=64))
    buffer_load  |    21.5 |  4240 | G2LB[28:32] = buffer_load_dwordx4(...)

Usage::

    python -m rocasm.profile_dump --att-dir /tmp/att_output --map /path/to/_mainloop.map.json -o profile.txt
    python -m rocasm.profile_dump --att-dir /tmp/att_output --map /path/to/_mainloop.map.json  # stdout
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rocasm.att import ATTProfile
from rocasm.source_map import MainloopSourceMap
from rocasm.profile import profile_mainloop, ProfiledMainloop, _classify_instruction


def format_profile(profiled: ProfiledMainloop) -> str:
    """Format a ProfiledMainloop as a pipe-delimited annotated listing."""
    lines = []
    lines.append("# TYPE         | AVG_LAT | STALL | PYTHON")

    for pl in profiled.lines:
        itype = _classify_instruction(pl.python_text)
        if itype == "other":
            itype = _classify_instruction(pl.asm_text)
        avg_lat = pl.avg_latency
        lines.append(
            f"{itype:<14s} | {avg_lat:7.1f} | {pl.stall:5d} | {pl.python_text}"
        )

    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Dump per-line rocasm profiling data from an ATT trace.")
    att_group = parser.add_mutually_exclusive_group(required=True)
    att_group.add_argument("--att-dir",
                           help="Path to rocprofv3 --att output directory")
    att_group.add_argument("--att-json",
                           help="Path to code.json file directly")
    parser.add_argument("--map", required=True,
                        help="Path to _mainloop.map.json source map file")
    parser.add_argument("-o", "--output", default=None,
                        help="Output file (default: stdout)")
    args = parser.parse_args(argv)

    # Staleness check: if the source map is newer than the ATT trace,
    # the kernel was rebuilt but not re-profiled. The profile data would
    # be from the old schedule applied to the new Python positions — wrong.
    map_path = Path(args.map)
    if args.att_json:
        att_path = Path(args.att_json)
    else:
        # Find the code.json inside the ATT directory
        att_dir = Path(args.att_dir)
        ui_dirs = list(att_dir.glob("ui_output_agent_*"))
        att_path = (ui_dirs[0] / "code.json") if ui_dirs else (att_dir / "code.json")

    if map_path.exists() and att_path.exists():
        if map_path.stat().st_mtime > att_path.stat().st_mtime:
            print(
                f"Error: source map ({map_path.name}) is newer than ATT trace "
                f"({att_path.name}).\n"
                f"The kernel was rebuilt but not re-profiled. Re-run rocprofv3 "
                f"before generating the profile dump.",
                file=sys.stderr,
            )
            sys.exit(1)

    if args.att_json:
        att = ATTProfile.from_code_json(args.att_json)
    else:
        att = ATTProfile.from_att_directory(args.att_dir)
    sm = MainloopSourceMap.from_json(args.map)
    profiled = profile_mainloop(att, sm)
    text = format_profile(profiled)

    if args.output:
        Path(args.output).write_text(text)
        print(f"Wrote {len(profiled.lines)} profiled lines to {args.output}",
              file=sys.stderr)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
