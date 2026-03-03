#!/usr/bin/env python3
"""
Parse rocgdb output. For each instruction line followed by 4 thread arrays,
prints the instruction, register, and the concatenated 256-element array.

Reads from stdin or a file argument.
"""

import re
import sys


INSTR_RE = re.compile(r'=>\s+0x[0-9a-f]+\s+<[^>]+>:\s+(.+)')
REG_RE   = re.compile(r'^# (\$\w+)$', re.MULTILINE)
ARRAY_RE = re.compile(r'\$\d+\s*=\s*\{([^}]*)\}', re.DOTALL)


def format_array(values):
    return ', '.join(str(v) for v in values)


def main():
    text = open(sys.argv[1]).read() if len(sys.argv) > 1 else sys.stdin.read()

    # Collect (position, kind, value) events in document order
    events = []
    for m in INSTR_RE.finditer(text):
        events.append((m.start(), 'instr', m.group(1).strip()))
    for m in REG_RE.finditer(text):
        events.append((m.start(), 'reg', m.group(1)))
    for m in ARRAY_RE.finditer(text):
        nums = [int(x.strip()) for x in m.group(1).split(',') if x.strip()]
        events.append((m.start(), 'array', nums))
    events.sort()

    # Walk events: each instruction starts a new group; flush when group has 4 arrays
    groups = []  # list of (instruction, reg, joined_array)
    current_instr = None
    current_reg   = None
    current_arrays = []

    for _, kind, value in events:
        if kind == 'instr':
            current_instr = value
            current_reg   = None
            current_arrays = []
        elif kind == 'reg':
            current_reg = value
        elif kind == 'array':
            current_arrays.append(value)
            if len(current_arrays) == 4:
                joined = [v for a in current_arrays for v in a]
                groups.append((current_instr, current_reg, joined))
                current_instr = None
                current_reg   = None
                current_arrays = []

    for instr, reg, values in groups:
        print(f'// {instr}  ({reg})')
        print('{' + format_array(values) + '}')
        print()


if __name__ == '__main__':
    main()
