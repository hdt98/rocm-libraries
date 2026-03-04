import os
import re

import gdb

FUNCTION = os.environ["KERNEL_NAME"]

gdb.execute("set pagination off")
gdb.execute("set breakpoint pending on")

gdb.Breakpoint(FUNCTION + "_exec_begin", temporary=True)
gdb.execute("run")

disasm = gdb.execute(f"disassemble {FUNCTION}", to_string=True)

ds_instructions = []
for line in disasm.splitlines():
    # Match lines like:  <+1156>:   ds_write_b32 v3, v1
    m = re.search(r"<\+(\d+)>:\s+(ds_\S+)\s+(.*)", line)
    if not m:
        continue
    offset = int(m.group(1))
    mnemonic = m.group(2)
    operands = [op.strip().split()[0] for op in m.group(3).split(",")]
    if mnemonic.startswith("ds_write"):
        addr_reg = operands[0]
    elif mnemonic.startswith("ds_read"):
        addr_reg = operands[1]
    else:
        raise ValueError(f"Unhandled ds_ instruction: {mnemonic}")
    ds_instructions.append({"offset": offset, "reg": addr_reg, "mnemonic": mnemonic})

THREADS = None

for instr in ds_instructions:
    gdb.Breakpoint(f"*{FUNCTION}+{instr['offset']}", temporary=True)
    gdb.execute("continue")
    gdb.execute("x/i $pc")
    print(f"# ${instr['reg']} ({instr['mnemonic']} @ +{instr['offset']})")

    if THREADS is None:
        # Discover threads from whichever thread stopped, plus the next 3
        # Generally this is the next 3 waves in the workgroup
        hit_thread = gdb.selected_thread().num
        all_threads = sorted(t.num for t in gdb.inferiors()[0].threads())
        idx = all_threads.index(hit_thread)
        THREADS = all_threads[idx : idx + 4]

    for t in THREADS:
        gdb.execute(f"thread {t}")
        gdb.execute(f"p ${instr['reg']}")

gdb.execute("set confirm off")
gdb.execute("quit")
