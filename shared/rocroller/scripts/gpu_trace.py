# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
GPU instruction tracer for rocGDB. Supports LDS (ds_*) and global (buffer_*) instructions.

Usage (run from build_release/):
  rocgdb --batch -ex "source ../scripts/gpu_trace.py" -ex "gpu_trace --help"
  rocgdb --batch -ex "source ../scripts/gpu_trace.py" -ex 'gpu_trace --kernel <label>' --args /path/to/executable [exe-args...]

  Preferred: use rocgdb's --args to keep the executable and its arguments together.
  Alternatively, pass executable arguments via --run= (note: use = when args start with '-').
  NOTE: --kernel is the base kernel symbol name; _exec_begin is appended internally for the entry breakpoint.

Examples (run from build_release/):

  rocgdb --batch -ex "source ../scripts/gpu_trace.py" -ex 'gpu_trace --workgroup 0,0,0 --families lds buffer --output trace.jsonl --kernel GEMMTest_GEMMTestSuiteGPU_GEMM_DataType_FP32_Basic_0_kernel' --args ./test/rocroller-tests --gtest_filter='*GPU_GEMM_DataType_FP32_Basic/0'

  # Paste from rrperf client command adjusted with --num_warmup=0 --num_outer=1 --num_inner=1
  rocgdb --batch -ex "source ../scripts/gpu_trace.py" -ex 'gpu_trace --families lds buffer --output trace.jsonl --kernel RRGEMM_TN_mxfp4_stE8M0_bs32_mxfp4_stE8M0_bs32_half_half_float_PreSWPreSwizzleScale_WGTS256x256x256_WGS128x2_WGMXCC0_LABufferToLDS_LBBufferToLDS_SDVGPRToGlobalMemoryWithBuffer_LSABufferToL_c03209e827442f0a' --args client/rocroller-gemm --mac_m=256 --mac_n=256 --mac_k=256 --wave_m=16 --wave_n=16 --wave_k=128 --wave_b=1 --workgroup_size_x=128 --workgroup_size_y=2 --workgroupRemapXCC=False --workgroupRemapXCCValue=-1 --load_A=BufferToLDS --load_B=BufferToLDS --store=VGPRToGlobalMemoryWithBuffer --betaInFma=True --padLDS_A=0,0 --padLDS_B=0,0 --scheduler=Priority --schedulerCost=LinearWeightedSimple --prefetch=True --prefetchInFlight=2 --prefetchLDSFactor=1 --prefetchMixMemOps=False --loadScale_A=BufferToLDS --loadScale_B=BufferToLDS --swizzleScale=True --sts=64x8/64x8 --prefetchScale=False --pretileScale=False --streamK=None --numWGs=0 --tailLoops=True --M=4096 --N=4096 --K=32768 --alpha=2.0 --beta=0.0 --type_A=fp4 --type_B=fp4 --type_C=half --type_D=half --type_acc=float --trans_A=T --trans_B=N --scale_A=Separate --scaleType_A=E8M0 --scale_B=Separate --scaleType_B=E8M0 --scaleBlockSize=-1 --scaleSkipPermlane=PreSwizzleScale --scaleValue_A=1.0 --scaleValue_B=1.0 --initMode_A=Bounded --initMode_B=Bounded --initMode_C=Bounded --workgroupMappingDim=-1 --workgroupMappingValue=-1 --num_warmup=0 --num_outer=1 --num_inner=1
"""

import argparse
import json
import re
import shlex
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import gdb


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def workgroup_arg(s):
    try:
        x, y, z = map(int, s.split(","))
        return (x, y, z)
    except Exception:
        raise argparse.ArgumentTypeError("Workgroup must be x,y,z (e.g. 0,0,0)")


def work_coordinates():
    """Return ((wgx, wgy, wgz), wave_index) for the current thread, or (None, None)."""
    pos = gdb.convenience_variable("_dispatch_pos").string()
    m = re.match(r"\((\d+),(\d+),(\d+)\)/(\d+)", pos)
    if m:
        return (int(m[1]), int(m[2]), int(m[3])), int(m[4])
    return None, None


def read_uint32(frame, regname, uint32_type):
    return int(frame.read_register(regname).cast(uint32_type))


def read_vgpr_lanes(frame, reg_idx, uint32_type):
    """Read all 64 lanes of a VGPR, return list of ints."""
    val = frame.read_register(f"v{reg_idx}")
    return [int(val[i].cast(uint32_type)) for i in range(64)]


# ---------------------------------------------------------------------------
# Instruction Families
# ---------------------------------------------------------------------------


@dataclass
class FoundInstruction:
    family: str
    mnemonic: str
    instruction: str
    address: int
    meta: dict = field(default_factory=dict)


class InstructionFamily(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def match(self, mnemonic: str) -> bool: ...

    @abstractmethod
    def parse(
        self, mnemonic: str, asm: str, address: int
    ) -> Optional[FoundInstruction]: ...

    @abstractmethod
    def read_lanes(
        self, frame, instr: "FoundInstruction", uint32_type, addr_fmt
    ) -> dict:
        """Return {"derived": {...}, "vgpr": {...}, "sgpr": {...}}.
        addr_fmt is a callable applied to all address values (int or hex).
        Omit "sgpr" if the instruction has no SGPR operands of interest."""
        ...


class LDSFamily(InstructionFamily):
    """Handles ds_read* and ds_write* (LDS) instructions.

    Extracts the per-lane LDS address register for each instruction.
      ds_write: ds_write_b32 v<addr>, v<data>  -> addr_reg = operand[0]
      ds_read:  ds_read_b32  v<dst>,  v<addr>  -> addr_reg = operand[1]
    """

    _OPERANDS_RE = re.compile(r"\S+\s+(.*)")

    @property
    def name(self):
        return "lds"

    def match(self, mnemonic):
        return mnemonic.startswith("ds_read") or mnemonic.startswith("ds_write")

    def parse(self, mnemonic, asm, address):
        m = self._OPERANDS_RE.match(asm)
        if not m:
            return None
        # Take the first token of each comma-separated operand (strips offsets/flags)
        operands = [op.strip().split()[0] for op in m.group(1).split(",")]
        addr_op = operands[0] if mnemonic.startswith("ds_write") else operands[1]
        rm = re.match(r"v(\d+)", addr_op)
        if not rm:
            return None
        offset_m = re.search(r"\boffset:(\d+)", asm)
        return FoundInstruction(
            family=self.name,
            mnemonic=mnemonic,
            instruction=asm,
            address=address,
            meta={
                "addr_reg": int(rm.group(1)),
                "offset": int(offset_m.group(1)) if offset_m else 0,
            },
        )

    def read_lanes(self, frame, instr, uint32_type, addr_fmt):
        reg = f"v{instr.meta['addr_reg']}"
        offset = instr.meta["offset"]
        raw = read_vgpr_lanes(frame, instr.meta["addr_reg"], uint32_type)
        return {
            "derived": {"effective_addr": [addr_fmt(v + offset) for v in raw]},
            "vgpr": {reg: raw},
        }


class BufferFamily(InstructionFamily):
    """Handles buffer_load* and buffer_store* (global memory) instructions.

    Decodes the 128-bit Shader Resource Descriptor (4 SGPRs) and per-lane
    VGPR offset to compute each lane's effective address.

    Matched format: buffer_<load|store>_<width> <dst>, v<voff>, s[<srd_base>:...], ...
    """

    _RE = re.compile(r"buffer_(?:load|store)_\S+\s+\S+,\s*v(\d+),\s*s\[(\d+):")

    @property
    def name(self):
        return "buffer"

    def match(self, mnemonic):
        return mnemonic.startswith("buffer_load") or mnemonic.startswith("buffer_store")

    def parse(self, mnemonic, asm, address):
        m = self._RE.match(asm)
        if not m:
            return None
        return FoundInstruction(
            family=self.name,
            mnemonic=mnemonic,
            instruction=asm,
            address=address,
            meta={
                "voffset_reg": int(m.group(1)),
                "srd_base_reg": int(m.group(2)),
            },
        )

    def read_lanes(self, frame, instr, uint32_type, addr_fmt):
        voffset_idx = instr.meta["voffset_reg"]
        srd_base = instr.meta["srd_base_reg"]
        voffsets = read_vgpr_lanes(frame, voffset_idx, uint32_type)
        s = [read_uint32(frame, f"s{srd_base + i}", uint32_type) for i in range(4)]
        base_pointer = (s[1] << 32) + s[0]
        srd_raw = (s[3] << 96) | (s[2] << 64) | (s[1] << 32) | s[0]
        return {
            "derived": {
                "base_pointer": addr_fmt(base_pointer),
                "size": s[2],
                "buffer_descriptor": addr_fmt(srd_raw),
                "effective_addr": [
                    addr_fmt(base_pointer + voffsets[i]) for i in range(64)
                ],
            },
            "vgpr": {f"v{voffset_idx}": voffsets},
            "sgpr": {f"s[{srd_base}:{srd_base+3}]": s},
        }


ALL_FAMILIES: dict = {
    "lds": LDSFamily(),
    "buffer": BufferFamily(),
}


# ---------------------------------------------------------------------------
# Disassembly Walker
# ---------------------------------------------------------------------------


def walk_disassembly(families: list) -> list:
    """Walk disassembly from the current PC using structured arch.disassemble().

    Stops at s_endpgm, an 'illegal' instruction, or a memory access error.
    Returns a list of FoundInstruction in program order.
    """
    arch = gdb.selected_frame().architecture()
    pc = gdb.selected_frame().pc()
    found = []
    while True:
        try:
            inst = arch.disassemble(pc)[0]
        except gdb.MemoryError:
            break
        asm = inst["asm"].strip()
        mnemonic = asm.split()[0] if asm.split() else ""
        for family in families:
            if family.match(mnemonic):
                fi = family.parse(mnemonic, asm, inst["addr"])
                if fi:
                    found.append(fi)
                break
        if mnemonic == "s_endpgm" or "illegal" in asm:
            break
        pc += inst["length"]
    return found


# ---------------------------------------------------------------------------
# Trace Loop
# ---------------------------------------------------------------------------


def trace_loop(
    instructions, workgroups, waves, trace_count, uint32_type, addr_fmt, emit
):
    """Trace instructions sequentially, one breakpoint at a time.

    Sets a single breakpoint per instruction and collects data before moving
    to the next. This avoids rocgdb's per-queue displaced stepping limit,
    which is exceeded when many breakpoints are set concurrently.

    trace_count controls how many hits to collect per instruction (1 = first
    hit only, 0 = unlimited until the inferior exits).
    """
    inferior = gdb.selected_inferior()

    for instr in instructions:
        if not inferior.is_valid():
            break

        family = ALL_FAMILIES[instr.family]
        bp = gdb.Breakpoint(f"*{hex(instr.address)}", temporary=(trace_count == 1))
        hits = 0
        gdb.execute("continue")

        while inferior.is_valid() and (trace_count == 0 or hits < trace_count):
            at_our_bp = False
            for thread in inferior.threads():
                thread.switch()
                frame = gdb.selected_frame()
                if frame.pc() != instr.address:
                    continue
                at_our_bp = True
                wg, wave = work_coordinates()
                if workgroups is not None and (
                    wg not in workgroups or wave not in waves
                ):
                    continue
                emit(
                    {
                        "family": instr.family,
                        "instruction": instr.instruction,
                        "address": addr_fmt(instr.address),
                        "workgroup": list(wg),
                        "wave": wave,
                        **family.read_lanes(frame, instr, uint32_type, addr_fmt),
                    }
                )
                hits += 1

            if not at_our_bp:
                # Stopped somewhere unrelated to our breakpoint; don't loop.
                break
            if trace_count > 0 and hits >= trace_count:
                break
            gdb.execute("continue")

        if bp.is_valid():
            bp.delete()


# ---------------------------------------------------------------------------
# GDB Command
# ---------------------------------------------------------------------------


class GPUTrace(gdb.Command):
    """Trace GPU memory instructions (LDS / buffer) from within rocGDB."""

    def __init__(self):
        super().__init__("gpu_trace", gdb.COMMAND_USER)
        self.parser = argparse.ArgumentParser(
            prog="gpu_trace",
            description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        self.parser.add_argument(
            "--kernel",
            required=True,
            help="Base kernel symbol name. The script appends '_exec_begin' to set the breakpoint.",
        )
        self.parser.add_argument(
            "--run",
            required=False,
            default=None,
            dest="run_command",
            help="Arguments passed to the executable. If omitted, the executable and its "
            "arguments should be specified via rocgdb's --args flag instead. "
            "Use = syntax when args start with '-': --run=\"--gtest_filter=*MyTest/0\"",
        )
        self.parser.add_argument(
            "--families",
            nargs="+",
            choices=["lds", "buffer", "all"],
            default=["all"],
            help="Instruction families to trace (default: all)",
        )
        self.parser.add_argument(
            "--workgroup",
            type=workgroup_arg,
            action="append",
            default=None,
            metavar="X,Y,Z",
            help="Workgroup to trace, e.g. 0,0,0. Repeatable. Default: accept any workgroup.",
        )
        self.parser.add_argument(
            "--wave",
            type=int,
            nargs="+",
            default=[0],
            metavar="N",
            help="Wave indices within the workgroup to trace (default: 0). Multiple allowed.",
        )
        self.parser.add_argument(
            "--output",
            type=str,
            default=None,
            help="JSON Lines output file (default: stdout)",
        )
        self.parser.add_argument(
            "--trace_count",
            type=int,
            default=1,
            help="Max hits per instruction (default: 1, 0 = unlimited).",
        )
        self.parser.add_argument(
            "--hex",
            action="store_true",
            default=False,
            help="Print addresses in hex (default: base 10).",
        )

    def invoke(self, args_str, from_tty):
        args = self.parser.parse_args(shlex.split(args_str))

        families = (
            list(ALL_FAMILIES.values())
            if "all" in args.families
            else [ALL_FAMILIES[f] for f in args.families]
        )
        workgroups = set(map(tuple, args.workgroup)) if args.workgroup else None
        waves = set(args.wave)

        # --- Get to kernel ---
        gdb.execute("set pagination off")
        gdb.execute("set breakpoint pending on")
        gdb.Breakpoint(args.kernel + "_exec_begin", temporary=True)
        gdb.execute(
            f"run {args.run_command}" if args.run_command is not None else "run"
        )

        # --- Walk disassembly ---
        instructions = walk_disassembly(families)
        print(f"Found {len(instructions)} matching instructions:")
        for instr in instructions:
            print(
                f"  [{instr.family:6s}] {instr.instruction:50s} @ {hex(instr.address)}"
            )

        if not instructions:
            print("Nothing to trace.")
            return

        uint32_type = gdb.selected_frame().architecture().integer_type(32, False)

        # --- Set up output ---
        out = open(args.output, "w") if args.output else None

        def emit(record):
            line = json.dumps(record)
            if out:
                out.write(line + "\n")
                out.flush()
            else:
                print(line)

        # --- Trace ---
        try:
            addr_fmt = hex if args.hex else lambda x: x
            trace_loop(
                instructions,
                workgroups,
                waves,
                args.trace_count,
                uint32_type,
                addr_fmt,
                emit,
            )
        except Exception as e:
            print(f"Stopping early: {e}")
        finally:
            if out:
                out.close()
            gdb.execute("set confirm off")
            gdb.execute("quit")

    def complete(self, text, word):
        return gdb.COMPLETE_NONE


GPUTrace()
