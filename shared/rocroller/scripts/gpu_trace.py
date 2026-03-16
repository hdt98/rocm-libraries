# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
GPU instruction tracer for rocGDB. Supports LDS (ds_*) and global (buffer_*) instructions.

Usage:
  rocgdb --batch -ex "source ../scripts/gpu_trace.py" -ex "gpu_trace --help"
  rocgdb --batch -ex "source ../scripts/gpu_trace.py" -ex 'gpu_trace --kernel <label>' --args /path/to/executable [exe-args...]

Examples:
  rocgdb --batch -ex "source ../scripts/gpu_trace.py" -ex 'gpu_trace --workgroup 0,0,0 --output trace1.jsonl --kernel GEMMTest_GEMMTestSuiteGPU_GEMM_DataType_FP32_Basic_0_kernel' --args ./test/rocroller-tests --gtest_filter='*GPU_GEMM_DataType_FP32_Basic/0'

  # Pasted from rrperf client command adjusted with --num_warmup=0 --num_outer=1 --num_inner=1
  rocgdb --batch -ex "source ../scripts/gpu_trace.py" -ex 'gpu_trace --workgroup 15,15,0 --output trace4.jsonl --kernel RRGEMM_TN_mxfp4_stE8M0_bs32_mxfp4_stE8M0_bs32_half_half_float_PreSWPreSwizzleScale_WGTS256x256x256_WGS128x2_WGMXCC0_LABufferToLDS_LBBufferToLDS_SDVGPRToGlobalMemoryWithBuffer_LSABufferToL_c03209e827442f0a' --args client/rocroller-gemm --mac_m=256 --mac_n=256 --mac_k=256 --wave_m=16 --wave_n=16 --wave_k=128 --wave_b=1 --workgroup_size_x=128 --workgroup_size_y=2 --workgroupRemapXCC=False --workgroupRemapXCCValue=-1 --load_A=BufferToLDS --load_B=BufferToLDS --store=VGPRToGlobalMemoryWithBuffer --betaInFma=True --padLDS_A=0,0 --padLDS_B=0,0 --scheduler=Priority --schedulerCost=LinearWeightedSimple --prefetch=True --prefetchInFlight=2 --prefetchLDSFactor=1 --prefetchMixMemOps=False --loadScale_A=BufferToLDS --loadScale_B=BufferToLDS --swizzleScale=True --sts=64x8/64x8 --prefetchScale=False --pretileScale=False --streamK=None --numWGs=0 --tailLoops=True --M=4096 --N=4096 --K=32768 --alpha=2.0 --beta=0.0 --type_A=fp4 --type_B=fp4 --type_C=half --type_D=half --type_acc=float --trans_A=T --trans_B=N --scale_A=Separate --scaleType_A=E8M0 --scale_B=Separate --scaleType_B=E8M0 --scaleBlockSize=-1 --scaleSkipPermlane=PreSwizzleScale --scaleValue_A=1.0 --scaleValue_B=1.0 --initMode_A=Bounded --initMode_B=Bounded --initMode_C=Bounded --workgroupMappingDim=-1 --workgroupMappingValue=-1 --num_warmup=0 --num_outer=1 --num_inner=1
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

    @staticmethod
    def _instr_width_bytes(mnemonic: str) -> int:
        """Return the data width in bytes for a ds_read/ds_write mnemonic."""
        m = re.search(r"_b(\d+)", mnemonic)
        return int(m.group(1)) // 8 if m else 4

    def read_lanes(self, frame, instr, uint32_type, addr_fmt):
        reg = f"v{instr.meta['addr_reg']}"
        offset = instr.meta["offset"]
        raw = read_vgpr_lanes(frame, instr.meta["addr_reg"], uint32_type)
        effective = [v + offset for v in raw]

        # LDS has 64 banks, each 4 bytes (1 dword) wide.
        # first_bank[lane]        = (effective_addr // 4) % 64
        # In units of the instruction width (e.g. dwordx4 for ds_read_b128):
        #   dwords_per_instr       = instr_width_bytes / 4
        #   first_bank_in_units   = first_bank // dwords_per_instr
        width_bytes = self._instr_width_bytes(instr.mnemonic)
        dwords_per_instr = max(1, width_bytes // 4)
        first_bank = [((addr // 4) % 64) // dwords_per_instr for addr in effective]

        return {
            "derived": {
                "effective_addr": [addr_fmt(v) for v in effective],
                "first_bank": first_bank,
            },
            "vgpr": {reg: raw},
        }


class BufferFamily(InstructionFamily):
    """Handles buffer_load* and buffer_store* (global memory) instructions.

    Decodes the 128-bit Shader Resource Descriptor (4 SGPRs) and per-lane
    VGPR offset to compute each lane's effective address.

    Two operand layouts are supported:
      Normal:     buffer_<load|store>_<width> <dst>, v<voff>, s[<srd_base>:...], ...
      LDS-direct: buffer_load_<width>          v<voff>, s[<srd_base>:...], ...  lds
    In the LDS-direct form there is no destination VGPR; data flows from
    global memory straight into LDS. The LDS write address per lane is:
      lds_effective_addr[lane] = M0 + lane * width_bytes
    """

    # Normal: has a destination operand before the voffset register.
    _RE = re.compile(r"buffer_(?:load|store)_\S+\s+\S+,\s*v(\d+),\s*s\[(\d+):")
    # LDS-direct: voffset is the first (and only) operand before the SRD.
    _RE_LDS_DIRECT = re.compile(r"buffer_load_\S+\s+v(\d+),\s*s\[(\d+):")

    @staticmethod
    def _instr_width_bytes(mnemonic: str) -> int:
        """Return the data width in bytes for a buffer_load/store mnemonic."""
        suffix = mnemonic.rsplit("_", 1)[-1]  # e.g. "dwordx4", "dword", "short"
        if suffix in ("byte", "ubyte"):
            return 1
        if suffix in ("short", "ushort"):
            return 2
        if suffix == "dword":
            return 4
        m = re.match(r"dwordx(\d+)$", suffix)
        if m:
            return int(m.group(1)) * 4
        return 4  # fallback

    @property
    def name(self):
        return "buffer"

    def match(self, mnemonic):
        return mnemonic.startswith("buffer_load") or mnemonic.startswith("buffer_store")

    def parse(self, mnemonic, asm, address):
        lds_direct = bool(re.search(r"\blds\b", asm))
        m = (self._RE_LDS_DIRECT if lds_direct else self._RE).match(asm)
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
                "lds_direct": lds_direct,
            },
        )

    def read_lanes(self, frame, instr, uint32_type, addr_fmt):
        voffset_idx = instr.meta["voffset_reg"]
        srd_base = instr.meta["srd_base_reg"]
        voffsets = read_vgpr_lanes(frame, voffset_idx, uint32_type)
        s = [read_uint32(frame, f"s{srd_base + i}", uint32_type) for i in range(4)]
        base_pointer = (s[1] << 32) + s[0]
        srd_raw = (s[3] << 96) | (s[2] << 64) | (s[1] << 32) | s[0]
        derived = {
            "base_pointer": addr_fmt(base_pointer),
            "size": s[2],
            "buffer_descriptor": addr_fmt(srd_raw),
            "effective_addr": [addr_fmt(base_pointer + voffsets[i]) for i in range(64)],
        }
        sgpr = {f"s[{srd_base}:{srd_base+3}]": s}
        if instr.meta["lds_direct"]:
            m0 = read_uint32(frame, "m0", uint32_type)
            width = self._instr_width_bytes(instr.mnemonic)
            derived["lds_direct"] = True
            derived["lds_effective_addr"] = [
                addr_fmt(m0 + lane * width) for lane in range(64)
            ]
            sgpr["m0"] = m0
        return {
            "derived": derived,
            "vgpr": {f"v{voffset_idx}": voffsets},
            "sgpr": sgpr,
        }


ALL_FAMILIES: dict = {
    "lds": LDSFamily(),
    "buffer": BufferFamily(),
}


# ---------------------------------------------------------------------------
# Disassembly Walker
# ---------------------------------------------------------------------------


_DISASM_LINE_RE = re.compile(r"^\s+(0x[0-9a-f]+)\s+<[^>]+>:\s+(.*)")


def walk_disassembly(families: list) -> list:
    """Walk disassembly of the current function using gdb's disassemble command.

    Parses the text output of 'disassemble' rather than using arch.disassemble(),
    which can produce spurious <illegal instruction> results on some rocgdb builds.
    Returns a list of FoundInstruction in program order.
    """
    text = gdb.execute("disassemble", to_string=True)
    found = []
    for line in text.splitlines():
        m = _DISASM_LINE_RE.match(line)
        if not m:
            continue
        addr = int(m.group(1), 16)
        asm = m.group(2).strip()
        mnemonic = asm.split()[0] if asm.split() else ""
        for family in families:
            if family.match(mnemonic):
                fi = family.parse(mnemonic, asm, addr)
                if fi:
                    found.append(fi)
                break
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
                if workgroups is not None and wg not in workgroups:
                    continue
                if waves is not None and wave not in waves:
                    continue
                emit(
                    {
                        "family": instr.family,
                        "instruction": instr.instruction,
                        "address": hex(instr.address),
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
            help="Base kernel symbol name.",
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
            default=None,
            metavar="N",
            help="Wave indices within the workgroup to trace. Multiple allowed. Default: all waves.",
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
            help="Max hits per instruction -- e.g. due to loops (default: 1, 0 = unlimited).",
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
        waves = set(args.wave) if args.wave else None

        # --- Get to kernel ---
        gdb.execute("set pagination off")
        gdb.execute("set breakpoint pending on")
        gdb.Breakpoint(args.kernel, temporary=True)
        # rocRoller kernels have extra label for after kernel argument setup
        gdb.Breakpoint(args.kernel + "_exec_begin", temporary=True)
        gdb.execute("run")

        # After stopping, switch to a GPU thread so that walk_disassembly
        # disassembles from the GPU frame rather than a CPU host frame.
        # $_dispatch_pos is only defined on GPU threads.
        for t in gdb.selected_inferior().threads():
            t.switch()
            wg, _ = work_coordinates()
            if wg is not None:
                break
        else:
            print("No GPU thread found after reaching kernel.")
            return

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
