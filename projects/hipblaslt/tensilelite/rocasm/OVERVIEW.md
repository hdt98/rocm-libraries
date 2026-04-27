# rocasm: A Python Representation Layer for CDNA Kernel Schedules

## What it is

rocasm is a Python representation layer for AMD CDNA GPU instruction sequences. It sits on top of rocisa (ROCm ISA library) and expresses GPU assembly as readable Python — register slices as array indices, instructions as function calls, and the schedule as sequential Python statements. The result is a format that is simultaneously executable (it emits real assembly via rocisa), readable (an engineer can follow the data flow), and editable (reordering Python lines reorders instructions).

rocasm currently targets the mainloop of GEMM kernels in TensileLite, where the instruction schedule — the ordering of matrix multiply, memory load, and synchronization instructions — is the primary determinant of performance.

## From ISA to Python

A CDNA GEMM mainloop is a sequence of ~200 instructions repeated thousands of times. Here's what three representative instructions look like at each level:

**CDNA3 assembly** (what the GPU executes):
```asm
s_waitcnt lgkmcnt(7)
v_mfma_f32_16x16x16_bf16 a[0:3], v[48:49], v[16:17], a[0:3]
ds_read_b128 v[32:35], v[14] offset:64
```

**rocisa Module code** (how TensileLite's code generator emits it):
```python
module = Module("loop_body")
module.add(SWaitCnt(lgkmcnt=7))
module.add(VMfma(dst=AccVgprRange(0, 4), src0=VgprRange(48, 2),
                 src1=VgprRange(16, 2), src2=AccVgprRange(0, 4),
                 variant="f32_16x16x16_bf16"))
module.add(DSReadB128(dst=VgprRange(32, 4), addr=Vgpr(14),
                      ds=DSModifiers(offset=64)))
```

**rocasm** (what the engineer reads and edits):
```python
s_waitcnt(dscnt=7)
Acc[0:4] = vmfma_f32_16x16x16bf16_1k(B0[0:2], A0[0:2], Acc[0:4])
A2[0:4] = ds_read_b128(LocalReadAddrA[0:1], ds=DSModifiers(offset=64))
```

The rocasm version uses named register arrays (`Acc`, `B0`, `A0`, `A2`, `LocalReadAddrA`) that are declared once at the top of the function and map to physical register ranges. The instruction `Acc[0:4] = vmfma(B0[0:2], A0[0:2], Acc[0:4])` reads as "accumulate a 16x16 matrix multiply of B-tile column 0 and A-tile column 0 into accumulator tiles 0-3." The full 216-line mainloop reads as a schedule — the ordering of these statements *is* the schedule.

## The optimization workflow

The practical motivation for rocasm is **iterative schedule optimization** — the process of reordering instructions within a mainloop to reduce pipeline stalls, guided by hardware profiling data.

The workflow operates in a loop:

```
 +---------------------------------------------+
 |  rocasm Python module                       |
 |  (the instruction schedule)                 |
 +----------------------+-----------------------+
                        |  Tensile/bin/Tensile builds kernel
                        v
 +---------------------------------------------+
 |  Compiled kernel (.co)                      |
 |  + source map (.map.json)                   |
 +----------------------+-----------------------+
                        |  rocprofv3 --att captures thread trace
                        v
 +---------------------------------------------+
 |  ATT profile (per-instruction timing)       |
 |  hitcount, latency, stall, idle per instr   |
 +----------------------+-----------------------+
                        |  profile_dump joins via source map
                        v
 +---------------------------------------------+
 |  Annotated profile (profile.txt)            |
 |  Each Python line <- stall cycles           |
 +----------------------+-----------------------+
                        |  Human or agent reads profile,
                        |  reorders Python lines
                        v
                        +-------- back to top --+
```

The source map (`.map.json`) is generated alongside the kernel and records the 1:1 correspondence between each rocasm Python line and its emitted assembly instruction. When profiling data arrives from `rocprofv3 --att`, the `profile_dump` tool joins the per-instruction timing back to the Python source, producing output like:

```
# TYPE         | AVG_LAT | STALL | PYTHON
s_waitcnt      |    19.7 |  4896 | s_waitcnt(dscnt=7)
mfma           |     4.0 |     0 | Acc[0:4] = vmfma(B0[0:2], A0[0:2], Acc[0:4])
ds_read        |     4.0 |     4 | A2[0:4] = ds_read_b128(LocalReadAddrA, offset=64)
mfma           |    16.0 |  2976 | Acc[76:80] = vmfma(B0[16:18], A0[12:14], Acc[76:80])
mfma           |    16.0 |  2976 | Acc[80:84] = vmfma(B0[20:22], A0[0:2], Acc[80:84])
```

A `STALL` of 2976 on consecutive MFMAs means the matrix pipeline is idle for 12 cycles per iteration — there's no independent work between those instructions to keep the pipeline fed. The fix is to move a memory operation (a `ds_read`, `buffer_load`, or `ds_write`) between them. In rocasm, that's moving a line of Python.

This loop can be driven by a human engineer using the `rocprof-cli` TUI (a Textual-based viewer that color-codes each line by instruction type and stall severity), or by an AI agent following scheduling rules codified as a Claude Code command. Either way, the unit of work is the same: read the profile, identify a stall pattern, reorder Python lines, rebuild, re-profile, check.

## Toward DSL compatibility

Most Python-level DSLs for GPU kernel development — Triton, Gluon, Mojo, torch.compile — compile *down* through increasingly hardware-specific intermediate representations until they reach ISA. rocasm comes from the opposite direction: it represents ISA and reaches *up* toward Python. This makes it a natural complement to top-down DSLs — an "assembly layer for DSLs" where a DSL's compiler could hand off its inner loop to a pre-optimized rocasm schedule.

The integration point would be somewhere in the DSL's lowering pipeline — likely at the LLVM IR or MLIR level — where the DSL has already decided on tiling and memory layout but hasn't yet committed to an instruction schedule. At that boundary, the DSL provides a contract (which data is in which buffers, what computation is needed) and rocasm provides the schedule (the precise ordering of loads, stores, matrix multiplies, and synchronization). The DSL handles everything outside the inner loop — grid dispatch, prologue, epilogue, activation functions — while rocasm handles the performance-critical core.

This is still exploratory, but the register-named-array abstraction in rocasm (`A0`, `B0`, `Acc` rather than `v[16:31]`, `v[48:79]`, `a[0:127]`) is a step toward the kind of symbolic register interface that would make this practical. A future version of rocasm where register arrays carry logical identities rather than physical base addresses could slot into a compiler's register allocation pass while preserving the hand-crafted instruction order that makes the schedule fast.
