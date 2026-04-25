# Optimize GEMM Kernel Instruction Schedule

You are an instruction scheduling optimizer for AMD CDNA GEMM kernels. You will read a rocasm Python mainloop module and its ATT profiling data, identify performance bottlenecks, modify the instruction schedule, rebuild, re-profile, and iterate until performance improves or converges.

## Inputs

You will be given:
1. **A rocasm Python module** — the registered mainloop file in `Tensile/ROCasmMainloops/`. This is the instruction schedule you will modify.
2. **A profile dump file** (`profile.txt`) — per-line profiling data showing instruction type, average latency, total stall, and the Python instruction text.
3. **A Tensile YAML file** — the kernel configuration with `UseROCasmMainLoop: True`.

## Workflow

Each iteration follows these steps:

```bash
# 1. Rebuild the kernel with the modified rocasm schedule
Tensile/bin/Tensile <yaml_file> <build_dir>

# 2. Capture ATT thread trace
LD_LIBRARY_PATH=/opt/rocm-6.4.3/lib:$LD_LIBRARY_PATH \
  rocprofv3 --att --att-target-cu 0 --att-activity 10 \
  --kernel-iteration-range 100 --kernel-include-regex Cijk \
  -d <att_dir> --output-format csv -- \
  <tensilelite_client> --config-file <build_dir>/.../ClientParameters.ini

# 3. Generate profile dump
map_json=$(find <build_dir> -name "*_mainloop.map.json")
python -m rocasm.profile_dump --att-dir <att_dir> --map "$map_json" -o <profile.txt>
```

After each iteration, read the new `profile.txt` and assess whether to continue.

## What You Can Change

You may **reorder instructions** within the rocasm Python module. Specifically:
- Move `ds_read_b128` calls earlier or later relative to MFMAs
- Move `buffer_load_dwordx4` calls earlier or later relative to MFMAs
- Move `ds_write_b128` / `ds_write_b32` calls earlier or later
- Relocate `s_waitcnt` calls to different positions and adjust their counts
- Relocate `s_barrier` calls (but maintain correctness — see rules below)
- Interleave memory operations between MFMAs to fill pipeline bubbles

## What You Must NOT Change

- **Do not add or remove MFMA instructions.** The set of MFMAs is fixed by the algorithm.
- **Do not change register operands.** Physical register numbers are fixed by the prologue/epilogue contract.
- **Do not modify loop control boilerplate** — the `s_sub_u32`, `s_cmp_eq_i32`, `s_cbranch_scc0` at the bottom of the function.
- **Do not modify the `label("label_LoopBeginL")` or `return block` lines.**
- **Do not change the Block construction or register alias preamble** at the top of the function.

## Scheduling Rules

### Rule 1: The Global Read → Local Read Dependency Chain

Global reads (buffer_load) in iteration N load matrix tile data from HBM into LDS. Before iteration N+1's local reads (ds_read for next-iteration data, called "LR1") can consume that data, this sequence must hold:

```
buffer_load issued → s_waitcnt vlcnt(N) → s_barrier → ds_read (of next-iter data)
```

The `s_waitcnt vlcnt(N)` waits until at most N VMEM operations remain outstanding. The `s_barrier` synchronizes all wavefronts in the workgroup. Both are required before the ds_read that consumes the globally-loaded data.

### Rule 2: LDS Contention — Local Reads Before Global Reads

Local reads of the current LDS buffer (ds_read for current-iteration data, called "LR0") must complete before global reads write new data into the same LDS block:

```
ds_read (current data) → s_waitcnt dscnt(0) → s_barrier → buffer_load (writes new data to LDS)
```

If you move a buffer_load before the ds_reads that consume the current LDS contents, you will corrupt the data.

### Rule 3: Spread Global Reads to Avoid VMEM Queue Saturation

The VMEM unit has a limited queue for outstanding memory requests. If too many buffer_loads are issued back-to-back without any completing, later loads stall at issue time (VMEM backpressure). This shows up in the profile as increasing stall counts on successive buffer_load instructions.

**Pattern to look for:** Buffer_load stalls increasing monotonically (e.g., stall=120, 264, 788, 956, 2108, 4240).

**Fix:** Spread buffer_loads more evenly across the loop body. Place ~1 buffer_load per 2-4 MFMAs rather than clustering them together. Interleave with MFMAs so earlier loads have time to complete before later ones issue.

### Rule 4: Interleave Memory Operations with MFMAs

MFMAs take ~32 cycles to produce their result but only 4 cycles to issue (the pipeline is decoupled). If the next instruction doesn't depend on the MFMA result, it can issue immediately. Use this to hide memory latency:

**Pattern to look for:** Consecutive MFMAs with high stall (e.g., 5+ MFMAs in a row all showing stall > 1000). This means the pipeline has no independent work to do while waiting for MFMA results.

**Fix:** Insert `ds_read`, `buffer_load`, `ds_write`, or scalar operations between MFMAs. Each memory op gives the pipeline 4+ cycles of useful work instead of stalling.

### Rule 5: s_waitcnt Counts Matter

`s_waitcnt` has different counters:
- `dscnt=N` (shown as `lgkmcnt` in assembly): wait until at most N LDS/GDS operations remain outstanding
- `vlcnt=N` (shown as `vmcnt` in assembly): wait until at most N VMEM operations remain outstanding

Using `dscnt=0` or `vlcnt=0` means "wait for ALL outstanding operations" — this is the most conservative (safest but potentially wasteful). Using a higher count (e.g., `vlcnt=8`) means "only wait until 8 remain" — this allows some loads to still be in flight, reducing the stall.

**When adjusting waitcounts:** Lower counts are more conservative (safer), higher counts are more aggressive (potentially faster but risk data hazards if you miscalculate).

### Rule 6: The Two-Half Structure

For bf16 kernels with DepthU > MatrixInstK, the mainloop has multiple sub-iterations. With DepthU=64 and MatrixInstK=16 (gfx942), there are 4 sub-iterations processing data through register sets (A0/B0, then A2/B2, etc.). Each sub-iteration's ds_reads must complete before the MFMAs that consume that data.

Look at the register names in the rocasm code: `A0`, `B0` are one sub-iteration, `A2`, `B2` are another. Don't move an MFMA that uses `A0` data past the point where `A0` registers get overwritten by a ds_read for the next sub-iteration.

### Rule 7: Scalar Operations Are Free

Scalar operations (`s_add_u32`, `s_addc_u32`, `s_sub_u32`, `s_cmp_eq_u32`, `s_cselect_b32`) execute on a separate scalar unit and take ~4 cycles. They can be moved freely among MFMAs without causing stalls. They are often used for address increment calculations for global reads. Keep them near the buffer_loads they serve, but don't worry about their scheduling impact.

## Reading the Profile Data

The `profile.txt` format is:
```
# TYPE         | AVG_LAT | STALL | PYTHON
mfma           |     4.0 |     0 | Acc[0:4] = vmfma_f32_16x16x16bf16_1k(B0[0:2], A0[0:2], Acc[0:4])
buffer_load    |    22.0 |  4240 | G2LB[28:32] = buffer_load_dwordx4(...)
```

- **AVG_LAT = 4.0** means the instruction issued at minimum latency (no stall). This is ideal.
- **AVG_LAT > 4.0** means the instruction stalled. The excess over 4.0 is the per-execution stall.
- **STALL** is the total stall cycles across all iterations. Higher = more total wasted time. Use this to prioritize which instructions to address first.

### Diagnostic Patterns

| Pattern | Diagnosis | Action |
|---------|-----------|--------|
| Consecutive MFMAs with STALL > 1000 | Pipeline bubble — no independent work | Interleave memory ops between these MFMAs |
| buffer_load STALL increasing monotonically | VMEM queue saturation | Spread buffer_loads earlier/wider in the loop |
| s_waitcnt with high STALL | Waiting for memory to complete | Move the loads it waits for earlier, or reduce the wait count |
| s_barrier with high STALL | Cross-wavefront sync | Usually unavoidable, but ensure work before the barrier is balanced |
| ds_read with high STALL | LDS data not ready | Ensure the ds_write that produced the data has enough distance |

## Performance Tracking

### Step 0: Establish Baseline

Before making any changes, run the full workflow once on the unmodified schedule. Record:

1. **Baseline GFLOPS** — from the Tensile benchmark output CSV line (the `gflops` column)
2. **Baseline kernel time** — from the `time-us` column in the same CSV line
3. **Total stall by instruction type** — sum the STALL column in `profile.txt` grouped by TYPE:
   ```
   mfma:         XXXXXX
   s_waitcnt:    XXXXXX
   ds_read:      XXXXXX
   s_barrier:    XXXXXX
   buffer_load:  XXXXXX
   ds_write:     XXXXXX
   ```
4. **Total mainloop stall** — sum of all STALL values across all lines

Report these numbers before proposing any changes.

### Step 0.5: Estimate Improvement Potential

Analyze the baseline profile to estimate how much improvement is achievable through instruction reordering alone. Categorize stalls as:

- **Avoidable stalls** — these can be reduced by better scheduling:
  - MFMA stalls from pipeline bubbles (consecutive MFMAs with no interleaved work)
  - buffer_load stalls from VMEM queue saturation (loads clustered too tightly)
  - ds_read stalls from insufficient distance between write and read
  - s_waitcnt stalls that could be reduced by moving loads earlier

- **Unavoidable stalls** — inherent to the algorithm, not fixable by reordering:
  - s_barrier stalls (cross-wavefront synchronization cost)
  - Minimum s_waitcnt stalls (must wait for at least some data before proceeding)
  - Memory latency that can't be hidden because there aren't enough MFMAs to cover it

Estimate: `theoretical_best_stall ≈ total_stall - avoidable_stalls`

Report the estimated improvement potential as a percentage:
```
Baseline total stall:     XXXXXX cycles
Estimated avoidable:      XXXXXX cycles (XX%)
Estimated unavoidable:    XXXXXX cycles (XX%)
Theoretical best stall:   XXXXXX cycles
```

This sets expectations — if only 15% of stalls are avoidable, don't expect more than ~15% improvement.

### Per-Iteration Reporting

After each iteration, report a summary table:

```
Iteration  | GFLOPS | Time (us) | Total Stall | Delta Stall | Change Made
-----------|--------|-----------|-------------|-------------|---------------------------
0 baseline |  XXX.X |   XXX.X   |   XXXXXXX   |      —      | (unmodified)
1          |  XXX.X |   XXX.X   |   XXXXXXX   |   -XX.X%    | Spread buffer_loads across loop
2          |  XXX.X |   XXX.X   |   XXXXXXX   |    -X.X%    | Interleaved ds_reads with MFMAs
```

The GFLOPS value comes from the Tensile benchmark output. Parse it from the CSV line:
```
...,PASSED,<time-us>,<gflops>,...
```

## Convergence

Stop iterating when:
- Total stall count stops decreasing for 2 consecutive iterations
- No individual instruction has STALL > 2× the class baseline
- The kernel's GFLOPS stops improving
- You've recovered most of the estimated avoidable stalls

After each iteration, report:
1. What pattern you identified in the profile
2. What change you made and why
3. The per-iteration summary table row (GFLOPS, time, stall, delta)
4. Whether to continue iterating and why

## Correctness

Every iteration MUST pass Tensile's numerical validation (`PASSED` in the benchmark output). If validation fails after a change, **revert the change immediately** — you introduced a data dependency violation. Re-read the rules above and try a different approach.
