# Tune Kernel — Orchestrator

Orchestrates the full kernel tuning loop for a single convolution shape.
Delegates to the specialist agents in sequence and manages the feedback loop.

## Usage

Provide one of:
- A MIOpenDriver command line for a specific shape to tune
- `worst-i2v` / `worst-t2v` — automatically pick the lowest-TFLOPs shape from the baseline files
- `all-i2v` / `all-t2v` — iterate over all shapes below a TFLOPs threshold (ask the user for the threshold)

## Loop

```
[Profile] → identify target shape
    ↓
[Engineer] → analyze shape, propose instance, add to .inc example, compile example
    ↓  ← (compile error: update INSTANCE_CONSTRAINTS.md, re-propose)
    ↓  (performance check via example binary)
[Test] → verify correctness via example binary (verify=2)
    ↓  ← (fail: report back to Engineer)
[Engineer] → add final instance to device_grouped_conv_fwd_xdl_instance.hpp, compile ckProfiler
    ↓
[Profile] → benchmark single shape, compare vs baseline
    ↓
Report improvement (or lack thereof) and ask whether to continue tuning this shape or move to the next
```

## Step-by-step

### 1. Identify target shape

If `worst-i2v` or `worst-t2v`:
- Read `projects/composablekernel/build-gfx950/data/i2v_baseline.txt` or `t2v_baseline.txt`
- Find the shape with the lowest `tflops:` value
- Present it to the user and confirm before proceeding

If a MIOpenDriver command is given directly, use that shape.

### 2. Engineer the kernel

Invoke the Kernel Engineer role:
> "Acting as the Kernel Engineer (see `.claude/commands/engineer-kernel.md`), analyze the
> following shape and propose a new instance: [shape]"

Iterate until compilation succeeds (max 5 attempts before asking the user to intervene).

### 3. Test correctness

Invoke the Kernel Tester role:
> "Acting as the Kernel Tester (see `.claude/commands/test-kernel.md`), verify the candidate
> kernel for the shape: [shape]. Use args: [example binary args with verify=2]"

If the test fails, return to step 2 with the failure details.

### 4. Add to instance header and compile profiler

Once the tester passes, return to the Engineer to complete Steps 7–10:
add the instance to `device_grouped_conv_fwd_xdl_instance.hpp` and compile `ckProfiler`.

### 5. Benchmark

Run the profiler on the single shape and extract the new TFLOPs. Compare against the
baseline entry for the same shape.

### 6. Report

Present a summary:
```
Shape:         <description>
Baseline:      X.XX TFLOPs  (instance: <name>)
New instance:  Y.YY TFLOPs  (instance: <name>)
Improvement:   +Z.ZZ TFLOPs (+P%)
Constraints discovered: <list any new entries added to INSTANCE_CONSTRAINTS.md>
```

Ask the user whether to continue to the next shape or stop.

## Key files (shared state)

| File | Purpose |
|---|---|
| `build-gfx950/data/i2v_baseline.txt` | Reference performance numbers |
| `build-gfx950/data/t2v_baseline.txt` | Reference performance numbers |
| `INSTANCE_CONSTRAINTS.md` | Constraints discovered during compilation — always keep up to date |
| `library/include/.../device_grouped_conv_fwd_xdl_instance.hpp` | The file being tuned |
