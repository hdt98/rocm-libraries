# Tune Kernel — Orchestrator

Orchestrates the full kernel tuning loop for a single convolution shape.
Delegates to the specialist agents in sequence and manages the feedback loop.
Maintains an up-to-date list of the WAN model shapes where their performance in terms TFLOPs 
is recorded and shapes are ranked in terms of their TFLOPs performance (worst shape is first). 

## Usage

Provide one of:
- A MIOpenDriver command line for a specific shape to tune
- `worst-i2v` / `worst-t2v` — automatically pick the lowest-TFLOPs shape from the latest set of benchmarked shapes.
- `all-i2v` / `all-t2v` — iterate over all shapes below a TFLOPs threshold (ask the user for the threshold; a reasonable default is 50% of the theoretical peak)

## Loop

```
[Profile] → identify target shape
    ↓
[Engineer] → analyze shape, propose instance, add to .inc example, compile example
    ↓  ← (compile error: update INSTANCE_CONSTRAINTS.md, re-propose)
    ↓  (performance check via example binary, verify=0)
    ↓  ← (no improvement: optionally invoke Kernel Profiler for hardware insights, then re-propose)
    ↓  (improvement found)
[Test] → verify correctness via example binary (verify=2)
    ↓  ← (fail: report back to Engineer)
[Engineer] → add final instance to device_grouped_conv_fwd_xdl_instance.hpp, compile ckProfiler
    ↓
[Orchestrator] snapshots i2v_current.txt → i2v_<iter>.txt, runs profiler on shape, updates i2v_current.txt
    ↓
Report improvement (or lack thereof) and ask whether to continue tuning this shape or move to the next.
If `all-i2v` / `all-t2v` argument is provided, move to the next shape.
```

Maintain an internal loop count. The loop count should start from zero, unless snapshot files
`projects/composablekernel/build-gfx950/data/i2v_<N>.txt` already exist — in that case,
find the highest `<N>` among existing files and start from `<N> + 1`.

## Step-by-step

### 1. Identify target shape

If `worst-i2v` or `worst-t2v`:
- Check that `projects/composablekernel/build-gfx950/data/i2v_current.txt` (or `t2v_current.txt`) exists.
  If it does not exist, **stop and report an error** — do not proceed without the current ranking file.
- Read the ranking and find the shape with the lowest `tflops:` value that is not yet marked as processed.
- Present it to the user and confirm before proceeding.

If a MIOpenDriver command is given directly, use that shape.

If given `all-i2v` / `all-t2v` proceed automatically from the lowest unoptimized shape. 
Then continue processing all unoptimized shapes.

### 2. Engineer the kernel

Invoke the Kernel Engineer role:
> "Acting as the Kernel Engineer (see `.claude/commands/engineer-kernel.md`), analyze the
> following shape and propose a new instance: [shape]"

Iterate until compilation succeeds (max 15 attempts before asking the user to intervene).
Input may include optional profiling feedback from the Kernel Profiler.

### 3. Profile the kernel (optional — only if Engineer cannot improve performance)

If the Kernel Engineer reports no improvement after a candidate attempt, invoke the Kernel Profiler
to identify hardware bottlenecks:

> "Acting as the Kernel Profiler (see `.claude/commands/profile-kernel.md`), profile the
> kernel that the Kernel Engineer provided for the shape we are currently working on.
> Provide feedback for the Kernel Engineer: [shape]"

Once the profiling feedback is ready, go back to step 2 with that feedback. Run at most 10
Kernel Engineer ↔ Kernel Profiler iterations before concluding that the shape is already optimized.

### 4. Test correctness

Invoke the Kernel Tester role:
> "Acting as the Kernel Tester (see `.claude/commands/test-kernel.md`), verify the candidate
> kernel for the shape: [shape]. Use args: [example binary args with verify=2]"

If the test fails, return to step 2 with the failure details.

### 5. Add to instance header and compile profiler

Once the tester passes, return to the Engineer to complete Steps 7–10:
add the instance to `device_grouped_conv_fwd_xdl_instance.hpp` and compile `ckProfiler`.

### 6. Benchmark

Run the profiler on the single shape and extract the new TFLOPs. Compare against the
baseline entry for the same shape.

**File bookkeeping (orchestrator responsibility):**

1. Snapshot the current ranking before updating:
   - Copy `data/i2v_current.txt` → `data/i2v_<iter>.txt` (use the internal iteration number)
   - Copy `data/t2v_current.txt` → `data/t2v_<iter>.txt`

2. Update `data/i2v_current.txt` / `data/t2v_current.txt` with the new benchmark result for
   this shape, preserving all other entries unchanged.

3. Mark the shape as processed in `_current.txt` so it is skipped in future `worst-*` / `all-*` runs.

4. Increment the internal iteration counter.

### 7. Report

Present a summary:
```
Shape:         <description>
Baseline:      X.XX TFLOPs  (instance: <name>)
New instance:  Y.YY TFLOPs  (instance: <name>)
Improvement:   +Z.ZZ TFLOPs (+P%)
Constraints discovered: <list any new entries added to INSTANCE_CONSTRAINTS.md>
```

Ask the user whether to continue to the next shape or stop. If provided `all-i2v` / `all-t2v`, proceed automatcally
to the next shape.

## Key files (shared state)

| File | Purpose |
|---|---|
| `build-gfx950/data/i2v_baseline.txt` | Reference performance numbers |
| `build-gfx950/data/t2v_baseline.txt` | Reference performance numbers |
| `build-gfx950/data/i2v_current.txt`  | Current performance numbers   |
| `build-gfx950/data/t2v_current.txt`  | Current performance numbers   |
| `INSTANCE_CONSTRAINTS.md` | Constraints discovered during compilation — always keep up to date |
| `projects/composablekernel/library/include/ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_xdl_instance.hpp` | Generic XDL instances to be tuned |
|  `projects/composablekernel/library/include/ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_xdl_large_tensor_instance.hpp` | Large tensor instances |
|  `projects/composablekernel/library/include/ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_xdl_mem_instance.hpp` | Instances for memory bound cases |
|  `projects/composablekernel/library/include/ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_xdl_merged_groups_instance.hpp` | Instances for merging multiple conv groups into a single GEMM batch |
|  `projects/composablekernel/library/include/ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_xdl_comp_instance.hpp` | Instances for compute bound cases |
