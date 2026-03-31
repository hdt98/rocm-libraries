# Agents Setup

This document describes the Claude Code agent skills set up for the WAN grouped forward
convolution tuning project. Skills are defined as slash commands in `.claude/commands/`
at the repo root and are available in any Claude Code session opened in this repository.

## Agents

| Skill | Command | Role |
|---|---|---|
| Profiler | `/profile-shapes` | Runs ckProfiler over WAN shapes; ranks by TFLOPs; identifies worst-performing shape classes |
| Kernel Engineer | `/engineer-kernel` | Analyzes a shape, proposes XDL tile parameters, writes the candidate instance, compiles, records constraints, and ultimately adds the final instance to the instance header |
| Kernel Profiler | `/profile-kernel` | Profiles a candidate instance (or two instances for comparison) using `rocprof-compute`; reports MFMA utilization and memory transfer bottlenecks |
| Kernel Tester | `/test-kernel` | Verifies numerical correctness of a candidate instance using the fwd conv example binary in verify=2 mode |
| Orchestrator | `/tune-kernel` | Drives the full loop end-to-end for one or more shapes |

## Workflow

The iterative tuning loop is:

```
[/profile-shapes] ──► identify worst-performing shape
        │
        ▼
[/engineer-kernel]
  Step 1: analyze shape → compute M/N/K, select ConvSpec
  Step 2: propose template parameters
  Step 3: add candidate to run_grouped_conv_fwd_example.inc
  Step 4: compile example binary (fast; catches static_assert errors)
           ↑ loop back on compile error (record in INSTANCE_CONSTRAINTS.md)
           run example for performance check (no verify)
           ↑ loop back to Step 2 if no improvement
           │  (optionally consult /profile-kernel for bottleneck analysis
           │   or baseline-vs-candidate comparison before re-proposing)
  Step 5: hand off to tester
        │
        ▼
[/test-kernel]
  Build example binary (if needed)
  Run with verify=2 → report max err / pass / fail
  Pass → hand back to engineer │ Fail → hand back to engineer
        │
        ▼
[/engineer-kernel] (Steps 7–10)
  Step 7: add instance to device_grouped_conv_fwd_xdl_instance.hpp
  Step 8: compile ckProfiler
  Step 9: handle compile errors (record constraints, loop back)
  Step 10: hand off to profiler
        │
        ▼
[/profile-shapes] ──► benchmark shape(s), compare vs baseline
        │
        └──► iterate to next worst shape, or stop
```

### `/profile-kernel` interaction (optional)

`/profile-kernel` can be invoked by `/engineer-kernel` at Step 4 in two modes:

- **Single instance**: given `<args>` for the example binary, runs `rocprof-compute` and
  reports MFMA utilization and memory transfer overhead for the candidate
- **Comparison**: given a baseline `.inc` and a candidate `.inc` plus `<args>`, profiles both
  separately and reports the most relevant differences in the analysis output

The result is passed back to `/engineer-kernel` to inform the next parameter adjustment.

## Key shared files

| File | Written by | Read by |
|---|---|---|
| `INSTANCE_CONSTRAINTS.md` | Kernel Engineer | Kernel Engineer (every session) |
| `PROBLEM.md` | Human | All agents |
| `build-gfx950/data/*_baseline.txt` | Profiler | Engineer, Orchestrator |
| `build-gfx950/data/*_results_iter_N.txt` | Profiler | Human, Orchestrator |
| `example/30_.../run_grouped_conv_fwd_example.inc` | Kernel Engineer | Kernel Tester, Kernel Profiler |
| `library/include/.../device_grouped_conv_fwd_xdl_instance.hpp` | Kernel Engineer | ckProfiler build |

## Design notes

- The **example binary** (`example_grouped_conv_fwd_xdl_bf16`) is used as a fast compile-and-test
  sandbox before touching the instance header. It avoids the much longer ckProfiler build during
  the iterate-on-compile-errors phase.
- **`/profile-kernel`** is an optional but valuable step when a candidate compiles and runs but
  doesn't beat the baseline — hardware counter data from `rocprof-compute` can reveal whether
  the bottleneck is MFMA underutilisation, LDS bandwidth, or global memory throughput.
- **`INSTANCE_CONSTRAINTS.md`** is the persistent knowledge base — every new compile-time rule
  found must be recorded there so future sessions don't repeat the same failed attempts.
- **Profiling all shapes** (batch mode) is expensive and should only be done when a new instance
  is expected to be broadly applicable. Prefer single-shape mode during active tuning.
- The Profiler numbers results files as `*_iter_N.txt` starting from 0, incrementing each time
  the Engineer is given a new shape to work on.
