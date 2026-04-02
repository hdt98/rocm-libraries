# Collecting ATT Traces for rocRoller GEMM Runs

This guide teaches an agent how to collect AMDGPU AMD Thread Trace (ATT) files
for rocRoller GEMM kernels using `rocprofv3`, the command-line profiling tool
shipped with ROCm.

---

## Prerequisites

- ROCm ≥ 6.x installed at `ROCM_PATH` (default `/opt/rocm`)
- `rocprofv3` on `PATH` (ships with ROCm: `${ROCM_PATH}/bin/rocprofv3`)
- `rocprof-trace-decoder` shared library for decoding `.att` binary output:

```bash
# Ubuntu 22.04
wget -O /tmp/rocprof-trace-decoder.deb \
  https://github.com/ROCm/rocprof-trace-decoder/releases/download/0.1.4/rocprof-trace-decoder-ubuntu-22.04-0.1.4-Linux.deb
sudo apt-get install -y /tmp/rocprof-trace-decoder.deb
rm /tmp/rocprof-trace-decoder.deb
```

The decoder is searched in `LD_LIBRARY_PATH` and `/opt/rocm/lib`. For a custom
install location pass `--att-library-path <dir>` or set
`ROCPROF_ATT_LIBRARY_PATH`.

---

## Build the rocRoller GEMM Client

```bash
cmake --preset default:release -B build -S .
cmake --build build -j -- rocroller-gemm
```

The client executable is `build/client/rocroller-gemm`.

---

## Basic ATT Collection

Always apply `--kernel-include-regex "GEMM"` so that only the GEMM dispatch is
traced (see [Targeting a Specific Kernel](#targeting-a-specific-kernel)). The
output is written to `att_output/` (created if it does not exist). Because the
output directory name contains a variable component (the agent PID), and previous
traces may already exist there, snapshot the directory contents before and after
the run to reliably identify the new trace:

```bash
# Snapshot existing trace directories
before=$(find att_output -maxdepth 1 -name 'ui_output_agent_*' -type d 2>/dev/null | sort)

rocprofv3 --att \
    --kernel-include-regex "GEMM" \
    --output-directory ./att_output -- \
    ./build/client/rocroller-gemm \
        --M=1024 --N=1024 --K=1024 \
        --type_A=half --type_B=half --type_C=half --type_D=half --type_acc=float \
        --trans_A=N --trans_B=T \
        generate validate

# Identify the newly created directory
trace_dir=$(comm -23 \
    <(find att_output -maxdepth 1 -name 'ui_output_agent_*' -type d | sort) \
    <(echo "$before"))

rocprof-compute-viewer "$trace_dir"
```

The `generate validate` subcommands generate the kernel and launch it exactly once for
correctness checking. Prefer this over `benchmark` when tracing, as `benchmark` launches
the kernel many times and only the first dispatch is captured.

### Output files

| File | Contents |
|---|---|
| `stats_*.csv` | Per-instruction hitcount, latency, stall, idle per kernel |
| `ui_output_agent_*_dispatch_*/` | JSON traces for ROCprof Compute Viewer |
| `*.att` | Raw SQTT binary (decoded automatically by `rocprofv3`) |
| `*.out` | Code object binary (used for ISA correlation) |

The `stats_*.csv` file is the primary artifact for instruction-level analysis.
Its columns are:

| Column | Meaning |
|---|---|
| `Codeobj` | Code-object load ID assigned by rocprofiler-sdk |
| `Vaddr` | ELF virtual address of the instruction |
| `Hitcount` | Total executions across all traced waves |
| `Latency` | Stall + Issue cycles (gfx9) or Stall + Execute cycles (gfx10+) |
| `Stall` | Cycles the hardware pipe could not issue (LDS/TCP backpressure, etc.) |
| `Idle` | Cycles between instruction completion and the next issue |
| `Source` | Source line (requires kernel compiled with debug symbols) |

---

## Targeting a Specific Kernel

Without a kernel filter, the first dispatch captured is typically `fillBuffer`
(data movement to device). The GEMM kernel itself is usually the second dispatch.
Use `--kernel-include-regex "GEMM"` to target it directly by name, as shown in
[Basic ATT Collection](#basic-att-collection).

---

## Configuring the Trace Parameters

Thread trace is collected from a **single CU per shader engine**. For small GEMM
workloads there is a risk that no wave lands on the target CU. The flags below
control this:

| Flag | Default | Recommendation |
|---|---|---|
| `--att-target-cu` | 1 | CU index (0–15) within the selected SE |
| `--att-shader-engine-mask` | `0x1` | Bitmask of SEs; start with `0x1`, widen if empty |
| `--att-simd-select` | `0xF` | SIMD bitmask within the CU |
| `--att-buffer-size` | system default | Bytes; increase for long kernels |

Example with explicit parameters:

```bash
rocprofv3 --att \
    --att-target-cu 1 \
    --att-shader-engine-mask 0x1 \
    --att-simd-select 0xF \
    --att-buffer-size 0x6000000 \
    --output-directory ./att_output -- \
    ./build/client/rocroller-gemm --M=1024 --N=1024 --K=1024 \
        --type_A=half --type_B=half --type_C=half --type_D=half --type_acc=float \
        --trans_A=N --trans_B=T generate validate
```

Alternatively supply these as a JSON input file:

```json
{
  "jobs": [{
    "advanced_thread_trace": true,
    "att_target_cu": 1,
    "att_shader_engine_mask": "0x1",
    "att_simd_select": "0xF",
    "att_buffer_size": "0x6000000"
  }]
}
```

```bash
rocprofv3 --input att_config.json --output-directory ./att_output -- \
    ./build/client/rocroller-gemm --M=1024 --N=1024 --K=1024 \
        --type_A=half --type_B=half --type_C=half --type_D=half --type_acc=float \
        --trans_A=N --trans_B=T generate validate
```

---

## Concentrating Waves on the Target CU

For small GEMM kernels that may not dispatch enough waves to reach the target CU, it may receive no waves.
`HSA_CU_MASK` restricts HW scheduling so that most or all waves land on the
traced CU:

```bash
# Allow only CU 1 in SE 0 (bitmask format is GPU-model specific; consult `rocminfo`)
HSA_CU_MASK=0x2 rocprofv3 --att --att-target-cu 1 --output-directory ./att_output -- \
    ./build/client/rocroller-gemm --M=1024 --N=1024 --K=1024 \
        --type_A=half --type_B=half --type_C=half --type_D=half --type_acc=float \
        --trans_A=N --trans_B=T generate validate
```

`HSA_CU_MASK` is a hex bitmask where each bit is a CU. Set all bits except the
target to 0. This forces the GPU scheduler to assign waves to the unmasked CU,
making the ATT capture reliable even for small tile sizes.

---

## Tracing Multiple Dispatches

By default `rocprofv3` captures the **first matching dispatch** only.

```bash
# Trace dispatches 2 through 5 of the matched kernel
rocprofv3 --att \
    --kernel-include-regex "GEMM" \
    --kernel-iteration-range 2:5 \
    --output-directory ./att_output -- \
    ./build/client/rocroller-gemm --M=1024 --N=1024 --K=1024 \
        --type_A=half --type_B=half --type_C=half --type_D=half --type_acc=float \
        --trans_A=N --trans_B=T generate validate

# Merge N consecutive dispatches into one output file
rocprofv3 --att \
    --att-consecutive-kernels 4 \
    --output-directory ./att_output -- \
    ./build/client/rocroller-gemm ...
```

---

## Adding SQ Performance Counters

To stream hardware counters into the ATT buffer alongside instruction data:

```bash
rocprofv3 --att \
    --att-perfcounter-ctrl 3 \
    --att-perfcounters "SQ_VALU_MFMA_BUSY_CYCLES SQ_INSTS_VALU SQ_INSTS_MFMA SQ_INST_LEVEL_LDS" \
    --output-directory ./att_output -- \
    ./build/client/rocroller-gemm --M=1024 --N=1024 --K=1024 \
        --type_A=half --type_B=half --type_C=half --type_D=half --type_acc=float \
        --trans_A=N --trans_B=T generate validate
```

---

## Viewing Results

### CSV (quick inspection)

```bash
column -t -s, ./att_output/stats_*.csv | less
```

### ROCprof Compute Viewer (interactive)

```bash
pip install rocprof-compute-viewer   # or use the ROCm package
rocprof-compute-viewer "$trace_dir"
```

where `trace_dir` is the newly created `ui_output_agent_*_dispatch_1/` directory
identified by diffing the output directory before and after the run (see
[Basic ATT Collection](#basic-att-collection)).

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `stats_*.csv` is empty / missing rows | Small GEMM dispatched no waves to target CU | Set `HSA_CU_MASK` to pin waves; try `--att-target-cu 0` |
| Packet loss / truncated trace | Buffer too small or too many SEs traced | Increase `--att-buffer-size`; narrow `--att-shader-engine-mask` |
| `INVALID_SHADER_DATA` error | AQL profile / trace decoder version mismatch | Update `rocprof-trace-decoder` to match ROCm version |
| All kernels traced (slow) | No kernel filter applied | Add `--kernel-include-regex` |
| Decoder library not found | `rocprof-trace-decoder` not installed | Install deb (see Prerequisites); or set `ROCPROF_ATT_LIBRARY_PATH` |
| rocprofv3 not found | ROCm not on PATH | `export PATH=/opt/rocm/bin:$PATH` |
