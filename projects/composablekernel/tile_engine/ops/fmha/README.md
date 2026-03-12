# FMHA Tile Engine

Benchmarking and kernel enumeration for Fused Multi-Head Attention via the CK dispatcher's JIT pipeline.

## Quick Start

```bash
# Minimal CI test (16 kernels, ~1 min)
python fmha_benchmark.py configs/fwd_ci.json --workers 128 --verify

# Full receipt-0 sweep (11,980 kernels, ~35 min with 256 workers)
python fmha_benchmark.py configs/receipt0_fwd.json --workers 256 --compile-only

# Count configs without building
python fmha_instance_builder.py configs/receipt0_fwd.json --count-only
```

## Architecture

```
fmha/
  fmha_instance_builder.py   # Kernel enumeration (JSON config + pipeline rules)
  fmha_benchmark.py          # JIT compile + GPU benchmark runner
  CMakeLists.txt             # CMake targets (benchmark_fmha, benchmark_fmha_ci, etc.)
  configs/                   # Sweep definitions (JSON)
    receipt0_fwd.json        # Full receipt-0: 11,980 kernels on gfx950
    fwd_ci.json              # Minimal CI: fp16, qr_async, batch, no features
    fwd.json                 # Forward variants
    bwd.json                 # Backward variants
    splitkv.json             # Split-KV
    appendkv.json            # Append-KV
    pagedkv.json             # Paged-KV
    batch_prefill.json       # Batch prefill
  filters/                   # Sample Python filter files
    h128_no_dropout.py       # Example: keep only h128 without dropout
```

The kernel enumeration pipeline:

```
JSON config (trait_config allow-list)
  --> fmha_pipeline_rules.py (self-contained CK parity rules)
    --> fmha_arch_specs.json (tile tables per arch/dtype/hdim)
      --> FmhaKernelConfig list (11,980 for receipt-0 gfx950)
        --> optional --filter / --filter-file
          --> setup_multiple_fmha_dispatchers() (3-stage pipelined JIT)
            --> GPU benchmark
```

## JSON Config Format

Each JSON config specifies a `variant` and an optional `trait_config` that acts as an allow-list filter over the pipeline rules output.

```json
{
  "variant": "fwd",
  "trait_config": {
    "data_type": {"values": ["fp16"]},
    "pipeline": {"values": ["qr_async"]},
    "mask": {"values": ["no"]},
    "bias": {"values": ["no"]},
    "mode": {"values": ["batch"]},
    "lse": {"values": [false]},
    "dropout": {"values": [false]},
    "logits": {"values": [false]},
    "sink": {"values": [false]}
  }
}
```

If a trait key is absent, all values pass (no filtering on that dimension). The `receipt0_fwd.json` config only specifies `data_type` to exclude fp32, letting everything else through for the full 11,980-kernel set.

## Filtering

### CLI expression filter

```bash
# Only h128 qr_async kernels
python fmha_benchmark.py configs/receipt0_fwd.json \
    --filter "c.hdim_q == 128 and c.pipeline == 'qr_async'"

# Only fp8 kernels with blockscale
python fmha_instance_builder.py configs/receipt0_fwd.json \
    --filter "c.qscale == 'blockscale'" --count-only
```

The expression has access to `c` (the `FmhaKernelConfig` dataclass) with fields: `data_type`, `mode`, `hdim_q`, `hdim_v`, `pipeline`, `tile_m0`, `tile_n0`, `tile_k0`, `pad_s`, `pad_sk`, `pad_d`, `pad_dv`, `mask`, `bias`, `lse`, `dropout`, `logits`, `sink`, `skip_min_seqlen_q`, `qscale`.

### Python file filter

```bash
python fmha_benchmark.py configs/receipt0_fwd.json \
    --filter-file filters/h128_no_dropout.py
```

The file must define `filter_config(c) -> bool`. See `filters/h128_no_dropout.py` for a template.

Both `--filter` and `--filter-file` can be combined (AND logic).

## Parity with CK

The dispatcher's `fmha_pipeline_rules.py` reproduces the exact kernel filtering logic from CK Tile's `01_fmha/codegen/ops/fmha_fwd.py` -- including per-arch tile constraints, pipeline selection rules, and receipt filters. Run the parity test to verify:

```bash
python dispatcher/tests/validate_arch_specs_parity.py --arch gfx950 --receipt 0
# PASS: 11,980 kernels, 37 categories all match
```

## CMake Targets

```bash
make benchmark_fmha          # Forward sweep
make benchmark_fmha_ci       # Quick CI validation
make benchmark_fmha_bwd      # Backward sweep
make benchmark_fmha_all      # All variants
make benchmark_fmha_splitkv  # Split-KV only
```

## Benchmark Output

```bash
python fmha_benchmark.py configs/fwd_ci.json --workers 128 --verify --best
```

Produces per-kernel timing and optional CPU reference validation:

```
  Kernel                                 Time(ms)  TFLOPS  MaxErr  Status
  fmha_fwd_fp16_batch_h128_qr_async...     0.013   40.55  9.7e-06  PASS
  fmha_fwd_fp16_batch_h256_qr_async...     0.024   22.72  9.7e-06  PASS
```

Use `--csv` or `--json` to export results for analysis.
