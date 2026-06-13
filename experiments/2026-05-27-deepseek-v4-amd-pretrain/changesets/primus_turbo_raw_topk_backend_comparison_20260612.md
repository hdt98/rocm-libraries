# Primus-Turbo Raw Top-K Backend Comparison

Date: 2026-06-12

## Question

Redo the baseline table correctly between:

- no-helper standard EP, meaning TorchTitan's standard/PyTorch all-to-all EP path;
- no-helper Primus-Turbo raw top-k EP, using the real lower backend selector;
- standard-EP top-8 hot-helper, as the retained intermediate feature gate.

The desired future target is still native hot-helper with MORI or Primus-Turbo,
but that should be measured against the correct raw/no-helper and hot-helper
baselines.

## Backend Identification

The Primus-Turbo no-helper runs were not generic "DeepEP" runs. They used the
Primus-Turbo normal top-k hooks:

```text
CANARY_STANDARD_EP_BACKEND_NORMAL_TOPK_DISPATCH_PERMUTE=1
CANARY_MORI_AITER_BALANCED_MOE_PLAN_BACKEND=primus_turbo
```

The path is:

```text
TorchTitan normal token-major [tokens, top_k] routes
-> primus_turbo.pytorch.ops.moe.balanced_moe
-> dispatch_permute_normal_topk_tokens(...)
-> moe_dispatch(...)
-> Primus-Turbo moe_permute / moe_unpermute
-> moe_combine(...)
```

Source check:

```text
primus_turbo/pytorch/kernels/moe/moe_dispatch_combine_impl.py
  TurboEPBackend     in-tree Primus-Turbo DeepEP backend
  DeepEPBackend      optional external deep_ep package backend
  default selector   TURBO
```

Runtime smoke on `do-sonle5-mi350-gpu` in
`onenexus/nexus-titan:rocm722-pytorch-nightly-mori` with ROCm devices mounted:

```text
env None
TurboEPBackend available True
External DeepEPBackend available False
selected default name TURBO
selected default type TurboEPBackend
deep_ep spec None
deepep spec None
```

So the retained Primus-Turbo rows below should be read as in-tree
`TurboEPBackend` raw top-k/no-helper EP, not optional external
`DeepEPBackend`. External `DeepEPBackend` remains not measured on this surface.

### External DeepEP Availability Check

Artifact:

```text
run_artifacts/torchtitan_primus_external_deepep_backend_availability_20260612.json
```

Fresh checks on `do-sonle5-mi350-gpu` and `do-vunguyen-mi350-gpu` confirm that
the current TorchTitan MORI image exposes `primus_turbo`, but not `deep_ep`,
`deepep`, `uccl`, or `rocshmem`. The `rocm/primus:v26.2` image also exposes
`primus_turbo` and the `DEEP_EP` enum, but reports `HAVE_DEEP_EP=False` and is
missing the same external packages. Source-wise, `DeepEPBackend` is only the
optional external-package wrapper; `TurboEPBackend` is the in-tree backend that
we already measured.

Decision: do not launch a 20-step `PRIMUS_TURBO_MOE_DISPATCH_COMBINE_BACKEND=DEEP_EP`
training run in the current images. It would fail before producing a fair
metric. A real external-DeepEP column requires a MI350-compatible image with
`deep_ep`/`uccl`/rocSHMEM installed, followed by a one-step backend smoke and
then the same 20-step no-helper ladder.

## Corrected Table

All rows are 20-step, 8xMI350, S4096, GBS128, FSDP8/EP8,
attention-only selective activation checkpointing, CE8, AdamW. Values are
steps 2-20 tok/GPU/s unless the standard-EP DeepSeek MBS8 no-helper row fails
before the first metric.

| Model surface | MBS | Plain no-helper standard EP | Primus-Turbo `TurboEPBackend` no-helper raw top-k EP | Standard-EP top-8 hot-helper | Primus vs plain | Primus vs helper |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DeepSeek-V4 Flash 12-layer | 1 | `3,822.38` | `5,082.97` | `4,202.72` | `+32.98%` | `+20.94%` |
| DeepSeek-V4 Flash 12-layer | 2 | `4,321.64` | `5,898.16` | `5,254.38` | `+36.48%` | `+12.25%` |
| DeepSeek-V4 Flash 12-layer | 4 | `5,058.13` | `6,720.95` | `6,550.00` | `+32.87%` | `+2.61%` |
| DeepSeek-V4 Flash 12-layer | 8 | fails before metric | `6,910.74` | `8,632.78` | changes fit boundary | `-19.95%` |
| Qwen3.5-397B-A17B 8-layer | 1 | `7,857.68` | `9,355.32` | `8,379.57` | `+19.06%` | `+11.64%` |
| Qwen3.5-397B-A17B 8-layer | 2 | `10,058.39` | `12,473.87` | `11,174.09` | `+24.01%` | `+11.63%` |
| Qwen3.5-397B-A17B 8-layer | 4 | `12,106.35` | `15,745.63` | `13,914.22` | `+30.06%` | `+13.16%` |
| Qwen3.5-397B-A17B 8-layer | 8 | `12,683.54` | `11,650.90` | `15,833.91` | `-8.14%` | `-26.42%` |

### Qwen MBS Scaling Audit

The first Qwen Primus-Turbo audit correctly marked MBS8 as suspicious, but it
did not explain the apparent MBS2 -> MBS4 drop. A matched MBS2/MBS4 profile
pair plus a clean 20-step MBS4 rerun now closes that gap: the MBS4 drop does
not reproduce, and the old `11,799.62` MBS4 row is demoted as stale or
non-comparable evidence.

Correction artifact:

```text
run_artifacts/torchtitan_qwen397b_primus_turbo_mbs2_mbs4_rerun_profile_20260612.json
```

Fresh MBS4 retained row:

```text
qwen397b_primus_turbo_nohelper_mbs4_rerun_steps20_20260612T1602
steps 2-20:  15,745.63 tok/GPU/s, 269.61 TFLOPs/GPU
steps 13-20: 15,746.38 tok/GPU/s, 269.62 TFLOPs/GPU
peak memory: 262.80 GiB
allocator retry warnings: 0
```

The low-intrusion MBS2/MBS4 profile explains the direction:

| Profile row | GA | Dispatch records/rank/step | Rows/dispatch record | Dispatch+permute rank-max/step | Unpermute+combine rank-max/step | Train step rank-max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MBS2 | 8 | 64 | `81,920` | `177.76 ms` | `415.36 ms` | `5151.89 ms` |
| MBS4 | 4 | 32 | `163,840` | `164.68 ms` | `369.94 ms` | `4210.42 ms` |

So MBS4 doubles rows per EP call but halves GA and halves the number of EP
records per training step. The measured total EP edge and total train step both
drop. The old MBS4 row should not be used for scaling conclusions.

The Qwen Primus-Turbo MBS8 row remains suspicious, and the raw log audit
confirms that it should not be treated as a clean high-MBS scaling baseline yet.
The configs line up, but the MBS8 run is allocator-pressure contaminated:

- `MBS1`, `MBS2`, and `MBS4` log zero CUDA memory allocation retry warnings.
- `MBS8` logs `45` CUDA memory allocation retry warnings and reaches
  `277.96 GiB` max logged memory.
- `MBS8` has high bursts (`13,868` tok/GPU/s at step 2), but also slow steps
  at `8,936`, `7,167`, `9,136`, and `10,030` tok/GPU/s.

Audit artifact:

```text
run_artifacts/torchtitan_qwen397b_primus_turbo_mbs_scaling_audit_20260612.json
```

| Model surface | MBS | Primus steps 2-20 | Primus steps 13-20 | Read |
| --- | ---: | ---: | ---: | --- |
| DeepSeek-V4 Flash 12-layer | 4 | `6,720.95` | `6,638.38` | MBS4 remains above late MBS8 |
| DeepSeek-V4 Flash 12-layer | 8 | `6,910.74` | `6,837.13` | MBS8 fits but still trails helper |
| Qwen3.5-397B-A17B 8-layer | 4 | `15,745.63` | `15,746.38` | clean rerun; old row demoted |
| Qwen3.5-397B-A17B 8-layer | 8 | `11,650.90` | `12,149.56` | contaminated by allocator retries; diagnostic only |

So the Qwen MBS8 retained average should not be read as proof that larger MBS
is intrinsically slower for TurboEPBackend, and it also should not be used as
evidence that raw TurboEP scales cleanly at high MBS. It should be read as:
raw top-k TurboEP has a real high-MBS memory/headroom or adapter-pressure
problem on this Qwen surface. The next fair gate is an MBS8 rerun/profile that
splits Turbo dispatch/permute/combine, expert compute, weighted combine,
allocator retries, and memory snapshots.

Artifacts:

```text
run_artifacts/torchtitan_dsv4_flash12_standard_ep_hothelper_mbs1_mbs2_ce8_ladder_20step_20260612.json
run_artifacts/torchtitan_qwen397b_a17b_standard_ep_hothelper_mbs1_mbs8_20step_20260612.json
run_artifacts/torchtitan_primus_turbo_nohelper_cross_model_mbs1_mbs8_20step_20260612.json
run_artifacts/torchtitan_dsv4_flash12_primus_deepep_nohelper_mbs4_mbs8_20step_20260612.json
run_artifacts/torchtitan_qwen397b_a17b_primus_deepep_nohelper_mbs4_mbs8_20step_20260612.json
run_artifacts/torchtitan_qwen397b_primus_turbo_mbs2_mbs4_rerun_profile_20260612.json
```

## Read

Primus-Turbo `TurboEPBackend` is a real raw-top-k/no-helper EP candidate. It
helps DeepSeek MBS1/2/4 and makes DeepSeek MBS8 fit when plain standard EP
no-helper fails before metrics. It also helps Qwen MBS1/2/4.

It does not replace hot-helper:

- DeepSeek MBS8 still trails standard-EP top-8 hot-helper by `19.95%`.
- Qwen MBS4 is corrected to a positive raw-TurboEP row, but Qwen MBS8 still
  regresses versus both plain no-helper and top-8 hot-helper.
- The model/batch dependence is real, but the Qwen MBS8 row is diagnostic
  rather than clean scaling evidence until rerun or profiled with more memory
  headroom.

The next performance target is therefore not just "turn on DeepEP." It is:

```text
preserve the hot/cold/helper owner-compact partition
-> make MORI or Primus-Turbo consume that partition natively
-> avoid re-expanding it into raw token-major top-k routes or dense selected-order adapters
```

The hot-helper standard-EP row remains the intermediate feature gate. Native
MORI/Primus-Turbo hot-helper should beat that by reducing transport,
materialization, return, and helper-VJP overhead while keeping the same
load-placement advantage.
