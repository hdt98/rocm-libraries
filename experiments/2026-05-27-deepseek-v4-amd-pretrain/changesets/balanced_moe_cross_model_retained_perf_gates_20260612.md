# Balanced-MoE Cross-Model Retained Perf Gates

Date: 2026-06-12

## Purpose

The native backend goals for MORI and Primus-Turbo should not be promoted from
DeepSeek-only evidence.  The retained standard-EP hot-helper feature needs a
cross-model baseline first, then each backend-native implementation must beat
or at least reproduce the same mechanism on the same model surfaces.

This changeset records the current evidence and the remaining retained gates.

## Latest Robust Speedup

The robust retained feature measurement is the standard-EP top-8 hot-helper
ladder.  This is the target mechanism both MORI and Primus-Turbo need to
reproduce natively before backend-specific transport improvements are promoted.

All rows below use 8xMI350, `S=4096`, `GBS=128`, FSDP8/EP8, standard EP
`grouped_mm`, selective attention-only activation checkpointing, CE8, AdamW,
and 20 non-profiled steps.  Retained throughput windows exclude the first cold
step.

| Model | MBS | No helper tok/GPU/s | Top-8 helper tok/GPU/s | Uplift | Memory read |
| --- | ---: | ---: | ---: | ---: | --- |
| DeepSeek-V4 Flash 12-layer | 1 | `3,822.38` | `4,202.72` | `+9.95%` | retained feature evidence |
| DeepSeek-V4 Flash 12-layer | 2 | `4,321.64` | `5,254.38` | `+21.58%` | retained feature evidence |
| DeepSeek-V4 Flash 12-layer | 4 | `5,058.13` | `6,550.00` | `+29.49%` | retained feature evidence |
| DeepSeek-V4 Flash 12-layer | 8 | fails before first metric | `8,632.78` | fit-boundary change | helper makes MBS8 fit |
| Qwen3.5-397B-A17B 8-layer | 1 | `7,857.68` | `8,379.57` | `+6.64%` | `137.96 -> 136.19 GiB` |
| Qwen3.5-397B-A17B 8-layer | 2 | `10,058.39` | `11,174.09` | `+11.09%` | `171.06 -> 148.93 GiB` |
| Qwen3.5-397B-A17B 8-layer | 4 | `12,106.35` | `13,914.22` | `+14.93%` | `226.01 -> 178.09 GiB` |
| Qwen3.5-397B-A17B 8-layer | 8 | `12,683.54` | `15,833.91` | `+24.84%` | `281.42 -> 242.22 GiB` |
| Kimi-K2.6 6-layer | pending | no retained run | no retained run | not measured | ABI/route-shape only |

Read: the retained feature is now robust on DeepSeek and Qwen.  Kimi is still
the missing throughput gate; it has a route-shape ABI check, not a retained
training perf row.  Until that Kimi row exists, backend-native MORI or
Primus-Turbo claims should be worded as DeepSeek+Qwen-supported rather than
fully cross-model.

Coverage decision: Qwen retained perf is now part of the robustness baseline,
not a side note. Kimi remains a required retained perf gate, but it must be a
real Kimi training surface; the ABI/route-shape check is useful for backend
planner compatibility and is not a substitute for a tok/s row.

## Cross-Model Retained Perf Matrix

| Model surface | Reduced topology | Retained standard-EP top-8 helper result | Status |
| --- | --- | --- | --- |
| DeepSeek-V4 Flash | 12 layers, 256 experts, top-k 6, first-12 compress ratios `[1,1,4,128,4,128,4,128,4,128,4,128]` | MBS1/2/4 uplifts `+9.95%`, `+21.58%`, `+29.49%`; MBS8 no-helper fails before metric while top-8 reaches `8,632.78` tok/GPU/s late-window | retained feature evidence |
| Qwen3.5-397B-A17B | 8 layers, 512 routed experts, top-k 10 + 1 shared, `hidden=4096`, GQA `32Q/2KV`, `head_dim=256` | steps 2-20 uplifts `+6.64%`, `+11.09%`, `+14.93%`, `+24.84%` at MBS1/2/4/8; MBS8 memory improves `281.42 -> 242.22 GiB` | retained cross-model evidence |
| Kimi-K2.6 | target 6 layers, 384 routed experts, top-k 8 + 1 shared, `moe_inter=2048`, MLA, native `hidden=7168` with `4096` fallback if needed | no retained tok/s yet; only ABI/route-shape parity is covered by `tools/check_balanced_moe_model_shapes.py` | blocked on TorchTitan Kimi training model/config surface |

The Kimi row is intentionally not proxied through DeepSeek or Qwen. The next
Kimi step is a real reduced TorchTitan model/config surface, then the same
20-step no-helper versus top-8 helper ladder.

## Current Retained Feature Evidence

The robust retained measurement started as DeepSeek-V4 Flash only:

```text
8xMI350, S4096, GBS128, FSDP8/EP8
activation checkpointing: selective attention_only
chunked CE: CE8
standard EP grouped_mm
balanced_moe: top-8 helper execute
```

20-step late-window ladder:

| Model | MBS | No helper tok/GPU/s | Top-8 helper tok/GPU/s | Uplift | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| DeepSeek-V4 Flash 12-layer | 1 | `3,822.38` | `4,202.72` | `+9.95%` | retained feature evidence |
| DeepSeek-V4 Flash 12-layer | 2 | `4,321.64` | `5,254.38` | `+21.58%` | retained feature evidence |
| DeepSeek-V4 Flash 12-layer | 4 | `5,058.13` | `6,550.00` | `+29.49%` | retained feature evidence |
| DeepSeek-V4 Flash 12-layer | 8 | fails before first metric | `8,632.78` | fit-boundary change | retained feature evidence |

Full artifact:

```text
run_artifacts/torchtitan_dsv4_flash12_standard_ep_hothelper_mbs1_mbs2_ce8_ladder_20step_20260612.json
```

Machine-readable cross-model status, including the Kimi missing-gate row:

```text
run_artifacts/balanced_moe_cross_model_retained_perf_status_20260612.json
```

Qwen3.5-397B-A17B reduced 8-layer retained rows were added on the same
8xMI350 node and policy. The original MBS1/MBS2 artifact remains in
`run_artifacts/`, and the superseding full ladder artifact is:

```text
run_artifacts/torchtitan_qwen397b_a17b_standard_ep_hothelper_mbs1_mbs8_20step_20260612.json
```

20-step retained windows:

| Model | MBS | Window | No helper tok/GPU/s | Top-8 helper tok/GPU/s | Uplift | No helper peak GiB | Top-8 helper peak GiB | Status |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Qwen3.5-397B-A17B 8-layer | 1 | steps 2-20 | `7,857.68` | `8,379.57` | `+6.64%` | `137.96` | `136.19` | retained feature evidence |
| Qwen3.5-397B-A17B 8-layer | 1 | steps 13-20 | `7,900.73` | `8,434.08` | `+6.75%` | `137.96` | `136.19` | retained feature evidence |
| Qwen3.5-397B-A17B 8-layer | 2 | steps 2-20 | `10,058.39` | `11,174.09` | `+11.09%` | `171.06` | `148.93` | retained feature evidence |
| Qwen3.5-397B-A17B 8-layer | 2 | steps 13-20 | `9,923.47` | `11,166.19` | `+12.52%` | `171.06` | `148.93` | retained feature evidence |
| Qwen3.5-397B-A17B 8-layer | 4 | steps 2-20 | `12,106.35` | `13,914.22` | `+14.93%` | `226.01` | `178.09` | retained feature evidence |
| Qwen3.5-397B-A17B 8-layer | 4 | steps 13-20 | `12,220.66` | `13,850.00` | `+13.33%` | `226.01` | `178.09` | retained feature evidence |
| Qwen3.5-397B-A17B 8-layer | 8 | steps 2-20 | `12,683.54` | `15,833.91` | `+24.84%` | `281.42` | `242.22` | retained feature evidence |
| Qwen3.5-397B-A17B 8-layer | 8 | steps 13-20 | `13,174.91` | `15,676.33` | `+18.99%` | `278.91` | `242.22` | retained feature evidence |

Read: the hot-helper feature now has retained throughput evidence on DeepSeek
and Qwen.  On this Qwen surface the uplift grows with MBS, and the
memory saving becomes material at larger microbatches: `-47.92 GiB` at `MBS=4`
and `-39.20 GiB` at `MBS=8` over steps 2-20.  Unlike DeepSeek, Qwen MBS8
no-helper still completes, but it is close to the ceiling and logs CUDA memory
allocation retries; top-8 helper lowers peak memory to `242.22 GiB` and is
`+24.84%` faster over steps 2-20.  It does not yet establish Kimi robustness.

## Required Remaining Retained Gates

Kimi still needs a real TorchTitan model/config surface before a retained perf
gate is meaningful:

| Model | Required reduced shape | Retained hot-helper perf | Status |
| --- | --- | --- | --- |
| Qwen3.5-397B-A17B | `512` routed experts, top-10 routed + 1 shared, `moe_inter=1024`, `hidden=4096`, GQA `32Q/2KV`, `head_dim=256`, `60 -> 8` layers | MBS1/MBS2/MBS4/MBS8 retained rows complete | retained cross-model evidence |
| Kimi-K2.6 | `384` routed experts, top-8 + 1 shared, `moe_inter=2048`, MLA, `61 -> 6` layers, `hidden=4096` fallback if native `7168` OOMs | missing | blocked until a TorchTitan Kimi model/config surface exists |

The Kimi row is not optional. It is the remaining robustness gate before a
backend-native MORI or Primus-Turbo speedup can be called broad across all
three model families.

Current source availability check:

- TorchTitan WIP now has a Qwen3.5-style retained-gate config:
  `qwen3_5_397b_a17b_reduced_8layer_hothelper_gate`.
  It instantiates `397B-A17B-reduced-8layer` with `hidden=4096`,
  `32Q/2KV`, `head_dim=256`, `512` routed experts, top-k `10`,
  routed expert intermediate size `1024`, and one shared FFN expert with
  intermediate size `1024`.
- TorchTitan WIP has no Kimi training model directory or Kimi config in
  `torchtitan/models`, so the Kimi-K2.6 retained perf gate first needs a
  reduced TorchTitan model/config surface or a clearly scoped imported target.

## Qwen Retained-Perf Run Surface

The Qwen gate should use the same retained-feature comparison policy as the
DeepSeek ladder:

```text
module/config: qwen3 / qwen3_5_397b_a17b_reduced_8layer_hothelper_gate
data: c4_test
sequence length: 4096
global batch size: 128
parallelism: FSDP8 / EP8 / TP1 / PP1 / CP1
activation checkpointing: selective, with TORCHTITAN_SELECTIVE_AC_SCOPE=attention_only
loss: ChunkedCELoss CE8
optimizer: AdamW
runtime: 20 non-profiled steps
ladder: no-helper versus top-8 helper, with MBS=1/2/4/8 as fit allows
```

Completed 2026-06-12 retained gates:

```text
qwen397b_a17b_mbs1_helper0_20step_20260612T0626
qwen397b_a17b_mbs1_helper8_20step_20260612T0626
qwen397b_a17b_mbs2_helper0_20step_20260612T0636
qwen397b_a17b_mbs2_helper8_20step_20260612T0636
qwen397b_a17b_mbs4_helper0_20step_20260612T0654
qwen397b_a17b_mbs4_helper8_20step_20260612T0655
qwen397b_a17b_mbs8_helper0_20step_20260612T0659
qwen397b_a17b_mbs8_helper8_20step_20260612T0702
```

Raw logs remain outside the repo under:

```text
/scratch/sonle5/dsv4_pretrain_canary_20260527/runs/<run_id>/logs/train.log
```

The first step is excluded from retained windows because it includes cold start
and AITER JIT/import effects.  Qwen MBS8 does not show the same fit/no-fit
boundary as DeepSeek: both no-helper and helper complete, but no-helper reaches
`281.42 GiB` peak logged memory with CUDA allocation retries while helper
finishes at `242.22 GiB` and higher throughput.

## Kimi Retained-Perf Run Surface

Kimi-K2.6 should not be proxied through Qwen or DeepSeek.  The intended gate is:

```text
layers: 61 -> 6
hidden: 7168 native, falling back to 4096 only if native hidden OOMs
routed experts: 384
top-k: 8 routed + 1 shared expert
moe intermediate: 2048
attention: MLA
parallelism/runtime: same 8xMI350 GBS128 20-step ladder policy as above
```

The current TorchTitan WIP has no Kimi model implementation, so this remains a
model-surface task before it can become a retained throughput task.

## Added ABI Shape Gate

Added:

```text
tools/check_balanced_moe_model_shapes.py
```

The tool is intentionally not a benchmark.  It builds deterministic imbalanced
route tensors for:

Added follow-up source-coverage guard:

```text
tools/check_balanced_moe_model_surfaces.py
```

This guard is intentionally separate from the route-shape ABI check. It parses
the TorchTitan WIP source tree and verifies that each retained perf gate has a
real model module, config registry function, and model flavor. It passes for
DeepSeek-V4 Flash 12-layer and Qwen3.5-397B-A17B reduced 8-layer, and reports
`kimi_k2_6_reduced_6layer` as missing unless that missing surface is explicitly
allowed:

```text
python3 tools/check_balanced_moe_model_surfaces.py \
  --allow-missing kimi_k2_6_reduced_6layer
```

Read: Kimi remains a real model-surface task. Do not promote a Kimi throughput
proxy built from DeepSeek or Qwen configs; the perf gate needs an actual
TorchTitan Kimi module/config before the 20-step ladder is meaningful.

```text
DeepSeek-V4 Flash 12-layer: 256 experts, top-k 6
Qwen3.5-397B-A17B 8-layer: 512 experts, top-k 10
Kimi-K2.6 6-layer: 384 experts, top-k 8
```

Then it verifies that MORI and Primus-Turbo produce the same:

```text
route counts
hot expert plan
source hot/cold/helper partition
owner-compact exchange metadata
runtime layout
```

This prevents the two backend-native implementations from drifting before
backend-native perf runs are compared on each model family.

Runtime result in the local ROCm/PyTorch container:

```text
docker run --rm \
  -v /Users/sonle5/.codex/worktrees/0a57/one-mount-ai-platform/sdks/mlops/playground:/work \
  -w /work \
  rocm/sgl-dev:rocm720-mi35x-f96ac98-20260527-DSv4 \
  python3 experiments/2026-05-27-deepseek-v4-amd-pretrain/tools/check_balanced_moe_model_shapes.py --json
```

passed for all required model-family route shapes:

| Model | Experts | Top-k | Selected hot experts | Remote helper rows | Modeled max-load reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| DeepSeek-V4 Flash 12-layer | `256` | `6` | `8` | `6,929` | `59.86%` |
| Qwen3.5-397B-A17B 8-layer | `512` | `10` | `8` | `6,391` | `48.93%` |
| Kimi-K2.6 6-layer | `384` | `8` | `8` | `6,563` | `53.37%` |

This is ABI evidence only. It proves the MORI and Primus-Turbo route planner
and runtime layout agree for the required model-family shapes; it does not
replace the remaining Kimi throughput gate.

## Decision

For both MORI and Primus-Turbo:

1. Keep DeepSeek-V4 Flash 12-layer as the first throughput optimization surface.
2. Add Kimi retained standard-EP hot-helper 20-step ladders before claiming
   full cross-model perf; Qwen MBS1/MBS2/MBS4/MBS8 retained rows are now
   complete.
3. Use `check_balanced_moe_model_shapes.py` as the cheap ABI gate for all three
   model families before remote runs.

## Backend-Native Status

The current backend code has a native balanced-MoE layout ABI in both MORI and
Primus-Turbo:

```text
hot expert planner
source hot/cold/helper partition
owner-compact exchange metadata
owner-compact runtime layout
autograd-correct owner-compact row exchange
normal top-k dispatch tensor construction for backend raw-top-k EP paths
```

The current code does **not** yet have promoted native backend transport for
the hot-helper row movement:

| Backend | Native layout ABI | Hot-helper row transport | Promotion status |
| --- | --- | --- | --- |
| MORI | yes | default `torch_distributed_all_to_all_single`; opt-in `mori_sdma_padded_all2all` exists but is not promoted e2e | not promoted |
| Primus-Turbo | yes | owner-compact helper exchange still uses `torch_distributed_all_to_all_single`; normal/cold raw-topk EP wrapper now calls Primus-Turbo `moe_dispatch` / `moe_combine` | not promoted |

This distinction is now explicit in both backend modules through
`balanced_moe_backend_capabilities()`.  The model-shape ABI gate includes that
capability block in its JSON report. The backend parity gate verifies both
backends expose the same planner/layout capability set, while keeping
transport claims backend-specific: MORI advertises the opt-in padded SDMA
transport; Primus-Turbo advertises a raw-topk normal/cold EP wrapper but does
not yet advertise native hot-helper owner-compact transport. A future
Primus-Turbo promotion needs TorchTitan to route normal/cold rows through that
wrapper and win on the same 20-step retained gates. A future MORI promotion
needs the existing SDMA transport to win on the same gates, not only on a
microstage.

## Validation

Local syntax and diff hygiene:

```text
python3 -m py_compile \
  tools/check_balanced_moe_model_shapes.py \
  tools/check_balanced_moe_goal_scope.py \
  tools/check_balanced_moe_backend_parity.py \
  sources/wip/torchtitan/torchtitan/models/qwen3/__init__.py \
  sources/wip/torchtitan/torchtitan/models/qwen3/config_registry.py \
  sources/wip/torchtitan/torchtitan/models/qwen3/model.py \
  sources/references/mori/python/mori/ops/balanced_moe.py \
  sources/references/mori/tests/python/ops/test_balanced_moe.py \
  sources/wip/primus-turbo/primus_turbo/pytorch/ops/moe/balanced_moe.py \
  sources/wip/primus-turbo/tests/pytorch/ops/test_balanced_moe.py

python3 -m json.tool \
  run_artifacts/torchtitan_qwen397b_a17b_standard_ep_hothelper_mbs1_mbs8_20step_20260612.json

git diff --check -- \
  run_artifacts/torchtitan_qwen397b_a17b_standard_ep_hothelper_mbs1_mbs2_20step_20260612.json \
  run_artifacts/torchtitan_qwen397b_a17b_standard_ep_hothelper_mbs1_mbs8_20step_20260612.json \
  sources/wip/torchtitan/torchtitan/models/qwen3/__init__.py \
  sources/wip/torchtitan/torchtitan/models/qwen3/config_registry.py \
  sources/wip/torchtitan/torchtitan/models/qwen3/model.py \
  tools/check_balanced_moe_model_shapes.py \
  tools/check_balanced_moe_goal_scope.py \
  changesets/balanced_moe_backend_parity_harness_20260612.md \
  changesets/balanced_moe_cross_model_retained_perf_gates_20260612.md
```

Both passed.

Scope guard:

```text
python3 tools/check_balanced_moe_goal_scope.py --no-dirty --max-print 20
```

passed with no committed branch drift:

```text
committed main...HEAD: allowed=0 out_of_scope=0
combined: allowed=0 out_of_scope=0
```

The full dirty-tree scope guard is still intentionally failing:

```text
dirty working tree: allowed=27 out_of_scope=4133
combined: allowed=27 out_of_scope=4133
```

Runtime note: the desktop Python environment does not have `torch`, so
`check_balanced_moe_model_shapes.py` must be run inside a Torch/ROCm container
or remote MI350 environment before promotion.

Qwen model-spec smoke:

```text
docker run --rm \
  -v /Users/sonle5/.codex/worktrees/0a57/one-mount-ai-platform/sdks/mlops/playground:/work \
  -w /work/experiments/2026-05-27-deepseek-v4-amd-pretrain/sources/wip/torchtitan \
  rocm/sgl-dev:rocm720-mi35x-f96ac98-20260527-DSv4 \
  python3 -c '<stub tyro, import model_registry("397B-A17B-reduced-8layer")>'
```

passed with:

```text
8 4096 32 2 256 10 512 True
```

That confirms the reduced Qwen flavor has `8` layers, hidden `4096`,
`32` query heads, `2` KV heads, head dim `256`, routed top-k `10`,
`512` routed experts, and a non-null shared expert path.

Backend parity and cross-model ABI gates:

```text
docker run --rm \
  -v /Users/sonle5/.codex/worktrees/0a57/one-mount-ai-platform/sdks/mlops/playground:/work \
  -w /work \
  rocm/sgl-dev:rocm720-mi35x-f96ac98-20260527-DSv4 \
  python3 experiments/2026-05-27-deepseek-v4-amd-pretrain/tools/check_balanced_moe_backend_parity.py

docker run --rm \
  -v /Users/sonle5/.codex/worktrees/0a57/one-mount-ai-platform/sdks/mlops/playground:/work \
  -w /work \
  rocm/sgl-dev:rocm720-mi35x-f96ac98-20260527-DSv4 \
  python3 experiments/2026-05-27-deepseek-v4-amd-pretrain/tools/check_balanced_moe_model_shapes.py --json
```

Both passed.  The model-shape JSON now records that MORI and Primus-Turbo both
cover the planner/layout/autograd owner-compact exchange ABI and the normal
top-k dispatch tensor helper. Transport and normal/cold EP support are
intentionally different:

```text
MORI:
  owner_compact_exchange_transports=[
    torch_distributed_all_to_all_single,
    mori_sdma_padded_all2all
  ]
  native_hot_helper_transport=true
  native_hot_helper_transport_status=opt_in_padded_sdma_all2all
  normal_topk_ep_dispatch=false
  normal_topk_ep_dispatch_backend=not_implemented

Primus-Turbo:
  owner_compact_exchange_transports=[
    torch_distributed_all_to_all_single
  ]
  native_hot_helper_transport=false
  native_hot_helper_transport_status=not_implemented
  normal_topk_ep_dispatch=true
  normal_topk_ep_dispatch_backend=primus_turbo_moe_dispatch
```

So the retained perf target is still the standard-EP hot-helper feature until a
backend transport path promotes on the same 20-step ladders. MORI has a native
candidate transport but not a promoted e2e result; Primus-Turbo has the first
raw-topk normal/cold EP wrapper, but still needs TorchTitan to route the
normal/cold path through that wrapper and then promote on the retained 20-step
gates.

Focused backend unit tests:

```text
docker run --rm \
  -v /Users/sonle5/.codex/worktrees/0a57/one-mount-ai-platform/sdks/mlops/playground:/work \
  -w /work/experiments/2026-05-27-deepseek-v4-amd-pretrain/sources/references/mori \
  -e PYTHONPATH=. \
  rocm/sgl-dev:rocm720-mi35x-f96ac98-20260527-DSv4 \
  python3 -m pytest tests/python/ops/test_balanced_moe.py

docker run --rm \
  -v /Users/sonle5/.codex/worktrees/0a57/one-mount-ai-platform/sdks/mlops/playground:/work \
  -w /work/experiments/2026-05-27-deepseek-v4-amd-pretrain/sources/wip/primus-turbo \
  -e PYTHONPATH=. \
  rocm/sgl-dev:rocm720-mi35x-f96ac98-20260527-DSv4 \
  python3 -m pytest tests/pytorch/ops/test_balanced_moe.py
```

Results:

```text
MORI: 21 passed, 2 skipped
Primus-Turbo: 12 passed
```

The same tests should be run from their backend roots with `PYTHONPATH=.`.
A combined invocation from the experiment root collected the MORI package with
the wrong import root and failed before running test bodies with
`ModuleNotFoundError: tests.python`.

Remote retained Qwen gates completed on `do-vunguyen-mi350-gpu`:

```text
MBS=1 STEPS=20 HELPER_HOT=0 RUN_LABEL=qwen397b_a17b_mbs1_helper0_20step_20260612T0626
MBS=1 STEPS=20 HELPER_HOT=8 RUN_LABEL=qwen397b_a17b_mbs1_helper8_20step_20260612T0626
MBS=2 STEPS=20 HELPER_HOT=0 RUN_LABEL=qwen397b_a17b_mbs2_helper0_20step_20260612T0636
MBS=2 STEPS=20 HELPER_HOT=8 RUN_LABEL=qwen397b_a17b_mbs2_helper8_20step_20260612T0636
```

All four completed 20 steps with finite loss/grad metrics.

Dependency pin changes: none.
