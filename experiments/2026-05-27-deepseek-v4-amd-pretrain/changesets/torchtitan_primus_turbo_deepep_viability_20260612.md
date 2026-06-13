# Primus-Turbo / PyTorch DeepEP Viability Check

Date: 2026-06-12

## Question

Can we use `primus_turbo.pytorch.deep_ep` or a PyTorch/DeepEP backend natively,
without introducing adapter overhead, before continuing the native MORI
balanced-MoE work?

## Findings

### 1. Primus-Turbo DeepEP is present in the training images

With ROCm devices mounted, these images can import
`primus_turbo.pytorch.deep_ep`:

```text
onenexus/nexus-titan:rocm722-pytorch-nightly
onenexus/nexus-titan:rocm722-pytorch-nightly-mori
rocm/primus:v26.2
```

The import check saw all 8 MI350 devices and loaded:

```text
primus_turbo.pytorch.deep_ep
```

Fresh containers paid an AITER JIT import/build cost:

```text
onenexus/nexus-titan:rocm722-pytorch-nightly       module_aiter_core ~86.5s
onenexus/nexus-titan:rocm722-pytorch-nightly-mori  module_aiter_core ~148.6s
rocm/primus:v26.2                                  module_aiter_enum ~13.6s
```

That cost is cacheable, but it means any quick throughput gate must mount a
persistent JIT cache or warm the image first.

### 2. TorchTitan's current `deepep` backend does not use Primus-Turbo

The current WIP TorchTitan backend selector describes:

```text
deepep   -> DeepEP custom kernels for H100/NVLink Switch
hybridep -> HybridEP with TMA optimization for GB200/NVLink72
mori     -> AMD MORI EP boundary
standard -> PyTorch all-to-all collectives
```

Source paths:

```text
sources/wip/torchtitan/torchtitan/models/common/config_utils.py
sources/wip/torchtitan/torchtitan/models/common/token_dispatcher.py
sources/wip/torchtitan/torchtitan/distributed/deepep/deepep.py
```

`DeepEPTokenDispatcher` imports:

```python
from torchtitan.distributed.deepep import deepep
```

and that wrapper imports:

```python
from deep_ep import Buffer, EventHandle, EventOverlap
```

The tested images do not contain standalone `deep_ep` or `deepep`:

```text
deep_ep_spec None
deepep_spec None
```

So setting `CANARY_MOE_COMM_BACKEND=deepep` today would not exercise the
installed Primus-Turbo DeepEP package.

### 3. Primus-Turbo DeepEP's ABI is still raw DeepEP dispatch/combine

The packaged API is:

```python
from primus_turbo.pytorch.deep_ep import Buffer, EventOverlap
```

and its `Buffer` methods consume raw route/layout inputs:

```text
get_dispatch_layout(topk_idx, num_experts, ...)
dispatch(x, topk_idx, topk_weights, num_tokens_per_rank,
         is_token_in_rank, num_tokens_per_expert, ...)
combine(x, handle, topk_weights, ...)
```

It also has a `num_worst_tokens` intranode path to avoid CPU sync, and async
event plumbing for overlap. That is useful, but it is not the same ABI as the
winning standard-EP hot-helper layout:

```text
saved owner-compact hot/cold/helper partition
-> cold/normal compact rows
-> helper-hot owner-compact rows
-> compact dX/dTopK return
-> hot dW reduce by owner slot
```

In other words, Primus-Turbo DeepEP can plausibly accelerate a raw/no-helper
EP dispatch-combine path after an adapter, but it does not natively consume the
already-partitioned MindSpeed-style hot/cold/helper layout that made MBS8 fit.

### 4. There is a higher-level Primus-Turbo PyTorch MoE API

The package also exposes a more useful PyTorch-side wrapper than the raw
`Buffer` ABI:

```text
primus_turbo.pytorch.ops.moe.moe_dispatch(...)
primus_turbo.pytorch.ops.moe.moe_combine(...)
primus_turbo.pytorch.modules.moe.DeepEPTokenDispatcher
```

The relevant installed files are:

```text
primus_turbo/pytorch/ops/moe/moe_dispatch_combine.py
primus_turbo/pytorch/kernels/moe/moe_dispatch_combine_impl.py
primus_turbo/pytorch/modules/moe/token_dispatcher.py
```

That wrapper has an internal `TurboEPBackend` using
`primus_turbo.pytorch.deep_ep`, and an optional `DeepEPBackend` using an
external `deep_ep` package. Since the tested images have the in-tree
Primus-Turbo backend but not standalone `deep_ep`, a TorchTitan adapter should
target the `TurboEPBackend`/`turbo.ops.moe_dispatch` path, not the current
TorchTitan `deepep` backend.

This makes the Primus-Turbo backend goal concrete:

```text
standard/no-helper EP route tensors
-> primus_turbo.pytorch.ops.moe_dispatch
-> Primus-Turbo moe_permute into expert-major rows
-> local grouped expert compute
-> Primus-Turbo moe_unpermute + moe_combine
```

The first gate should establish the plain/raw-top-k Primus-Turbo baseline. The
second, equivalent-to-MORI goal is to extend the ABI so Primus-Turbo consumes
the saved owner-compact hot/cold/helper partition directly, instead of wrapping
that layout around raw top-k dispatch with extra copies and masks.

### 5. Deeper source scan: no installed balanced-MoE hot-helper path

The installed Primus-Turbo package on
`onenexus/nexus-titan:rocm722-pytorch-nightly` has:

```text
primus_turbo.pytorch.kernels.moe.moe_dispatch_combine_impl.TurboEPBackend
primus_turbo.pytorch.ops.moe.moe_dispatch
primus_turbo.pytorch.ops.moe.moe_combine
primus_turbo.pytorch.ops.moe.moe_permute
primus_turbo.pytorch.ops.moe.moe_unpermute
primus_turbo.pytorch.modules.moe.DeepEPTokenDispatcher
```

The actual dispatch path is:

```text
get_dispatch_layout(topk_idx, num_experts)
-> dispatch(x, topk_idx, token_weights, layout)
-> moe_permute(dispatched_rows, dispatched_indices)
-> expert compute
-> moe_unpermute(expert_output)
-> combine(expert_output, handle)
```

A scan for `balanced`, `hot`, `owner_compact`, and `helper` under
`primus_turbo/pytorch` did not find a MindSpeed-style hot expert helper
implementation. The only "balanced" hits were grouped-GEMM profiling helpers,
not runtime expert relocation.

So Primus-Turbo can be used as a native raw top-k EP candidate, but it is not a
drop-in replacement for the proven hot-helper feature unless we add a new
balanced-MoE ABI to Primus-Turbo too.

## Decision

Do not treat `CANARY_MOE_COMM_BACKEND=deepep` as a drop-in next gate on the
current TorchTitan WIP. It points to the wrong package (`deep_ep`) and the
wrong mechanism for the current best surface.

The first reasonable experiment branch is:

```text
TorchTitan Primus-Turbo EP adapter
-> import primus_turbo.pytorch.ops.moe
-> use moe_dispatch/moe_combine through TurboEPBackend
-> warm/cache AITER/Primus JIT
-> compare against no-helper standard EP and plain MORI EP first
```

This is worth a raw-EP baseline, especially to see whether raw/no-helper EP can
improve the plain no-helper surface. But it is not the whole Primus-Turbo goal:
it will not preserve the MBS8 MindSpeed-style fit advantage unless
Primus-Turbo also learns a compact-row hot/cold/helper ABI.

The two backend goals stay equivalent after that:

```text
MORI or Primus-Turbo owns the saved hot/cold/helper partition
-> native compact cold/normal dispatch/combine
-> helper owner-compact grouped MLP/VJP
-> compact dX/dTopK return
-> overlapped hot dW reduce
```

## Follow-up After MBS1/MBS2 Hot-Helper Ladder

The new 20-step ladder makes the backend contract stronger. On the same
8xMI350 `flash_12layer_ceiling_probe`, `GBS=128`, `S4096`, attention-only AC,
and CE8 surface, top-8 hot-helper improves late-window throughput at every
comparable MBS and reduces memory:

```text
MBS1: 3822.38 -> 4202.72 tok/GPU/s, +9.95%,  -10.35 GiB
MBS2: 4321.64 -> 5254.38 tok/GPU/s, +21.58%, -24.98 GiB
MBS4: 5058.13 -> 6550.00 tok/GPU/s, +29.49%, -48.87 GiB
MBS8: no-helper fails before first metric; hot-helper completes 20 steps
```

Artifact:

```text
run_artifacts/torchtitan_dsv4_flash12_standard_ep_hothelper_mbs1_mbs2_ce8_ladder_20step_20260612.json
changesets/torchtitan_standard_ep_hothelper_mbs1_mbs2_ce8_ladder_20260612.md
```

Read: both MORI and Primus-Turbo should be measured against this same feature
contract. The fair comparisons are plain backend EP versus the same backend
with native balanced-MoE, using robust 20-step throughput gates and short
profile gates on the matching shape.

That means the feature we must preserve is not "a faster all-to-all"; it is the
MindSpeed-style layout change:

```text
raw top-k routes
-> remove remote-hot rows from the normal EP path
-> execute remote-hot rows on helper/source ranks in owner-compact order
-> return dX/dTopK to token owners
-> reduce hot dW back to true expert owners
```

The current Primus-Turbo/TurboEP path still does not consume that layout
natively. The code-level entry points confirm this:

```text
primus_turbo/pytorch/deep_ep/buffer.py:
  get_dispatch_layout(topk_idx, num_experts)
  dispatch(x, topk_idx, topk_weights, num_tokens_per_rank, ...)
  combine(x, handle, topk_weights, ...)

primus_turbo/pytorch/kernels/moe/moe_dispatch_combine_impl.py:
  TurboEPBackend dispatch first calls get_dispatch_layout(topk_idx, ...)

primus_turbo/pytorch/modules/moe/token_dispatcher.py:
  DeepEPTokenDispatcher warns that DeepEP only accepts [num_tokens, router_topk]
  token_indices/topk form, then runs moe_dispatch -> moe_permute ->
  moe_unpermute -> moe_combine.
```

The local Primus-Turbo `balanced_moe.py` file now contains the shared
owner-compact planning/runtime-layout utilities, but that is the ABI we added
for future native integration. It is not yet wired into the TurboEP
dispatch/combine primitive as a compact-row balanced-MoE execution mode.

Updated recommendation:

1. Keep MORI and Primus-Turbo as equivalent backend goals. Each backend needs
   a plain/no-helper EP baseline and a native balanced-MoE candidate that
   consumes the proven hot/cold/helper owner-compact partition.
2. Add a TorchTitan `comm_backend=primus_turbo` raw top-k adapter as the plain
   Primus-Turbo baseline comparator against plain standard EP and plain MORI
   EP.
3. Do not wrap the winning hot-helper layout around raw TurboEP. That would add
   the same masks, copies, selected-order adapters, and post-hoc row transforms
   that the native backend work is trying to remove.
4. The Primus-Turbo balanced-MoE candidate requires a new TurboEP/Primus native
   compact-row balanced-MoE ABI, not just calling the existing raw
   `moe_dispatch` / `moe_combine` wrapper.

## External References

- AMD-AGI Primus-Turbo README: `https://github.com/AMD-AGI/Primus-Turbo`
- Primus-Turbo DeepEP example: `https://github.com/AMD-AGI/Primus-Turbo/blob/main/docs/examples.md#4-deepep`
- PyTorch blog on AMD TorchTitan MoE scaling with Primus-Turbo/DeepEP:
  `https://pytorch.org/blog/efficient-moe-pre-training-at-scale-with-torchtitan/`

## Validation

```text
rg -n "deepep|DeepEP|hybridep|mori|comm_backend" \
  sources/wip/torchtitan/torchtitan/models/common/config_utils.py \
  sources/wip/torchtitan/torchtitan/models/common/token_dispatcher.py \
  sources/wip/torchtitan/torchtitan/distributed

docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
  --ipc=host --network=host --entrypoint python \
  onenexus/nexus-titan:rocm722-pytorch-nightly \
  -c 'import primus_turbo.pytorch.deep_ep as de; print(de.__file__)'

docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
  --ipc=host --network=host --entrypoint python \
  onenexus/nexus-titan:rocm722-pytorch-nightly \
  -c 'import importlib.util as u; print(u.find_spec("deep_ep"))'
```

Dependency pin changes: none.
