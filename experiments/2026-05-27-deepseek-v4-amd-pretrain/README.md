# DeepSeek V4 AMD Pretrain

Status: migrated into `/Users/sonle5/rocm-libraries` as a source-review and
training-canary experiment. Source snapshots now live at the repo root under
`projects/` and `references/`; this experiment directory keeps only docs,
minimal harness code, and compact artifact summaries.

## Goal

- Build an AMD MI355X pretraining path for DeepSeek V4, starting from the local Primus training stack and comparing against the NVIDIA Day-0 Miles/Megatron-LM RL training implementation.
- Keep the editable WIP source surface focused on root-level `projects/primus`,
  `projects/primus-turbo`, `projects/torchtitan`, and `projects/megatron-LM`.
- Preserve comparison references under root-level `references/{amd,ascend,nvidia}`
  so AMD pretraining changes can be traced back to Day-0, Ascend, and AMD kernel
  assumptions.

## Layout

- `changesets/`: local patches or patch notes required to reproduce the experiment.
- `docs/`: active notes, roadmap, and status checkpoints.
- `launchers/`: canonical rerun scripts once a first AMD canary shape is selected.
- `run_artifacts/`: compact pins and artifact pointers; raw logs, checkpoints, and traces stay outside the repo.
- `tools/`: host-side helper scripts if source comparison or launch preparation needs them.

Source trees are vendored snapshots, not nested Git repos or submodules. See
[`docs/source_migration_20260613.md`](docs/source_migration_20260613.md) for the
root-level source layout and migration provenance.

## Source Pins

WIP snapshots:

| Repo | Commit | Role |
| --- | --- | --- |
| `AMD-AGI/Primus` | `05ec3231288f3932cdef789a06149d26825736ff` | AMD training framework and orchestration surface |
| `pytorch/torchtitan` | `7fcd9beacde733f3ceca4a4da7e2f0518d64f68e` | pretraining backend candidate and reference implementation |
| `NVIDIA/Megatron-LM` | `873678adafa71702ef7e7a405a7f4a3377dd2727` | Megatron backend surface used by Primus and for DSv4 model porting |

Reference snapshots:

| Source | Commit or URL | Role |
| --- | --- | --- |
| LMSYS Day-0 RL section | `https://www.lmsys.org/blog/2026-04-25-deepseek-v4/#reinforcement-learning-miles-support` | DSv4 training feature and result summary |
| `radixark/miles#1046` | `https://github.com/radixark/miles/issues/1046` | public DeepSeek V4 RL roadmap and launch shape |
| `yueming-yuan/miles:deepseek-v4` | `032721cd61bf7164955f084425eb9f315352fd26` | Miles PR `radixark/miles#1045` source reference |
| `yueming-yuan/Megatron-LM:deepseek-v4` | `1c6e5b7bcde0097ebe193b6258115ff3558f69d6` | Megatron-LM PR `radixark/Megatron-LM#28` source reference |

## Current Notes

- The NVIDIA Day-0 RL stack reports DSv4 support across DP/TP/SP/EP/PP/CP, compressed-attention TP/SP/CP handling, mHC p2p shape changes, TileLang kernels for indexer/sparse MLA/mHC, FP8 rollout, BF16/FP8 training, FP8 attention QAT, R3, and experimental indexer replay.
- The issue roadmap records a H200/B200 Docker image, a 4-layer Flash smoke command, and an 8-node H200 Flash 284B full-train command. Treat these as NVIDIA reference shapes, not AMD rerun commands.
- Existing AMD Primus pretraining baselines in this repo are Qwen-family topology and runtime proofs. This experiment has not yet selected a DSv4 AMD shape, dataset, checkpoint format, or first-node canary.

## Next Checkpoint

1. Diff the DSv4 model and training changes in `sources/references/megatron-lm-deepseek-v4-pr28` against `sources/wip/megatron-LM`.
2. Map required Primus backend changes in `sources/wip/primus` for model config, checkpoint conversion, distributed optimizer precision, and CPU-load/offload behavior.
3. Decide whether the first AMD canary should target a small synthetic DSv4 layer stack or a Flash 284B topology with mock data.
4. Add only the minimum launch script once the command shape is concrete.
