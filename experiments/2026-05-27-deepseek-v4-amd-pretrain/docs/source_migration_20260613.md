# Source Migration 2026-06-13

## Purpose

This migration makes `/Users/sonle5/rocm-libraries` the canonical local repo for
DeepSeek-V4 AMD pretraining source snapshots and experiments. The large vendor
snapshot baseline belongs on `origin/main`; active balanced-MoE work belongs on
`users/sonle5/balanced_moe` so Codex app diffs do not walk the vendor tree.

## Baseline

- Target repo: `/Users/sonle5/rocm-libraries`
- Baseline branch: `users/sonle5/dev`
- Base commit: `origin/main` at `c7fe831061a4`
- Source platform repo snapshot: local `main` at `c3df890a7657`
- Vendor format: plain tracked snapshots, with nested `.git` and transient
  caches stripped.
- Baseline migration commits:
  - `ad4237094b`: vendor/source baseline layout.
  - `6b1d5b2fb3`: compact top-level run-artifact JSON baseline.
  - `fbe8ed7b31`: forward correction that removes active Primus-Turbo
    hot-helper work from the baseline snapshot.
- Updated `origin/main`: fast-forwarded to `fbe8ed7b31` without force-push.

## Active Work Branch

- Branch: `users/sonle5/balanced_moe`
- Base: corrected `origin/main` at `fbe8ed7b31`
- Purpose: carry only active TorchTitan policy/layer/autograd integration,
  Primus-Turbo native hot-helper ABI work, MORI SDMA transport integration when
  present, and concise experiment docs/harness changes.

## Layout

- `references/ascend/`: Ascend CANN, MindSpeed, CANN ops, and TorchTitan-NPU
  references.
- `references/nvidia/`: Radix/Miles and Megatron-LM DeepSeek-V4 references.
- `references/amd/`: AMD/HIP references such as MORI and HipKittens.
- `projects/`: active WIP vendor repos used by the canary work:
  `primus`, `primus-turbo`, `torchtitan`, and `megatron-LM`.
- `experiments/2026-05-27-deepseek-v4-amd-pretrain/`: docs, recipe, launchers,
  changesets, minimal tools, and compact top-level JSON artifacts only.

## Available Vendor SHAs

Some migrated snapshots had nested Git metadata before migration. Those SHAs are
recorded here before stripping `.git`:

- `references/ascend/ascend-mindspeed`: `bb583c908bc3`
- `references/ascend/ascend-mindspeed-llm-upstream`: `df924c584961`
- `references/ascend/ascend-mindspeed-upstream-gitee-lite`: `bb583c908bc3`
- `references/ascend/ascend-mindspeed-upstream-github`: `829b232a5470`
- `references/ascend/cann-ops-adv`: `e0f6933182bc`
- `references/amd/hipkittens`: `cd090ae98ee4`
- `projects/primus-turbo`: `abacf7403da3`

The clean `main` source tree provided the tracked baseline snapshots for
`projects/primus`, `projects/torchtitan`, `projects/megatron-LM`,
`references/nvidia/megatron-lm-deepseek-v4-pr28`, and
`references/nvidia/miles-deepseek-v4-pr1045`.
