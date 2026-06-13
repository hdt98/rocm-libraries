# Local Diff Hygiene 2026-06-13

## Scope

The branch history itself is not the noisy part: `git diff main...HEAD` was
empty at the time of this cleanup. The large diff is dirty working-tree state,
mostly under this experiment plus old generated/vendor trees.

## Findings

- Local Codex/Git process storms came from the Codex app parent spawning many
  Git helper children while inspecting dirty vendored trees.
- The first sampled storm was `git hash-object --no-filters` against the old
  `experiments/2026-05-06-deepseek-v4-amd-port-pr23882-node54/sources/references`
  tree, including SGLang/FlashInfer reference files.
- After restoring that old experiment tree to HEAD, the storm cleared
  immediately, confirming that stale dirty state outside the new experiment was
  a direct trigger.
- Scoped status over the active experiment can still trigger helper scans while
  `sources/wip/torchtitan` has large dirty source changes. Treat broad Git UI
  scans and repeated local launcher loops as unsafe until the dirty source
  surface is reduced or committed.

## Cleanup Done

- Fast-forwarded `sonle5_dsv4_amd_pretrain` to local `main` and set the branch
  upstream to local `main`. This is local Git config/history alignment only; it
  does not update or touch `origin/main`.
- Restored the old `2026-05-06` experiment tree to HEAD because changes outside
  `experiments/2026-05-27-deepseek-v4-amd-pretrain/` should not be part of this
  branch.
- Removed generated `tools/__pycache__`.
- Collapsed `recipe.yaml` from a run ledger into a concise recipe contract.
- Added the process-hygiene root cause to `docs/status.md`.

## Current Policy

- Compare this work against local `main`, not `origin/main`. `origin/main`
  remains a remote-tracking reference and is intentionally not updated by this
  cleanup.
- Keep raw logs, traces, checkpoints, Docker caches, and remote run outputs under
  `/scratch/sonle5/dsv4_pretrain_canary_20260527`, not in the repository.
- Keep `tools/` limited to the balanced-MoE table harness and small static/smoke
  checks.
- Use path-scoped Git commands only:

```bash
GIT_OPTIONAL_LOCKS=0 git -c core.fsmonitor=false status --short -- \
  experiments/2026-05-27-deepseek-v4-amd-pretrain
```

- If helpers respawn from the Codex app parent, kill only exact read-only helper
  children such as `git hash-object --no-filters` or
  `git config --null --get core.fsmonitor`. If the parent keeps respawning them,
  restart the Codex app parent instead of killing broader processes.

## Next Cleanup Target

Classify the remaining active `sources/wip/torchtitan` dirty source into:

- TorchTitan policy/layer integration to keep,
- Primus-Turbo/MORI backend ABI hooks to move out of TorchTitan when possible,
- generated or obsolete diagnostics to restore.

Do not remove those source changes blindly; they are the active native
Primus-Turbo/MORI balanced-MoE implementation surface.
