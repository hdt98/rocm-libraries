# Roadmap

## Source Review

- Compare `sources/references/megatron-lm-deepseek-v4-pr28` with `sources/wip/megatron-LM` for DSv4 model architecture, mHC, compressed attention, CP/SP/TP handling, checkpoint save/load, and optimizer precision changes.
- Inspect `sources/references/miles-deepseek-v4-pr1045` for launch topology, R3/indexer replay assumptions, rollout/training precision contracts, and checkpoint conversion flow.
- Map those changes onto `sources/wip/primus` before choosing whether the AMD first pass should use the Megatron backend directly or route through a Primus abstraction.
- Use `sources/wip/torchtitan` as the alternate pretraining backend reference if Megatron-LM integration is slower than a smaller DSv4 canary.

## AMD Canary Candidates

- Synthetic DSv4 small-layer model with mock data to validate model construction, forward/backward, optimizer, and checkpoint save.
- Flash 284B mock-data topology if model config and checkpoint conversion are ready enough to test distributed layout early.
- Kernel-only probes for compressed attention, sparse MLA, mHC, and indexer paths if a training canary blocks on CUDA-only kernels.

## Promotion Bar

- Document exact command shape, node layout, environment knobs, and dependency SHAs.
- Retain core training metrics in this README or a future `metrics.md`: completed iterations, skipped/NaN counts, loss movement, throughput, and checkpoint markers.
- Keep launchers short once the rerun path is stable. Long source-analysis notes and troubleshooting stay in `docs/` or `changesets/`.
