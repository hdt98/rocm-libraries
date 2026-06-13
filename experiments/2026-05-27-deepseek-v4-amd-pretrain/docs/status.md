# Status

## 2026-05-27

- Created the DeepSeek V4 AMD pretraining experiment scaffold.
- Vendored WIP source snapshots for Primus, TorchTitan, and NVIDIA Megatron-LM under `sources/wip/`.
- Vendored NVIDIA-side Day-0 DSv4 RL reference snapshots for Miles PR `1045` and Megatron-LM PR `28` under `sources/references/`.
- Recorded the LMSYS Day-0 RL section and radixark Miles roadmap issue as source references.
- No AMD pretraining run has been launched from this experiment directory.

## Open Items

- Select the first AMD canary shape.
- Decide the initial backend path: Primus with Megatron-LM, Primus with TorchTitan, or a minimal Megatron-LM source probe.
- Add a launcher only after the rerun command is concrete.
- Add patch files under `changesets/` once source deltas are known.
