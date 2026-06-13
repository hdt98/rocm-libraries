# Tools

This directory is intentionally minimal.  It only contains the balanced-MoE
baseline/candidate harness and small ABI/smoke checks needed to reproduce the
current Primus-Turbo/MORI hot-helper work.  Raw logs, traces, caches, and old
kernel probes stay outside the repo.

- `run_balanced_moe_table_sweep.py`: compact comparison-table harness for
  DeepSeek-V4 12-layer, Qwen3.5 8-layer, and Kimi-K2.6 6-layer balanced-MoE
  runs across MBS values and backend variants.
- `check_balanced_moe_backend_parity.py`: CPU-only guard for MORI and
  Primus-Turbo balanced-MoE ABI parity, native transport capability
  advertisement, source partitions, and owner-compact exchange metadata.
- `check_balanced_moe_model_shapes.py`: CPU-only route-shape guard for the
  DeepSeek, Qwen, and Kimi reduced model surfaces.
- `check_balanced_moe_model_surfaces.py`: source-inspection guard that verifies
  retained model/config surfaces exist before backend-native performance claims
  are promoted.
- `check_balanced_moe_goal_scope.py`: guard that the experiment docs and
  harness still describe the requested backend comparison scope.
- `mori_hot_helper_pack_smoke.py` and
  `smoke_mori_owner_compact_exchange_gloo.py`: small local transport/layout
  smoke checks for owner-compact hot-helper exchange semantics.

Archived local-only probe trees from earlier sparse-MLA, optimizer, and kernel
experiments were moved out of the repo to
`/tmp/dsv4_pretrain_tools_archive_20260613/`.
