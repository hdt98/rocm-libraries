# Changesets

Keep only concise patch notes needed for the current balanced-MoE migration and
backend comparison work. Raw logs, traces, and old sparse-MLA/kernel/optimizer
probe journals stay outside this repository.

Current retained notes:

- `local_diff_hygiene_20260613.md`: local branch cleanup and Codex/Git helper
  storm root cause.
- `active_diff_classification_20260613.md`: local-main-based active diff
  classification before migrating into `rocm-libraries`.
- `native_primus_turbo_hot_helper_scope_20260613.md`: ownership split for
  native Primus-Turbo hot-helper with MORI SDMA transport.
- `balanced_moe_cross_model_retained_perf_gates_20260612.md`: retained
  DeepSeek/Qwen/Kimi hot-helper evidence and promotion gates.
- `primus_turbo_raw_topk_backend_comparison_20260612.md`: corrected framing for
  Primus-Turbo `TurboEPBackend` raw top-k no-helper comparisons.
- `torchtitan_primus_turbo_deepep_viability_20260612.md`: viability audit for
  Primus-Turbo TurboEP/external DeepEP integration.

Each future note should name the source snapshot, base commit, files changed,
why the change is needed for AMD DeepSeek-V4 pretraining, and the validation
command or source comparison that supports it.
