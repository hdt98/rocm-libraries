# Native Primus-Turbo Hot-Helper Scope

Date: 2026-06-13

## Intent

Implement native hot-helper balanced-MoE as a Primus-Turbo ABI, while keeping
TorchTitan limited to policy/layer integration and MORI limited to SDMA
transport primitives.

## Branch Cleanup Classification

Kept in scope:

- `tools/`: minimal harness only.
  - `run_balanced_moe_table_sweep.py`
  - backend/model parity checks
  - small MORI owner-compact smoke checks
- TorchTitan WIP: policy, model/layer integration, backend selector wiring,
  model surfaces, and training harness support.
- Primus-Turbo WIP: balanced-MoE planner/runtime ABI, normal top-k TurboEP
  handoff, owner-compact exchange contract, and native MORI SDMA transport
  delegation.
- MORI reference source: SDMA owner-compact transport primitive provider.
- Experiment docs and compact run-artifact summaries.

Removed from the active `tools/` surface:

- old sparse-attention microbench/probe directories;
- exploratory local probe folders not used by the table harness;
- generated Python bytecode.

Archived local-only probe material:

- `/tmp/dsv4_pretrain_tools_archive_20260613/`

## Native ABI Status

Implemented:

- Primus-Turbo capability advertises `native_hot_helper_transport=true`.
- `exchange_owner_compact_needed_rows(..., transport="mori_sdma")` normalizes
  to `mori_sdma_padded_all2all`.
- Primus-Turbo delegates owner-compact needed hot-weight exchange to
  `mori.ops.balanced_moe.exchange_owner_compact_needed_rows`.
- TorchTitan selects the Primus-Turbo balanced-MoE ABI and passes the saved
  owner-compact hot/cold/helper plan.

Not claimed native yet:

- compact-row dispatch/combine. These remain reference/not-promoted paths and
  their capability flags stay false until a backend-owned compact-row
  dispatch/combine path is implemented and measured.

## Active Measurement Gate

Remote table run:

- session: `bm_table_native_fresh1`
- node: `do-vunguyen-mi350-gpu`
- raw root: `/scratch/sonle5/dsv4_pretrain_canary_20260527/runs/balanced_moe_backend_table_native_20260613_fresh1`
- compact summary: `/scratch/sonle5/dsv4_pretrain_canary_20260527/sweep_snapshots/balanced_moe_backend_table_native_20260613_fresh1/playground/experiments/2026-05-27-deepseek-v4-amd-pretrain/run_artifacts/balanced_moe_backend_table_native_20260613_fresh1.json`
- local partial copy:
  `run_artifacts/balanced_moe_backend_table_native_20260613_fresh1.partial.json`

Rows:

- baseline model surfaces: DeepSeek-V4 12-layer MTP1,
  DeepSeek-V4 12-layer no-MTP historical control, Qwen3.5 8-layer,
  Kimi-K2.6 6-layer
- MBS: 1, 2, 4, 8
- baseline variants: standard EP no-helper, Primus-Turbo TurboEP no-helper,
  standard EP top-8 helper with MORI SDMA
- native candidate variant: Primus-Turbo native helper with MORI SDMA
  owner-compact exchange, run as a follow-up candidate table after the
  corrected baseline table is complete
- DeepSeek topology is now explicit MTP1 for the corrected table:
  `CANARY_DSV4_NUM_MTP_MODULES=1`,
  `CANARY_DSV4_MTP_LOSS_WEIGHT=0.3`. The historical 20260608 `~8.5k`
  helper-MBS8 row is no-MTP/stale for this comparison: its raw log reports
  `81,244,314,368` parameters, while current-source MTP1 fresh4 reports
  `87,079,924,032` parameters and `dsv4_num_mtp_modules=default`.
  Use a separate `CANARY_DSV4_NUM_MTP_MODULES=0` control only to reproduce the
  old artifact, not as the corrected baseline table.

Promotion rule:

- Native Primus-Turbo helper must beat the strongest baseline by at least 3%
  on DeepSeek MBS4/MBS8.
- Qwen and Kimi must not regress by more than 2% versus their strongest
  baseline unless the run changes the memory fit boundary.

Status:

- `bm_table_native_fresh1` exited with code `0`, but stopped early after
  DeepSeek/helper8/MBS2 because postflight clean-node validation returned
  `postflight_idle=false`.
- The artifact is partial: `7/48` planned rows completed. The top-level JSON
  status says `complete`, so consumers must check `len(results)` against
  `len(planned)` before treating a table as complete.
- Completed rows cover DeepSeek MBS1 for all four variants and DeepSeek MBS2
  for plain/Primus/helper8. Native Primus helper8 MBS2, all MBS4/MBS8 rows,
  Qwen, and Kimi were not reached.
- No promotion decision can be made from fresh1. Relaunch the full matrix after
  the node is clean, or adjust the postflight guard to ignore unrelated
  non-table lighthouse containers only when they have no GPU allocation and do
  not share the table run root.

Harness fix:

- `tools/run_balanced_moe_table_sweep.py` now records
  `completion.planned_rows`, `completion.result_rows`, `completion.complete`,
  failed-row counts, and blocked-row counts.
- Final table status is now `incomplete` when the loop stops before all planned
  rows or hits a clean-node block, and the process returns nonzero for that
  incomplete case.
- The clean-node guard can ignore a configured non-GPU lighthouse container
  name pattern only when ROCm reports zero GPU utilization and zero VRAM
  allocation. GPU-active torchft/train containers still block the table.
- Validation: Python compile passed, and dry-run planning for DeepSeek MBS1/2
  across plain/Primus/helper8/native-helper generated the expected `8` rows.
- After the MTP audit, the harness exposes `deepseek_mtp1` and
  `deepseek_nomtp` as separate model surfaces and sets the default baseline
  matrix to `deepseek_mtp1,deepseek_nomtp,qwen,kimi` x `1,2,4,8` x
  `plain,primus,helper8`.

Current block:

- `do-vunguyen-mi350-gpu` and `do-sonle5-mi350-gpu` are both occupied by the
  `torchft_32b_fix_20260613_052200` two-node job family, with active GPU use.
  The full 48-row rerun should wait for one node to be clean.

## Current Validation

Passed before full table:

- Python compile for Primus-Turbo balanced-MoE module, tests, and table harness.
- Docker/runtime backend parity, model shape, and model surface checks.
- First table row, DeepSeek/plain/MBS1, completed 20 steps and reproduced the
  expected current-source baseline band.

Local note:

- The Codex app spawned Git helper storms during broad dirty-tree inspection.
  The sweep now runs inside remote `tmux`, and local follow-up should avoid
  repeated broad `git status`/`git diff` scans.
