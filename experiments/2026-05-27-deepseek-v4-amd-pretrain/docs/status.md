# Status

## 2026-06-13

- Local Git/Codex hygiene finding: broad diff/status scans over this dirty
  experiment can spawn thousands of `git hash-object --no-filters` helpers.
  The concrete offender observed locally was a Codex/Git parent scanning the
  old large reference tree under
  `experiments/2026-05-06-deepseek-v4-amd-port-pr23882-node54/sources/references/sglang-amd-deepseek-v4`.
  The huge dirty diff contributes by forcing Git to hash many modified files,
  but the main trigger is unscoped Git/Codex inspection across vendored and
  generated trees. Cleanup killed only those local read-only helper children;
  current local state is `0` zombies and no active helper storm. Use scoped
  commands such as `GIT_OPTIONAL_LOCKS=0 git -c core.fsmonitor=false status
  --short -- experiments/2026-05-27-deepseek-v4-amd-pretrain` and keep large
  run logs/traces outside the worktree.
- Branch-base correction: this branch was also still being compared by some
  tools against `origin/main`, which is the wrong baseline for this experiment
  because local `main` already contains the vendor snapshot baseline and should
  not be pushed/touched through `origin/main` yet. The branch has now been
  fast-forwarded to local `main` and configured to track local `main`; upstream
  versus HEAD is `0/0`, while `origin/main` remains intentionally divergent.
- Native balanced-MoE work is scoped to the backend split requested for this branch:
  TorchTitan owns policy/layer integration, Primus-Turbo owns the hot-helper ABI
  and normal top-k TurboEP path, and MORI owns SDMA owner-compact transport.
- Minimal harness is under `tools/`; old local probe directories were moved out
  of the repo to `/tmp/dsv4_pretrain_tools_archive_20260613/`.
- Static runtime gates passed on `do-vunguyen-mi350-gpu` in
  `onenexus/nexus-titan:rocm722-pytorch-nightly-mori`:
  `check_balanced_moe_backend_parity.py`,
  `check_balanced_moe_model_shapes.py`, and
  `check_balanced_moe_model_surfaces.py`.
- Full 48-row table attempt `bm_table_native_fresh1` is no longer running.
  It produced a partial artifact only: `7/48` planned rows completed, all on
  DeepSeek MBS1 plus DeepSeek MBS2 plain/Primus/helper8. The driver stopped
  after `balanced_moe_backend_table_native_20260613_fresh1_deepseek_helper8_mbs2_steps20`
  because the postflight clean-node guard returned `postflight_idle=false`.
  At poll time a separate `torchtitan-ft-lighthouse` Docker container was
  present, and new GPU-holder PIDs were visible, so no unrelated cleanup was
  attempted. The partial compact summary was copied to
  `run_artifacts/balanced_moe_backend_table_native_20260613_fresh1.partial.json`.
  Raw logs are under
  `/scratch/sonle5/dsv4_pretrain_canary_20260527/runs/balanced_moe_backend_table_native_20260613_fresh1/`.
  Remote compact artifact is
  `/scratch/sonle5/dsv4_pretrain_canary_20260527/sweep_snapshots/balanced_moe_backend_table_native_20260613_fresh1/playground/experiments/2026-05-27-deepseek-v4-amd-pretrain/run_artifacts/balanced_moe_backend_table_native_20260613_fresh1.json`.
- The table matrix is DeepSeek-V4 12-layer, Qwen3.5 8-layer, and
  Kimi-K2.6 6-layer across MBS `1,2,4,8` and four options:
  plain standard EP, Primus-Turbo TurboEP no-helper, standard-EP top-8 helper
  with MORI SDMA, and native Primus-Turbo top-8 helper with MORI SDMA.
- After the MTP audit, the harness default baseline matrix is corrected to
  `48` rows: DeepSeek MTP1, DeepSeek no-MTP historical control, Qwen, and
  Kimi across MBS `1,2,4,8` and the three baseline options
  `plain,primus,helper8`. The native Primus helper remains implemented as an
  opt-in variant for the follow-up candidate table, not part of the default
  baseline rerun.
- DeepSeek helper MBS8 is explicitly part of this run because previous
  retained evidence around `~8.5k tok/GPU/s` did not reproduce cleanly in the
  later table attempts.
- Source/log audit explains the DeepSeek MBS8 discrepancy. The 20260608
  `~8.5k` run built an `81,244,314,368` parameter model and did not log MTP.
  Current source defaults `TORCHTITAN_DSV4_NUM_MTP_MODULES` to `1` when unset;
  fresh4 logged `dsv4_num_mtp_modules=default` and built an
  `87,079,924,032` parameter model. That is `+5,835,609,664` parameters
  (`+7.18%`) and roughly `+19.4 GiB` peak memory versus the old run. Treat
  the old row as historical no-MTP evidence; reproduce it only through the
  explicit `deepseek_nomtp` model surface. The corrected table harness now has
  explicit `deepseek_mtp1` and `deepseek_nomtp` surfaces instead of relying on
  the source default.
- Completed fresh1 rows:
  - DeepSeek MBS1 plain: `4214.37` tok/GPU/s steps 2-20, `4264.88` late,
    `208.64` TFLOPs/GPU, `135.56 GiB`, no allocator retries.
  - DeepSeek MBS1 Primus raw top-k: `5009.10` tok/GPU/s steps 2-20,
    `5113.62` late, `247.98` TFLOPs/GPU, `123.69 GiB`, no retries.
  - DeepSeek MBS1 standard helper8 SDMA: `4574.58` tok/GPU/s steps 2-20,
    `4614.25` late, `226.48` TFLOPs/GPU, `127.11 GiB`, no retries.
  - DeepSeek MBS1 native Primus helper8 SDMA: `4810.02` tok/GPU/s steps 2-20,
    `4861.12` late, `238.13` TFLOPs/GPU, `126.74 GiB`, no retries.
  - DeepSeek MBS2 plain: `4835.59` tok/GPU/s steps 2-20, `4827.38` late,
    `239.39` TFLOPs/GPU, `166.89 GiB`, no retries.
  - DeepSeek MBS2 Primus raw top-k: `5765.91` tok/GPU/s steps 2-20,
    `5671.11` late, `285.45` TFLOPs/GPU, `141.62 GiB`, no retries.
  - DeepSeek MBS2 standard helper8 SDMA: `5966.37` tok/GPU/s steps 2-20,
    `6063.89` late, `295.38` TFLOPs/GPU, `148.12 GiB`, no retries.
  This partial run is not a valid full baseline or promotion table; rerun the
  full matrix under a clean-node guard that distinguishes unrelated lighthouse
  jobs from table-run leftovers.
- Harness hardening after fresh1:
  `tools/run_balanced_moe_table_sweep.py` now writes explicit completion
  metadata, marks short tables as `status=incomplete`, returns nonzero for
  incomplete runs, and permits a configured non-GPU lighthouse container only
  when ROCm reports zero GPU/VRAM activity. Validation: `py_compile` passed,
  and dry-run planning for DeepSeek MBS1/2 across all four variants produced
  the expected `8` planned rows.
- Harness correction after the MTP audit:
  `tools/run_balanced_moe_table_sweep.py` now has separate
  `deepseek_mtp1` and `deepseek_nomtp` model surfaces. The `deepseek` alias
  remains MTP1 for compatibility. Validation: `py_compile` passed and local
  env inspection showed DeepSeek MBS8 emits `CANARY_DSV4_NUM_MTP_MODULES=1`,
  `CANARY_DSV4_MTP_LOSS_WEIGHT=0.3`, and CE8 for all baseline/native variants.
- Current node state blocks a fair relaunch. `do-vunguyen-mi350-gpu` is running
  a `torchft_32b_fix_20260613_052200` job with active GPUs, and
  `do-sonle5-mi350-gpu` is running the paired `torchft_32b_fix_20260613_052200`
  node job plus lighthouse; GPUs are at roughly `99-100%` use. No unrelated
  containers or PIDs were stopped.

## 2026-05-27

- Created the DeepSeek V4 AMD pretraining experiment scaffold.
- Migrated WIP source snapshots for Primus, Primus-Turbo, TorchTitan, and
  NVIDIA Megatron-LM to repo-root `projects/`.
- Migrated comparison references to repo-root `references/`, split into
  `references/ascend/`, `references/nvidia/`, and `references/amd/`.
- Recorded the LMSYS Day-0 RL section and radixark Miles roadmap issue as source references.
- No AMD pretraining run has been launched from this experiment directory.

## Open Items

- Select the first AMD canary shape.
- Decide the initial backend path: Primus with Megatron-LM, Primus with TorchTitan, or a minimal Megatron-LM source probe.
- Add a launcher only after the rerun command is concrete.
- Add patch files under `changesets/` once source deltas are known.
