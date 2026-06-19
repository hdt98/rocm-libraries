# DSV4 Megatron Measurements

Reduced 12-layer DSV4 Megatron pretrain evidence. Completed rows use no MTP,
Lightning/DSA indexer, Sparse MLA, AdamW, seq_len 4096, 256 experts, top-k 6,
and `DISABLE_SAVE=1`.

## Correctness

| Candidate | Oracle | Config | Tokens | Logprob diff | Evidence |
|---|---|---|---:|---|---|
| Standard EP Megatron | PyTorch reference Sparse MLA and Lightning Indexer branches in Miles DSV4 plugin | TP=1 EP=4 FSDP=4 MBS=2 GBS=8, mHC off, fixed rank-0 batch, matched weights | 8192 | max 0.1168928; mean 0.0144115; RMS 0.0199401 | `/local/data/sonle5/dsv4_pretrain_rl/runs/dsv4_standard_pytorch_oracle_mbs2_20260618T204647Z/correctness_vs_candidate.json` |

The correctness comparison JSON records matched `tokens`, `labels`,
`loss_mask`, and `position_ids`, each with shape `[2,4096]`. Candidate
logprobs are flattened to `[8192]`; oracle logprobs are `[2,4096]`.

Candidate artifact:
`/local/data/sonle5/dsv4_pretrain_rl/runs/dsv4_standard_logprob_dump_mbs2_cachefix_20260618T202953Z/candidate_logprobs.pt`

Oracle artifact:
`/local/data/sonle5/dsv4_pretrain_rl/runs/dsv4_standard_pytorch_oracle_mbs2_20260618T204647Z/oracle_logprobs.pt`

## Best Rows

| Scope | Best run | Config | Perf | Evidence |
|---|---|---|---|---|
| 4x mHC off | `megatron_standard_no_mtp_mhcoff_tp1ep4_mbs2_gbs128_4xmi350_postprune1617_20260619T140503Z` | TP=1 EP=4 FSDP=4 MBS=2 GBS=128, Standard EP `alltoall`, CPU optimizer offload | iter20 138.2; avg 2-20 133.15; avg 13-20 138.59 TFLOP/s/GPU; peak 235275.25/241054 MB | `/local/data/sonle5/dsv4_pretrain_rl/runs/megatron_standard_no_mtp_mhcoff_tp1ep4_mbs2_gbs128_4xmi350_postprune1617_20260619T140503Z/measurement.json` |
| 4x mHC on | `megatron_standard_no_mtp_mhcon_tp1ep4_mbs2_gbs128_4xmi350_4567_20260618T181309Z` | TP=1 EP=4 FSDP=4 MBS=2 GBS=128, Standard EP `alltoall`, CPU optimizer offload | iter20 126.0; avg 2-20 121.36; avg 13-20 126.65 TFLOP/s/GPU; peak 235308.06/241154 MB | `/local/data/sonle5/dsv4_pretrain_rl/runs/megatron_standard_no_mtp_mhcon_tp1ep4_mbs2_gbs128_4xmi350_4567_20260618T181309Z/measurement.json` |
| 8x TP=1 mHC off | `megatron_standard_no_mtp_mhcoff_tp1ep8_mbs4_gbs128_8xmi350_postprimus_current_r2_20260619T172546Z` | TP=1 EP=8 FSDP=8 MBS=4 GBS=128, Standard EP `alltoall`, TE fused Adam, validation fingerprint `3d0b...`; post-cleanup `934d...` validation below | iter20 228.8; avg 2-20 215.19; avg 13-20 230.31 TFLOP/s/GPU; peak 239449.48/254006 MB | `/local/data/sonle5/dsv4_pretrain_rl/runs/megatron_standard_no_mtp_mhcoff_tp1ep8_mbs4_gbs128_8xmi350_postprimus_current_r2_20260619T172546Z/measurement.json` |
| 8x TP=1 mHC on | `megatron_standard_no_mtp_mhcon_tp1ep8_mbs4_gbs128_8xmi350_cpuoffload_fp16main_20260618T191007Z` | TP=1 EP=8 FSDP=8 MBS=4 GBS=128, Standard EP `alltoall`, CPU optimizer offload, fp16 main params | iter20 113.5; avg 2-20 99.22; avg 13-20 106.91 TFLOP/s/GPU; peak 210404.47/221692 MB | `/local/data/sonle5/dsv4_pretrain_rl/runs/megatron_standard_no_mtp_mhcon_tp1ep8_mbs4_gbs128_8xmi350_cpuoffload_fp16main_20260618T191007Z/measurement.json` |
| 8x TP=8 MBS=4 | `megatron_standard_no_mtp_mhcoff_tp8ep8_mbs4_gbs128_8xmi350_tp8ep8_fsdp1_block64_norecompute_20step_20260618T222957Z` | TP=8 EP=8 FSDP=1 MBS=4 GBS=128, Standard EP `alltoall`, no recompute | iter20 79.8; avg 2-20 75.39; avg 13-20 80.30 TFLOP/s/GPU; peak 209183.20/214160 MB | `/local/data/sonle5/dsv4_pretrain_rl/runs/megatron_standard_no_mtp_mhcoff_tp8ep8_mbs4_gbs128_8xmi350_tp8ep8_fsdp1_block64_norecompute_20step_20260618T222957Z/measurement.json` |
| 8x TP=8 MBS=8 | `megatron_standard_no_mtp_mhcoff_tp8ep8_mbs8_gbs128_8xmi350_tp8ep8_fsdp1_mbs8_block64_norecompute_20step_20260618T230344Z` | TP=8 EP=8 FSDP=1 MBS=8 GBS=128, Standard EP `alltoall`, no recompute | iter20 89.5; avg 2-20 53.83; avg 13-20 66.49 TFLOP/s/GPU; peak 209361.36/240024 MB | `/local/data/sonle5/dsv4_pretrain_rl/runs/megatron_standard_no_mtp_mhcoff_tp8ep8_mbs8_gbs128_8xmi350_tp8ep8_fsdp1_mbs8_block64_norecompute_20step_20260618T230344Z/measurement.json` |

Each completed row's `measurement.json` records `tflops_windows`,
`memory_metrics`, `peak_memory_mb`, `peak_reserved_mb`, and
`allocator_retries`. `allocator_retries` is `null` for these rows because the
Megatron logs did not emit a retry count. The runner now writes
`measurement.json` automatically after every attempt and includes exact
`host_env.txt` and `container_env.txt` contents under the JSON `env` key.
Older row directories keep the available `run_metadata.json` and `host_env.txt`
or `host_env.json` captured by the earlier wrapper.

Post-prune validation completed.

- 4x wrapper validation for fingerprint
  `1617c7fe0abf83741ed3cbd5f1d99863096f4badc66f09b088067dde42028d36`:
  `megatron_standard_no_mtp_mhcoff_tp1ep4_mbs2_gbs128_4xmi350_postprune1617_20260619T140503Z`
  reached iter20 138.2, avg 2-20 133.15, avg 13-20 138.59
  TFLOP/s/GPU, peak 235275.25/241054 MB.
- Previous 4x Standard EP validation for fingerprint
  `3d0b4bf22a1031d17340e17278be71f4717c4c34dbc3a7ac7f413a38984b2bfa`:
  `megatron_standard_no_mtp_mhcoff_tp1ep4_mbs2_gbs128_4xmi350_postprimus_current_20260619T162534Z`
  reached exit 0, iter20 138.1, avg 2-20 130.44, avg 13-20 136.40
  TFLOP/s/GPU, peak 235274.14/241124 MB.
- Post-cleanup 4x Standard EP validation for recorded fingerprint
  `934d9325a94231d939900eb6343b47ac5fcb80edf848253648b856a45de640e7`:
  `megatron_standard_no_mtp_mhcoff_tp1ep4_mbs2_gbs128_4xmi350_postprune934d_20260619T181446Z`
  reached exit 0, iter20 136.3, avg 2-20 131.18, avg 13-20 136.31
  TFLOP/s/GPU, peak 235274.26/241120 MB.
- Previous 8x post-prune validation for fingerprint
  `ade7b3851976c66b2ee20db65898cf95e4e6a87e56106d8ab8b8faecca285734`:
  `megatron_standard_no_mtp_mhcoff_tp1ep8_mbs4_gbs128_8xmi350_postpruneade7_20260619T151907Z`
  reached iter20 227.6, avg 2-20 215.59, avg 13-20 226.70
  TFLOP/s/GPU, peak 239449.85/255036 MB.
- Previous 8x validation for fingerprint
  `3d0b4bf22a1031d17340e17278be71f4717c4c34dbc3a7ac7f413a38984b2bfa`:
  first run
  `megatron_standard_no_mtp_mhcoff_tp1ep8_mbs4_gbs128_8xmi350_postprimus_current_20260619T171443Z`
  reached exit 0, iter20 221.8, avg 2-20 211.42, avg 13-20 220.09
  TFLOP/s/GPU, peak 239448.72/255034 MB. Repeat
  `megatron_standard_no_mtp_mhcoff_tp1ep8_mbs4_gbs128_8xmi350_postprimus_current_r2_20260619T172546Z`
  reached exit 0, iter20 228.8, avg 2-20 215.19, avg 13-20 230.31
  TFLOP/s/GPU, peak 239449.48/254006 MB, so the current harness is not
  slower than the previous 8x post-prune best within run-to-run variance.
- Post-cleanup 8x Standard EP validation for recorded fingerprint
  `934d9325a94231d939900eb6343b47ac5fcb80edf848253648b856a45de640e7`:
  `megatron_standard_no_mtp_mhcoff_tp1ep8_mbs4_gbs128_8xmi350_postprune934d_r2_20260619T183810Z`
  reached exit 0, iter20 222.6, avg 2-20 217.96, avg 13-20 226.41
  TFLOP/s/GPU, peak 239449.85/255044 MB. The first `934d` 8x launch failed
  before Megatron init because `main_grads_dtype=bf16` requires
  `ACCUMULATE_ALLREDUCE_GRADS_IN_FP32=0`; the corrected `r2` launch is the
  validation row.
- A previous post-prune 4x mHC-on validation for fingerprint
  `ade7b3851976c66b2ee20db65898cf95e4e6a87e56106d8ab8b8faecca285734`,
  `megatron_standard_no_mtp_mhcon_tp1ep4_mbs2_gbs128_4xmi350_mhcon_postpruneade7_20260619T153341Z`,
  reached iter20 121.9, avg 2-20 122.26, avg 13-20 126.65
  TFLOP/s/GPU, peak 235308.33/241218 MB.
- Previous 4x mHC-on validation for fingerprint
  `3d0b4bf22a1031d17340e17278be71f4717c4c34dbc3a7ac7f413a38984b2bfa`:
  `megatron_standard_no_mtp_mhcon_tp1ep4_mbs2_gbs128_4xmi350_mhcon_postprimus_current_20260619T164858Z`
  observed all 20 iterations, iter20 122.5, avg 2-20 120.46, avg 13-20
  124.91 TFLOP/s/GPU, peak 235305.44/241162 MB. The host wrapper exited
  before Docker completion, so `exit_code` is not captured in this reconstructed
  `measurement.json`; no failure marker appeared in the captured Docker log.

Evidence:
`/local/data/sonle5/dsv4_pretrain_rl/runs/megatron_standard_no_mtp_mhcoff_tp1ep4_mbs2_gbs128_4xmi350_postprune1617_20260619T140503Z/measurement.json`;
`/local/data/sonle5/dsv4_pretrain_rl/runs/megatron_standard_no_mtp_mhcoff_tp1ep4_mbs2_gbs128_4xmi350_postprimus_current_20260619T162534Z/measurement.json`;
`/local/data/sonle5/dsv4_pretrain_rl/runs/megatron_standard_no_mtp_mhcoff_tp1ep4_mbs2_gbs128_4xmi350_postprune934d_20260619T181446Z/measurement.json`;
`/local/data/sonle5/dsv4_pretrain_rl/runs/megatron_standard_no_mtp_mhcoff_tp1ep8_mbs4_gbs128_8xmi350_postpruneade7_20260619T151907Z/measurement.json`;
`/local/data/sonle5/dsv4_pretrain_rl/runs/megatron_standard_no_mtp_mhcoff_tp1ep8_mbs4_gbs128_8xmi350_postprimus_current_20260619T171443Z/measurement.json`;
`/local/data/sonle5/dsv4_pretrain_rl/runs/megatron_standard_no_mtp_mhcoff_tp1ep8_mbs4_gbs128_8xmi350_postprimus_current_r2_20260619T172546Z/measurement.json`;
`/local/data/sonle5/dsv4_pretrain_rl/runs/megatron_standard_no_mtp_mhcoff_tp1ep8_mbs4_gbs128_8xmi350_postprune934d_20260619T183626Z/measurement.json`;
`/local/data/sonle5/dsv4_pretrain_rl/runs/megatron_standard_no_mtp_mhcoff_tp1ep8_mbs4_gbs128_8xmi350_postprune934d_r2_20260619T183810Z/measurement.json`;
`/local/data/sonle5/dsv4_pretrain_rl/runs/megatron_standard_no_mtp_mhcon_tp1ep4_mbs2_gbs128_4xmi350_mhcon_postpruneade7_20260619T153341Z/measurement.json`;
`/local/data/sonle5/dsv4_pretrain_rl/runs/megatron_standard_no_mtp_mhcon_tp1ep4_mbs2_gbs128_4xmi350_mhcon_postprimus_current_20260619T164858Z/measurement.json`.

## Open / Failed

| Target | Status | Evidence |
|---|---|---|
| Primus TurboEP | Wired in launcher. Custom Miles+Primus image reaches training then segfaults in `primus_turbo::deep_ep::layout::get_dispatch_layout`. Standalone official `rocm/primus:v26.3` dispatch/combine succeeds for DSV4-shaped EP4/EP8, tokens 4096, hidden 4096, experts 256, top-k 6/8, so the DSV4 route shape is not itself the native-Turbo blocker. Official `rocm/primus:v26.3` plus the DSV4 Megatron root parses `--experimental-attention-variant dsv4`, builds the model/optimizer/dataset, enables Primus TurboEP, and enters the training step, but full Megatron remains blocked by Transformer Engine GEMM coverage at DSV4 `wq_a`; `NVTE_USE_HIPBLASLT=0`, `TURBO_DEEPEP_NUM_CU=20`, and GPUs 4-7 still fail with `RuntimeError: Unable to find any suitable algorithms`. `--use-turbo-grouped-gemm` changes an earlier path to a grouped-linear split-count assertion; `--moe-use-legacy-grouped-gemm` reaches the same DSV4 attention projection TE dense GEMM blocker. A scratch-only DSV4 PrimusTurboLinear route plus lazy `ops.gemm` import fixes the earlier Python `module object is not callable` binding, but both registered Primus dense GEMM backends fail this projection: HIPBLASLt reports no valid algorithm, while `PRIMUS_TURBO_GEMM_BACKEND=TRITON` fails in the Origami selector with `ValueError: vector::reserve`. | Custom SIGSEGV: `/local/data/sonle5/dsv4_pretrain_rl/runs/megatron_primus_turboep_no_mtp_mhcoff_tp1ep4_mbs1_gbs4_4xmi350_primus_forcelb_20260618T234215Z/train.log`; official standalone dispatch/combine success: `/local/data/sonle5/dsv4_pretrain_rl/runs/primus_deepep_official_dsv4_matrix_20260619T011108Z/summary.jsonl`; official HIPBLASLt-disabled TE blocker: `/local/data/sonle5/dsv4_pretrain_rl/runs/megatron_primus_turboep_no_mtp_mhcoff_tp1ep4_mbs1_gbs4_4xmi350_officialprimus_hipblaslt0b_20260619T180109Z/measurement.json`; official router fallback diagnostic: `/local/data/sonle5/dsv4_pretrain_rl/runs/megatron_primus_turboep_no_mtp_mhcoff_tp1ep4_mbs1_gbs4_4xmi350_officialprimus_tefallback_20260619T134059Z/measurement.json`; official TE grouped GEMM blocker: `/local/data/sonle5/dsv4_pretrain_rl/runs/megatron_primus_turboep_no_mtp_mhcoff_tp1ep4_mbs1_gbs4_4xmi350_officialprimus_tefallback2_20260619T134947Z/measurement.json`; official Turbo grouped GEMM split-count blocker: `/local/data/sonle5/dsv4_pretrain_rl/runs/megatron_primus_turboep_no_mtp_mhcoff_tp1ep4_mbs1_gbs4_4xmi350_officialprimus_turbogemm_20260619T135510Z/measurement.json`; official legacy grouped GEMM TE dense blocker: `/local/data/sonle5/dsv4_pretrain_rl/runs/megatron_primus_turboep_no_mtp_mhcoff_tp1ep4_mbs1_gbs4_4xmi350_officialprimus_legacygg_20260619T143136Z/measurement.json`; scratch TurboLinear bad binding: `/local/data/sonle5/dsv4_pretrain_rl/runs/megatron_primus_turboep_no_mtp_mhcoff_tp1ep4_mbs1_gbs4_4xmi350_officialprimus_turbolinear_gemm_legacygg_20260619T145752Z/measurement.json`; scratch TurboLinear HIPBLASLt blocker: `/local/data/sonle5/dsv4_pretrain_rl/runs/megatron_primus_turboep_no_mtp_mhcoff_tp1ep4_mbs1_gbs4_4xmi350_20260619T150514Z/measurement.json`; scratch TurboLinear Triton blocker: `/local/data/sonle5/dsv4_pretrain_rl/runs/megatron_primus_turboep_no_mtp_mhcoff_tp1ep4_mbs1_gbs4_4xmi350_officialprimus_tritongemm_legacygg_20260619T151231Z/measurement.json` |
| Primus TurboEP DeepEP-only | Default launcher path omits `--enable-primus-turbo` so dense DSV4 linears stay on the Miles/Megatron TE path. With the remote scratch Miles linear override disabled, the Miles+Primus image builds model/optimizer/dataset and enters `training ...`. `PRIMUS_TURBO_MOE_DISPATCH_COMBINE_BACKEND=TURBO` then SIGSEGVs before iter1 in native DeepEP dispatch. A narrowed rerun with `TURBO_DEEPEP_DISABLE_ASYNC_FINISH=1` and `TURBO_DEEPEP_DISABLE_ALLOCATE_ON_COMM_STREAM=1` still SIGSEGVs in `primus_turbo::deep_ep::layout::get_dispatch_layout`, so the crash is not the async-finish or comm-stream allocation path. `PRIMUS_TURBO_MOE_DISPATCH_COMBINE_BACKEND=DEEP_EP`, `TURBO_DEEPEP_NUM_CU=32` fails cleanly at first dispatch because `deep_ep` is not installed. Import check: `rocm/primus:v26.3` has `primus_turbo` and `origami`, but not `deep_ep`; the Miles+Primus image also lacks `deep_ep`. | Turbo backend SIGSEGV: `/local/data/sonle5/dsv4_pretrain_rl/runs/megatron_primus_turboep_no_mtp_mhcoff_tp1ep4_mbs1_gbs4_4xmi350_deepeponly7ae3b_20260619T161049Z/measurement.json`; async/allocation-disabled SIGSEGV: `/local/data/sonle5/dsv4_pretrain_rl/runs/megatron_primus_turboep_no_mtp_mhcoff_tp1ep4_mbs1_gbs4_4xmi350_20260619T174702Z/measurement.json`; DEEP_EP backend missing package: `/local/data/sonle5/dsv4_pretrain_rl/runs/megatron_primus_turboep_no_mtp_mhcoff_tp1ep4_mbs1_gbs4_4xmi350_deeponly_deepep32_0cb2_20260619T161715Z/measurement.json` |
| 4x TP=4 EP=4 FSDP=1 | Tried. `BLOCK_I=16` fails Sparse MLA layout inference; `BLOCK_I=64` reaches optimizer. No-offload OOMs in TE Adam master params, CPU-offload reaches iter13 then hangs. | No-offload OOM: `/local/data/sonle5/dsv4_pretrain_rl/runs/megatron_standard_no_mtp_mhcoff_tp4ep4_mbs2_gbs128_4xmi350_tp4ep4_fsdp1_block64_smoke_20260618T210008Z/train.log`; CPU-offload hang: `/local/data/sonle5/dsv4_pretrain_rl/runs/megatron_standard_no_mtp_mhcoff_tp4ep4_mbs2_gbs128_4xmi350_tp4ep4_fsdp1_block64_cpuoffload_20step_20260618T211718Z/train.log` |
| 8x TP=1 EP=8 MBS=8 | Failed before iter1 at CE/head memory pressure in early attempts; no clean 20-step row. | `/local/data/sonle5/dsv4_pretrain_rl/runs/megatron_standard_no_mtp_mhcoff_tp1ep8_mbs8_gbs128_8xmi350_1234_cpuoffload_dtensor_scratchinit_pruned_20260618T161822Z/train.log`; `/local/data/sonle5/dsv4_pretrain_rl/runs/megatron_standard_no_mtp_mhcoff_tp1ep8_mbs8_gbs128_8xmi350_fsdp_paadamfix_nopg_20260618T125953Z/train.log` |
| BF16 Sparse MLA global `dKV` accumulation | Rejected by kernel-oracle correctness; keep FP32. | `/local/data/sonle5/dsv4_pretrain_rl/runs/sparse_mla_bwd_dkv_bf16_cas_probe_sweep_20260618T173522Z` |

## Provenance

- standard image: `sonle5/dsv4-pr1300-megatron-pretrain:rocm720-mi35x-20260618`
- Primus image tried: `sonle5/dsv4-pr1300-megatron-primus-turbo:rocm720-mi35x-20260618`
- official Primus diagnostic image: `rocm/primus:v26.3`
- Miles path/SHA: `/local/data/sonle5/dsv4_pretrain_rl/deps/miles-pr1300-full`, `0dc10df6488aab5a08d883d7eebb1565303158fd`
- current local runtime fingerprint, excluding `MEASUREMENTS.md`, `README.md`,
  and `__pycache__`:
  `b5cd21ad449d395f2f8d58acb4025d940df2009a87fd1e2ac06f152afe281b2e`
- latest Primus async/allocation diagnostic run recorded source fingerprint:
  `af487ec4383019f25e1ee2d3f40a796904077e25798f6641e476bf3f09f2cea1`
- remote source mount: `/local/data/sonle5/dsv4_pretrain_rl/source`
- completed rows ran on `son-le5-mlops-ubuntu-gpu-mi350x8-2304gb-ric1`
