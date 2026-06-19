# DSV4 Megatron Pretrain Harness

Lean Megatron pretraining port of the Miles PR1300 DeepSeek-V4 recipe for MI350
benchmark rows. The faithful PR1300 recipe snapshot remains under
`references/amd/miles-deepseek-v4-pr1300`; this directory is the runnable
experiment layer.

## Shape

- 12 layers, hidden 4096, vocab 129280, 64 attention heads
- q_lora 1024, kv_lora 512, qk_rope/nope 64/448, v_head_dim 512
- 256 MoE experts, MoE hidden 2048, router top-k 6
- seq_len 4096, compress ratios `0,0,4,128,4,128,4,128,4,128,4,128`
- no MTP, Lightning/DSA indexer on, Sparse MLA on, AdamW

## Runtime

Standard EP rows use:

- image: `sonle5/dsv4-pr1300-megatron-pretrain:rocm720-mi35x-20260618`
- Miles source: `/local/data/sonle5/dsv4_pretrain_rl/deps/miles-pr1300-full`
- source mount: `/local/data/sonle5/dsv4_pretrain_rl/source`

The launcher applies only the overlays needed for this shape: Megatron DSV4
parser/init fixes, Sparse MLA MI350 tile knobs, mHC reshape fix, DTensor
CPU-offload fix, optional Primus TurboEP bridge, and rank-0 logprob dump hook.
`scripts/run_megatron_perf_row.sh` writes `run_metadata.json`,
`host_env.txt`, `container_env.txt`, `train.log`, and `measurement.json` under
the run directory.

## Launch

4x TP=1 EP=4 FSDP=4 MBS=2 GBS=128:

```bash
BASE=/local/data/sonle5/dsv4_pretrain_rl \
GPUS=0,1,2,3 LOGICAL_GPUS=0,1,2,3 NUM_GPUS_PER_NODE=4 GPU_LABEL=4xmi350 \
TP=1 EP=4 MBS=2 GBS=128 MHC=off TRAIN_ITERS=20 DISABLE_SAVE=1 \
USE_MEGATRON_FSDP=1 DATA_PARALLEL_SHARDING_STRATEGY=optim_grads_params \
USE_PRECISION_AWARE_OPTIMIZER=1 \
MAIN_GRADS_DTYPE=fp32 MAIN_PARAMS_DTYPE=fp32 \
EXP_AVG_DTYPE=fp32 EXP_AVG_SQ_DTYPE=fp32 \
OPTIMIZER_CPU_OFFLOAD=1 OVERLAP_CPU_OPTIMIZER_D2H_H2D=1 \
experiments/amd/dsv4-pretrain-megatron/scripts/run_megatron_perf_row.sh
```

8x TP=1 EP=8 FSDP=8 MBS=4 GBS=128:

```bash
GPUS=0,1,2,3,4,5,6,7 LOGICAL_GPUS=0,1,2,3,4,5,6,7 \
NUM_GPUS_PER_NODE=8 GPU_LABEL=8xmi350 TP=1 EP=8 MBS=4 GBS=128
```

TP-heavy probes use `TP=4 EP=4 NUM_GPUS_PER_NODE=4` or
`TP=8 EP=8 NUM_GPUS_PER_NODE=8`, `USE_MEGATRON_FSDP=0`, Sparse MLA
`BLOCK_I=64 BWD_BLOCK_SIZE=64 BWD_BLOCK_H=16`, and for completed TP=8 rows
`DISABLE_RECOMPUTE=1`.

## Notes

- TP=1 FSDP rows use selective recompute over `mlp moe mla_up_proj layernorm`.
- TP=8 EP=8 FSDP=1 completed rows disable recompute.
- Keep Sparse MLA backward global `dKV` accumulation in FP32.
- Primus TurboEP is wired but has no completed perf row yet. Custom
  Miles+Primus reaches training then segfaults in native DeepEP layout;
  official `rocm/primus:v26.3` passes native Turbo dispatch/combine but is
  blocked in Transformer Engine BF16 `generic_gemm` at `wq_a`.
  The launcher defaults to DeepEP-only for this backend; the current
  DeepEP-only smoke gets through model/dataloader setup and fails with SIGSEGV
  before iter1 with the `TURBO` dispatch/combine backend. The `DEEP_EP`
  dispatch/combine backend fails cleanly because neither tried Primus image
  contains the `deep_ep` Python package. Set `PRIMUS_TURBO_ENABLE_FULL_STACK=1`
  to reproduce the broader Primus TESpecProvider/linear replacement path.
- Correctness/perf rows, failure evidence, artifact paths, and source
  fingerprints are in `MEASUREMENTS.md`.
