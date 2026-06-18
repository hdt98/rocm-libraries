#!/usr/bin/env bash
set -euo pipefail

RECIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${RECIPE_DIR}/../../.." && pwd)"

MEGATRON_PATH="${MEGATRON_PATH:-${REPO_ROOT}/projects/megatron-LM}"
MILES_PLUGIN_PATH="${MILES_PLUGIN_PATH:-${REPO_ROOT}/references/nvidia/miles-deepseek-v4-pr1045}"
OUTPUT_PATH="${OUTPUT_PATH:-${REPO_ROOT}/.artifacts/dsv4_pretrain_megatron}"
DATA_PATH="${DATA_PATH:-}"

NUM_NODES="${NUM_NODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-8}"

TP="${TP:-8}"
PP="${PP:-1}"
CP="${CP:-1}"
EP="${EP:-8}"
ETP="${ETP:-1}"
MBS="${MBS:-1}"
GBS="${GBS:-8}"
TRAIN_ITERS="${TRAIN_ITERS:-10}"
SEQ_LEN="${SEQ_LEN:-4096}"
SEED="${SEED:-1234}"
DRY_RUN="${DRY_RUN:-0}"

export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export NVTE_ALLOW_NONDETERMINISTIC_ALGO="${NVTE_ALLOW_NONDETERMINISTIC_ALGO:-0}"
export USE_ROCM="${USE_ROCM:-1}"
export USE_CUDA="${USE_CUDA:-0}"
export ROCM_HOME="${ROCM_HOME:-/opt/rocm}"
export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
export HIP_FORCE_DEV_KERNARG="${HIP_FORCE_DEV_KERNARG:-1}"
export HSA_NO_SCRATCH_RECLAIM="${HSA_NO_SCRATCH_RECLAIM:-1}"
export NCCL_IB_TC="${NCCL_IB_TC:-160}"
export NCCL_IB_TIMEOUT="${NCCL_IB_TIMEOUT:-22}"
export NCCL_IB_RETRY_CNT="${NCCL_IB_RETRY_CNT:-7}"
export NCCL_IB_QPS_PER_CONNECTION="${NCCL_IB_QPS_PER_CONNECTION:-2}"
export NCCL_IB_SPLIT_DATA_ON_QPS="${NCCL_IB_SPLIT_DATA_ON_QPS:-0}"
export NCCL_MIN_NCHANNELS="${NCCL_MIN_NCHANNELS:-32}"
export NCCL_MAX_NCHANNELS="${NCCL_MAX_NCHANNELS:-32}"
export NCCL_ALGO="${NCCL_ALGO:-Ring}"
export NCCL_PXN_DISABLE="${NCCL_PXN_DISABLE:-0}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-2}"
export MILES_DSV4_CKPT_VERSION="${MILES_DSV4_CKPT_VERSION:-2604}"
export MILES_DSV4_2604_SUBMODE="${MILES_DSV4_2604_SUBMODE:-2604B}"
export MEGATRON_USE_KV_QAT="${MEGATRON_USE_KV_QAT:-1}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-$(seq -s, 0 $((NUM_GPUS_PER_NODE - 1)))}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${HIP_VISIBLE_DEVICES}}"

PYTHONPATH_ENTRIES=("${MEGATRON_PATH}")
if [ -d "${MILES_PLUGIN_PATH}/miles_plugins" ]; then
  PYTHONPATH_ENTRIES+=("${MILES_PLUGIN_PATH}")
fi
if [ -n "${PYTHONPATH:-}" ]; then
  PYTHONPATH_ENTRIES+=("${PYTHONPATH}")
fi
export PYTHONPATH="$(IFS=:; echo "${PYTHONPATH_ENTRIES[*]}")"

COMPRESS_RATIOS=(${COMPRESS_RATIOS:-})
source "${RECIPE_DIR}/scripts/models/deepseek-v4-flash-reduced-12l.sh"

DATA_ARGS=(--mock-data --tokenizer-type NullTokenizer)
if [ -n "${DATA_PATH}" ]; then
  DATA_ARGS=(--data-path "${DATA_PATH}" --split 99,1,0 --tokenizer-type HuggingFaceTokenizer --tokenizer-model "${TOKENIZER_MODEL:-deepseek-ai/DeepSeek-V3}")
fi

DISTRIBUTED_ARGS=(
  --nproc-per-node "${NUM_GPUS_PER_NODE}"
  --nnodes "${NUM_NODES}"
  --node-rank "${NODE_RANK}"
  --master-addr "${MASTER_ADDR}"
  --master-port "${MASTER_PORT}"
)

TRAIN_ARGS=(
  "${MODEL_ARGS[@]}"
  --distributed-timeout-minutes 60
  --tensor-model-parallel-size "${TP}"
  --pipeline-model-parallel-size "${PP}"
  --context-parallel-size "${CP}"
  --expert-model-parallel-size "${EP}"
  --expert-tensor-parallel-size "${ETP}"
  --sequence-parallel
  --use-mcore-models
  --use-distributed-optimizer
  --overlap-grad-reduce
  --overlap-param-gather
  --seq-length "${SEQ_LEN}"
  --max-position-embeddings "${SEQ_LEN}"
  --micro-batch-size "${MBS}"
  --global-batch-size "${GBS}"
  --train-iters "${TRAIN_ITERS}"
  --eval-iters 0
  --eval-interval "${TRAIN_ITERS}"
  --lr 1e-6
  --min-lr 1e-7
  --lr-decay-style constant
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.98
  --clip-grad 1.0
  --init-method-std 0.02
  --bf16
  --recompute-granularity selective
  --recompute-modules mlp moe mla_up_proj layernorm
  --transformer-impl transformer_engine
  --attention-softmax-in-fp32
  --grad-reduce-in-bf16
  --accumulate-allreduce-grads-in-fp32
  --seed "${SEED}"
  --data-cache-path "${OUTPUT_PATH}/cache"
  "${DATA_ARGS[@]}"
  --save "${OUTPUT_PATH}/checkpoints"
  --save-interval "${TRAIN_ITERS}"
  --tensorboard-dir "${OUTPUT_PATH}/tensorboard"
  --log-throughput
  --log-interval 1
  --no-check-for-nan-in-loss-and-grad
  --manual-gc
  --manual-gc-interval 10
)

CMD=(torchrun "${DISTRIBUTED_ARGS[@]}" "${MEGATRON_PATH}/pretrain_gpt.py" "${TRAIN_ARGS[@]}")

printf 'Megatron-LM path: %s\n' "${MEGATRON_PATH}"
printf 'Output path: %s\n' "${OUTPUT_PATH}"
printf 'Command:\n'
printf ' %q' "${CMD[@]}"
printf '\n'

if [ "${DRY_RUN}" = "1" ]; then
  exit 0
fi

mkdir -p "${OUTPUT_PATH}/checkpoints" "${OUTPUT_PATH}/tensorboard" "${OUTPUT_PATH}/cache"
cd "${MEGATRON_PATH}"
"${CMD[@]}"
