#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TORCHTITAN_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${TORCHTITAN_ROOT}/../.." && pwd)"

OUTPUT_PATH="${OUTPUT_PATH:-${REPO_ROOT}/.artifacts/dsv4_pretrain_torchtitan}"
NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-8}"
TP="${TP:-8}"
EP="${EP:-8}"
CP="${CP:-1}"
PP="${PP:-1}"
LOCAL_BATCH_SIZE="${LOCAL_BATCH_SIZE:-1}"
SEQ_LEN="${SEQ_LEN:-4096}"
STEPS="${STEPS:-10}"
SEED="${SEED:-1234}"
DRY_RUN="${DRY_RUN:-0}"

export USE_ROCM="${USE_ROCM:-1}"
export USE_CUDA="${USE_CUDA:-0}"
export ROCM_HOME="${ROCM_HOME:-/opt/rocm}"
export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
export HIP_FORCE_DEV_KERNARG="${HIP_FORCE_DEV_KERNARG:-1}"
export HSA_NO_SCRATCH_RECLAIM="${HSA_NO_SCRATCH_RECLAIM:-1}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export NVTE_ALLOW_NONDETERMINISTIC_ALGO="${NVTE_ALLOW_NONDETERMINISTIC_ALGO:-0}"
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
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-$(seq -s, 0 $((NUM_GPUS_PER_NODE - 1)))}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${HIP_VISIBLE_DEVICES}}"

CMD=(
  "${TORCHTITAN_ROOT}/run_train.sh"
  --dump-folder "${OUTPUT_PATH}"
  --training.local_batch_size "${LOCAL_BATCH_SIZE}"
  --training.seq_len "${SEQ_LEN}"
  --training.steps "${STEPS}"
  --training.dtype bfloat16
  --training.mixed_precision_param bfloat16
  --parallelism.tensor_parallel_degree "${TP}"
  --parallelism.expert_parallel_degree "${EP}"
  --parallelism.context_parallel_degree "${CP}"
  --parallelism.pipeline_parallel_degree "${PP}"
  --parallelism.enable_sequence_parallel
  --debug.seed "${SEED}"
  --debug.deterministic
  --metrics.log_freq 1
  --checkpoint.interval "${STEPS}"
)

printf 'TorchTitan path: %s\n' "${TORCHTITAN_ROOT}"
printf 'Output path: %s\n' "${OUTPUT_PATH}"
printf 'Command:\n'
printf ' %q' NGPU="${NUM_GPUS_PER_NODE}" MODULE=graph_trainer.deepseek_v3 CONFIG=graph_trainer_deepseek_v4_reduced_12l "${CMD[@]}"
printf '\n'

if [ "${DRY_RUN}" = "1" ]; then
  exit 0
fi

mkdir -p "${OUTPUT_PATH}"
cd "${TORCHTITAN_ROOT}"
NGPU="${NUM_GPUS_PER_NODE}" MODULE=graph_trainer.deepseek_v3 CONFIG=graph_trainer_deepseek_v4_reduced_12l "${CMD[@]}"
