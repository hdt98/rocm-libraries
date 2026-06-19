#!/usr/bin/env bash
set -euo pipefail

RECIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${RECIPE_DIR}/../../.." && pwd)"

DEFAULT_MEGATRON_PATH="/root/Megatron-LM"
if [ ! -d "${DEFAULT_MEGATRON_PATH}" ] && [ -d "${REPO_ROOT}/projects/megatron-LM" ]; then
  DEFAULT_MEGATRON_PATH="${REPO_ROOT}/projects/megatron-LM"
fi

MEGATRON_PATH="${MEGATRON_PATH:-${DEFAULT_MEGATRON_PATH}}"
MILES_PLUGIN_PATH="${MILES_PLUGIN_PATH:-/root/miles}"
PRIMUS_PATH="${PRIMUS_PATH:-${REPO_ROOT}/projects/primus}"
OUTPUT_PATH="${OUTPUT_PATH:-${REPO_ROOT}/.artifacts/dsv4_pretrain_megatron}"

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
SAVE_INTERVAL="${SAVE_INTERVAL:-1000000}"
DISABLE_SAVE="${DISABLE_SAVE:-1}"
DRY_RUN="${DRY_RUN:-0}"
DSV4_EP_BACKEND="${DSV4_EP_BACKEND:-standard}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-}"
USE_MEGATRON_FSDP="${USE_MEGATRON_FSDP:-0}"
DATA_PARALLEL_SHARDING_STRATEGY="${DATA_PARALLEL_SHARDING_STRATEGY:-optim_grads_params}"
USE_PRECISION_AWARE_OPTIMIZER="${USE_PRECISION_AWARE_OPTIMIZER:-0}"
MAIN_GRADS_DTYPE="${MAIN_GRADS_DTYPE:-fp32}"
MAIN_PARAMS_DTYPE="${MAIN_PARAMS_DTYPE:-fp32}"
EXP_AVG_DTYPE="${EXP_AVG_DTYPE:-fp32}"
EXP_AVG_SQ_DTYPE="${EXP_AVG_SQ_DTYPE:-fp32}"
OPTIMIZER_CPU_OFFLOAD="${OPTIMIZER_CPU_OFFLOAD:-0}"
USE_TORCH_OPTIMIZER_FOR_CPU_OFFLOAD="${USE_TORCH_OPTIMIZER_FOR_CPU_OFFLOAD:-0}"
OVERLAP_CPU_OPTIMIZER_D2H_H2D="${OVERLAP_CPU_OPTIMIZER_D2H_H2D:-0}"
GRAD_REDUCE_IN_BF16="${GRAD_REDUCE_IN_BF16:-1}"
ACCUMULATE_ALLREDUCE_GRADS_IN_FP32="${ACCUMULATE_ALLREDUCE_GRADS_IN_FP32:-1}"
OVERLAP_GRAD_REDUCE="${OVERLAP_GRAD_REDUCE:-1}"
OVERLAP_PARAM_GATHER="${OVERLAP_PARAM_GATHER:-1}"
CROSS_ENTROPY_LOSS_FUSION="${CROSS_ENTROPY_LOSS_FUSION:-0}"
CROSS_ENTROPY_FUSION_IMPL="${CROSS_ENTROPY_FUSION_IMPL:-native}"
DISABLE_RECOMPUTE="${DISABLE_RECOMPUTE:-0}"
DISABLE_COMPILE_DEPENDENCIES="${DISABLE_COMPILE_DEPENDENCIES:-1}"
EXTRA_MEGATRON_ARGS="${EXTRA_MEGATRON_ARGS:-}"

MEGATRON_PATCHES_DEFAULT="${RECIPE_DIR}/patches/megatron_callable_spec_overlay.patch ${RECIPE_DIR}/patches/megatron_moe_recompute_input_ids_overlay.patch ${RECIPE_DIR}/patches/megatron_hash_router_scratch_overlay.patch ${RECIPE_DIR}/patches/megatron_dsv4_padded_vocab_overlay.patch ${RECIPE_DIR}/patches/megatron_dsv4_scratch_init_overlay.patch ${RECIPE_DIR}/patches/megatron_checkpoint_without_output_backward_fallback_overlay.patch ${RECIPE_DIR}/patches/megatron_router_expert_bias_skip_hash_layers_overlay.patch ${RECIPE_DIR}/patches/megatron_logprob_dump_overlay.patch"
MILES_PATCHES_DEFAULT="${RECIPE_DIR}/patches/miles_dsv4_scratch_init_overlay.patch ${RECIPE_DIR}/patches/miles_dsv4_pytorch_oracle_overlay.patch ${RECIPE_DIR}/patches/miles_sparse_mla_mi350_tile_overlay.patch"
TILE_PATCHES_DEFAULT="${RECIPE_DIR}/patches/tile_kernels_mhc_pre_apply_mix_reshape_overlay.patch"
TE_PATCHES_DEFAULT="${RECIPE_DIR}/patches/te_fused_adam_dtensor_precision_aware_overlay.patch"
PRIMUS_MEGATRON_PATCHES_DEFAULT="${RECIPE_DIR}/patches/megatron_primus_turboep_overlay.patch"
PRIMUS_PACKAGE_PATCHES_DEFAULT="${RECIPE_DIR}/patches/primus_turbo_deepep_lazy_modules_overlay.patch"
DTENSOR_CPU_OFFLOAD_PATCHES_DEFAULT="${RECIPE_DIR}/patches/megatron_hybrid_optimizer_dtensor_cpuoffload_overlay.patch"

export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-$([ "${USE_MEGATRON_FSDP}" = "1" ] && echo 8 || echo 1)}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
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
export SGLANG_OPT_USE_TILELANG_MHC_PRE="${SGLANG_OPT_USE_TILELANG_MHC_PRE:-false}"
export SGLANG_OPT_USE_TILELANG_MHC_POST="${SGLANG_OPT_USE_TILELANG_MHC_POST:-false}"
export SGLANG_OPT_USE_AITER_MHC_PRE="${SGLANG_OPT_USE_AITER_MHC_PRE:-true}"
export SGLANG_OPT_USE_AITER_MHC_POST="${SGLANG_OPT_USE_AITER_MHC_POST:-true}"
export MILES_DSV4_SPARSE_MLA_BLOCK_I="${MILES_DSV4_SPARSE_MLA_BLOCK_I:-16}"
export MILES_DSV4_SPARSE_MLA_BWD_BLOCK_SIZE="${MILES_DSV4_SPARSE_MLA_BWD_BLOCK_SIZE:-16}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-$(seq -s, 0 $((NUM_GPUS_PER_NODE - 1)))}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${HIP_VISIBLE_DEVICES}}"
export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-${HIP_VISIBLE_DEVICES}}"
export GPU_DEVICE_ORDINAL="${GPU_DEVICE_ORDINAL:-${HIP_VISIBLE_DEVICES}}"

PYTHONPATH_ENTRIES=("${MEGATRON_PATH}")
[ -d "${MILES_PLUGIN_PATH}/miles_plugins" ] && PYTHONPATH_ENTRIES+=("${MILES_PLUGIN_PATH}")
[ -d "${PRIMUS_PATH}/primus" ] && PYTHONPATH_ENTRIES+=("${PRIMUS_PATH}")
[ -n "${DSV4_EXTRA_PYTHONPATH:-}" ] && PYTHONPATH_ENTRIES+=("${DSV4_EXTRA_PYTHONPATH}")
[ -n "${PYTHONPATH:-}" ] && PYTHONPATH_ENTRIES+=("${PYTHONPATH}")
export PYTHONPATH="$(IFS=:; echo "${PYTHONPATH_ENTRIES[*]}")"

apply_overlays() {
  local label="$1"
  local root="$2"
  shift 2
  local patch_file
  for patch_file in "$@"; do
    [ -z "${patch_file}" ] && continue
    [ -f "${patch_file}" ] || { echo "${label} overlay patch not found: ${patch_file}" >&2; exit 2; }
    if patch --dry-run -p1 -d "${root}" < "${patch_file}" >/dev/null; then
      patch -p1 -d "${root}" < "${patch_file}"
    else
      echo "${label} overlay patch skipped or already applied: ${patch_file}"
    fi
  done
}

patch_array() {
  local value="$1"
  local -n out="$2"
  read -r -a out <<< "${value}"
}

patch_array "${MEGATRON_OVERLAY_PATCHES:-${MEGATRON_PATCHES_DEFAULT}}" MEGATRON_PATCHES
if [ "${DSV4_EP_BACKEND}" = "primus_turboep" ]; then
  patch_array "${PRIMUS_TURBOEP_OVERLAY_PATCHES:-${PRIMUS_MEGATRON_PATCHES_DEFAULT}}" PRIMUS_MEGATRON_PATCHES
  MEGATRON_PATCHES+=("${PRIMUS_MEGATRON_PATCHES[@]}")
elif [ "${DSV4_EP_BACKEND}" != "standard" ]; then
  echo "Unsupported DSV4_EP_BACKEND=${DSV4_EP_BACKEND}" >&2
  exit 2
fi
if [ "${OPTIMIZER_CPU_OFFLOAD}" = "1" ]; then
  patch_array "${DTENSOR_CPU_OFFLOAD_OVERLAY_PATCHES:-${DTENSOR_CPU_OFFLOAD_PATCHES_DEFAULT}}" DTENSOR_PATCHES
  MEGATRON_PATCHES+=("${DTENSOR_PATCHES[@]}")
fi
apply_overlays Megatron "${MEGATRON_PATH}" "${MEGATRON_PATCHES[@]}"

[ -d "${MILES_PLUGIN_PATH}" ] || { echo "Miles plugin path not found: ${MILES_PLUGIN_PATH}" >&2; exit 2; }
patch_array "${MILES_OVERLAY_PATCHES:-${MILES_PATCHES_DEFAULT}}" MILES_PATCHES
apply_overlays Miles "${MILES_PLUGIN_PATH}" "${MILES_PATCHES[@]}"

TILE_KERNELS_PATH="${TILE_KERNELS_PATH:-$(python3 -c 'import pathlib, tile_kernels; print(pathlib.Path(tile_kernels.__file__).resolve().parent)')}"
patch_array "${TILE_KERNELS_OVERLAY_PATCHES:-${TILE_PATCHES_DEFAULT}}" TILE_PATCHES
apply_overlays TileKernels "${TILE_KERNELS_PATH}" "${TILE_PATCHES[@]}"

TE_SITE_PACKAGES="$(python3 -c 'import pathlib, transformer_engine; print(pathlib.Path(transformer_engine.__file__).resolve().parent.parent)')"
patch_array "${TE_OVERLAY_PATCHES:-${TE_PATCHES_DEFAULT}}" TE_PATCHES
apply_overlays TransformerEngine "${TE_SITE_PACKAGES}" "${TE_PATCHES[@]}"

if [ "${DSV4_EP_BACKEND}" = "primus_turboep" ]; then
  PRIMUS_TURBO_ROOT="$(python3 -c 'import importlib.util,pathlib; spec=importlib.util.find_spec("primus_turbo"); assert spec and spec.origin; print(pathlib.Path(spec.origin).resolve().parent.parent)')"
  patch_array "${PRIMUS_TURBO_PACKAGE_OVERLAY_PATCHES:-${PRIMUS_PACKAGE_PATCHES_DEFAULT}}" PRIMUS_PACKAGE_PATCHES
  apply_overlays PrimusTurbo "${PRIMUS_TURBO_ROOT}" "${PRIMUS_PACKAGE_PATCHES[@]}"
  [ "${TP}" = "1" ] || { echo "DSV4_EP_BACKEND=primus_turboep requires TP=1" >&2; exit 2; }
  MOE_TOKEN_DISPATCHER_TYPE="flex"
  MOE_FLEX_DISPATCHER_BACKEND="deepep"
fi

COMPRESS_RATIOS=(${COMPRESS_RATIOS:-})
source "${RECIPE_DIR}/scripts/models/deepseek-v4-flash-reduced-12l.sh"

DISTRIBUTED_ARGS=(
  --nproc-per-node "${NUM_GPUS_PER_NODE}"
  --nnodes "${NUM_NODES}"
  --node-rank "${NODE_RANK}"
  --master-addr "${MASTER_ADDR}"
  --master-port "${MASTER_PORT}"
)

RECOMPUTE_ARGS=()
if [ "${DISABLE_RECOMPUTE}" != "1" ]; then
  read -r -a RECOMPUTE_MODULE_ARRAY <<< "${RECOMPUTE_MODULES:-mlp moe mla_up_proj layernorm}"
  RECOMPUTE_ARGS=(--recompute-granularity "${RECOMPUTE_GRANULARITY:-selective}" --recompute-modules "${RECOMPUTE_MODULE_ARRAY[@]}")
fi

OPTIMIZER_ARGS=()
[ "${USE_PRECISION_AWARE_OPTIMIZER}" = "1" ] && OPTIMIZER_ARGS=(--use-precision-aware-optimizer --main-grads-dtype "${MAIN_GRADS_DTYPE}" --main-params-dtype "${MAIN_PARAMS_DTYPE}" --exp-avg-dtype "${EXP_AVG_DTYPE}" --exp-avg-sq-dtype "${EXP_AVG_SQ_DTYPE}")
[ "${OPTIMIZER_CPU_OFFLOAD}" = "1" ] && OPTIMIZER_ARGS+=(--optimizer-cpu-offload)
[ "${USE_TORCH_OPTIMIZER_FOR_CPU_OFFLOAD}" = "1" ] && OPTIMIZER_ARGS+=(--use-torch-optimizer-for-cpu-offload)
[ "${OVERLAP_CPU_OPTIMIZER_D2H_H2D}" = "1" ] && OPTIMIZER_ARGS+=(--overlap-cpu-optimizer-d2h-h2d)

FSDP_ARGS=()
[ "${USE_MEGATRON_FSDP}" = "1" ] && FSDP_ARGS=(--use-megatron-fsdp --data-parallel-sharding-strategy "${DATA_PARALLEL_SHARDING_STRATEGY}" --ckpt-format fsdp_dtensor)

SAVE_ARGS=()
[ "${DISABLE_SAVE}" != "1" ] && SAVE_ARGS=(--save "${OUTPUT_PATH}/checkpoints" --save-interval "${SAVE_INTERVAL}")

GRAD_DTYPE_ARGS=()
[ "${GRAD_REDUCE_IN_BF16}" = "1" ] && GRAD_DTYPE_ARGS+=(--grad-reduce-in-bf16)
[ "${ACCUMULATE_ALLREDUCE_GRADS_IN_FP32}" = "1" ] && GRAD_DTYPE_ARGS+=(--accumulate-allreduce-grads-in-fp32)

OPTIONAL_ARGS=()
[ "${OVERLAP_GRAD_REDUCE}" = "1" ] && OPTIONAL_ARGS+=(--overlap-grad-reduce)
[ "${OVERLAP_PARAM_GATHER}" = "1" ] && OPTIONAL_ARGS+=(--overlap-param-gather)
[ "${CROSS_ENTROPY_LOSS_FUSION}" = "1" ] && OPTIONAL_ARGS+=(--cross-entropy-loss-fusion --cross-entropy-fusion-impl "${CROSS_ENTROPY_FUSION_IMPL}")
[ -n "${ATTENTION_BACKEND}" ] && OPTIONAL_ARGS+=(--attention-backend "${ATTENTION_BACKEND}")
if [ "${DSV4_EP_BACKEND}" = "primus_turboep" ]; then
  [ "${PRIMUS_TURBO_ENABLE_FULL_STACK:-0}" = "1" ] && OPTIONAL_ARGS+=(--enable-primus-turbo)
  OPTIONAL_ARGS+=(--use-turbo-deepep --turbo-deepep-num-cu "${TURBO_DEEPEP_NUM_CU:-64}")
  [ "${DISABLE_COMPILE_DEPENDENCIES}" = "1" ] && OPTIONAL_ARGS+=(--disable-compile-dependencies)
  [ "${USE_TURBO_GROUPED_GEMM:-0}" = "1" ] && OPTIONAL_ARGS+=(--use-turbo-grouped-gemm)
  [ "${TURBO_DEEPEP_USE_COMM_STREAM:-0}" = "1" ] && OPTIONAL_ARGS+=(--turbo-deepep-use-comm-stream)
fi
[ -n "${EXTRA_MEGATRON_ARGS}" ] && read -r -a EXTRA_ARGS <<< "${EXTRA_MEGATRON_ARGS}" && OPTIONAL_ARGS+=("${EXTRA_ARGS[@]}")

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
  "${FSDP_ARGS[@]}"
  "${OPTIONAL_ARGS[@]}"
  --seq-length "${SEQ_LEN}"
  --max-position-embeddings "${SEQ_LEN}"
  --micro-batch-size "${MBS}"
  --global-batch-size "${GBS}"
  --train-iters "${TRAIN_ITERS}"
  --eval-iters 0
  --eval-interval "${TRAIN_ITERS}"
  --optimizer adam
  --lr 1e-6
  --min-lr 1e-7
  --lr-decay-style constant
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.98
  "${OPTIMIZER_ARGS[@]}"
  --clip-grad 1.0
  --init-method-std 0.02
  --bf16
  "${RECOMPUTE_ARGS[@]}"
  --transformer-impl transformer_engine
  --attention-softmax-in-fp32
  "${GRAD_DTYPE_ARGS[@]}"
  --seed "${SEED}"
  --data-cache-path "${OUTPUT_PATH}/cache"
  --mock-data
  --tokenizer-type NullTokenizer
  "${SAVE_ARGS[@]}"
  --tensorboard-dir "${OUTPUT_PATH}/tensorboard"
  --log-throughput
  --log-interval 1
  --no-check-for-nan-in-loss-and-grad
  --manual-gc
  --manual-gc-interval 10
)

CMD=(torchrun "${DISTRIBUTED_ARGS[@]}" "${MEGATRON_PATH}/pretrain_gpt.py" "${TRAIN_ARGS[@]}")
printf 'Megatron-LM path: %s\nOutput path: %s\nCommand:\n' "${MEGATRON_PATH}" "${OUTPUT_PATH}"
printf ' %q' "${CMD[@]}"
printf '\n'
[ "${DRY_RUN}" = "1" ] && exit 0

mkdir -p "${OUTPUT_PATH}/tensorboard" "${OUTPUT_PATH}/cache"
[ "${DISABLE_SAVE}" != "1" ] && mkdir -p "${OUTPUT_PATH}/checkpoints"
cd "${MEGATRON_PATH}"
"${CMD[@]}"
