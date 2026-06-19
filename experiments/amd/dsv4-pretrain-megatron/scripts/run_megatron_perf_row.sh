#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-/local/data/sonle5/dsv4_pretrain_rl}"
SRC="${SRC:-${BASE}/source}"
IMAGE="${IMAGE:-sonle5/dsv4-pr1300-megatron-pretrain:rocm720-mi35x-20260618}"

GPUS="${GPUS:-0,5,6,7}"
LOGICAL_GPUS="${LOGICAL_GPUS:-0,1,2,3}"
EP_BACKEND="${EP_BACKEND:-standard}"
MHC="${MHC:-on}"
HC_MULT="${HC_MULT:-4}"
MBS="${MBS:-2}"
GBS="${GBS:-128}"
EP="${EP:-4}"
TP="${TP:-1}"
NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-4}"
GPU_LABEL="${GPU_LABEL:-${NUM_GPUS_PER_NODE}xmi350}"
TRAIN_ITERS="${TRAIN_ITERS:-5}"
DISABLE_RECOMPUTE="${DISABLE_RECOMPUTE:-0}"
ATTN_BACKEND="${ATTN_BACKEND:-}"

[ "${MHC}" = "off" ] && HC_MULT=1

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LABEL="megatron_${EP_BACKEND}_no_mtp_mhc${MHC}_tp${TP}ep${EP}_mbs${MBS}_gbs${GBS}_${GPU_LABEL}_${STAMP}"
RUN_DIR="${BASE}/runs/${LABEL}"
mkdir -p "${RUN_DIR}"

git_sha_or_none() {
  local path="$1"
  [ -d "${path}/.git" ] && git -C "${path}" rev-parse HEAD 2>/dev/null || true
}

source_fingerprint_or_none() {
  local path="$1"
  [ -d "${path}/experiments/amd/dsv4-pretrain-megatron" ] || return 0
  cd "${path}"
  find experiments/amd/dsv4-pretrain-megatron -type f ! -path '*/__pycache__/*' ! -path '*/MEASUREMENTS.md' ! -path '*/README.md' -print \
    | sort | xargs sha256sum | sha256sum | awk '{print $1}'
}

SOURCE_SHA="${SOURCE_SHA:-$(git_sha_or_none "${SRC}")}"
SOURCE_FINGERPRINT="${SOURCE_FINGERPRINT:-$(source_fingerprint_or_none "${SRC}")}"
MILES_SHA="${MILES_SHA:-$(git_sha_or_none "${BASE}/deps/miles-pr1300-full")}"
PRIMUS_SHA="${PRIMUS_SHA:-$(git_sha_or_none "${SRC}/projects/primus")}"
HOSTNAME_VALUE="$(hostname)"

cat > "${RUN_DIR}/run_metadata.json" <<JSON
{
  "label": "${LABEL}",
  "run_dir": "${RUN_DIR}",
  "node": "${HOSTNAME_VALUE}",
  "image": "${IMAGE}",
  "source_sha": "${SOURCE_SHA}",
  "source_fingerprint": "${SOURCE_FINGERPRINT}",
  "miles_sha": "${MILES_SHA}",
  "primus_sha": "${PRIMUS_SHA}",
  "selected_gpus": "${GPUS}",
  "logical_gpus": "${LOGICAL_GPUS}",
  "ep_backend": "${EP_BACKEND}",
  "mhc": "${MHC}",
  "hc_mult": "${HC_MULT}",
  "tp": ${TP},
  "ep": ${EP},
  "mbs": ${MBS},
  "gbs": ${GBS},
  "train_iters": ${TRAIN_ITERS}
}
JSON

cp "${RUN_DIR}/run_metadata.json" "${RUN_DIR}/status.json"
env | sort > "${RUN_DIR}/host_env.txt"
{
  echo "run_dir=${RUN_DIR}"
  echo "image=${IMAGE}"
  echo "selected_gpus=${GPUS}"
  echo "ep_backend=${EP_BACKEND} mhc=${MHC} hc_mult=${HC_MULT} tp=${TP} ep=${EP} mbs=${MBS} gbs=${GBS} train_iters=${TRAIN_ITERS}"
  date -u +%FT%TZ
} | tee "${RUN_DIR}/launcher.log"

DOCKER_ENV_ARGS=(
  -e HIP_VISIBLE_DEVICES="${LOGICAL_GPUS}"
  -e CUDA_VISIBLE_DEVICES="${LOGICAL_GPUS}"
  -e ROCR_VISIBLE_DEVICES="${GPUS}"
  -e GPU_DEVICE_ORDINAL="${LOGICAL_GPUS}"
  -e MILES_DSV4_SPARSE_MLA_BLOCK_I="${MILES_DSV4_SPARSE_MLA_BLOCK_I:-16}"
  -e MILES_DSV4_SPARSE_MLA_BWD_BLOCK_SIZE="${MILES_DSV4_SPARSE_MLA_BWD_BLOCK_SIZE:-16}"
  -e MILES_DSV4_SPARSE_MLA_BWD_BLOCK_H="${MILES_DSV4_SPARSE_MLA_BWD_BLOCK_H:-32}"
)

pass_env() {
  local name
  for name in "$@"; do
    [ -n "${!name:-}" ] && DOCKER_ENV_ARGS+=(-e "${name}=${!name}")
  done
  return 0
}

pass_env \
  HIP_LAUNCH_BLOCKING AMD_SERIALIZE_KERNEL TORCH_SHOW_CPP_STACKTRACES NCCL_DEBUG RCCL_DEBUG \
  NVTE_USE_HIPBLASLT \
  PYTORCH_CUDA_ALLOC_CONF HSA_ENABLE_SDMA HSA_NO_SCRATCH_RECLAIM PYTORCH_TUNABLEOP_ENABLED \
  DRY_RUN DISABLE_SAVE DISABLE_COMPILE_DEPENDENCIES DSV4_LOGPROB_DUMP_PATH MILES_DSV4_INDEXER_IMPL MILES_DSV4_SPARSE_MLA_IMPL \
  RECOMPUTE_MODULES RECOMPUTE_GRANULARITY USE_MEGATRON_FSDP DATA_PARALLEL_SHARDING_STRATEGY \
  USE_PRECISION_AWARE_OPTIMIZER MAIN_GRADS_DTYPE MAIN_PARAMS_DTYPE EXP_AVG_DTYPE EXP_AVG_SQ_DTYPE \
  OPTIMIZER_CPU_OFFLOAD USE_TORCH_OPTIMIZER_FOR_CPU_OFFLOAD OVERLAP_CPU_OPTIMIZER_D2H_H2D \
  GRAD_REDUCE_IN_BF16 ACCUMULATE_ALLREDUCE_GRADS_IN_FP32 \
  OVERLAP_GRAD_REDUCE OVERLAP_PARAM_GATHER CROSS_ENTROPY_LOSS_FUSION CROSS_ENTROPY_FUSION_IMPL \
  MEGATRON_OVERLAY_PATCHES MILES_OVERLAY_PATCHES TILE_KERNELS_OVERLAY_PATCHES TE_OVERLAY_PATCHES \
  DSV4_EXTRA_PYTHONPATH MEGATRON_PATH EXTRA_MEGATRON_ARGS PRIMUS_TURBOEP_OVERLAY_PATCHES \
  PRIMUS_SOURCE_OVERLAY_PATCHES PRIMUS_TURBO_PACKAGE_OVERLAY_PATCHES \
  PRIMUS_TURBO_IMPORT_FULL_MODULES PRIMUS_TURBO_IMPORT_GEMM \
  PRIMUS_TURBO_ENABLE_FULL_STACK TURBO_DEEPEP_NUM_CU \
  TURBO_DEEPEP_USE_COMM_STREAM TURBO_DEEPEP_DISABLE_ASYNC_FINISH \
  TURBO_DEEPEP_DISABLE_ALLOCATE_ON_COMM_STREAM USE_TURBO_GROUPED_GEMM PRIMUS_TURBO_GEMM_BACKEND \
  PRIMUS_TURBO_GROUPED_GEMM_BACKEND PRIMUS_TURBO_MOE_DISPATCH_COMBINE_BACKEND \
  PRIMUS_TURBO_DEEPEP_TIMEOUT PRIMUS_TURBO_AUTO_TUNE

set +e
docker run --rm \
  --name "${LABEL}" \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  "${DOCKER_ENV_ARGS[@]}" \
  -v "${SRC}":/workspace \
  -v "${BASE}":"${BASE}" \
  -w /workspace \
  "${IMAGE}" \
  bash -lc "
    set -euo pipefail
    env | sort > ${RUN_DIR}/container_env.txt
    export MILES_PLUGIN_PATH=${BASE}/deps/miles-pr1300-full
    OUTPUT_PATH=${RUN_DIR}/output \
    NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE} TP=${TP} EP=${EP} MBS=${MBS} GBS=${GBS} TRAIN_ITERS=${TRAIN_ITERS} \
    DSV4_EP_BACKEND=${EP_BACKEND} DSV4_HC_MULT=${HC_MULT} MTP_NUM_LAYERS=0 \
    ATTENTION_BACKEND=${ATTN_BACKEND} DISABLE_RECOMPUTE=${DISABLE_RECOMPUTE} \
    experiments/amd/dsv4-pretrain-megatron/run_pretrain_dsv4_megatron.sh
  " 2>&1 | tee -a "${RUN_DIR}/train.log"
rc=${PIPESTATUS[0]}
set -e

python3 - <<PY
import json
import pathlib

path = pathlib.Path("${RUN_DIR}/status.json")
status = json.loads(path.read_text())
status["exit_code"] = ${rc}
path.write_text(json.dumps(status, indent=2) + "\n")
PY

RUN_DIR_FOR_PARSER="${RUN_DIR}" python3 - <<'PY'
import json
import os
import pathlib
import re

run_dir = pathlib.Path(os.environ["RUN_DIR_FOR_PARSER"])
train_log = run_dir / "train.log"
metadata_path = run_dir / "run_metadata.json"
status_path = run_dir / "status.json"

def load_json(path):
    return json.loads(path.read_text()) if path.exists() else {}

def load_env(path):
    env = {}
    if not path.exists():
        return env
    for line in path.read_text(errors="ignore").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            env[key] = value
    return env

text = train_log.read_text(errors="ignore") if train_log.exists() else ""
iteration_re = re.compile(
    r"iteration\s+(?P<iteration>\d+)/\s*(?P<total>\d+).*?"
    r"elapsed time per iteration \(ms\):\s*(?P<elapsed_ms>[0-9.+-eE]+).*?"
    r"throughput per GPU \(TFLOP/s/GPU\):\s*(?P<tflops>[0-9.+-eE]+)"
)
memory_re = re.compile(
    r"\[Rank\s+(?P<rank>\d+)\]\s+\(after\s+(?P<after_iter>\d+)\s+iterations\)\s+"
    r"memory \(MB\) \| allocated:\s*(?P<allocated>[0-9.+-eE]+)\s+\| "
    r"max allocated:\s*(?P<max_allocated>[0-9.+-eE]+)\s+\| "
    r"reserved:\s*(?P<reserved>[0-9.+-eE]+)\s+\| "
    r"max reserved:\s*(?P<max_reserved>[0-9.+-eE]+)"
)
retry_re = re.compile(
    r"(?:allocator|allocation|cudaMalloc|hipMalloc).*?"
    r"(?:retr(?:y|ies)|retry count).*?([0-9]+)",
    re.IGNORECASE,
)

iterations = [
    {
        "iteration": int(m.group("iteration")),
        "total_iterations": int(m.group("total")),
        "elapsed_ms": float(m.group("elapsed_ms")),
        "tflops_per_gpu": float(m.group("tflops")),
    }
    for m in iteration_re.finditer(text)
]
memory_metrics = [
    {
        "rank": int(m.group("rank")),
        "after_iter": int(m.group("after_iter")),
        "allocated_mb": float(m.group("allocated")),
        "max_allocated_mb": float(m.group("max_allocated")),
        "reserved_mb": float(m.group("reserved")),
        "max_reserved_mb": float(m.group("max_reserved")),
    }
    for m in memory_re.finditer(text)
]

by_iter = {item["iteration"]: item for item in iterations}

def window(start, end):
    vals = [by_iter[i]["tflops_per_gpu"] for i in range(start, end + 1) if i in by_iter]
    return {
        "start": start,
        "end": end,
        "count": len(vals),
        "avg_tflops_per_gpu": (sum(vals) / len(vals)) if vals else None,
        "iterations": [i for i in range(start, end + 1) if i in by_iter],
    }

retry_matches = [int(m.group(1)) for m in retry_re.finditer(text)]
measurement = {
    **load_json(metadata_path),
    "status": load_json(status_path),
    "exit_code": load_json(status_path).get("exit_code"),
    "iterations": iterations,
    "tflops_windows": {
        "steps_2_20": window(2, 20),
        "steps_13_20": window(13, 20),
    },
    "iter20_tflops_per_gpu": by_iter.get(20, {}).get("tflops_per_gpu"),
    "memory_metrics": memory_metrics,
    "peak_memory_mb": max((m["max_allocated_mb"] for m in memory_metrics), default=None),
    "peak_reserved_mb": max((m["max_reserved_mb"] for m in memory_metrics), default=None),
    "allocator_retries": max(retry_matches) if retry_matches else None,
    "env": {
        "host": load_env(run_dir / "host_env.txt"),
        "container": load_env(run_dir / "container_env.txt"),
    },
    "env_files": {
        "host": str(run_dir / "host_env.txt"),
        "container": str(run_dir / "container_env.txt"),
    },
    "log_path": str(train_log),
}
(run_dir / "measurement.json").write_text(json.dumps(measurement, indent=2, sort_keys=True) + "\n")
PY

exit "${rc}"
