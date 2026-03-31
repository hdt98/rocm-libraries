#!/usr/bin/env bash
set -u
set -o pipefail

# Source-tree runner for gfx1250 MXFP8 POC suite.
#
# Example:
#   ./run_mxfp8_gfx1250_poc.sh
#   ./run_mxfp8_gfx1250_poc.sh --exec-folder /path/to/clients --samples 1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PERF_ENTRY="${SCRIPT_DIR}/hipblaslt-perf"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DEFAULT_EXEC_FOLDER="${REPO_ROOT}/build/release/clients"
WORKSPACE="${REPO_ROOT}/hipBLASLt_benchmark/mxfp8_gfx1250_poc"
EXEC_FOLDER="${DEFAULT_EXEC_FOLDER}"
SAMPLES=1
TARGET_ARCH="gfx1250"
TAG="local"

# Keep virtualenv python/tools ahead to satisfy generated client scripts.
export PATH="/opt/venv/bin:/opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel/lib/llvm/bin:${PATH}"

usage() {
  cat <<'EOF'
run_mxfp8_gfx1250_poc.sh options:
  --workspace <dir>      Output workspace (default: <repo>/hipBLASLt_benchmark/mxfp8_gfx1250_poc)
  --exec-folder <dir>    Folder containing hipblaslt-bench (default: <repo>/build/release/clients)
  --samples <n>          Number of perf samples (default: 1)
  --target-arch <arch>   targetArch passed to hipblaslt-perf (default: gfx1250)
  --tag <name>           Output tag subfolder (default: local)
  --help                 Show this message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workspace) WORKSPACE="$2"; shift 2 ;;
    --exec-folder) EXEC_FOLDER="$2"; shift 2 ;;
    --samples) SAMPLES="$2"; shift 2 ;;
    --target-arch) TARGET_ARCH="$2"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    --help) usage; exit 0 ;;
    *)
      echo "[ERROR] Unknown argument: $1"
      usage
      exit 2
      ;;
  esac
done

if [[ ! -x "${PERF_ENTRY}" ]]; then
  echo "[ERROR] hipblaslt-perf not executable: ${PERF_ENTRY}"
  exit 3
fi

if [[ ! -x "${EXEC_FOLDER}/hipblaslt-bench" ]]; then
  echo "[ERROR] hipblaslt-bench not found in exec folder: ${EXEC_FOLDER}"
  echo "[HINT] Build clients first or pass --exec-folder /path/to/clients"
  exit 4
fi

mkdir -p "${WORKSPACE}"

echo "[INFO] repo root     : ${REPO_ROOT}"
echo "[INFO] perf entry    : ${PERF_ENTRY}"
echo "[INFO] exec folder   : ${EXEC_FOLDER}"
echo "[INFO] workspace     : ${WORKSPACE}"
echo "[INFO] target arch   : ${TARGET_ARCH}"
echo "[INFO] samples       : ${SAMPLES}"
echo "[INFO] tag           : ${TAG}"
echo "[INFO] suite         : mxfp8_gfx1250_poc"

"${PERF_ENTRY}" \
  -w "${WORKSPACE}" \
  -e "${EXEC_FOLDER}" \
  --suite mxfp8_gfx1250_poc \
  --targetArch "${TARGET_ARCH}" \
  --samples "${SAMPLES}" \
  --tag "${TAG}" \
  --csv

rc=$?
if [[ ${rc} -ne 0 ]]; then
  echo "[FAIL] hipblaslt-perf exited with ${rc}"
  exit ${rc}
fi

latest_csv="$(ls -1t "${WORKSPACE}"/hipBLASLt_PTS_Benchmarks_matmul/"${TAG}"/*.csv 2>/dev/null | awk 'NR==1{print;exit}')"
if [[ -z "${latest_csv}" ]]; then
  echo "[FAIL] CSV output not found under ${WORKSPACE}/hipBLASLt_PTS_Benchmarks_matmul/${TAG}"
  exit 5
fi

line_count="$(awk 'END{print NR}' "${latest_csv}")"
if [[ "${line_count}" -le 1 ]]; then
  echo "[FAIL] CSV has no data rows: ${latest_csv}"
  exit 6
fi

echo "[PASS] Generated ${latest_csv} (rows=${line_count})"
