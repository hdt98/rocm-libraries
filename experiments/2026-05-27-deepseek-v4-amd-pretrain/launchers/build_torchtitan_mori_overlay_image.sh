#!/usr/bin/env bash
set -euo pipefail

TARGET_HOST="${TARGET_HOST:-do-sonle5-mi350-gpu}"
BASE_IMAGE="${BASE_IMAGE:-onenexus/nexus-titan:rocm722-pytorch-nightly}"
IMAGE_TAG="${IMAGE_TAG:-onenexus/nexus-titan:rocm722-pytorch-nightly-mori}"
MORI_GPU_ARCHS="${MORI_GPU_ARCHS:-gfx950}"
MORI_ENABLE_PROFILER="${MORI_ENABLE_PROFILER:-OFF}"
MORI_REMOTE="${MORI_REMOTE:-https://github.com/ROCm/mori.git}"
MORI_COMMIT="${MORI_COMMIT:-5dd02d6815e9113381a72f6de6b22034fc09b5a3}"
MORI_SOURCE_MODE="${MORI_SOURCE_MODE:-git}"
REMOTE_ROOT="${REMOTE_ROOT:-/scratch/sonle5/dsv4_pretrain_canary_20260527}"
REMOTE_BUILD_DIR="${REMOTE_BUILD_DIR:-${REMOTE_ROOT}/deps/torchtitan_mori_overlay_image_git}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOCAL_MORI_DIR="${LOCAL_MORI_DIR:-${EXP_DIR}/sources/references/mori}"
LOCAL_DOCKERFILE="${LOCAL_DOCKERFILE:-${SCRIPT_DIR}/Dockerfile.torchtitan_mori_overlay}"

if [[ ! -d "${LOCAL_MORI_DIR}/python/mori" ]]; then
  echo "Missing local MORI source snapshot: ${LOCAL_MORI_DIR}" >&2
  exit 2
fi
if [[ ! -f "${LOCAL_DOCKERFILE}" ]]; then
  echo "Missing Dockerfile: ${LOCAL_DOCKERFILE}" >&2
  exit 2
fi

ssh "${TARGET_HOST}" "mkdir -p '${REMOTE_BUILD_DIR}'"
rsync -a "${LOCAL_DOCKERFILE}" "${TARGET_HOST}:${REMOTE_BUILD_DIR}/Dockerfile"

case "${MORI_SOURCE_MODE}" in
  git)
    ssh "${TARGET_HOST}" \
      "REMOTE_BUILD_DIR='${REMOTE_BUILD_DIR}' MORI_REMOTE='${MORI_REMOTE}' MORI_COMMIT='${MORI_COMMIT}' bash -s" <<'REMOTE'
set -euo pipefail
cd "${REMOTE_BUILD_DIR}"
if [[ -d mori && ! -d mori/.git ]]; then
  echo "Existing non-git mori directory in ${REMOTE_BUILD_DIR}; use a fresh REMOTE_BUILD_DIR or clean it manually." >&2
  exit 2
fi
if [[ ! -d mori/.git ]]; then
  git clone "${MORI_REMOTE}" mori
fi
git -C mori fetch origin "${MORI_COMMIT}"
git -C mori checkout "${MORI_COMMIT}"
git -C mori submodule update --init --recursive 3rdparty/spdlog 3rdparty/msgpack-c
REMOTE
    ;;
  local)
    ssh "${TARGET_HOST}" \
      "REMOTE_BUILD_DIR='${REMOTE_BUILD_DIR}' MORI_REMOTE='${MORI_REMOTE}' MORI_COMMIT='${MORI_COMMIT}' bash -s" <<'REMOTE'
set -euo pipefail
cd "${REMOTE_BUILD_DIR}"
if [[ -d mori && ! -d mori/.git ]]; then
  echo "Existing non-git mori directory in ${REMOTE_BUILD_DIR}; use a fresh REMOTE_BUILD_DIR or clean it manually." >&2
  exit 2
fi
if [[ ! -d mori/.git ]]; then
  git clone "${MORI_REMOTE}" mori
fi
git -C mori fetch origin "${MORI_COMMIT}"
git -C mori checkout "${MORI_COMMIT}"
git -C mori submodule update --init --recursive 3rdparty/spdlog 3rdparty/msgpack-c
REMOTE
    rsync -a \
      --exclude='.git' \
      --exclude='3rdparty/spdlog' \
      --exclude='3rdparty/msgpack-c' \
      "${LOCAL_MORI_DIR}/" "${TARGET_HOST}:${REMOTE_BUILD_DIR}/mori/"
    ssh "${TARGET_HOST}" "REMOTE_BUILD_DIR='${REMOTE_BUILD_DIR}' bash -s" <<'REMOTE'
set -euo pipefail
cd "${REMOTE_BUILD_DIR}/mori"
if [[ -d python/mori/_jit-sources/src && -d src ]]; then
  rsync -a src/ python/mori/_jit-sources/src/
fi
REMOTE
    ;;
  *)
    echo "Invalid MORI_SOURCE_MODE=${MORI_SOURCE_MODE}; expected git or local" >&2
    exit 2
    ;;
esac

ssh "${TARGET_HOST}" \
  "cd '${REMOTE_BUILD_DIR}' && docker build --pull=false --build-arg BASE_IMAGE='${BASE_IMAGE}' --build-arg MORI_GPU_ARCHS='${MORI_GPU_ARCHS}' --build-arg MORI_ENABLE_PROFILER='${MORI_ENABLE_PROFILER}' -t '${IMAGE_TAG}' -f Dockerfile ."

ssh "${TARGET_HOST}" \
  "docker run --rm --network=host --ipc=host --entrypoint /bin/bash '${IMAGE_TAG}' -lc 'python3 - <<\"PY\"
import aiter
import mori

print(\"mori\", mori.__version__, mori.__file__)
print(\"aiter\", getattr(aiter, \"__file__\", \"builtin\"))
PY'"

echo "Built ${IMAGE_TAG} on ${TARGET_HOST}"
