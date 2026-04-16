#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Usage: dispatcher_ci.sh <mode>
#   mode = cpu | gpu | stress | heuristics
#
# Configures, builds, and runs dispatcher tests for the requested mode.
# Invoked from the CK Jenkinsfile (one wrapper per dispatcher sub-stage),
# but also runnable locally for reproducing CI failures.
#
# Working directory: must be a fresh build/ directory under
# projects/composablekernel/ (the script invokes cmake with `..`).
#
# Environment overrides:
#   BUILD_COMPILER   Path to clang++ (default: /opt/rocm/llvm/bin/clang++)
#   NUM_THREADS      Build parallelism (default: $(nproc))
#   GPU_TARGET       GPU arch (default: gfx950 — only validated arch)

set -eu
set -o pipefail 2>/dev/null | true

MODE="${1:?Usage: dispatcher_ci.sh <cpu|gpu|stress|heuristics>}"
BUILD_COMPILER="${BUILD_COMPILER:-/opt/rocm/llvm/bin/clang++}"
NUM_THREADS="${NUM_THREADS:-$(nproc)}"
GPU_TARGET="${GPU_TARGET:-gfx950}"

# Heuristics is a pure-Python pytest run with no cmake build.
if [ "$MODE" = "heuristics" ]; then
    cd ../dispatcher
    pip install -r requirements-ml.txt
    exec python3 -m pytest heuristics/tests/ -v --tb=short
fi

# Common cmake args for all build-based modes. BUILD_TESTING=OFF skips CK's
# test/ subdir; dispatcher's own enable_testing() still registers tests in
# build/dispatcher/CTestTestfile.cmake.
COMMON_CMAKE_ARGS=(
    -G Ninja
    -D CMAKE_PREFIX_PATH=/opt/rocm
    -D "CMAKE_CXX_COMPILER=${BUILD_COMPILER}"
    -D "CMAKE_HIP_COMPILER=${BUILD_COMPILER}"
    -D CMAKE_BUILD_TYPE=Release
    -D "GPU_TARGETS=${GPU_TARGET}"
    -D BUILD_CK_DISPATCHER=ON
    -D BUILD_CK_DEVICE_INSTANCES=OFF
    -D BUILD_CK_TILE_ENGINE=OFF
    -D BUILD_CK_EXAMPLES=OFF
    -D BUILD_CK_TUTORIALS=OFF
    -D BUILD_CK_PROFILER=OFF
    -D BUILD_TESTING=OFF
    -D BUILD_DISPATCHER_TESTS=ON
)

case "$MODE" in
    cpu)
        cmake "${COMMON_CMAKE_ARGS[@]}" \
            -D BUILD_DISPATCHER_REAL_KERNEL_TESTS=OFF ..
        ninja -j"${NUM_THREADS}"
        cd dispatcher
        # Multi -L AND-intersects; -LE excludes by label. Skips stress and
        # integration on PR runs (they have dedicated stages).
        ctest --output-on-failure -L dispatcher -L cpp
        ctest --output-on-failure -L dispatcher -L python -LE 'stress|integration'
        ;;

    gpu)
        # cpp;gpu tests skipped pending dispatcher SINGLE_KERNEL_HEADER fix.
        # -LE stress excludes dispatcher_stress_test (labels include integration).
        cmake "${COMMON_CMAKE_ARGS[@]}" \
            -D BUILD_DISPATCHER_REAL_KERNEL_TESTS=OFF \
            -D BUILD_DISPATCHER_EXAMPLES=ON ..
        ninja -j"${NUM_THREADS}"
        cd dispatcher
        ctest --output-on-failure -L dispatcher -L python -L integration -LE stress
        ;;

    stress)
        cmake "${COMMON_CMAKE_ARGS[@]}" \
            -D BUILD_DISPATCHER_REAL_KERNEL_TESTS=OFF ..
        ninja -j"${NUM_THREADS}"
        cd dispatcher
        ctest --output-on-failure -L dispatcher -L stress
        ;;

    *)
        echo "Error: unknown mode '$MODE'" >&2
        echo "Usage: dispatcher_ci.sh <cpu|gpu|stress|heuristics>" >&2
        exit 2
        ;;
esac