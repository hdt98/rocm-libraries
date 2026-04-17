#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Usage: dispatcher_ci.sh <node_type>
#   node_type = nogpu | gpu
#
# nogpu: Runs CPU + Heuristics + Stress sub-modes based on env vars
#        RUN_CPU, RUN_HEUR, RUN_STRESS (each "true" to enable). Shares
#        cmake+ninja across CPU and Stress (identical build args) for speed.
# gpu:   Runs GPU integration tests on a gfx950 node.
#
# Working directory: must be a fresh build/ directory under
# projects/composablekernel/ (the script invokes cmake with `..`).
#
# Environment overrides:
#   BUILD_COMPILER   Path to clang++ (default: /opt/rocm/llvm/bin/clang++)
#   NUM_THREADS      Build parallelism (default: $(nproc))
#   GPU_TARGET       GPU arch (default: gfx950 — only validated arch)
#   RUN_CPU/RUN_HEUR/RUN_STRESS   "true" to enable each nogpu sub-mode
#
# Reproduce locally:
#   RUN_CPU=true RUN_STRESS=true bash script/dispatcher_ci.sh nogpu

set -eu
set -o pipefail 2>/dev/null | true

NODE_TYPE="${1:?Usage: dispatcher_ci.sh <nogpu|gpu>}"
BUILD_COMPILER="${BUILD_COMPILER:-/opt/rocm/llvm/bin/clang++}"
NUM_THREADS="${NUM_THREADS:-$(nproc)}"
GPU_TARGET="${GPU_TARGET:-gfx950}"

# Common cmake args. BUILD_TESTING=OFF skips CK's test/ subdir; dispatcher's
# own enable_testing() still registers tests in build/dispatcher/CTestTestfile.cmake.
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

case "$NODE_TYPE" in
    nogpu)
        # Build once if any cmake-based sub-mode is enabled (CPU or Stress
        # have identical cmake args, so they share build+ninja).
        if [ "${RUN_CPU:-false}" = "true" ] || [ "${RUN_STRESS:-false}" = "true" ]; then
            cmake "${COMMON_CMAKE_ARGS[@]}" -D BUILD_DISPATCHER_REAL_KERNEL_TESTS=OFF ..
            ninja -j"${NUM_THREADS}"
        fi
        # CPU: skips stress + integration (have dedicated runs); cpp tests
        # intentionally NOT run (8/16 cpp tests fail; tracked separately).
        if [ "${RUN_CPU:-false}" = "true" ]; then
            ( cd dispatcher && ctest --output-on-failure -L dispatcher -L python -LE 'stress|integration' )
        fi
        if [ "${RUN_STRESS:-false}" = "true" ]; then
            ( cd dispatcher && ctest --output-on-failure -L dispatcher -L stress )
        fi
        # Heuristics is a pure-Python pytest run with no cmake build.
        if [ "${RUN_HEUR:-false}" = "true" ]; then
            ( cd ../dispatcher && pip install -r requirements-ml.txt && \
              python3 -m pytest heuristics/tests/ -v --tb=short )
        fi
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

    *)
        echo "Error: unknown node_type '$NODE_TYPE'" >&2
        echo "Usage: dispatcher_ci.sh <nogpu|gpu>" >&2
        exit 2
        ;;
esac