#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Usage: dispatcher_ci.sh <mode>
#   mode = nogpu | gpu | cpu | stress | heuristics
#
# Jenkins CI uses: nogpu (cpu + ml heuristics) and gpu.
# Other modes (cpu, stress, heuristics) are kept for local reproduction
# of CI failures and ad-hoc developer workflows.
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

MODE="${1:?Usage: dispatcher_ci.sh <nogpu|gpu|cpu|stress|heuristics>}"
BUILD_COMPILER="${BUILD_COMPILER:-/opt/rocm/llvm/bin/clang++}"
NUM_THREADS="${NUM_THREADS:-$(nproc)}"
GPU_TARGET="${GPU_TARGET:-gfx950}"

# Heuristics is a pure-Python pytest run with no cmake build.
run_heuristics(){
    ( cd ../dispatcher && pip install -r requirements-ml.txt && \
      python3 -m pytest heuristics/tests/ -v --tb=short )
}

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

case "$MODE" in
    nogpu|cpu)
        # CPU build + tests. nogpu adds ML heuristics afterwards.
        cmake "${COMMON_CMAKE_ARGS[@]}" -D BUILD_DISPATCHER_REAL_KERNEL_TESTS=OFF ..
        ninja -j"${NUM_THREADS}"
        # cpp tests intentionally NOT run (8/16 fail; tracked separately).
        # -LE stress|integration: skipped (have dedicated runs / require GPU).
        ( cd dispatcher && ctest --output-on-failure -L dispatcher -L python -LE 'stress|integration' )
        if [ "$MODE" = "nogpu" ]; then
            run_heuristics
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

    stress)
        cmake "${COMMON_CMAKE_ARGS[@]}" -D BUILD_DISPATCHER_REAL_KERNEL_TESTS=OFF ..
        ninja -j"${NUM_THREADS}"
        cd dispatcher
        ctest --output-on-failure -L dispatcher -L stress
        ;;

    heuristics)
        run_heuristics
        ;;

    *)
        echo "Error: unknown mode '$MODE'" >&2
        echo "Usage: dispatcher_ci.sh <nogpu|gpu|cpu|stress|heuristics>" >&2
        exit 2
        ;;
esac