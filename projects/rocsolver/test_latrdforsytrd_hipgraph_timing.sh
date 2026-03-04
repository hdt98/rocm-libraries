#!/bin/bash
# Test script to compare HIP graph timing with different modes
#
# Benchmarks two functions side-by-side for each build:
#   latrd_forsytrd  -- LATRD-level HIP graph caching (existing)
#   sytrd           -- SYTRD-level HIP graph caching (new, SYTRD_GRAPH_MODE=1)
#
# LATRD_GRAPH_MODE values:
#   0   = no HIP graphs (baseline)
#   1   = one graph per LATRD iteration
#   4   = four iterations per graph
#  -1   = single graph for all iterations
#
# SYTRD_GRAPH_MODE values:
#   0   = disabled (default; falls through to LATRD-level caching if set)
#   1   = cache entire SYTRD call as a single graph

set -e

ROCSOLVER_DIR="/home/koneal/open_dev/rocm-libraries/projects/rocsolver"
BUILD_DIR="$ROCSOLVER_DIR/build"
RELEASE_DIR="$BUILD_DIR/release"
TEST_BINARY="$RELEASE_DIR/clients/staging/rocsolver-bench"

echo "======================================================================="
echo "LATRD_FORSYTRD vs SYTRD HIP GRAPH CACHING COMPARISON"
echo "======================================================================="
echo ""

# Test parameters - sweep multiple sizes and buffer intrinsics modes
TEST_SIZES=(1024 2048 4096)
LATRD_ITERS=1   # latrd_forsytrd: LATRD is called many times internally per iter
SYTRD_ITERS=1  # sytrd: outer benchmark iterations to exercise the cache
BUFFER_INTRINSICS_MODES=(0 1)  # 0 = disabled, 1 = enabled

echo "Test configuration:"
echo "  Matrix sizes: ${TEST_SIZES[@]}"
echo "  latrd_forsytrd bench iters: ${LATRD_ITERS}"
echo "  sytrd bench iters         : ${SYTRD_ITERS}"
echo "  Buffer intrinsics modes: ${BUFFER_INTRINSICS_MODES[@]}"
echo ""

# -----------------------------------------------------------------------
# rebuild_with_mode <CMAKE_FLAGS> <LATRD_MODE> <SYTRD_MODE> <MODE_NAME> <BUFFER_INTRINSICS> <BUFFER_NAME>
#
# Rebuilds rocSOLVER with the given CXX flags then runs both
# latrd_forsytrd and sytrd benchmarks for every test size.
#
# CMAKE_FLAGS  : full "-DFOO=x -DBAR=y" string forwarded to CMAKE_CXX_FLAGS
# LATRD_MODE   : numeric LATRD_GRAPH_MODE value (for [TEST_CONFIG] tag)
# SYTRD_MODE   : numeric SYTRD_GRAPH_MODE value (for [TEST_CONFIG] tag)
# MODE_NAME    : human-readable label printed in headers
# -----------------------------------------------------------------------
rebuild_with_mode() {
    local CMAKE_FLAGS=$1
    local LATRD_MODE=$2
    local SYTRD_MODE=$3
    local MODE_NAME=$4
    local BUFFER_INTRINSICS=$5
    local BUFFER_NAME=$6

    echo "======================================================================="
    echo "Building: ${MODE_NAME}"
    echo "          LATRD_USE_BUFFER_INTRINSICS=${BUFFER_INTRINSICS} (${BUFFER_NAME})"
    echo "          CMAKE_CXX_FLAGS=${CMAKE_FLAGS}"
    echo "======================================================================="

    # Remove build directory for clean rebuild
    rm -rf "${BUILD_DIR}"

    cd "${ROCSOLVER_DIR}"

    echo ""
    echo "Running install.sh with options:"
    echo "  -c (build clients)"
    echo "  -n (no optimizations)"
    echo "  -a gfx950 (architecture)"
    echo "  --cmake-arg \"-DCMAKE_CXX_FLAGS=${CMAKE_FLAGS} -DLATRD_USE_BUFFER_INTRINSICS=${BUFFER_INTRINSICS}\""
    echo ""

    ./install.sh -cn -a gfx950 \
        --cmake-arg "-DCMAKE_CXX_FLAGS=${CMAKE_FLAGS} -DLATRD_USE_BUFFER_INTRINSICS=${BUFFER_INTRINSICS}"

    if [ ! -f "$TEST_BINARY" ]; then
        echo "Error: Test binary not found at $TEST_BINARY"
        exit 1
    fi

    echo ""
    echo "Build complete: ${MODE_NAME}, BUFFER_INTRINSICS=${BUFFER_INTRINSICS}"
    echo ""

    # Run both benchmarks for each size
    for SIZE in "${TEST_SIZES[@]}"; do
        echo ""
        echo "-----------------------------------------------------------------------"
        echo "latrd_forsytrd | ${MODE_NAME} | BUFFER_INTRINSICS=${BUFFER_INTRINSICS} | n=${SIZE}"
        echo "-----------------------------------------------------------------------"
        # [TEST_CONFIG] tag is parsed by parse_latrd_graph_timing.py
        echo "[TEST_CONFIG] LATRD_GRAPH_MODE=${LATRD_MODE}, SYTRD_GRAPH_MODE=${SYTRD_MODE}, LATRD_USE_BUFFER_INTRINSICS=${BUFFER_INTRINSICS}, FUNCTION=latrd_forsytrd, SIZE=${SIZE}"
        # --iters 1: one outer benchmark call; LATRD is invoked many times internally by SYTRD
        "$TEST_BINARY" -f latrd_forsytrd -n ${SIZE} -k 64 --iters ${LATRD_ITERS} 2>&1

        echo ""
        echo "-----------------------------------------------------------------------"
        echo "sytrd          | ${MODE_NAME} | BUFFER_INTRINSICS=${BUFFER_INTRINSICS} | n=${SIZE}"
        echo "-----------------------------------------------------------------------"
        # [TEST_CONFIG] tag is parsed by parse_latrd_graph_timing.py
        echo "[TEST_CONFIG] LATRD_GRAPH_MODE=${LATRD_MODE}, SYTRD_GRAPH_MODE=${SYTRD_MODE}, LATRD_USE_BUFFER_INTRINSICS=${BUFFER_INTRINSICS}, FUNCTION=sytrd, SIZE=${SIZE}"
        # --iters 1: one outer benchmark call; LATRD is invoked many times internally by SYTRD
        "$TEST_BINARY" -f sytrd -n ${SIZE} --iters ${SYTRD_ITERS} 2>&1

        echo ""
    done
}

# -----------------------------------------------------------------------
# Main sweep
# -----------------------------------------------------------------------
echo ""
echo "======================================================================="
echo "# TEST SUITE: LATRD graph modes + SYTRD graph mode vs Buffer Intrinsics"
echo "======================================================================="
echo ""

for BUFFER_MODE in "${BUFFER_INTRINSICS_MODES[@]}"; do
    if [ "$BUFFER_MODE" -eq 0 ]; then
        BUFFER_NAME="Buffer intrinsics disabled"
    else
        BUFFER_NAME="Buffer intrinsics enabled"
    fi

    echo ""
    echo "======================================================================"
    echo "TESTING WITH BUFFER_INTRINSICS=${BUFFER_MODE} (${BUFFER_NAME})"
    echo "======================================================================"
    echo ""

    # ── LATRD-level modes (existing) ────────────────────────────────────
    # Args: CMAKE_FLAGS          LATRD_MODE  SYTRD_MODE  MODE_NAME                            BUFFER  BUFFER_NAME
    rebuild_with_mode \
        "-DLATRD_GRAPH_MODE=0"  0           0 \
        "LATRD: No graphs (baseline)" \
        ${BUFFER_MODE} "${BUFFER_NAME}"

    rebuild_with_mode \
        "-DLATRD_GRAPH_MODE=1"  1           0 \
        "LATRD: One graph per iteration" \
        ${BUFFER_MODE} "${BUFFER_NAME}"

    rebuild_with_mode \
        "-DLATRD_GRAPH_MODE=4"  4           0 \
        "LATRD: Four iterations per graph" \
        ${BUFFER_MODE} "${BUFFER_NAME}"

    rebuild_with_mode \
        "-DLATRD_GRAPH_MODE=-1" -1          0 \
        "LATRD: Single graph for all iterations" \
        ${BUFFER_MODE} "${BUFFER_NAME}"

    # ── SYTRD-level mode (new) ───────────────────────────────────────────
    # LATRD_GRAPH_MODE=0 so that LATRD does not independently try to use graphs
    # inside the SYTRD-level capture region.
    rebuild_with_mode \
        "-DSYTRD_GRAPH_MODE=1 -DLATRD_GRAPH_MODE=0"  0  1 \
        "SYTRD: Full-call graph cache (SYTRD_GRAPH_MODE=1)" \
        ${BUFFER_MODE} "${BUFFER_NAME}"

done

echo "======================================================================="
echo "TIMING COMPARISON COMPLETE"
echo "======================================================================="
echo ""
echo "Summary:"
echo "  LATRD modes tested : 0 (baseline), 1, 4, -1"
echo "  SYTRD modes tested : 1 (full-call cache)"
echo "  Sizes              : ${TEST_SIZES[@]}"
echo "  Buffer intrinsics  : ${BUFFER_INTRINSICS_MODES[@]}"
echo "  Total builds       : $((5 * ${#BUFFER_INTRINSICS_MODES[@]}))"
echo ""
echo "Look for:"
echo "  [LATRD_GRAPH TIMING]     -- LATRD graph path active"
echo "  [LATRD_NON-GRAPH TIMING] -- LATRD non-graph (baseline) path"
echo "  [SYTRD CACHE HIT/MISS]   -- SYTRD-level graph cache activity"
echo "  cpu_time_us / gpu_time_us rows -- rocsolver-bench wall-clock results"
echo ""
echo "Expected findings:"
echo "  1. LATRD modes: compare capture / instantiate / launch overhead per iteration"
echo "  2. SYTRD mode:  single cache entry per (n, uplo) vs ~n/64 LATRD entries"
echo "  3. Buffer intrinsics: independent impact on each caching mode"
echo ""
