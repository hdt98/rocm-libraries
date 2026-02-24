#!/bin/bash
# Test script to compare HIP graph timing with different modes

set -e

ROCSOLVER_DIR="/home/koneal/open_dev/rocm-libraries/projects/rocsolver"
BUILD_DIR="$ROCSOLVER_DIR/build"
RELEASE_DIR="$BUILD_DIR/release"
TEST_BINARY="$RELEASE_DIR/clients/staging/rocsolver-bench"

echo "======================================================================="
echo "LATRD_FORSYTRD HIP GRAPH TIMING COMPARISON"
echo "======================================================================="
echo ""

# Test parameters - sweep multiple sizes and buffer intrinsics modes
TEST_SIZES=(1024 2048 4096)
ITERS=1
BUFFER_INTRINSICS_MODES=(0 1)  # 0 = disabled, 1 = enabled

echo "Test configuration:"
echo "  Matrix sizes: ${TEST_SIZES[@]}"
echo "  Iterations per test: ${ITERS}"
echo "  Buffer intrinsics modes: ${BUFFER_INTRINSICS_MODES[@]}"
echo ""

# Function to rebuild with specific graph mode and buffer intrinsics setting
rebuild_with_mode() {
    local MODE=$1
    local MODE_NAME=$2
    local BUFFER_INTRINSICS=$3
    local BUFFER_NAME=$4
    
    echo "======================================================================="
    echo "Building with LATRD_GRAPH_MODE=${MODE} (${MODE_NAME})"
    echo "              LATRD_USE_BUFFER_INTRINSICS=${BUFFER_INTRINSICS} (${BUFFER_NAME})"
    echo "======================================================================="
    
    # Remove build directory for clean rebuild
    rm -rf "${BUILD_DIR}"
    
    # Change to rocsolver directory
    cd "${ROCSOLVER_DIR}"
    
    # Build using install.sh with cmake-arg to pass both LATRD_GRAPH_MODE and LATRD_USE_BUFFER_INTRINSICS
    echo ""
    echo "Running install.sh with options:"
    echo "  -c (build clients)"
    echo "  -n (no optimizations)"
    echo "  -a gfx950 (architecture)"
    echo "  --cmake-arg -DCMAKE_CXX_FLAGS=-DLATRD_GRAPH_MODE=${MODE} -DLATRD_USE_BUFFER_INTRINSICS=${BUFFER_INTRINSICS}"
    echo ""
    
    ./install.sh -cn -a gfx950 --cmake-arg "-DCMAKE_CXX_FLAGS=-DLATRD_GRAPH_MODE=${MODE} -DLATRD_USE_BUFFER_INTRINSICS=${BUFFER_INTRINSICS}"
    
    # Verify test binary exists
    if [ ! -f "$TEST_BINARY" ]; then
        echo "Error: Test binary not found at $TEST_BINARY"
        exit 1
    fi
    
    echo ""
    echo "Build complete for MODE=${MODE}, BUFFER_INTRINSICS=${BUFFER_INTRINSICS}"
    echo ""
    
    # Run tests for each size
    for SIZE in "${TEST_SIZES[@]}"; do
        echo ""
        echo "-----------------------------------------------------------------------"
        echo "Running LATRD_FORSYTRD benchmark: MODE=${MODE}, BUFFER_INTRINSICS=${BUFFER_INTRINSICS}, SIZE=${SIZE}"
        echo "-----------------------------------------------------------------------"
        
        # Print the configuration as a parseable line for the analysis script
        echo "[TEST_CONFIG] LATRD_GRAPH_MODE=${MODE}, LATRD_USE_BUFFER_INTRINSICS=${BUFFER_INTRINSICS}, SIZE=${SIZE}"
        
        # Run the benchmark (--iters 1 means one benchmark run, but LATRD will still
        # be called multiple times internally as needed for the SYTRD algorithm)
        # Capture [LATRD timing lines and the benchmark timing table (header + following line)
        # Use grep -A1 to capture the line immediately after cpu_time_us header
        "$TEST_BINARY" -f latrd_forsytrd -n ${SIZE} --iters ${ITERS} 2>&1 | grep -E "\[LATRD|cpu_time_us" -A1
        
        echo ""
    done
}

# Test different modes
echo ""
echo "####################################################################### "
echo "# TEST SUITE: Comparing HIP Graph Modes with Buffer Intrinsics       #"
echo "####################################################################### "
echo ""

# Sweep through all combinations
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
    
    rebuild_with_mode 0 "No graphs (baseline)" ${BUFFER_MODE} "${BUFFER_NAME}"
    rebuild_with_mode 1 "One graph per iteration" ${BUFFER_MODE} "${BUFFER_NAME}"
    rebuild_with_mode 4 "Four iterations per graph (default)" ${BUFFER_MODE} "${BUFFER_NAME}"
    rebuild_with_mode -1 "Single graph for all iterations" ${BUFFER_MODE} "${BUFFER_NAME}"
done

echo "======================================================================="
echo "TIMING COMPARISON COMPLETE"
echo "======================================================================="
echo ""
echo "Summary:"
echo "  - Tested modes: 0 (baseline), 1, 4, -1"
echo "  - Tested sizes: ${TEST_SIZES[@]}"
echo "  - Tested buffer intrinsics: ${BUFFER_INTRINSICS_MODES[@]}"
echo "  - Total configurations: $((4 * ${#TEST_SIZES[@]} * ${#BUFFER_INTRINSICS_MODES[@]}))"
echo "  - Look for '[LATRD_GRAPH TIMING]' and '[LATRD_NON-GRAPH TIMING]' in output above"
echo ""
echo "Expected findings:"
echo "  1. Compare 'Total capture time' vs 'Total instantiation time' vs 'Total launch time'"
echo "  2. Compare graph execution time to non-graph execution time"
echo "  3. Identify which component (capture/instantiate/launch) dominates overhead"
echo "  4. Evaluate impact of buffer intrinsics on performance"
echo ""
