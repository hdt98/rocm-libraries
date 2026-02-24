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

# Test parameters - sweep multiple sizes
TEST_SIZES=(1024 2048 4096)
ITERS=1

echo "Test configuration:"
echo "  Matrix sizes: ${TEST_SIZES[@]}"
echo "  Iterations per test: ${ITERS}"
echo ""

# Function to rebuild with specific graph mode
rebuild_with_mode() {
    local MODE=$1
    local MODE_NAME=$2
    
    echo "======================================================================="
    echo "Building with LATRD_GRAPH_MODE=${MODE} (${MODE_NAME})"
    echo "======================================================================="
    
    # Remove build directory for clean rebuild
    rm -rf "${BUILD_DIR}"
    
    # Change to rocsolver directory
    cd "${ROCSOLVER_DIR}"
    
    # Build using install.sh with cmake-arg to pass LATRD_GRAPH_MODE
    echo ""
    echo "Running install.sh with options:"
    echo "  -c (build clients)"
    echo "  -n (no optimizations)"
    echo "  -a gfx950 (architecture)"
    echo "  --cmake-arg -DCMAKE_CXX_FLAGS=-DLATRD_GRAPH_MODE=${MODE}"
    echo ""
    
    ./install.sh -cn -a gfx950 --cmake-arg "-DCMAKE_CXX_FLAGS=-DLATRD_GRAPH_MODE=${MODE}"
    
    # Verify test binary exists
    if [ ! -f "$TEST_BINARY" ]; then
        echo "Error: Test binary not found at $TEST_BINARY"
        exit 1
    fi
    
    echo ""
    echo "Build complete for MODE=${MODE}"
    echo ""
    
    # Run tests for each size
    for SIZE in "${TEST_SIZES[@]}"; do
        echo ""
        echo "-----------------------------------------------------------------------"
        echo "Running LATRD_FORSYTRD benchmark: MODE=${MODE}, SIZE=${SIZE}"
        echo "-----------------------------------------------------------------------"
        
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
echo "# TEST SUITE: Comparing HIP Graph Modes Across Multiple Sizes        #"
echo "####################################################################### "
echo ""

rebuild_with_mode 0 "No graphs (baseline)"
rebuild_with_mode 1 "One graph per iteration"
rebuild_with_mode 4 "Four iterations per graph (default)"
rebuild_with_mode -1 "Single graph for all iterations"

echo "======================================================================="
echo "TIMING COMPARISON COMPLETE"
echo "======================================================================="
echo ""
echo "Summary:"
echo "  - Tested modes: 0 (baseline), 1, 4, -1"
echo "  - Tested sizes: ${TEST_SIZES[@]}"
echo "  - Look for '[LATRD_GRAPH TIMING]' and '[LATRD_NON-GRAPH TIMING]' in output above"
echo ""
echo "Expected findings:"
echo "  1. Compare 'Total capture time' vs 'Total instantiation time' vs 'Total launch time'"
echo "  2. Compare graph execution time to non-graph execution time"
echo "  3. Identify which component (capture/instantiate/launch) dominates overhead"
echo ""
