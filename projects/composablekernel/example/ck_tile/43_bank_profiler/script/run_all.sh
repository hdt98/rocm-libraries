#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROFILER_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${CK_BUILD_DIR:-}"

if [ -z "$BUILD_DIR" ]; then
    echo "Usage: CK_BUILD_DIR=/path/to/build bash $0"
    echo "  or:  bash $0 /path/to/build"
    if [ -n "$1" ]; then
        BUILD_DIR="$1"
    else
        echo "ERROR: No build directory specified."
        exit 1
    fi
fi

echo "========================================"
echo "CK Tile GEMM Bank Profiler"
echo "========================================"
echo "Build dir: $BUILD_DIR"
echo "Profiler dir: $PROFILER_DIR"
echo

# Step 1: Build
echo "=== Step 1: Build all profiler binaries ==="
cd "$BUILD_DIR"
make tile_bank_profiler -j$(nproc)
echo

# Step 2: Probe hardware
echo "=== Step 2: Probe hardware bank structure ==="
cd "$PROFILER_DIR"
python3 script/bank_solver.py --build-dir="$BUILD_DIR" --mode=2
echo
python3 script/phase_solver.py --build-dir="$BUILD_DIR" --mode=2
echo

# Step 3: Profile GEMM kernels
echo "=== Step 3: Profile GEMM kernels ==="
python3 script/gemm_profiler.py --build-dir="$BUILD_DIR"
echo

# Step 4: Analyze
echo "=== Step 4: Analyze results ==="
python3 script/analyze_results.py
echo

echo "=== Done! Results in $PROFILER_DIR/out/ ==="
