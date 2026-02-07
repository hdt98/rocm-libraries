#!/bin/bash
# Test build a single target and capture error

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${REPO_ROOT}/build"
TARGET="${1:-test_include_pipeline_gemm_pipeline_problem}"

cd "$BUILD_DIR"

echo "Building $TARGET..."
ninja "$TARGET" 2>&1 | tee "${REPO_ROOT}/build_error_${TARGET}.log"
