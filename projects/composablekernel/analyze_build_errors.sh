#!/bin/bash
# Analyze build errors for failed include tests

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${REPO_ROOT}/build"
FAILED_FILE="${REPO_ROOT}/failed_include_test_targets.txt"
ERROR_DIR="${REPO_ROOT}/build_errors"

mkdir -p "$ERROR_DIR"
cd "$BUILD_DIR"

echo "Analyzing build errors..."
echo ""

# Try to build each failed target and capture the first error
while IFS= read -r target; do
    if [ -n "$target" ]; then
        echo "Analyzing $target..."
        ERROR_FILE="${ERROR_DIR}/${target}_error.txt"

        ninja "$target" 2>&1 | tee "$ERROR_FILE" | grep -A 5 "error:" | head -20

        echo "Error log saved to: $ERROR_FILE"
        echo "---"
    fi
done < "$FAILED_FILE"

echo ""
echo "Analysis complete. Error logs in: $ERROR_DIR"
