#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Script to generate self-contained include tests for all headers in include/ck_tile/ops/gemm/

# Get the repository root (3 levels up from this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
GEMM_INCLUDE_DIR="$REPO_ROOT/include/ck_tile/ops/gemm"
TEST_OUTPUT_DIR="$REPO_ROOT/test/ck_tile/gemm/include_tests"

# Create output directory
mkdir -p "$TEST_OUTPUT_DIR"

echo "Generating include tests for headers in: $GEMM_INCLUDE_DIR"
echo "Output directory: $TEST_OUTPUT_DIR"

# Find all .hpp files in the gemm directory
find "$GEMM_INCLUDE_DIR" -name "*.hpp" | while read -r header_file; do
    # Get relative path from include/ck_tile/ops/gemm/
    rel_path="${header_file#$GEMM_INCLUDE_DIR/}"

    # Create a sanitized name for the test file (replace / with _ and remove .hpp)
    test_name=$(echo "$rel_path" | sed 's/\//_/g' | sed 's/\.hpp$//')
    test_file="$TEST_OUTPUT_DIR/test_include_${test_name}.cpp"

    # Get the include path relative to the include/ directory
    include_path="ck_tile/ops/gemm/$rel_path"

    # Generate the test file
    cat > "$test_file" <<EOF
// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Test that $include_path is self-contained
#include "$include_path"

int main()
{
    // Minimal test - if this compiles, the header is self-contained
    return 0;
}
EOF

    echo "Generated: $test_file"
done

echo ""
echo "Done! Generated tests in: $TEST_OUTPUT_DIR"
echo ""
echo "Now update CMakeLists.txt to add these targets."
