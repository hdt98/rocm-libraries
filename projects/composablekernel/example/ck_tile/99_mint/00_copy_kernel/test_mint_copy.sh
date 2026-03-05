#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Test script for MINT copy kernel tutorial
# Runs various test cases to validate the implementation

set -e

EXEC=${1:-./bin/test_mint_copy}

echo "======================================"
echo "MINT Copy Kernel Tutorial Test Script"
echo "======================================"
echo ""

# Check if executable exists
if [ ! -f "$EXEC" ]; then
    echo "Error: Executable $EXEC not found!"
    echo "Please build the project first with: make mint_tutorial_copy"
    exit 1
fi

echo "Using executable: $EXEC"
echo ""

echo "========================================="
echo "Test Suite 1: Simple Kernel (kernel=0)"
echo "========================================="
echo ""

echo "Test 1: Simple kernel with various sizes"
$EXEC -kernel 0 -v 1 -warmup 10 -repeat 50
echo ""

echo "========================================="
echo "Test Suite 2: Tiled Kernel (kernel=1)"
echo "========================================="
echo ""

echo "Test 2: Tiled kernel with fixed sizes"
$EXEC -kernel 1 -v 1 -warmup 10 -repeat 50
echo ""

echo "========================================="
echo "Performance Comparison"
echo "========================================="
echo ""

echo "Simple kernel (high iteration count):"
$EXEC -kernel 0 -v 0 -warmup 100 -repeat 500
echo ""

echo "Tiled kernel (high iteration count):"
$EXEC -kernel 1 -v 0 -warmup 100 -repeat 500
echo ""

echo "======================================"
echo "All tests completed successfully!"
echo "======================================"
