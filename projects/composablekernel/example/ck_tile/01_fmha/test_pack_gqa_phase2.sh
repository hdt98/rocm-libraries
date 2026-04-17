#!/bin/bash
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#
# Phase 2 Testing: Pack-GQA with Dropout and Split-KV
# Tests comprehensive stride fix for randval and accumulator buffers

set -e

EXAMPLE=./example_fmha_fwd
PASSED=0
FAILED=0

run_test() {
    local test_name="$1"
    shift
    local args="$@"

    echo "===================================================================="
    echo "TEST: $test_name"
    echo "Args: $args"
    echo "--------------------------------------------------------------------"

    if $EXAMPLE $args; then
        echo "PASS: $test_name"
        ((PASSED++))
    else
        echo "FAIL: $test_name"
        ((FAILED++))
        return 1
    fi
    echo ""
}

echo "========================================================================"
echo "PHASE 2: Pack-GQA with Dropout and Split-KV Testing"
echo "========================================================================"
echo ""

# Category 1: Pack-GQA with Dropout (p_drop > 0)
echo "=== Category 1: Pack-GQA with Dropout (10 tests) ==="
echo ""

run_test "GQA 4:1 + dropout 0.1" \
    -b=2 -h=8 -h_k=2 -s=256 -d=128 -pack_gqa=1 -p_drop=0.1 -v=1

run_test "GQA 8:1 + dropout 0.1" \
    -b=2 -h=32 -h_k=4 -s=512 -d=128 -pack_gqa=1 -p_drop=0.1 -v=1

run_test "GQA 4:1 + dropout 0.2" \
    -b=4 -h=16 -h_k=4 -s=1024 -d=128 -pack_gqa=1 -p_drop=0.2 -v=1

run_test "MQA 8:1 + dropout 0.1" \
    -b=2 -h=8 -h_k=1 -s=256 -d=128 -pack_gqa=1 -p_drop=0.1 -v=1

run_test "GQA 4:1 fp16 + dropout 0.15" \
    -b=2 -h=8 -h_k=2 -s=256 -d=128 -prec=fp16 -pack_gqa=1 -p_drop=0.15 -v=1

run_test "GQA 4:1 d=64 + dropout 0.1" \
    -b=2 -h=8 -h_k=2 -s=256 -d=64 -pack_gqa=1 -p_drop=0.1 -v=1

run_test "GQA 4:1 d=256 + dropout 0.1" \
    -b=2 -h=8 -h_k=2 -s=512 -d=256 -pack_gqa=1 -p_drop=0.1 -v=1

run_test "GQA 4:1 large batch + dropout 0.1" \
    -b=8 -h=16 -h_k=4 -s=512 -d=128 -pack_gqa=1 -p_drop=0.1 -v=1

run_test "GQA 4:1 long seq + dropout 0.1" \
    -b=2 -h=8 -h_k=2 -s=2048 -d=128 -pack_gqa=1 -p_drop=0.1 -v=1

run_test "GQA 4:1 decode + dropout 0.1" \
    -b=16 -h=32 -h_k=8 -s=1 -d=128 -pack_gqa=1 -p_drop=0.1 -v=1

echo ""
echo "=== Category 2: Pack-GQA with Split-KV (num_splits > 1) (8 tests) ==="
echo ""

run_test "GQA 4:1 + split-kv 2" \
    -b=2 -h=8 -h_k=2 -s=512 -d=128 -pack_gqa=1 -num_splits=2 -v=1

run_test "GQA 8:1 + split-kv 4" \
    -b=2 -h=32 -h_k=4 -s=1024 -d=128 -pack_gqa=1 -num_splits=4 -v=1

run_test "GQA 4:1 + split-kv 8" \
    -b=4 -h=32 -h_k=8 -s=2048 -d=128 -pack_gqa=1 -num_splits=8 -v=1

run_test "MQA 8:1 + split-kv 2" \
    -b=2 -h=8 -h_k=1 -s=512 -d=128 -pack_gqa=1 -num_splits=2 -v=1

run_test "GQA 4:1 fp16 + split-kv 4" \
    -b=2 -h=16 -h_k=4 -s=1024 -d=128 -prec=fp16 -pack_gqa=1 -num_splits=4 -v=1

run_test "GQA 4:1 d=64 + split-kv 2" \
    -b=2 -h=8 -h_k=2 -s=512 -d=64 -pack_gqa=1 -num_splits=2 -v=1

run_test "GQA 4:1 large batch + split-kv 4" \
    -b=8 -h=32 -h_k=8 -s=1024 -d=128 -pack_gqa=1 -num_splits=4 -v=1

run_test "GQA 4:1 long seq + split-kv 8" \
    -b=2 -h=32 -h_k=8 -s=4096 -d=128 -pack_gqa=1 -num_splits=8 -v=1

echo ""
echo "=== Category 3: Pack-GQA with Dropout AND Split-KV (6 tests) ==="
echo ""

run_test "GQA 4:1 + dropout 0.1 + split-kv 2" \
    -b=2 -h=8 -h_k=2 -s=512 -d=128 -pack_gqa=1 -p_drop=0.1 -num_splits=2 -v=1

run_test "GQA 8:1 + dropout 0.15 + split-kv 4" \
    -b=2 -h=32 -h_k=4 -s=1024 -d=128 -pack_gqa=1 -p_drop=0.15 -num_splits=4 -v=1

run_test "GQA 4:1 + dropout 0.2 + split-kv 2" \
    -b=4 -h=16 -h_k=4 -s=1024 -d=128 -pack_gqa=1 -p_drop=0.2 -num_splits=2 -v=1

run_test "MQA 8:1 + dropout 0.1 + split-kv 2" \
    -b=2 -h=8 -h_k=1 -s=512 -d=128 -pack_gqa=1 -p_drop=0.1 -num_splits=2 -v=1

run_test "GQA 4:1 fp16 + dropout 0.1 + split-kv 4" \
    -b=2 -h=16 -h_k=4 -s=1024 -d=128 -prec=fp16 -pack_gqa=1 -p_drop=0.1 -num_splits=4 -v=1

run_test "GQA 4:1 large batch + dropout 0.1 + split-kv 4" \
    -b=8 -h=32 -h_k=8 -s=1024 -d=128 -pack_gqa=1 -p_drop=0.1 -num_splits=4 -v=1

echo ""
echo "=== Category 4: Production Workloads with Dropout/Split-KV (6 tests) ==="
echo ""

run_test "LLaMA-style + dropout 0.1" \
    -b=2 -h=32 -h_k=8 -s=2048 -d=128 -pack_gqa=1 -p_drop=0.1 -v=1

run_test "LLaMA-style + split-kv 4" \
    -b=2 -h=32 -h_k=8 -s=2048 -d=128 -pack_gqa=1 -num_splits=4 -v=1

run_test "Mistral-style + dropout 0.1 + split-kv 2" \
    -b=4 -h=32 -h_k=8 -s=1024 -d=128 -pack_gqa=1 -p_drop=0.1 -num_splits=2 -v=1

run_test "Prefill + split-kv 8" \
    -b=1 -h=32 -h_k=8 -s=4096 -d=128 -pack_gqa=1 -num_splits=8 -v=1

run_test "Decode batch=64 + dropout 0.05" \
    -b=64 -h=32 -h_k=8 -s=1 -d=128 -pack_gqa=1 -p_drop=0.05 -v=1

run_test "Mixed: GQA h=16/h_k=4 + dropout 0.15 + split-kv 4" \
    -b=4 -h=16 -h_k=4 -s=2048 -d=64 -pack_gqa=1 -p_drop=0.15 -num_splits=4 -v=1

echo ""
echo "========================================================================"
echo "PHASE 2 TEST SUMMARY"
echo "========================================================================"
echo "Total Tests: $((PASSED + FAILED))"
echo "Passed: $PASSED"
echo "Failed: $FAILED"
echo ""

if [ $FAILED -eq 0 ]; then
    echo "ALL TESTS PASSED"
    exit 0
else
    echo "SOME TESTS FAILED"
    exit 1
fi
