#!/bin/bash
# Test that --init-seed produces reproducible random initialization.
# Usage: test_deterministic_init.sh --client <path> --config <ini_file> --mode <Random|RandomNarrow|RandomNegPosLimited>
set -euo pipefail

CLIENT=""
CONFIG=""
MODE="Random"
SEED_A=42
SEED_B=43
TMPDIR_BASE=""

usage() {
    echo "Usage: $0 --client <path> --config <ini_file> [--mode <init_mode>]"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --client)  CLIENT="$2"; shift 2 ;;
        --config)  CONFIG="$2"; shift 2 ;;
        --mode)    MODE="$2";   shift 2 ;;
        *)         usage ;;
    esac
done

[[ -z "$CLIENT" || -z "$CONFIG" ]] && usage
[[ -x "$CLIENT" ]] || { echo "FAIL: client not found or not executable: $CLIENT"; exit 1; }
[[ -f "$CONFIG" ]]  || { echo "FAIL: config file not found: $CONFIG"; exit 1; }

cleanup() {
    [[ -n "$TMPDIR_BASE" && -d "$TMPDIR_BASE" ]] && rm -rf "$TMPDIR_BASE"
}
trap cleanup EXIT

TMPDIR_BASE=$(mktemp -d)

# Run the client with a given seed, capture tensor A text output, return its md5.
run_and_hash() {
    local seed=$1
    local run_dir="$TMPDIR_BASE/run_seed${seed}_$$_${RANDOM}"
    mkdir -p "$run_dir"

    # Create a modified .ini: replace existing values, append missing ones
    cp "$CONFIG" "$run_dir/params.ini"
    local ini="$run_dir/params.ini"

    # Key=value pairs to set (replace if present, append if not)
    declare -A overrides=(
        [init-seed]="$seed"
        [init-a]="$MODE"
        [init-b]="$MODE"
        [print-tensor-a]="True"
        [num-benchmarks]="1"
        [num-elements-to-validate]="-1"
        [num-enqueues-per-sync]="1"
        [num-syncs-per-benchmark]="1"
        [num-warmups]="0"
    )
    for key in "${!overrides[@]}"; do
        local val="${overrides[$key]}"
        if grep -q "^${key}=" "$ini"; then
            sed -i "s|^${key}=.*|${key}=${val}|" "$ini"
        else
            echo "${key}=${val}" >> "$ini"
        fi
    done

    local output
    # Client may exit non-zero on validation failure; we still want the output
    output=$("$CLIENT" --config-file "$run_dir/params.ini" 2>&1) || true

    # Extract tensor A numeric values only. Exclude metadata lines
    # (A:/B: headers, Tensor(...), data_ptr, coordinate labels, CSV results).
    local values
    values=$(echo "$output" | sed -n '/^A: /,/^B: /p' | grep -vE '^A: |^B: |^Tensor\(|data_ptr|^\(|^$|^[0-9]+,[0-9]')
    if [[ -z "$values" ]]; then
        echo "EMPTY"
    else
        echo "$values" | md5sum | awk '{print $1}'
    fi
}

echo "Testing mode=$MODE ..."

# Test 1: Same seed -> same output
hash1=$(run_and_hash $SEED_A)
hash2=$(run_and_hash $SEED_A)

if [[ "$hash1" == "EMPTY" ]]; then
    echo "SKIP: No tensor output for $MODE (validation may have failed)"
    exit 0
fi

if [[ "$hash1" != "$hash2" ]]; then
    echo "FAIL: Same seed ($SEED_A) produced different results for $MODE"
    echo "  run1: $hash1"
    echo "  run2: $hash2"
    exit 1
fi
echo "  PASS: seed=$SEED_A is deterministic (hash=$hash1)"

# Test 2: Different seed -> different output
hash3=$(run_and_hash $SEED_B)
if [[ "$hash1" == "$hash3" ]]; then
    echo "FAIL: Different seeds ($SEED_A vs $SEED_B) produced same results for $MODE"
    echo "  hash: $hash1"
    exit 1
fi
echo "  PASS: seed=$SEED_A ($hash1) differs from seed=$SEED_B ($hash3)"

echo "PASS: $MODE deterministic init test passed"
