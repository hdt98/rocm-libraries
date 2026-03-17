#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build_tmp"

# Build
echo "=== Building ==="
invoke build-client --gpu-targets=gfx942

cmake -S "${SCRIPT_DIR}/../" -B "${BUILD_DIR}" \
    --preset tensilelite \
    -DTENSILELITE_ENABLE_CLIENT=ON \
    -DTENSILELITE_BUILD_TESTING=ON \
    > /dev/null 2>&1

cmake --build "${BUILD_DIR}" --target cpu-gemm-driver > /dev/null 2>&1

# Run tests, capture output
echo "=== Running tests ==="
TEST_DIR="${BUILD_DIR}/tensilelite/tests"
OUTPUT=$(ctest --test-dir "${TEST_DIR}" -R CPUGemm --output-on-failure 2>&1) || true
echo "${OUTPUT}"

# Parse results into associative arrays keyed by base name (without fast_/slow_ prefix)
declare -A FAST_TIME SLOW_TIME FAST_STATUS SLOW_STATUS

while IFS= read -r line; do
    # Match lines like: "... CPUGemm.fast_f32_NN ...   Passed    0.05 sec"
    if [[ $line =~ CPUGemm\.(fast|slow)_([^ ]+)[[:space:]]+(\.+)[[:space:]]+(Passed|Failed)[[:space:]]+([0-9.]+)[[:space:]]+sec ]]; then
        path="${BASH_REMATCH[1]}"
        base="${BASH_REMATCH[2]}"
        status="${BASH_REMATCH[4]}"
        time="${BASH_REMATCH[5]}"

        if [[ $path == "fast" ]]; then
            FAST_TIME[$base]=$time
            FAST_STATUS[$base]=$status
        else
            SLOW_TIME[$base]=$time
            SLOW_STATUS[$base]=$status
        fi
    fi
done <<< "${OUTPUT}"

# Collect all base names and sort
mapfile -t ALL_BASES < <(printf '%s\n' "${!FAST_TIME[@]}" "${!SLOW_TIME[@]}" | sort -fu)

# Print table
echo ""
echo "=== Results ==="
printf "%-40s  %6s  %8s  %8s  %8s\n" "Test" "Status" "Fast(ms)" "Slow(ms)" "Saved(%)"
printf "%-40s  %6s  %8s  %8s  %8s\n" "$(printf '%0.s-' {1..40})" "------" "--------" "--------" "--------"

for base in "${ALL_BASES[@]}"; do
    ft="${FAST_TIME[$base]:-}"
    st="${SLOW_TIME[$base]:-}"
    fs="${FAST_STATUS[$base]:-}"
    ss="${SLOW_STATUS[$base]:-}"

    # Determine overall status
    if [[ -n $fs && -n $ss ]]; then
        if [[ $fs == "Passed" && $ss == "Passed" ]]; then
            status="PASS"
        else
            status="FAIL"
        fi
    elif [[ -n $ss ]]; then
        status=$( [[ $ss == "Passed" ]] && echo "PASS" || echo "FAIL" )
    else
        status=$( [[ $fs == "Passed" ]] && echo "PASS" || echo "FAIL" )
    fi

    # Format times
    ft_display="${ft:-"--"}"
    st_display="${st:-"--"}"

    # Calculate saved %
    if [[ -n $ft && -n $st ]]; then
        saved=$(awk "BEGIN { printf \"%.1f\", ($st - $ft) / $st * 100 }")
    else
        saved="--"
    fi

    printf "%-40s  %6s  %8s  %8s  %8s\n" "$base" "$status" "$ft_display" "$st_display" "$saved"
done
