#!/bin/bash
set -euo pipefail

usage() {
    cat <<'EOF'
BO vs BF Benchmark — compare Bayesian Optimization against Brute Force tuning.

Usage:
    bash scripts/benchmark_bo_vs_bf.sh [MODE]

Modes:
    full       Run both BF and BO for all 30 cases (default). Saves BF baseline.
    bo-only    Run only BO, reuse saved BF baseline from a previous full run.

Options:
    -h, --help    Show this help message and exit.

Output:
    benchmark_results/<timestamp>/results_<timestamp>.csv   Full comparison CSV
    benchmark_results/<timestamp>/bf_*.log                  BF logs per case
    benchmark_results/<timestamp>/bo_*.log                  BO logs per case
    benchmark_results/baseline_bf.csv                       Saved BF baseline

Columns (CSV):
    Speed%     Wall-clock: positive = BO faster search
    Kernel%    10-run avg: positive = BO found faster kernel
    Final%     1-run:      noisy single-run reference (less reliable)

Examples:
    # First run: build baseline (~2-3 hours)
    bash scripts/benchmark_bo_vs_bf.sh

    # After BO code changes: reuse BF baseline (~30-40 min)
    bash scripts/benchmark_bo_vs_bf.sh bo-only

Environment:
    MIOPEN_TUNING_SEARCH_METHOD    0=brute force, 1=bayesian (set automatically)
    MIOPEN_DEBUG_TUNING_BO_INITIAL Random initial probes for BO (default: 3)
EOF
    exit 0
}

case "${1:-}" in -h|--help) usage ;; esac

DRIVER="$(dirname "$0")/../build_bo/bin/MIOpenDriver"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_DIR="$(dirname "$0")/../benchmark_results/${TIMESTAMP}"
mkdir -p "${RESULT_DIR}"

COMMON_ENV=(
    "MIOPEN_FIND_ENFORCE=SEARCH_DB_UPDATE"
    "MIOPEN_DEBUG_DISABLE_FIND_DB=1"
    "MIOPEN_DISABLE_CACHE=1"
    "MIOPEN_LOG_LEVEL=5"
)

BO_ENV=("MIOPEN_TUNING_SEARCH_METHOD=1" "MIOPEN_DEBUG_TUNING_BO_INITIAL=3")
BF_ENV=("MIOPEN_TUNING_SEARCH_METHOD=0")

# ---- 30 Diverse Cases (10 per direction) ----
# Format: "ID|Dir|Description|driver_command"
#   Dir: F=forward, B=backward-data, W=backward-weight
#   -F 1=fwd, -F 2=bwd, -F 4=wrw
declare -a CASES=(
    # =========== Forward (10 cases) ===========

    # F01: 1x1 NHWC BF16, large channels, small spatial
    "F01_1x1_nhwc|F|N64_C1280_K1280_8x8_1x1_NHWC_BF16|convbfp16 -n 64 -c 1280 -H 8 -W 8 -k 1280 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 --in_layout NHWC --out_layout NHWC --fil_layout NHWC -F 1 -s 1 -t 1"

    # F02: 3x3 NHWC BF16, ResNet-like
    "F02_3x3_nhwc|F|N16_C64_K128_56x56_3x3_NHWC_BF16|convbfp16 -n 16 -c 64 -H 56 -W 56 -k 128 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 1 --in_layout NHWC --out_layout NHWC --fil_layout NHWC -F 1 -s 1 -t 1"

    # F03: 3x3 NCHW BF16, big batch (expensive per-eval)
    "F03_3x3_nchw|F|N128_C64_K64_56x56_3x3_NCHW_BF16|convbfp16 -n 128 -c 64 -H 56 -W 56 -k 64 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 1 -F 1 -s 1 -t 1"

    # F04: 1x1 NHWC FP16, bottleneck expand
    "F04_1x1_fp16|F|N64_C256_K1024_14x14_1x1_NHWC_FP16|convfp16 -n 64 -c 256 -H 14 -W 14 -k 1024 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F 1 -s 1 -t 1"

    # F05: 7x7 NHWC BF16, ResNet stem, stride-2
    "F05_7x7_nhwc|F|N32_C3_K64_112x112_7x7_NHWC_BF16|convbfp16 -n 32 -c 3 -H 112 -W 112 -k 64 -y 7 -x 7 -p 3 -q 3 -u 2 -v 2 -l 1 -j 1 -g 1 --in_layout NHWC --out_layout NHWC --fil_layout NHWC -F 1 -s 1 -t 1"

    # F06: 3x3 NHWC FP16, large batch
    "F06_3x3_fp16|F|N256_C128_K128_28x28_3x3_NHWC_FP16|convfp16 -n 256 -c 128 -H 28 -W 28 -k 128 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 1 --in_layout NHWC --out_layout NHWC --fil_layout NHWC -F 1 -s 1 -t 1"

    # F07: 1x1 NCHW BF16, bottleneck expand, small spatial
    "F07_1x1_nchw|F|N64_C512_K2048_7x7_1x1_NCHW_BF16|convbfp16 -n 64 -c 512 -H 7 -W 7 -k 2048 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F 1 -s 1 -t 1"

    # F08: 5x5 NHWC BF16, Inception-like
    "F08_5x5_nhwc|F|N16_C32_K64_56x56_5x5_NHWC_BF16|convbfp16 -n 16 -c 32 -H 56 -W 56 -k 64 -y 5 -x 5 -p 2 -q 2 -u 1 -v 1 -l 1 -j 1 -g 1 --in_layout NHWC --out_layout NHWC --fil_layout NHWC -F 1 -s 1 -t 1"

    # F09: 3x3 NCHW FP16, stride-2 downsample
    "F09_3x3_nchw|F|N128_C256_K256_14x14_3x3s2_NCHW_FP16|convfp16 -n 128 -c 256 -H 14 -W 14 -k 256 -y 3 -x 3 -p 1 -q 1 -u 2 -v 2 -l 1 -j 1 -g 1 -F 1 -s 1 -t 1"

    # F10: 1x1 NHWC BF16, small batch, huge channels
    "F10_1x1_nhwc|F|N4_C2048_K512_7x7_1x1_NHWC_BF16|convbfp16 -n 4 -c 2048 -H 7 -W 7 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 --in_layout NHWC --out_layout NHWC --fil_layout NHWC -F 1 -s 1 -t 1"

    # =========== Backward Data (10 cases) ===========

    # B01: 5x5 NHWC BF16, large filter
    "B01_5x5_nhwc|B|N256_C192_K32_28x28_5x5_NHWC_BF16|convbfp16 -n 256 -c 192 -H 28 -W 28 -k 32 -y 5 -x 5 -p 2 -q 2 -u 1 -v 1 -l 1 -j 1 -g 1 --in_layout NHWC --out_layout NHWC --fil_layout NHWC -F 2 -s 1 -t 1"

    # B02: 3x3 NCHW BF16, medium spatial
    "B02_3x3_nchw|B|N128_C256_K256_14x14_3x3_NCHW_BF16|convbfp16 -n 128 -c 256 -H 14 -W 14 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 1 -F 2 -s 1 -t 1"

    # B03: 1x1 NHWC BF16, large channels
    "B03_1x1_nhwc|B|N64_C512_K512_28x28_1x1_NHWC_BF16|convbfp16 -n 64 -c 512 -H 28 -W 28 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 --in_layout NHWC --out_layout NHWC --fil_layout NHWC -F 2 -s 1 -t 1"

    # B04: 3x3 NHWC FP16, large batch
    "B04_3x3_fp16|B|N256_C64_K64_56x56_3x3_NHWC_FP16|convfp16 -n 256 -c 64 -H 56 -W 56 -k 64 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 1 --in_layout NHWC --out_layout NHWC --fil_layout NHWC -F 2 -s 1 -t 1"

    # B05: 1x1 NCHW BF16, bottleneck
    "B05_1x1_nchw|B|N64_C1024_K256_14x14_1x1_NCHW_BF16|convbfp16 -n 64 -c 1024 -H 14 -W 14 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F 2 -s 1 -t 1"

    # B06: 7x7 NHWC BF16, stem backward, stride-2
    "B06_7x7_nhwc|B|N32_C3_K64_112x112_7x7s2_NHWC_BF16|convbfp16 -n 32 -c 3 -H 112 -W 112 -k 64 -y 7 -x 7 -p 3 -q 3 -u 2 -v 2 -l 1 -j 1 -g 1 --in_layout NHWC --out_layout NHWC --fil_layout NHWC -F 2 -s 1 -t 1"

    # B07: 3x3 NHWC BF16, stride-2
    "B07_3x3_nhwc|B|N128_C128_K128_28x28_3x3s2_NHWC_BF16|convbfp16 -n 128 -c 128 -H 28 -W 28 -k 128 -y 3 -x 3 -p 1 -q 1 -u 2 -v 2 -l 1 -j 1 -g 1 --in_layout NHWC --out_layout NHWC --fil_layout NHWC -F 2 -s 1 -t 1"

    # B08: 1x1 NHWC FP16, tiny spatial
    "B08_1x1_fp16|B|N16_C2048_K512_7x7_1x1_NHWC_FP16|convfp16 -n 16 -c 2048 -H 7 -W 7 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 --in_layout NHWC --out_layout NHWC --fil_layout NHWC -F 2 -s 1 -t 1"

    # B09: 3x3 NCHW FP16, small spatial
    "B09_3x3_nchw|B|N64_C512_K512_7x7_3x3_NCHW_FP16|convfp16 -n 64 -c 512 -H 7 -W 7 -k 512 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 1 -F 2 -s 1 -t 1"

    # B10: 5x5 NHWC BF16, large filter + large spatial
    "B10_5x5_nhwc|B|N32_C64_K128_56x56_5x5_NHWC_BF16|convbfp16 -n 32 -c 64 -H 56 -W 56 -k 128 -y 5 -x 5 -p 2 -q 2 -u 1 -v 1 -l 1 -j 1 -g 1 --in_layout NHWC --out_layout NHWC --fil_layout NHWC -F 2 -s 1 -t 1"

    # =========== Weight Gradient (10 cases) ===========

    # W01: 3x3 NHWC BF16, massive batch (split_k)
    "W01_3x3_nhwc|W|N512_C512_K512_28x28_3x3_NHWC_BF16|convbfp16 -n 512 -c 512 -H 28 -W 28 -k 512 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 1 --in_layout NHWC --out_layout NHWC --fil_layout NHWC -F 4 -s 1 -t 1"

    # W02: 3x3 NCHW BF16 (expensive per-eval)
    "W02_3x3_nchw|W|N128_C256_K256_14x14_3x3_NCHW_BF16|convbfp16 -n 128 -c 256 -H 14 -W 14 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 1 -F 4 -s 1 -t 1"

    # W03: 1x1 NHWC BF16, huge channels (BO edge case)
    "W03_1x1_nhwc|W|N256_C1280_K256_28x28_1x1_NHWC_BF16|convbfp16 -n 256 -c 1280 -H 28 -W 28 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 --in_layout NHWC --out_layout NHWC --fil_layout NHWC -F 4 -s 1 -t 1"

    # W04: 7x7 NHWC BF16, stem wrw, stride-2
    "W04_7x7_nhwc|W|N32_C3_K64_112x112_7x7s2_NHWC_BF16|convbfp16 -n 32 -c 3 -H 112 -W 112 -k 64 -y 7 -x 7 -p 3 -q 3 -u 2 -v 2 -l 1 -j 1 -g 1 --in_layout NHWC --out_layout NHWC --fil_layout NHWC -F 4 -s 1 -t 1"

    # W05: 1x1 NCHW FP16, bottleneck wrw
    "W05_1x1_nchw|W|N64_C256_K1024_14x14_1x1_NCHW_FP16|convfp16 -n 64 -c 256 -H 14 -W 14 -k 1024 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -F 4 -s 1 -t 1"

    # W06: 3x3 NHWC FP16, medium wrw
    "W06_3x3_fp16|W|N128_C128_K128_28x28_3x3_NHWC_FP16|convfp16 -n 128 -c 128 -H 28 -W 28 -k 128 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 1 --in_layout NHWC --out_layout NHWC --fil_layout NHWC -F 4 -s 1 -t 1"

    # W07: 5x5 NHWC BF16, large filter wrw
    "W07_5x5_nhwc|W|N64_C192_K32_28x28_5x5_NHWC_BF16|convbfp16 -n 64 -c 192 -H 28 -W 28 -k 32 -y 5 -x 5 -p 2 -q 2 -u 1 -v 1 -l 1 -j 1 -g 1 --in_layout NHWC --out_layout NHWC --fil_layout NHWC -F 4 -s 1 -t 1"

    # W08: 3x3 NCHW BF16, big batch wrw
    "W08_3x3_nchw|W|N256_C64_K64_56x56_3x3_NCHW_BF16|convbfp16 -n 256 -c 64 -H 56 -W 56 -k 64 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 1 -F 4 -s 1 -t 1"

    # W09: 1x1 NHWC BF16, huge C, small spatial
    "W09_1x1_nhwc|W|N16_C2048_K512_7x7_1x1_NHWC_BF16|convbfp16 -n 16 -c 2048 -H 7 -W 7 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 --in_layout NHWC --out_layout NHWC --fil_layout NHWC -F 4 -s 1 -t 1"

    # W10: 3x3 NHWC BF16, stride-2 wrw
    "W10_3x3_nhwc|W|N64_C256_K256_14x14_3x3s2_NHWC_BF16|convbfp16 -n 64 -c 256 -H 14 -W 14 -k 256 -y 3 -x 3 -p 1 -q 1 -u 2 -v 2 -l 1 -j 1 -g 1 --in_layout NHWC --out_layout NHWC --fil_layout NHWC -F 4 -s 1 -t 1"
)

N_CASES=${#CASES[@]}

MODE="${1:-full}"  # full = BF+BO (baseline), bo-only = reuse saved BF

BASELINE_CSV="$(dirname "$0")/../benchmark_results/baseline_bf.csv"
CSV="${RESULT_DIR}/results_${TIMESTAMP}.csv"

if [ "$MODE" = "bo-only" ] && [ ! -f "$BASELINE_CSV" ]; then
    echo "ERROR: No baseline found at ${BASELINE_CSV}. Run without arguments first."
    exit 1
fi

echo "case|direction|description|bf_wall_s|bo_wall_s|speedup|speed_pct|bf_search_ms|bo_search_ms|kernel_pct|bf_final_ms|bo_final_ms|final_diff_pct|bf_verify|bo_verify|bf_solver|bo_solver|bf_config|bo_config|same_solver|same_config" > "$CSV"

run_case() {
    local method="$1"
    local driver_cmd="$2"
    local logfile="$3"

    local -a env_vars=("${COMMON_ENV[@]}")
    if [ "$method" = "bo" ]; then
        env_vars+=("${BO_ENV[@]}")
    else
        env_vars+=("${BF_ENV[@]}")
    fi

    { time env "${env_vars[@]}" ${DRIVER} ${driver_cmd} ; } > "${logfile}" 2>&1 || true
}

parse_kernel_time() {
    grep "GPU Kernel Time" "$1" | head -1 | grep -oP '[\d.]+(?= ms)' || echo "N/A"
}

parse_verify() {
    grep -q "Verifies OK" "$1" && echo "OK" || echo "FAIL"
}

parse_wall_s() {
    grep "^real" "$1" | head -1 | sed 's/real\s*//' | awk -F'[ms]' '{
        if (NF >= 3) printf "%.0f", $1*60 + $2
        else printf "%.0f", $1
    }' || echo "N/A"
}

parse_bo_solver() {
    grep '>> \[BEST\]' "$1" | tail -1 | sed 's/.*\[BEST\] //' | sed 's/ | time:.*//' || echo "N/A"
}

parse_bo_config() {
    local solver_line
    solver_line=$(grep '>> \[BEST\]' "$1" | tail -1 | sed 's/.*\[BEST\] //' | sed 's/ | time:.*//')
    local solver_num
    solver_num=$(echo "$solver_line" | grep -oP 'Solver #\d+' | grep -oP '\d+')
    if [ -n "$solver_num" ]; then
        grep "config:" "$1" | tail -1 | sed 's/.*config: //' || echo "N/A"
    else
        echo "N/A"
    fi
}

parse_bo_search_time() {
    # Extract 10-run averaged time from BO's >> [BEST] ... | time: Xms line
    grep '>> \[BEST\]' "$1" | tail -1 | grep -oP 'time: \K[0-9.]+' || echo "N/A"
}

parse_bf_solver() {
    grep "Solution:" "$1" | head -1 | sed 's/.*Solution: //' || echo "N/A"
}

parse_bf_config() {
    # Multiple "Done:" lines (one per solver). Pick the one with lowest time.
    grep "Done:.*best #" "$1" | \
        sed 's/.*best #[0-9a-fA-F]* //' | \
        sort -n | head -1 | \
        sed 's/^[0-9.e+-]* //' || echo "N/A"
}

parse_bf_search_time() {
    # Extract 10-run averaged best time from BF's "Done: N/F/T, best #X TIME config" line
    # Pick the Done line with the lowest time across all solvers
    grep "Done:.*best #" "$1" | \
        sed 's/.*best #[0-9a-fA-F]* //' | \
        sort -n | head -1 | \
        grep -oP '^[0-9.e+-]+' || echo "N/A"
}

echo ""
echo "========================================================================"
echo "  BO vs BF Benchmark — ${N_CASES} cases (mode: ${MODE})"
echo "  $(date)"
echo "  Driver: ${DRIVER}"
echo "========================================================================"
echo ""

TOTAL_START=$(date +%s)

# Load BF baseline into associative array for bo-only mode
declare -A BF_WALL BF_SEARCH BF_KERNEL BF_VERIFY BF_SOLVER BF_CONFIG
if [ "$MODE" = "bo-only" ]; then
    while IFS='|' read -r cid bf_w bf_sk bf_k bf_v bf_s bf_c; do
        [ "$cid" = "case" ] && continue
        BF_WALL["$cid"]="$bf_w"
        BF_SEARCH["$cid"]="$bf_sk"
        BF_KERNEL["$cid"]="$bf_k"
        BF_VERIFY["$cid"]="$bf_v"
        BF_SOLVER["$cid"]="$bf_s"
        BF_CONFIG["$cid"]="$bf_c"
    done < "$BASELINE_CSV"
    echo "  Loaded BF baseline from ${BASELINE_CSV}"
    echo ""
fi

for i in "${!CASES[@]}"; do
    IFS='|' read -r case_id direction description driver_cmd <<< "${CASES[$i]}"
    case_num=$((i + 1))

    echo "--------------------------------------------------------------------"
    echo "[$case_num/${N_CASES}] ${case_id} (${direction}) ${description}"
    echo "--------------------------------------------------------------------"

    if [ "$MODE" = "bo-only" ]; then
        bf_wall="${BF_WALL[$case_id]:-N/A}"
        bf_search="${BF_SEARCH[$case_id]:-N/A}"
        bf_kernel="${BF_KERNEL[$case_id]:-N/A}"
        bf_verify="${BF_VERIFY[$case_id]:-N/A}"
        bf_solver="${BF_SOLVER[$case_id]:-N/A}"
        bf_config="${BF_CONFIG[$case_id]:-N/A}"
        echo "  [BF] baseline: ${bf_wall}s  search=${bf_search}ms  final=${bf_kernel}ms  solver=${bf_solver}  ${bf_verify}"
    else
        bf_log="${RESULT_DIR}/bf_${case_id}.log"
        echo "  [BF] running..."
        run_case "bf" "$driver_cmd" "$bf_log"
        bf_wall=$(parse_wall_s "$bf_log")
        bf_search=$(parse_bf_search_time "$bf_log")
        bf_kernel=$(parse_kernel_time "$bf_log")
        bf_verify=$(parse_verify "$bf_log")
        bf_solver=$(parse_bf_solver "$bf_log")
        bf_config=$(parse_bf_config "$bf_log")
        echo "  [BF] ${bf_wall}s  search=${bf_search}ms  final=${bf_kernel}ms  solver=${bf_solver}  ${bf_verify}"
        echo "  [BF] config: ${bf_config}"
    fi

    # Run BO
    bo_log="${RESULT_DIR}/bo_${case_id}.log"
    echo "  [BO] running..."
    run_case "bo" "$driver_cmd" "$bo_log"
    bo_wall=$(parse_wall_s "$bo_log")
    bo_search=$(parse_bo_search_time "$bo_log")
    bo_kernel=$(parse_kernel_time "$bo_log")
    bo_verify=$(parse_verify "$bo_log")
    bo_solver=$(parse_bo_solver "$bo_log")
    bo_config=$(parse_bo_config "$bo_log")
    echo "  [BO] ${bo_wall}s  search=${bo_search}ms  final=${bo_kernel}ms  solver=${bo_solver}  ${bo_verify}"
    echo "  [BO] config: ${bo_config}"

    # Check if same config
    if [ "$bf_config" = "$bo_config" ]; then
        same_config="YES"
    else
        same_config="NO"
    fi

    # Check if same solver — if same config, always same solver (same config = same instance)
    if [ "$same_config" = "YES" ]; then
        same_solver="YES"
    else
        bf_solver_name=$(echo "$bf_solver" | sed 's|^[0-9]*/||')
        bo_solver_name=$(echo "$bo_solver" | sed 's|^Solver #[0-9]* ||')
        if [ "$bf_solver_name" = "$bo_solver_name" ]; then
            same_solver="YES"
        else
            same_solver="NO"
        fi
    fi

    # Compute metrics — primary comparison uses search times (10-run avg)
    # Speed%: positive = BO faster search, negative = BO slower search
    speedup=$(awk "BEGIN { if ($bo_wall > 0 && $bf_wall > 0) printf \"%.1f\", $bf_wall/$bo_wall; else print \"N/A\" }")
    speed_pct=$(awk "BEGIN { if ($bf_wall > 0) printf \"%+.0f\", ($bf_wall - $bo_wall) / $bf_wall * 100; else print \"N/A\" }")
    # Kernel%: positive = BO found faster kernel, negative = BO found slower kernel
    kernel_pct=$(awk "BEGIN { if (\"$bf_search\" != \"N/A\" && \"$bo_search\" != \"N/A\" && $bf_search > 0) printf \"%+.1f\", ($bf_search - $bo_search) / $bf_search * 100; else print \"N/A\" }")
    final_diff=$(awk "BEGIN { if (\"$bf_kernel\" != \"N/A\" && \"$bo_kernel\" != \"N/A\" && $bf_kernel > 0) printf \"%+.1f\", ($bf_kernel - $bo_kernel) / $bf_kernel * 100; else print \"N/A\" }")

    if [ "$same_config" = "YES" ]; then
        match_str="SAME"
    elif [ "$same_solver" = "YES" ]; then
        match_str="diff cfg"
    else
        match_str="DIFF"
    fi

    printf "  ──▶  Speed: %s%% (%sx)  |  Kernel: %s%%  |  Match: %s\n\n" "$speed_pct" "$speedup" "$kernel_pct" "$match_str"

    echo "${case_id}|${direction}|${description}|${bf_wall}|${bo_wall}|${speedup}|${speed_pct}|${bf_search}|${bo_search}|${kernel_pct}|${bf_kernel}|${bo_kernel}|${final_diff}|${bf_verify}|${bo_verify}|${bf_solver}|${bo_solver}|${bf_config}|${bo_config}|${same_solver}|${same_config}" >> "$CSV"
done

TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$((TOTAL_END - TOTAL_START))

# Save BF baseline for future bo-only runs
if [ "$MODE" != "bo-only" ]; then
    echo "case|bf_wall_s|bf_search_ms|bf_final_ms|bf_verify|bf_solver|bf_config" > "$BASELINE_CSV"
    while IFS='|' read -r cid dir desc bf_w bo_w spd spd_pct bf_sk bo_sk kpct bf_k bo_k fdiff bf_v bo_v bf_s bo_s bf_c bo_c same_s same_c; do
        [ "$cid" = "case" ] && continue
        echo "${cid}|${bf_w}|${bf_sk}|${bf_k}|${bf_v}|${bf_s}|${bf_c}" >> "$BASELINE_CSV"
    done < "$CSV"
    echo "BF baseline saved to: ${BASELINE_CSV}"
fi

echo ""
echo "========================================================================"
echo "  RESULTS SUMMARY"
echo "========================================================================"
echo ""
printf "%-18s %3s %-35s %6s %6s %7s %8s %10s %10s %8s %10s %10s %8s %4s %4s %s\n" \
    "Case" "Dir" "Description" "BF(s)" "BO(s)" "Speedup" "Speed%" "BF-Srch" "BO-Srch" "Kernel%" "BF-Final" "BO-Final" "Final%" "Slvr" "Cfg" "Verify"
printf "%-18s %3s %-35s %6s %6s %7s %8s %10s %10s %8s %10s %10s %8s %4s %4s %s\n" \
    "------------------" "---" "-----------------------------------" "------" "------" "-------" "--------" "----------" "----------" "--------" "----------" "----------" "--------" "----" "----" "------"

total_bf=0
total_bo=0
while IFS='|' read -r cid dir desc bf_w bo_w spd spd_pct bf_sk bo_sk kpct bf_k bo_k fdiff bf_v bo_v bf_s bo_s bf_c bo_c same_s same_c; do
    [ "$cid" = "case" ] && continue
    printf "%-18s %3s %-35s %5ss %5ss %6sx %7s%% %9sms %9sms %7s%% %9sms %9sms %7s%% %4s %4s %s/%s\n" \
        "$cid" "$dir" "$desc" "$bf_w" "$bo_w" "$spd" "$spd_pct" "$bf_sk" "$bo_sk" "$kpct" "$bf_k" "$bo_k" "$fdiff" "$same_s" "$same_c" "$bf_v" "$bo_v"
    total_bf=$((total_bf + bf_w))
    total_bo=$((total_bo + bo_w))
done < "$CSV"

total_speedup=$(awk "BEGIN { if ($total_bo > 0) printf \"%.1f\", $total_bf/$total_bo; else print \"N/A\" }")
total_speed_pct=$(awk "BEGIN { if ($total_bf > 0) printf \"%+.0f\", ($total_bf - $total_bo) / $total_bf * 100; else print \"N/A\" }")
echo ""
printf "%-18s %3s %-35s %5ss %5ss %6sx %7s%%\n" "TOTAL" "" "" "$total_bf" "$total_bo" "$total_speedup" "$total_speed_pct"
echo ""
echo "Total wall time: ${TOTAL_ELAPSED}s"
echo "CSV: ${CSV}"
echo "Logs: ${RESULT_DIR}/"
echo ""
