#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# tune_im2win.sh — Benchmark all im2win configs for a given conv problem
#
# Usage:
#   cd <build-dir>
#   bash <src>/tune_im2win.sh [extra args passed to every binary]
#
# Example (target problem: G=32, N=32, K=4, C=4, 3×3, same-pad, 200×200):
#   bash .../tune_im2win.sh \
#       -prec=fp16 -in_layout=GNCHW -wei_layout=GKCYX -out_layout=NHWGK \
#       -g=32 -n=32 -k=4 -c=4 -y=3 -x=3 -h=200 -w=200 \
#       -lpad_h=1 -lpad_w=1 -rpad_h=1 -rpad_w=1 \
#       -v=0 -warmup=10 -repeat=50
# ═══════════════════════════════════════════════════════════════════════
set -euo pipefail

# Binary directory: defaults to ./bin relative to CWD (i.e., the build directory).
# Override with IM2WIN_BIN_DIR env var if needed.
BIN_DIR="${IM2WIN_BIN_DIR:-$(pwd)/bin}"

# All config names must match CMakeLists.txt
CONFIGS=(
    "CV3_M16N64K64"
    "CV3_M64N64K64"
    "CV3_M64N32K64"
    "Mem_M128N32K64"
    "Mem_M128N32K64_IW"
    "CV3_M128N128K64_Occ2"
    "CV3_M64N16K64"
    "CV3_M128N16K64"
    "Mem_M128N16K64"
    "Mem_M128N16K64_IW"
    "CV3_M64N16K64_Occ2"
    "Mem_M128N4K64"
    "Mem_M256N4K64"
)

ARGS=("$@")
RESULTS=()

echo "═══════════════════════════════════════════════════════════"
echo " im2win config sweep"
echo " Args: ${ARGS[*]}"
echo "═══════════════════════════════════════════════════════════"
printf "%-30s  %10s  %10s  %10s\n" "Config" "ms" "TFlops" "GB/s"
echo "───────────────────────────────────────────────────────────"

BEST_MS=1e9
BEST_NAME=""

for cfg in "${CONFIGS[@]}"; do
    BIN="${BIN_DIR}/tile_example_grouped_conv_fwd_im2win_${cfg}"
    if [[ ! -x "$BIN" ]]; then
        printf "%-30s  %-32s\n" "$cfg" "[binary not found, skipping]"
        continue
    fi

    # Run and capture timing line (format: "<ms> ms, <TFlops> TFlops, <GB/s> GB/s")
    OUTPUT=$("$BIN" "${ARGS[@]}" 2>&1) || true
    TIMING=$(echo "$OUTPUT" | grep -E "^[0-9]+\.[0-9]+ ms" | tail -1)

    if [[ -z "$TIMING" ]]; then
        # Check for unsupported/error
        ERR=$(echo "$OUTPUT" | grep -i "error\|unsupported\|not supported" | head -1)
        printf "%-30s  %-32s\n" "$cfg" "[SKIP: ${ERR:-no timing output}]"
        continue
    fi

    MS=$(echo    "$TIMING" | awk '{print $1}')
    TFLOPS=$(echo "$TIMING" | awk '{print $3}')
    GBS=$(echo   "$TIMING" | awk '{print $5}')

    printf "%-30s  %10s  %10s  %10s\n" "$cfg" "$MS" "$TFLOPS" "$GBS"
    RESULTS+=("$MS $cfg")

    if python3 -c "import sys; sys.exit(0 if float('$MS') < float('$BEST_MS') else 1)" 2>/dev/null; then
        BEST_MS="$MS"
        BEST_NAME="$cfg"
    fi
done

echo "═══════════════════════════════════════════════════════════"
echo " Best: ${BEST_NAME}  →  ${BEST_MS} ms"
echo "═══════════════════════════════════════════════════════════"
