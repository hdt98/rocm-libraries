#!/usr/bin/env bash
# Remove stale ROCROLLER_SAVE_ASSEMBLY (*.s) and gpu_trace (*.jsonl) files from the
# rocroller build directory, but keep the FP4 MI16x16x128 / 64x64x256 TN reference pair.
#
# Usage (from anywhere):
#   ROCROLLER_BUILD_DIR=/path/to/build_release shared/rocroller/scripts/clean_saved_kernel_artifacts.sh
# Or:
#   shared/rocroller/scripts/clean_saved_kernel_artifacts.sh /path/to/build_release

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="${1:-${ROCROLLER_BUILD_DIR:-${SCRIPT_DIR}/../build_release}}"
BUILD="$(cd "$BUILD" && pwd)"

cd "$BUILD"

shopt -s nullglob
removed=0

for f in GEMMTest_*.s; do
    case "$f" in
    *MI16x16x128_MT64x64x256_TN_0_kernel*.s) ;;
    *)
        rm -f -- "$f"
        removed=$((removed + 1))
        ;;
    esac
done

for f in *.jsonl; do
    case "$f" in
    trace_fp4_mi64_d2l.jsonl | trace_fp4_mi64_viavgpr.jsonl) ;;
    *)
        rm -f -- "$f"
        removed=$((removed + 1))
        ;;
    esac
done

shopt -u nullglob

echo "clean_saved_kernel_artifacts: dir=$BUILD removed=$removed (kept MI64 TN reference .s and trace_fp4_mi64_*.jsonl)"
