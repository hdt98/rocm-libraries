#!/bin/bash
# Sweep GEMM perf under a co-tenant kernel occupying N CUs.
#
# Phase 2: tensilelite-client now spawns the co-tenant in-process on a
# separate stream via --cotenant-library / --cotenant-cus, terminated cleanly
# via a host-pokable device flag (PersistentKernelHostStop). No separate
# --persistent process, no signals, no orphans.
#
# For each N in CU_RANGE: invoke Tensile.sh on the benchmark YAML once with
# the co-tenant flags forwarded through the client. Tensile.sh propagates
# unknown args to the client.
#
# Per-arch YAML pairs live next to this script as
#   persistent_{cotenant,bench}_<ARCH>.yaml
# and are committed alongside the script. Edit them to change tuning or
# problem shape; do not regenerate them from the script.

set -uo pipefail

PROG="$(basename "$0")"
SCRIPTDIR="$(cd "$(dirname "$0")" && pwd)"

# Per-arch defaults for --total-cus. Add new entries when you ship new YAMLs.
declare -A ARCH_TOTAL_CUS=(
    [gfx942]=304    # MI300X
    [gfx950]=256    # MI350 (verify against your SKU)
    [gfx1201]=64    # RX 9070-class
)

usage() {
    cat <<EOF
Usage: $PROG [options]

Sweep GEMM benchmark perf under an in-process co-tenant kernel occupying N CUs.

YAML pair selection:
  The script looks for a pair of YAMLs next to itself, matching the detected
  (or --arch overridden) GPU architecture:
    $SCRIPTDIR/persistent_cotenant_<ARCH>.yaml
    $SCRIPTDIR/persistent_bench_<ARCH>.yaml
  Currently shipped: $(ls "$SCRIPTDIR"/persistent_cotenant_*.yaml 2>/dev/null | sed 's|.*persistent_cotenant_||;s|\.yaml||' | tr '\n' ' ')
  Override either YAML path explicitly with --cotenant-yaml / --bench-yaml.

Options (all have env-var equivalents; CLI args win over env, which wins over default):
  -h, --help                 Show this help and exit.
  --arch ARCH                GPU arch tag, e.g. gfx942, gfx1201.
                             Default: auto-detect via rocminfo.            [env: ARCH]
  --total-cus N              Total CU count for the full-device benchmark grid.
                             Default: per-arch table baked into this script. [env: TOTAL_CUS]
  --cu-min N                 Sweep start. Default: 0 (no co-tenant baseline). [env: CU_MIN]
  --cu-max N                 Sweep end (inclusive). Default: --total-cus.  [env: CU_MAX]
  --cu-step N                Sweep step size. Default: 16.                 [env: CU_STEP]
  --cotenant-yaml PATH       Override the auto-resolved co-tenant YAML.    [env: COTENANT_YAML]
  --cotenant-out DIR         Co-tenant build dir. Default: scripts/cotenant_out_<ARCH>.
                                                                           [env: COTENANT_OUT]
  --bench-yaml PATH          Override the auto-resolved benchmark YAML.    [env: BENCH_YAML]
  --bench-out DIR            Benchmark build dir. Default: scripts/bench_out_<ARCH>.
                                                                           [env: BENCH_OUT]
  --tensile-sh PATH          Tensile.sh wrapper. Default: build_tmp/Tensile.sh.
                                                                           [env: TENSILE_SH]

One-time setup (before first run):
  invoke build-client    # produces build_tmp/tensilelite/client/tensilelite-client

Examples:
  $PROG                                    # auto-detect arch, baseline + sweep to TOTAL_CUS
  $PROG --arch gfx1201 --cu-step 8         # gfx1201 (64 CUs default), finer sweep
  $PROG --cu-min 16 --cu-max 64            # narrow range only
  ARCH=gfx942 $PROG --total-cus 304        # mix env and CLI

The first run for a given arch builds the co-tenant .co (roughly 1-2 min) into
scripts/cotenant_out_<ARCH>/. Subsequent runs reuse it. Delete the dir to
force a rebuild.
EOF
}

# Defaults (env vars override; CLI args override env).
ARCH="${ARCH:-}"
TOTAL_CUS="${TOTAL_CUS:-}"
CU_STEP="${CU_STEP:-16}"
CU_MIN="${CU_MIN:-0}"
CU_MAX="${CU_MAX:-}"
COTENANT_YAML="${COTENANT_YAML:-}"
COTENANT_OUT="${COTENANT_OUT:-}"
BENCH_YAML="${BENCH_YAML:-}"
BENCH_OUT="${BENCH_OUT:-}"
TENSILE_SH="${TENSILE_SH:-build_tmp/Tensile.sh}"

while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help)        usage; exit 0 ;;
        --arch)           ARCH="$2";          shift 2 ;;
        --total-cus)      TOTAL_CUS="$2";     shift 2 ;;
        --cu-min)         CU_MIN="$2";        shift 2 ;;
        --cu-max)         CU_MAX="$2";        shift 2 ;;
        --cu-step)        CU_STEP="$2";       shift 2 ;;
        --cotenant-yaml)  COTENANT_YAML="$2"; shift 2 ;;
        --cotenant-out)   COTENANT_OUT="$2";  shift 2 ;;
        --bench-yaml)     BENCH_YAML="$2";    shift 2 ;;
        --bench-out)      BENCH_OUT="$2";     shift 2 ;;
        --tensile-sh)     TENSILE_SH="$2";    shift 2 ;;
        --) shift; break ;;
        -*) echo "ERROR: unknown option: $1" >&2; echo "Run '$PROG --help' for usage." >&2; exit 2 ;;
        *)  echo "ERROR: unexpected positional arg: $1" >&2; echo "Run '$PROG --help' for usage." >&2; exit 2 ;;
    esac
done

cd "$SCRIPTDIR/.."   # tensilelite/

# ---- Arch resolution ----------------------------------------------------

if [ -z "$ARCH" ]; then
    ARCH="$(rocminfo 2>/dev/null | awk '/Name:.*gfx/ {print $2; exit}')"
fi
if [ -z "$ARCH" ]; then
    echo "ERROR: could not detect GPU arch via rocminfo. Pass --arch or set ARCH." >&2
    exit 1
fi

# ---- YAML pair resolution -----------------------------------------------

if [ -z "$COTENANT_YAML" ]; then
    COTENANT_YAML="$SCRIPTDIR/persistent_cotenant_${ARCH}.yaml"
fi
if [ -z "$BENCH_YAML" ]; then
    BENCH_YAML="$SCRIPTDIR/persistent_bench_${ARCH}.yaml"
fi

if [ ! -f "$COTENANT_YAML" ]; then
    echo "ERROR: no co-tenant YAML for arch $ARCH at $COTENANT_YAML" >&2
    echo "       Available pairs: $(ls "$SCRIPTDIR"/persistent_cotenant_*.yaml 2>/dev/null | xargs -n1 basename | sed 's/persistent_cotenant_//;s/\.yaml//' | tr '\n' ' ')" >&2
    echo "       Override with --cotenant-yaml PATH, or write a new YAML pair." >&2
    exit 1
fi
if [ ! -f "$BENCH_YAML" ]; then
    echo "ERROR: no benchmark YAML for arch $ARCH at $BENCH_YAML" >&2
    echo "       Override with --bench-yaml PATH, or write a new YAML pair." >&2
    exit 1
fi

# ---- Per-arch defaults --------------------------------------------------

if [ -z "$TOTAL_CUS" ]; then
    TOTAL_CUS="${ARCH_TOTAL_CUS[$ARCH]:-}"
fi
if [ -z "$TOTAL_CUS" ]; then
    echo "ERROR: no default --total-cus for arch $ARCH. Pass --total-cus N or set TOTAL_CUS." >&2
    echo "       Known arch defaults: ${!ARCH_TOTAL_CUS[*]}" >&2
    exit 1
fi

CU_MAX="${CU_MAX:-$TOTAL_CUS}"
COTENANT_OUT="${COTENANT_OUT:-$SCRIPTDIR/cotenant_out_${ARCH}}"
BENCH_OUT="${BENCH_OUT:-$SCRIPTDIR/bench_out_${ARCH}}"

COTENANT_LIB="$COTENANT_OUT/library/TensileLibrary.yaml"
COTENANT_CO="$COTENANT_OUT/library/TensileLibrary_${ARCH}.co"

[ -x "$TENSILE_SH" ] || { echo "ERROR: $TENSILE_SH not found / not executable" >&2; exit 1; }

# ---- Build the co-tenant .co if missing ---------------------------------

if [ ! -f "$COTENANT_CO" ] || [ ! -f "$COTENANT_LIB" ]; then
    echo "=== Building co-tenant .co (one-time) from $COTENANT_YAML -> $COTENANT_OUT ==="
    "$TENSILE_SH" "$COTENANT_YAML" "$COTENANT_OUT"
    [ -f "$COTENANT_CO" ]  || { echo "ERROR: build did not produce $COTENANT_CO" >&2; exit 1; }
    [ -f "$COTENANT_LIB" ] || { echo "ERROR: build did not produce $COTENANT_LIB" >&2; exit 1; }
fi

# ---- Sweep ---------------------------------------------------------------

echo "=== Sweep config ==="
echo "  ARCH=$ARCH  TOTAL_CUS=$TOTAL_CUS"
echo "  CU range: $CU_MIN..$CU_MAX step $CU_STEP"
echo "  cotenant: $COTENANT_LIB"
echo "  bench:    $BENCH_YAML"
echo

# In Phase 2, each iteration is one self-contained tensilelite-client process
# spawned by Tensile.sh. The co-tenant is launched in-process on a separate
# stream and torn down via the host-pokable stop flag — no orphans, no signals,
# no inter-process plumbing.
#
# Tensile.sh forwards unknown CLI args verbatim to the client. The
# --cotenant-* args are consumed by the client (Phase 2).
for ((N = CU_MIN; N <= CU_MAX; N += CU_STEP)); do
    if [ "$N" -eq 0 ]; then
        echo "=== Sweep N=0 (baseline, no co-tenant) ==="
        TENSILE_STREAMK_FIXED_GRID="$TOTAL_CUS" "$TENSILE_SH" "$BENCH_YAML" "$BENCH_OUT"
    else
        echo "=== Sweep N=$N CUs occupied by co-tenant ==="
        TENSILE_STREAMK_FIXED_GRID="$TOTAL_CUS" "$TENSILE_SH" "$BENCH_YAML" "$BENCH_OUT" \
            --cotenant-library "$COTENANT_LIB" \
            --cotenant-code-object "$COTENANT_CO" \
            --cotenant-cus "$N"
    fi
done
