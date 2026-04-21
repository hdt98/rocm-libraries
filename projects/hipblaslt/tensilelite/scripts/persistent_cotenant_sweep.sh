#!/bin/bash
# Sweep GEMM perf under a persistent co-tenant kernel occupying N CUs.
#
# For each N in CU_RANGE: launch a persistent Tensile kernel sized to occupy N
# CUs as a co-tenant (synthetic contention), run the benchmark on the full
# device with TENSILE_STREAMK_FIXED_GRID, kill the co-tenant, iterate.
#
# The benchmark kernel uses StreamK: 4 (dynamic work queue, added in commit
# 5ebff00cb0). The co-tenant kernel uses StreamK: 3 + PersistentKernelLoopForever.
#
# Both YAMLs are auto-written next to the script on first run if missing.
# Both .co's are auto-built if missing. Edit the YAMLs by hand to customize.

set -uo pipefail

PROG="$(basename "$0")"

usage() {
    cat <<EOF
Usage: $PROG [options]

Sweep GEMM benchmark perf under a persistent co-tenant kernel occupying N CUs.

Options (all have env-var equivalents; CLI args win over env, which wins over default):
  -h, --help                 Show this help and exit.
  --arch ARCH                GPU arch tag, e.g. gfx942.
                             Default: auto-detect via rocminfo.            [env: ARCH]
  --total-cus N              Total CU count for the full-device benchmark grid.
                             Default: 304.                                 [env: TOTAL_CUS]
  --cu-min N                 Sweep start. Default: 32.                     [env: CU_MIN]
  --cu-max N                 Sweep end (inclusive). Default: --total-cus.  [env: CU_MAX]
  --cu-step N                Sweep step size. Default: 16.                 [env: CU_STEP]
  --cotenant-yaml PATH       Co-tenant YAML path.
                             Default: scripts/persistent_cotenant.yaml.    [env: COTENANT_YAML]
  --cotenant-out DIR         Co-tenant build dir. Default: scripts/cotenant_out.
                                                                           [env: COTENANT_OUT]
  --bench-yaml PATH          Benchmark YAML path.
                             Default: scripts/persistent_bench.yaml.       [env: BENCH_YAML]
  --bench-out DIR            Benchmark build dir. Default: scripts/bench_out.
                                                                           [env: BENCH_OUT]
  --client PATH              tensilelite-client binary.
                             Default: build_tmp/tensilelite/client/tensilelite-client.
                                                                           [env: CLIENT]
  --tensile-sh PATH          Tensile.sh wrapper. Default: build_tmp/Tensile.sh.
                                                                           [env: TENSILE_SH]

One-time setup (before first run):
  invoke build-client    # produces build_tmp/tensilelite/client/tensilelite-client

Examples:
  $PROG                                    # all defaults, sweep 32..304 step 16
  $PROG --total-cus 104 --cu-step 8        # MI210: 104 CUs, finer sweep
  $PROG --cu-min 16 --cu-max 64            # narrow range only
  ARCH=gfx942 $PROG --total-cus 304        # mix env and CLI

The first run writes the two YAMLs next to the script and builds the co-tenant
.co (roughly 1-2 min). Subsequent runs reuse both. Hand-edit the YAMLs to change
the GEMM shape / tuning; delete the corresponding *_out dir to force a rebuild.
EOF
}

# Defaults (env vars override; CLI args override env).
ARCH="${ARCH:-}"
TOTAL_CUS="${TOTAL_CUS:-304}"
CU_STEP="${CU_STEP:-16}"
CU_MIN="${CU_MIN:-32}"
CU_MAX="${CU_MAX:-}"
COTENANT_YAML="${COTENANT_YAML:-scripts/persistent_cotenant.yaml}"
COTENANT_OUT="${COTENANT_OUT:-scripts/cotenant_out}"
BENCH_YAML="${BENCH_YAML:-scripts/persistent_bench.yaml}"
BENCH_OUT="${BENCH_OUT:-scripts/bench_out}"
CLIENT="${CLIENT:-build_tmp/tensilelite/client/tensilelite-client}"
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
        --client)         CLIENT="$2";        shift 2 ;;
        --tensile-sh)     TENSILE_SH="$2";    shift 2 ;;
        --) shift; break ;;
        -*) echo "ERROR: unknown option: $1" >&2; echo "Run '$PROG --help' for usage." >&2; exit 2 ;;
        *)  echo "ERROR: unexpected positional arg: $1" >&2; echo "Run '$PROG --help' for usage." >&2; exit 2 ;;
    esac
done

cd "$(dirname "$0")/.."   # tensilelite/

if [ -z "$ARCH" ]; then
    ARCH="$(rocminfo 2>/dev/null | awk '/Name:.*gfx/ {print $2; exit}')"
fi
if [ -z "$ARCH" ]; then
    echo "ERROR: could not detect GPU arch via rocminfo. Pass --arch or set ARCH." >&2
    exit 1
fi

# CU_MAX defaults to TOTAL_CUS only after both are resolved.
CU_MAX="${CU_MAX:-$TOTAL_CUS}"

COTENANT_LIB="$COTENANT_OUT/library/TensileLibrary.yaml"
COTENANT_CO="$COTENANT_OUT/library/TensileLibrary_${ARCH}.co"

[ -x "$CLIENT" ]     || { echo "ERROR: client not found at $CLIENT (run: invoke build-client)" >&2; exit 1; }
[ -x "$TENSILE_SH" ] || { echo "ERROR: $TENSILE_SH not found / not executable" >&2; exit 1; }

# ---- Write YAMLs if missing ----------------------------------------------

if [ ! -f "$COTENANT_YAML" ]; then
    echo "=== Writing $COTENANT_YAML ==="
    mkdir -p "$(dirname "$COTENANT_YAML")"
    cat > "$COTENANT_YAML" <<'EOF'
# Persistent co-tenant kernel.
# StreamK: 3 enables the persistent loop; PersistentKernelLoopForever: True
# replaces the loop's exit-check with an unconditional branch back to start.
# Termination is via process death (the --persistent client mode handles it).
#
# IMPORTANT: NumWarmups must be >= 1. The client's --persistent path lives
# inside the warmup launch block; with NumWarmups: 0 the client errors out at
# startup (the persistent branch would never run, and the code would fall
# through to the benchmark loop and hang on the first sync).
TestParameters:
  marks: [skip-gfx900, skip-gfx906, skip-gfx908, skip-gfx1010, skip-gfx1011, skip-gfx1012, skip-gfx1030, skip-gfx1100, skip-gfx1101, skip-gfx1102, skip-gfx1200, skip-gfx1201]

GlobalParameters:
  NumElementsToValidate: 0
  BoundsCheck: False
  KernelTime: False
  DataInitTypeAlpha: 1
  DataInitTypeBeta: 1
  DataInitTypeA: 12
  DataInitTypeB: 13
  DataInitTypeC: 12
  MaxWorkspaceSize: 134217728
  NumWarmups: 1
  EnqueuesPerSync: 0
  NumBenchmarks: 0

BenchmarkProblems:
  - # HGEMM NT
    - OperationType: GEMM
      DataType: h
      DestDataType: h
      ComputeDataType: s
      HighPrecisionAccumulate: True
      TransposeA: False
      TransposeB: True
      UseBeta: True
      Batched: True

    - InitialSolutionParameters:
      BenchmarkCommonParameters:
        - KernelLanguage: ["Assembly"]
        - PrefetchLocalRead: [True]
      ForkParameters:
        - 1LDSBuffer: [1]
        - DepthU: [64]
        - ExpandPointerSwap: [False]
        - GlobalReadVectorWidthA: [8]
        - GlobalReadVectorWidthB: [8]
        - GlobalSplitU: [0]
        - MatrixInstruction: [[16, 16, 16, 1, 1, 4,4, 2,2]]
        - MIArchVgpr: [0]
        - PrefetchGlobalRead: [2]
        - PrefetchLocalRead: [1]
        - ScheduleIterAlg: [3]
        - SourceSwap: [True]
        - StoreRemapVectorWidth: [0]
        - StreamK: [3]
        - PersistentKernelLoopForever: [True]
        - TransposeLDS: [0]
        - WorkGroupMapping: [1]
      BenchmarkForkParameters:
      JoinParameters:
      BenchmarkJoinParameters:
      BenchmarkFinalParameters:
        - ProblemSizes:
          - Exact: [4096, 4096, 1, 4096]
EOF
fi

if [ ! -f "$BENCH_YAML" ]; then
    echo "=== Writing $BENCH_YAML ==="
    mkdir -p "$(dirname "$BENCH_YAML")"
    cat > "$BENCH_YAML" <<'EOF'
# Benchmark kernel: StreamK: 4 (dynamic work queue, commit 5ebff00cb0).
# Run with TENSILE_STREAMK_FIXED_GRID=<TOTAL_CUS> to occupy the full device,
# while the co-tenant kernel holds N CUs.
TestParameters:
  marks: [skip-gfx900, skip-gfx906, skip-gfx908, skip-gfx1010, skip-gfx1011, skip-gfx1012, skip-gfx1030, skip-gfx1100, skip-gfx1101, skip-gfx1102, skip-gfx1200, skip-gfx1201]

GlobalParameters:
  NumElementsToValidate: 0
  BoundsCheck: False
  KernelTime: True
  DataInitTypeAlpha: 1
  DataInitTypeBeta: 1
  DataInitTypeA: 12
  DataInitTypeB: 13
  DataInitTypeC: 12
  MaxWorkspaceSize: 134217728
  NumWarmups: 3
  EnqueuesPerSync: 10
  NumBenchmarks: 3
  SyncsPerBenchmark: 1

BenchmarkProblems:
  - # HGEMM NT
    - OperationType: GEMM
      DataType: h
      DestDataType: h
      ComputeDataType: s
      HighPrecisionAccumulate: True
      TransposeA: False
      TransposeB: True
      UseBeta: True
      Batched: True

    - InitialSolutionParameters:
      BenchmarkCommonParameters:
        - KernelLanguage: ["Assembly"]
        - PrefetchLocalRead: [True]
      ForkParameters:
        - 1LDSBuffer: [1]
        - DepthU: [64]
        - ExpandPointerSwap: [False]
        - GlobalReadVectorWidthA: [8]
        - GlobalReadVectorWidthB: [8]
        - GlobalSplitU: [0]
        - MatrixInstruction: [[16, 16, 16, 1, 1, 4,4, 2,2]]
        - MIArchVgpr: [0]
        - PrefetchGlobalRead: [2]
        - PrefetchLocalRead: [1]
        - ScheduleIterAlg: [3]
        - SourceSwap: [True]
        - StoreRemapVectorWidth: [0]
        - StreamK: [4]
        - TransposeLDS: [0]
        - WorkGroupMapping: [1]
      BenchmarkForkParameters:
      JoinParameters:
      BenchmarkJoinParameters:
      BenchmarkFinalParameters:
        - ProblemSizes:
          - Exact: [4096, 4096, 1, 4096]
EOF
fi

# ---- Build the co-tenant .co if missing ---------------------------------

if [ ! -f "$COTENANT_CO" ] || [ ! -f "$COTENANT_LIB" ]; then
    echo "=== Building co-tenant .co (one-time) from $COTENANT_YAML -> $COTENANT_OUT ==="
    "$TENSILE_SH" "$COTENANT_YAML" "$COTENANT_OUT"
    [ -f "$COTENANT_CO" ]  || { echo "ERROR: build did not produce $COTENANT_CO" >&2; exit 1; }
    [ -f "$COTENANT_LIB" ] || { echo "ERROR: build did not produce $COTENANT_LIB" >&2; exit 1; }
fi

# ---- Sweep ---------------------------------------------------------------

# Cleanup: kill any surviving co-tenant process on script exit (Ctrl-C, normal exit, signals).
COTENANT_PID=
trap 'if [ -n "$COTENANT_PID" ]; then kill -TERM "$COTENANT_PID" 2>/dev/null; wait "$COTENANT_PID" 2>/dev/null; fi' EXIT INT TERM

echo "=== Sweep config ==="
echo "  ARCH=$ARCH  TOTAL_CUS=$TOTAL_CUS"
echo "  CU range: $CU_MIN..$CU_MAX step $CU_STEP"
echo "  cotenant: $COTENANT_CO"
echo "  bench:    $BENCH_YAML"
echo

for ((N = CU_MIN; N <= CU_MAX; N += CU_STEP)); do
    echo "=== Sweep N=$N CUs occupied by persistent co-tenant ==="

    # Launch persistent co-tenant. Direct client invocation (NOT through Tensile.sh) so
    # PR_SET_PDEATHSIG covers the bash -> client relationship in one parent hop.
    TENSILE_STREAMK_FIXED_GRID="$N" "$CLIENT" \
        --persistent \
        --library-file "$COTENANT_LIB" \
        --code-object "$COTENANT_CO" \
        > /dev/null 2>&1 &
    COTENANT_PID=$!

    # Steady-state settle: let the co-tenant's N workgroups become resident
    # before the benchmark workgroups race the scheduler for placement.
    sleep 0.2

    # Run the benchmark normally on the full device.
    TENSILE_STREAMK_FIXED_GRID="$TOTAL_CUS" "$TENSILE_SH" "$BENCH_YAML" "$BENCH_OUT"

    # Stop the co-tenant. SIGTERM -> sigwait returns -> client exit() -> HIP teardown.
    kill -TERM "$COTENANT_PID" 2>/dev/null
    wait "$COTENANT_PID" 2>/dev/null
    COTENANT_PID=
done
