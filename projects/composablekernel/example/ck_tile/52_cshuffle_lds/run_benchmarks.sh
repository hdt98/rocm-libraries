#!/bin/bash
#SBATCH --job-name=cshuffle-lds
#SBATCH --gres=gpu:gfx950-mi350x:1
#SBATCH --time=00:20:00
#SBATCH --container-image=/cluster/images/ck_ub24.04_rocm7.2.sqsh
#SBATCH --container-mount-home
#SBATCH --output=/home/AMD/mpodkory/workloads/cshuffle_lds_%j.log
#SBATCH --error=/home/AMD/mpodkory/workloads/cshuffle_lds_%j.err

# Exit on error, print commands
set -ex

# Configuration
BRANCH="${BRANCH:-users/tenpercent/ck/xor-lds-swizzle}"
REMOTE="https://github.com/ROCm/rocm-libraries.git"

# Directories (all in /tmp to avoid NFS issues)
WORKDIR=/tmp/$USER/cshuffle-lds-$$
REPO=$WORKDIR/ck
BUILD=$WORKDIR/build
RESULTS=$WORKDIR/results
FINAL_RESULTS=/home/AMD/mpodkory/workloads/cshuffle-lds-$(date +%Y%m%d_%H%M%S)

# Create all directories upfront
mkdir -p $WORKDIR $RESULTS "$FINAL_RESULTS"

# Helper to save results on error exit
save_results_on_error() {
    echo "Saving results on error to $FINAL_RESULTS..."
    cp -r $RESULTS/* "$FINAL_RESULTS/" 2>/dev/null || true
}
trap save_results_on_error EXIT

echo "=========================================="
echo "CShuffleLds Microbenchmark Suite"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Final results: $FINAL_RESULTS"
echo "=========================================="

rocminfo | grep -E "Marketing Name:" | head -1

#
# Step 1: Sparse checkout
#
echo "=========================================="
echo "Step 1: Cloning $BRANCH (sparse)..."
echo "=========================================="

export GIT_PROGRESS_DELAY=1000

if ! ( time git clone --progress --filter=blob:none --depth=1 --single-branch --no-tags --sparse -b "$BRANCH" "$REMOTE" "$REPO" ); then
    echo "ERROR: git clone failed"
    exit 1
fi

cd $REPO

if ! git sparse-checkout set projects/composablekernel; then
    echo "ERROR: sparse-checkout set failed"
    exit 1
fi

if ! ( time git checkout ); then
    echo "ERROR: git checkout failed"
    exit 1
fi

echo "Branch: $(git branch --show-current)"
echo "Commit: $(git rev-parse --short HEAD)"

# CK is inside the monorepo
CK=$REPO/projects/composablekernel
echo "CK directory: $CK"
echo "Step 1: SUCCESS"

#
# Step 2: CMake configure
#
echo "=========================================="
echo "Step 2: CMake configure..."
echo "=========================================="
mkdir -p $BUILD

# Time the configure step with profiling
if ! ( time cmake -G Ninja -B $BUILD -S $CK \
    --profiling-format=google-trace --profiling-output=$BUILD/cmake_trace.json \
    -DCMAKE_CXX_COMPILER=/opt/rocm/lib/llvm/bin/clang++ \
    -DGPU_TARGETS=gfx950 \
    -DBUILD_DEV=OFF \
    -DBUILD_CK_DEVICE_INSTANCES=OFF \
    -DBUILD_CK_TILE_ENGINE=OFF \
    -DBUILD_CK_EXAMPLES=ON \
    -DBUILD_CK_TUTORIALS=OFF \
    -DBUILD_CK_PROFILER=OFF \
    -DBUILD_CK_TILE_ENGINE_TESTS=OFF \
    -DBUILD_CK_TILE_FMHA_TESTS=OFF \
    ) 2>&1 | tee $BUILD/cmake_configure.log; then
    echo "ERROR: cmake configure failed"
    exit 1
fi
echo "Step 2: SUCCESS"

#
# Step 3: Build benchmarks
#
echo "=========================================="
echo "Step 3: Building benchmarks..."
echo "=========================================="

# Build single benchmark: 16x16x128 MFMA FP8->FP8 with 2x2 wave layout
TARGET="bench_lds_fp8_16x16x128_2x2_fp8"
echo "Target: $TARGET"

if ! ninja -C $BUILD $TARGET; then
    echo "ERROR: ninja build failed"
    exit 1
fi
echo "Step 3: SUCCESS"

#
# Step 4: Profile with rocprofv3
#
echo "=========================================="
echo "Step 4: Profiling..."
echo "=========================================="

PROF=$WORKDIR/prof
mkdir -p $PROF

cat > $PROF/counters.txt <<'EOF'
pmc: SQ_LDS_BANK_CONFLICT SQ_INSTS_LDS
EOF

BENCH_COUNT=0
for bench in $BUILD/bin/bench_lds_*; do
    if [ ! -x "$bench" ]; then
        echo "WARNING: $bench not executable, skipping"
        continue
    fi

    name=$(basename $bench)
    echo "--- $name ---"

    if rocprofv3 -i $PROF/counters.txt -d $PROF/$name -- $bench 2>&1; then
        BENCH_COUNT=$((BENCH_COUNT + 1))
    else
        echo "WARNING: rocprofv3 failed for $name"
    fi
done

if [ $BENCH_COUNT -eq 0 ]; then
    echo "ERROR: No benchmarks ran successfully"
    exit 1
fi
echo "Step 4: SUCCESS ($BENCH_COUNT benchmarks profiled)"

#
# Step 5: Analyze results
#
echo "=========================================="
echo "Step 5: Analyzing LDS Bank Conflicts..."
echo "=========================================="

python3 - "$PROF" "$RESULTS" <<'PYTHON'
import csv
import sys
import glob
import os
import sqlite3

prof_dir, results_dir = sys.argv[1], sys.argv[2]

results = []

for bench_dir in sorted(glob.glob(f'{prof_dir}/bench_lds_*')):
    bench_name = os.path.basename(bench_dir)

    # Find SQLite database files
    db_files = glob.glob(f'{bench_dir}/**/*_results.db', recursive=True)
    if not db_files:
        continue

    for db_file in db_files:
        try:
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()

            # Get table names (they have UUID suffixes)
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'rocpd_kernel_dispatch_%'")
            dispatch_table = cursor.fetchone()[0]
            pmc_event_table = dispatch_table.replace('kernel_dispatch', 'pmc_event')
            pmc_info_table = dispatch_table.replace('kernel_dispatch', 'info_pmc')
            kernel_table = dispatch_table.replace('kernel_dispatch', 'info_kernel_symbol')

            # Query kernel dispatch info joined with PMC events
            query = f"""
            SELECT
                k.display_name AS kernel_name,
                pmc_info.name AS event_name,
                pmc_event.value AS event_value
            FROM {dispatch_table} d
            JOIN {pmc_event_table} pmc_event ON d.dispatch_id = pmc_event.event_id
            JOIN {pmc_info_table} pmc_info ON pmc_event.pmc_id = pmc_info.id
            JOIN {kernel_table} k ON d.kernel_id = k.id
            WHERE pmc_info.name IN ('SQ_LDS_BANK_CONFLICT', 'SQ_INSTS_LDS')
            """

            cursor.execute(query)
            rows = cursor.fetchall()

            # Group by kernel and aggregate (sum across all dispatches)
            kernel_data = {}
            for kernel_name, event_name, event_value in rows:
                if kernel_name not in kernel_data:
                    kernel_data[kernel_name] = {}
                if event_name not in kernel_data[kernel_name]:
                    kernel_data[kernel_name][event_name] = 0
                kernel_data[kernel_name][event_name] += float(event_value)

            # Process each kernel
            for kernel_name, events in kernel_data.items():
                if 'storetile' in kernel_name.lower():
                    op = 'Store'
                elif 'loadtile' in kernel_name.lower():
                    op = 'Load'
                else:
                    continue

                conflicts = events.get('SQ_LDS_BANK_CONFLICT', 0)
                lds_insts = events.get('SQ_INSTS_LDS', 0)
                ratio = conflicts / lds_insts if lds_insts > 0 else 0

                results.append({
                    'benchmark': bench_name,
                    'operation': op,
                    'lds_insts': lds_insts,
                    'conflicts': conflicts,
                    'conflicts_per_inst': ratio
                })

            conn.close()
        except Exception as e:
            print(f"Warning: Failed to process {db_file}: {e}")

# Print summary
print(f"\n{'Benchmark':<45} {'Op':<6} {'LDS_Insts':>12} {'Conflicts':>12} {'Ratio':>10}")
print("=" * 90)

for r in sorted(results, key=lambda x: (x['benchmark'], x['operation'])):
    print(f"{r['benchmark']:<45} {r['operation']:<6} {r['lds_insts']:>12.0f} {r['conflicts']:>12.0f} {r['conflicts_per_inst']:>10.4f}")

# Write CSV
csv_file = os.path.join(results_dir, 'bank_conflicts.csv')
with open(csv_file, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['benchmark', 'operation', 'lds_insts', 'conflicts', 'conflicts_per_inst'])
    writer.writeheader()
    writer.writerows(results)

print(f"\nResults saved to: {csv_file}")
print(f"Total results: {len(results)}")
PYTHON

if [ $? -ne 0 ]; then
    echo "ERROR: Python analysis failed"
    exit 1
fi

# Copy raw profiling data
cp -r $PROF/* $RESULTS/ 2>/dev/null || true

echo "Step 5: SUCCESS"

#
# Done
#
echo "=========================================="
echo "All steps completed successfully!"
echo "Results in: $RESULTS"
ls -lh $RESULTS/
echo "=========================================="

# Save results before cleanup
echo "Copying results to $FINAL_RESULTS..."
cp -r $RESULTS/* "$FINAL_RESULTS/" 2>/dev/null || true
cp $BUILD/cmake_trace.json "$FINAL_RESULTS/" 2>/dev/null || true

# Cleanup temp work directory
rm -rf $WORKDIR

echo "=========================================="
echo "Final results saved to: $FINAL_RESULTS"
ls -lh "$FINAL_RESULTS/" || true
echo "=========================================="
