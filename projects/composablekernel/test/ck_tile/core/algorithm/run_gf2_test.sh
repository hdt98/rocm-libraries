#!/bin/bash
#SBATCH --job-name=gf2-test
#SBATCH --gres=gpu:gfx942-mi300x:1
#SBATCH --time=00:10:00
#SBATCH --container-image=/cluster/images/ck_ub24.04_rocm7.2.sqsh
#SBATCH --container-mount-home
#SBATCH --container-mounts=/scratch:/scratch
#SBATCH --output=/scratch/users/mpodkory/workloads/gf2_test_%j.log
#SBATCH --error=/scratch/users/mpodkory/workloads/gf2_test_%j.err

# Exit on error, print commands
set -ex

# Configuration
BRANCH="${BRANCH:-users/tenpercent/ck/gf2-linear-transform}"
REMOTE="https://github.com/ROCm/rocm-libraries.git"

# Directories (all in /tmp to avoid NFS issues)
WORKDIR=/tmp/$USER/gf2-test-$$
REPO=$WORKDIR/ck
BUILD=$WORKDIR/build

# Create all directories upfront
mkdir -p $WORKDIR

echo "=========================================="
echo "GF(2) Linear Transform Unit Test"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "=========================================="

rocminfo | grep -E "Marketing Name:" | head -1

#
# Step 1: Sparse checkout
#
echo "=========================================="
echo "Step 1: Cloning $BRANCH (sparse)..."
echo "=========================================="

export GIT_PROGRESS_DELAY=1000

if ! git clone --progress --filter=blob:none --no-checkout --depth=1 --no-tags --single-branch -b "$BRANCH" "$REMOTE" "$REPO"; then
    echo "ERROR: git clone failed"
    exit 1
fi

cd $REPO

if ! git sparse-checkout init --cone; then
    echo "ERROR: sparse-checkout init failed"
    exit 1
fi

if ! git sparse-checkout set projects/composablekernel; then
    echo "ERROR: sparse-checkout set failed"
    exit 1
fi

if ! git checkout; then
    echo "ERROR: git checkout failed"
    exit 1
fi

echo "Branch: $(git branch --show-current)"
echo "Commit: $(git rev-parse --short HEAD)"

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

if ! cmake -G Ninja -B $BUILD -S $CK \
    -DCMAKE_CXX_COMPILER=/opt/rocm/lib/llvm/bin/clang++ \
    -DGPU_TARGETS=gfx942 \
    -DBUILD_DEV=OFF \
    -DBUILD_TESTS=ON \
    -DBUILD_CK_DEVICE_INSTANCES=OFF \
    -DBUILD_CK_TILE_ENGINE=OFF \
    -DBUILD_CK_EXAMPLES=OFF \
    -DBUILD_CK_TUTORIALS=OFF \
    -DBUILD_CK_PROFILER=OFF \
    2>&1 | tee $BUILD/cmake_configure.log; then
    echo "ERROR: cmake configure failed"
    exit 1
fi
echo "Step 2: SUCCESS"

#
# Step 3: Build test
#
echo "=========================================="
echo "Step 3: Building GF(2) unit test..."
echo "=========================================="

TARGET="ck_tile_unit_gf2_linear_transform"
echo "Target: $TARGET"

if ! ninja -C $BUILD $TARGET; then
    echo "ERROR: ninja build failed"
    exit 1
fi
echo "Step 3: SUCCESS"

#
# Step 4: Run test
#
echo "=========================================="
echo "Step 4: Running test..."
echo "=========================================="

$BUILD/bin/$TARGET --gtest_color=yes

echo "=========================================="
echo "All steps completed successfully!"
echo "=========================================="

# Cleanup
rm -rf $WORKDIR
