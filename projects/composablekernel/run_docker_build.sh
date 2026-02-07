#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Script to run Docker container and build include tests

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Docker image ID
DOCKER_IMAGE="61474c2a0070"

echo "Starting Docker container..."
echo "Repository: $REPO_ROOT"
echo "Docker image: $DOCKER_IMAGE"
echo ""

# Run docker with GPU access and build the tests
docker run --rm -it \
    --network=host \
    --device=/dev/kfd \
    --device=/dev/dri \
    --security-opt seccomp=unconfined \
    --group-add video \
    -v "$REPO_ROOT:$REPO_ROOT" \
    -w "$REPO_ROOT" \
    "$DOCKER_IMAGE" \
    bash -c "
        echo 'Inside Docker container...'
        echo 'Working directory: \$(pwd)'
        echo ''

        # Run the build script
        ./build_include_tests.sh

        echo ''
        echo 'Build completed. Container will now exit.'
    "

echo ""
echo "Docker container stopped."
echo "Build artifacts are in: $REPO_ROOT/build"
