#!/bin/bash

# Script to list all convolution source files in the composablekernel project
# from the develop branch
# File extensions: .cpp, .in

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
BRANCH="develop"
OUTPUT_FILE="convolution_files_list.txt"

echo -e "${BLUE}Listing convolution source files from ${BRANCH} branch (library/src folder only)...${NC}"
echo ""

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "Error: Not in a git repository"
    exit 1
fi

# Get the git root directory and set project path
GIT_ROOT=$(git rev-parse --show-toplevel)
PROJECT_DIR="../library/src"

# Get the current branch to restore later
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)

# Use git ls-tree to list files from develop branch without checking it out
echo -e "${GREEN}Searching for convolution files in library/src folder with extensions: .cpp, .in${NC}"
echo ""

# Find all files matching the patterns from develop branch (library/src folder only)
git ls-tree -r --name-only "${BRANCH}" -- "${GIT_ROOT}/projects/composablekernel/library/src" | \
    grep -iE '(conv|convolution)' | \
    grep -E '\.(cpp|in)$' | \
    sort | \
    tee "${OUTPUT_FILE}"

# Count the files
FILE_COUNT=$(wc -l < "${OUTPUT_FILE}")

echo ""
echo -e "${GREEN}Found ${FILE_COUNT} convolution source files${NC}"
echo -e "${BLUE}Results saved to: ${OUTPUT_FILE}${NC}"
echo ""
echo "File breakdown by extension:"
echo "  .cpp files: $(grep -c '\.cpp$' "${OUTPUT_FILE}" || echo 0)"
echo "  .in files:  $(grep -c '\.in$' "${OUTPUT_FILE}" || echo 0)"
