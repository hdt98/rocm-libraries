#!/bin/bash

# Script to verify that all convolution files are referenced in CMakeLists.txt files
# Checks files from convolution_files_list.txt against CMakeLists.txt on current branch

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
INPUT_FILE="convolution_files_list.txt"
MISSING_FILES="missing_from_cmake.txt"
FOUND_FILES="found_in_cmake.txt"

echo -e "${BLUE}Checking CMakeLists.txt coverage for convolution files...${NC}"
echo ""

# Check if input file exists
if [ ! -f "${INPUT_FILE}" ]; then
    echo -e "${RED}Error: ${INPUT_FILE} not found${NC}"
    echo "Please run ./list_convolution_files.sh first"
    exit 1
fi

# Get current branch
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo -e "${BLUE}Current branch: ${CURRENT_BRANCH}${NC}"
echo ""

echo -e "${GREEN}Finding CMakeLists.txt files in convolution folders...${NC}"
# Get the git root directory
GIT_ROOT=$(git rev-parse --show-toplevel)
# Find CMakeLists.txt only in directories with "conv" in the path
TEMP_CMAKE_LIST=$(mktemp)
find "${GIT_ROOT}/projects/composablekernel/library/src" -type f -name "CMakeLists.txt" 2>/dev/null | grep -iE '(conv|quantization|utility)' > "$TEMP_CMAKE_LIST"
NUM_CMAKE=$(wc -l < "$TEMP_CMAKE_LIST")
echo -e "Found ${NUM_CMAKE} CMakeLists.txt files in convolution folders"
echo ""

# Pre-process CMakeLists.txt files to create a lookup map
echo -e "${GREEN}Building filename to CMakeLists.txt lookup map...${NC}"
TEMP_MAP=$(mktemp)

cmake_counter=0
while IFS= read -r cmake_file; do
    cmake_counter=$((cmake_counter + 1))

    # Show progress every 10 CMakeLists.txt files
    if [ $((cmake_counter % 10)) -eq 0 ] || [ $cmake_counter -eq $NUM_CMAKE ]; then
        echo -e "  Processing CMakeLists.txt: ${cmake_counter}/${NUM_CMAKE}..."
    fi

    # Extract all potential filenames from this CMakeLists.txt
    grep -oE '[a-zA-Z0-9_-]+\.(cpp|in)' "$cmake_file" 2>/dev/null | while read -r filename; do
        echo "$filename|$cmake_file" >> "$TEMP_MAP"
    done
done < "$TEMP_CMAKE_LIST"

echo -e "Built lookup map with $(wc -l < "$TEMP_MAP") entries"
echo ""

# Initialize output files
> "${MISSING_FILES}"
> "${FOUND_FILES}"

# Read the convolution files list
TOTAL_FILES=$(wc -l < "${INPUT_FILE}")
FOUND_COUNT=0
MISSING_COUNT=0

echo -e "${GREEN}Checking ${TOTAL_FILES} convolution files...${NC}"

# Check each file using the pre-built map
file_counter=0
while IFS= read -r filepath; do
    file_counter=$((file_counter + 1))

    # Show progress every 100 files
    if [ $((file_counter % 100)) -eq 0 ] || [ $file_counter -eq $TOTAL_FILES ]; then
        echo -e "  Progress: ${file_counter}/${TOTAL_FILES} files checked..."
    fi

    # Extract just the filename from the full path
    filename=$(basename "$filepath")

    # Look up the filename in the map
    found_in=$(grep -F "$filename|" "$TEMP_MAP" | head -1 | cut -d'|' -f2-)

    if [ -n "$found_in" ]; then
        echo "$filepath | $found_in" >> "${FOUND_FILES}"
        FOUND_COUNT=$((FOUND_COUNT + 1))
    else
        echo "$filepath" >> "${MISSING_FILES}"
        MISSING_COUNT=$((MISSING_COUNT + 1))
    fi
done < "${INPUT_FILE}"

# Clean up
rm -f "$TEMP_CMAKE_LIST" "$TEMP_MAP"

# Display results
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}           COVERAGE REPORT              ${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Calculate percentages
if [ $TOTAL_FILES -gt 0 ]; then
    FOUND_PERCENT=$(awk "BEGIN {printf \"%.1f\", ($FOUND_COUNT/$TOTAL_FILES)*100}")
    MISSING_PERCENT=$(awk "BEGIN {printf \"%.1f\", ($MISSING_COUNT/$TOTAL_FILES)*100}")
else
    FOUND_PERCENT="0.0"
    MISSING_PERCENT="0.0"
fi

echo -e "Total files checked:      ${TOTAL_FILES}"
echo -e "${GREEN}Files found in CMake:     ${FOUND_COUNT} (${FOUND_PERCENT}%)${NC}"
echo -e "${RED}Files missing from CMake: ${MISSING_COUNT} (${MISSING_PERCENT}%)${NC}"
echo ""

if [ $MISSING_COUNT -gt 0 ]; then
    echo -e "${YELLOW}Warning: Some files are not referenced in any CMakeLists.txt${NC}"
    echo -e "${YELLOW}Missing files saved to: ${MISSING_FILES}${NC}"
    echo ""
    echo -e "${YELLOW}First 20 missing files:${NC}"
    head -20 "${MISSING_FILES}"
    if [ $MISSING_COUNT -gt 20 ]; then
        echo -e "${YELLOW}... and $((MISSING_COUNT - 20)) more (see ${MISSING_FILES})${NC}"
    fi
else
    echo -e "${GREEN}✓ All convolution files are referenced in CMakeLists.txt!${NC}"
    rm -f "${MISSING_FILES}"
fi

echo ""
echo -e "${BLUE}Files found in CMake saved to: ${FOUND_FILES}${NC}"
echo -e "${BLUE}Format: <convolution_file> | <cmake_file_path>${NC}"
echo ""
