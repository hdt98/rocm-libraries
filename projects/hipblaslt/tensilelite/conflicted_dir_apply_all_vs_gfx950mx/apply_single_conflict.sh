#!/bin/bash
#
# Usage: ./conflicted_marker_command.sh <filename>
#   e.g. ./conflicted_marker_command.sh KernelWriterAssembly.py
#
# This script takes a file (basename or partial path) and adds conflict markers
# DIRECTLY into the file in the worktree, showing the differences between:
#   - Ours:   users/tomtang/gfx950_mx_rebase_tensilelite_to_1250_apply_all_conflicts => "apply_all"
#   - Theirs: gfx950_mx_rebase (from public ROCm repo)                               => "gfx950mx"
#
# The merge base (common ancestor) is used for 3-way merge.
# The result overwrites the file in the worktree with conflict markers.
#
# Prerequisites:
#   - You must have the apply_all branch checked out locally
#   - Internet access to fetch from the public ROCm repo (first run only)

set -euo pipefail

PUBLIC_REPO="https://github.com/ROCm/rocm-libraries"
PUBLIC_REMOTE_NAME="rocm-public"
GFX950_BRANCH_NAME="gfx950_mx_rebase"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFLICTED_FILES="${SCRIPT_DIR}/conflicted_files.txt"

REPO_ROOT=$(cd "$SCRIPT_DIR" && git rev-parse --show-toplevel)
OURS_BRANCH="users/tomtang/gfx950_mx_rebase_tensilelite_to_1250_apply_all_conflicts"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <filename>"
    echo "  e.g. $0 KernelWriterAssembly.py"
    echo ""
    echo "Available files:"
    cat "$CONFLICTED_FILES" | while read -r f; do basename "$f"; done
    exit 1
fi

INPUT="$1"

MATCH=$(grep -i "$INPUT" "$CONFLICTED_FILES" | head -1)
if [ -z "$MATCH" ]; then
    echo "ERROR: No matching file found for '$INPUT' in conflicted_files.txt"
    echo ""
    echo "Available files:"
    cat "$CONFLICTED_FILES" | while read -r f; do basename "$f"; done
    exit 1
fi

echo "Matched file: $MATCH"

cd "$REPO_ROOT"

GFX950_REF=""

for candidate in \
    "${GFX950_BRANCH_NAME}" \
    "remotes/${PUBLIC_REMOTE_NAME}/${GFX950_BRANCH_NAME}" \
    "remotes/gfx950_local/${GFX950_BRANCH_NAME}" \
    "remotes/public/${GFX950_BRANCH_NAME}" \
    "remotes/origin/${GFX950_BRANCH_NAME}"; do
    if git rev-parse --verify "$candidate" &>/dev/null; then
        GFX950_REF="$candidate"
        echo "Found gfx950_mx_rebase at: $GFX950_REF"
        break
    fi
done

if [ -z "$GFX950_REF" ]; then
    echo "gfx950_mx_rebase branch not found locally."
    echo "Adding public ROCm repo as remote '${PUBLIC_REMOTE_NAME}' and fetching..."

    if ! git remote get-url "$PUBLIC_REMOTE_NAME" &>/dev/null; then
        git remote add "$PUBLIC_REMOTE_NAME" "$PUBLIC_REPO"
        echo "Added remote: ${PUBLIC_REMOTE_NAME} -> ${PUBLIC_REPO}"
    else
        echo "Remote '${PUBLIC_REMOTE_NAME}' already exists."
    fi

    git fetch "$PUBLIC_REMOTE_NAME" "$GFX950_BRANCH_NAME"
    GFX950_REF="remotes/${PUBLIC_REMOTE_NAME}/${GFX950_BRANCH_NAME}"

    if ! git rev-parse --verify "$GFX950_REF" &>/dev/null; then
        echo "ERROR: Failed to fetch gfx950_mx_rebase from ${PUBLIC_REPO}"
        exit 1
    fi
    echo "Fetched gfx950_mx_rebase at: $GFX950_REF"
fi

MERGE_BASE=$(git merge-base "$OURS_BRANCH" "$GFX950_REF")
echo "Merge base: $MERGE_BASE"

TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

BASE_FILE="$TMPDIR/base"
OURS_FILE="$TMPDIR/ours"
THEIRS_FILE="$TMPDIR/theirs"

git show "${MERGE_BASE}:${MATCH}" > "$BASE_FILE" 2>/dev/null || echo "" > "$BASE_FILE"
git show "${OURS_BRANCH}:${MATCH}" > "$OURS_FILE" 2>/dev/null || echo "" > "$OURS_FILE"
git show "${GFX950_REF}:${MATCH}" > "$THEIRS_FILE" 2>/dev/null || echo "" > "$THEIRS_FILE"

TARGET_FILE="${REPO_ROOT}/${MATCH}"

git merge-file -p \
    --diff3 \
    -L "apply_all" \
    -L "base" \
    -L "gfx950mx" \
    "$OURS_FILE" "$BASE_FILE" "$THEIRS_FILE" > "$TARGET_FILE" || true

CONFLICT_COUNT=$(grep -c "^<<<<<<< apply_all" "$TARGET_FILE" 2>/dev/null || echo "0")

if [ "$CONFLICT_COUNT" -eq 0 ]; then
    echo "No conflicts found - files merge cleanly (auto-merged result written)."
else
    echo "Added $CONFLICT_COUNT conflict marker(s) to: $TARGET_FILE"
fi

echo "Done."
