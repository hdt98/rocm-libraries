#!/bin/bash
#
# Usage:
#   ./apply_all_conflicts.sh          Apply conflict markers to all non-rocisa files
#   ./apply_all_conflicts.sh --undo   Undo: restore all files to their original state
#
# This script applies 3-way merge conflict markers to all 78 non-rocisa conflicted
# files between:
#   - Ours:   users/tomtang/gfx950_mx_rebase_tensilelite_to_1250_apply_all_conflicts => "apply_all"
#   - Theirs: gfx950_mx_rebase                                                       => "gfx950mx"
#
# rocisa files (31 files) are SKIPPED because their conflicts are already resolved
# and are compatible with develop.
#
# The --undo option restores all files from the ours branch (apply_all).

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFLICTED_FILES="${SCRIPT_DIR}/conflicted_files.txt"
BACKUP_DIR="${SCRIPT_DIR}/.backup"

REPO_ROOT=$(cd "$SCRIPT_DIR" && git rev-parse --show-toplevel)

OURS_BRANCH="users/tomtang/gfx950_mx_rebase_tensilelite_to_1250_apply_all_conflicts"

PUBLIC_REPO="https://github.com/ROCm/rocm-libraries"
PUBLIC_REMOTE_NAME="rocm-public"
GFX950_BRANCH_NAME="gfx950_mx_rebase"

find_gfx950_ref() {
    for candidate in \
        "${GFX950_BRANCH_NAME}" \
        "remotes/${PUBLIC_REMOTE_NAME}/${GFX950_BRANCH_NAME}" \
        "remotes/gfx950_local/${GFX950_BRANCH_NAME}" \
        "remotes/public/${GFX950_BRANCH_NAME}" \
        "remotes/origin/${GFX950_BRANCH_NAME}"; do
        if git rev-parse --verify "$candidate" &>/dev/null; then
            echo "$candidate"
            return
        fi
    done

    echo "gfx950_mx_rebase not found locally. Fetching from public repo..." >&2
    if ! git remote get-url "$PUBLIC_REMOTE_NAME" &>/dev/null; then
        git remote add "$PUBLIC_REMOTE_NAME" "$PUBLIC_REPO"
    fi
    git fetch "$PUBLIC_REMOTE_NAME" "$GFX950_BRANCH_NAME" >&2
    echo "remotes/${PUBLIC_REMOTE_NAME}/${GFX950_BRANCH_NAME}"
}

do_undo() {
    echo "=== UNDO: Restoring files from branch $OURS_BRANCH ==="
    echo ""

    cd "$REPO_ROOT"

    count=0
    while IFS= read -r filepath; do
        [[ -z "$filepath" ]] && continue
        [[ "$filepath" == *rocisa* ]] && continue

        git show "${OURS_BRANCH}:${filepath}" > "${REPO_ROOT}/${filepath}" 2>/dev/null
        count=$((count + 1))
    done < "$CONFLICTED_FILES"

    echo "Restored $count files to their original state (from $OURS_BRANCH)."
    echo "Done."
}

do_apply() {
    echo "=== Applying conflict markers to all non-rocisa files ==="
    echo ""

    cd "$REPO_ROOT"

    GFX950_REF=$(find_gfx950_ref)
    echo "Using refs:"
    echo "  Ours:   $OURS_BRANCH ($(git log --oneline -1 "$OURS_BRANCH"))"
    echo "  Theirs: $GFX950_REF ($(git log --oneline -1 "$GFX950_REF"))"

    MERGE_BASE=$(git merge-base "$OURS_BRANCH" "$GFX950_REF")
    echo "  Base:   $MERGE_BASE ($(git log --oneline -1 "$MERGE_BASE"))"
    echo ""

    TMPDIR=$(mktemp -d)
    trap "rm -rf $TMPDIR" EXIT

    total=0
    applied=0
    clean=0
    skipped=0

    while IFS= read -r filepath; do
        [[ -z "$filepath" ]] && continue
        total=$((total + 1))

        if [[ "$filepath" == *rocisa* ]]; then
            skipped=$((skipped + 1))
            continue
        fi

        BASE_FILE="$TMPDIR/base"
        OURS_FILE="$TMPDIR/ours"
        THEIRS_FILE="$TMPDIR/theirs"

        git show "${MERGE_BASE}:${filepath}" > "$BASE_FILE" 2>/dev/null || echo "" > "$BASE_FILE"
        git show "${OURS_BRANCH}:${filepath}" > "$OURS_FILE" 2>/dev/null || echo "" > "$OURS_FILE"
        git show "${GFX950_REF}:${filepath}" > "$THEIRS_FILE" 2>/dev/null || echo "" > "$THEIRS_FILE"

        TARGET_FILE="${REPO_ROOT}/${filepath}"

        git merge-file -p \
            --diff3 \
            -L "apply_all" \
            -L "base" \
            -L "gfx950mx" \
            "$OURS_FILE" "$BASE_FILE" "$THEIRS_FILE" > "$TARGET_FILE" || true

        conflicts=$(grep -c "^<<<<<<< apply_all" "$TARGET_FILE" 2>/dev/null) || conflicts=0
        if [ "$conflicts" -gt 0 ]; then
            echo "  [CONFLICT]  $filepath  ($conflicts regions)"
            applied=$((applied + 1))
        else
            echo "  [CLEAN]     $filepath"
            clean=$((clean + 1))
        fi
    done < "$CONFLICTED_FILES"

    echo ""
    echo "========================================"
    echo "  Total files:     $total"
    echo "  Skipped (rocisa): $skipped"
    echo "  With conflicts:  $applied"
    echo "  Clean merges:    $clean"
    echo "========================================"
    echo ""
    echo "To undo:  $0 --undo"
    echo "Done."
}

if [ "${1:-}" = "--undo" ]; then
    do_undo
else
    do_apply
fi
