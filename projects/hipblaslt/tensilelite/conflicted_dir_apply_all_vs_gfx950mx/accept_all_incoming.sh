#!/bin/bash
#
# Usage:
#   ./accept_all_incoming.sh          Accept all incoming (gfx950mx) changes for non-rocisa files
#   ./accept_all_incoming.sh --undo   Undo: restore all files to their original state
#
# This script replaces each non-rocisa conflicted file with the gfx950_mx_rebase
# version directly -- no conflict markers, just accept theirs.
#
# rocisa files (31 files) are SKIPPED because their conflicts are already resolved.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFLICTED_FILES="${SCRIPT_DIR}/conflicted_files.txt"

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
        echo "  Restored: $filepath"
        count=$((count + 1))
    done < "$CONFLICTED_FILES"

    echo ""
    echo "Restored $count files to their original state (from $OURS_BRANCH)."
    echo "Done."
}

do_apply() {
    echo "=== Accepting all incoming (gfx950mx) changes for non-rocisa files ==="
    echo ""

    cd "$REPO_ROOT"

    GFX950_REF=$(find_gfx950_ref)
    echo "Using refs:"
    echo "  Ours (keeping as-is):  $OURS_BRANCH ($(git log --oneline -1 "$OURS_BRANCH"))"
    echo "  Theirs (accepting):    $GFX950_REF ($(git log --oneline -1 "$GFX950_REF"))"
    echo ""

    total=0
    accepted=0
    skipped=0
    not_in_theirs=0

    while IFS= read -r filepath; do
        [[ -z "$filepath" ]] && continue
        total=$((total + 1))

        if [[ "$filepath" == *rocisa* ]]; then
            skipped=$((skipped + 1))
            continue
        fi

        if ! git cat-file -e "${GFX950_REF}:${filepath}" 2>/dev/null; then
            echo "  [SKIP-NEW] $filepath  (only exists in ours, not in gfx950mx)"
            not_in_theirs=$((not_in_theirs + 1))
            continue
        fi

        TARGET_FILE="${REPO_ROOT}/${filepath}"
        git show "${GFX950_REF}:${filepath}" > "$TARGET_FILE"
        echo "  [ACCEPTED] $filepath"
        accepted=$((accepted + 1))
    done < "$CONFLICTED_FILES"

    echo ""
    echo "========================================"
    echo "  Total files:        $total"
    echo "  Skipped (rocisa):   $skipped"
    echo "  Skipped (new/ours): $not_in_theirs"
    echo "  Accepted theirs:    $accepted"
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
