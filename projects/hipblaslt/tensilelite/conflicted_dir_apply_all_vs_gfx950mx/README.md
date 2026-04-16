# conflicted_dir_apply_all_vs_gfx950mx

Conflict resolution toolkit for merging TensileLite changes between two branches:

- **Ours (apply_all):** `users/tomtang/gfx950_mx_rebase_tensilelite_to_1250_apply_all_conflicts`
- **Theirs (gfx950mx):** `gfx950_mx_rebase` (public ROCm repo)

109 files differ between the two branches: 31 are **rocisa** files (already resolved — all scripts skip them) and 78 are non-rocisa files.

---

## Files in this directory

| File | Description |
|------|-------------|
| `conflicted_files.txt` | List of all 109 differing files (full repo-relative paths) |
| `conflicted_files_with_diff_lines.txt` | Same list as a table, sorted ascending by diff line count |
| `apply_all_vs_gfx950mx.diff` | Full unified diff: ours → theirs |
| `gfx950mx_vs_apply_all.diff` | Full unified diff: theirs → ours |
| `apply_single_conflict.sh` | Add 3-way conflict markers to a **single** file |
| `apply_all_conflicts.sh` | Add 3-way conflict markers to **all** non-rocisa files at once |
| `accept_all_incoming.sh` | Overwrite **all** non-rocisa files with the gfx950mx version |

---

## Scripts

### 1. `apply_single_conflict.sh` — Single-file conflict markers

Adds 3-way merge conflict markers to one file so you can manually resolve it.

```bash
# By filename (searches conflicted_files.txt for a match)
./apply_single_conflict.sh KernelWriterAssembly.py

# By partial path
./apply_single_conflict.sh Components/GSU.py

# List all available files
./apply_single_conflict.sh
```

The conflict markers look like:

```
<<<<<<< apply_all
  (our version of the code)
||||||| base
  (common ancestor)
======= 
  (gfx950mx version of the code)
>>>>>>> gfx950mx
```

After running, open the file in your editor, resolve each conflict region, and remove the markers.

### 2. `apply_all_conflicts.sh` — Bulk conflict markers

Applies 3-way conflict markers to **all 78 non-rocisa files** at once. Useful when you want to review every conflict before choosing a resolution.

```bash
# Apply conflict markers to all non-rocisa files
./apply_all_conflicts.sh

# Undo — restore all files to their original state
./apply_all_conflicts.sh --undo
```

Output labels each file as `[CONFLICT]` (has conflict regions) or `[CLEAN]` (auto-merged cleanly).

### 3. `accept_all_incoming.sh` — Accept all gfx950mx changes

Replaces each non-rocisa file with the `gfx950_mx_rebase` version directly — no conflict markers, just accept theirs. 3 files that only exist in our branch (not in gfx950mx) are skipped automatically.

```bash
# Accept all incoming changes
./accept_all_incoming.sh

# Undo — restore all files to their original state
./accept_all_incoming.sh --undo
```

---

## First-time setup

All three scripts will **automatically** fetch `gfx950_mx_rebase` from the public ROCm repo if it isn't available locally. This requires internet access on the first run. Specifically, they:

1. Check for `gfx950_mx_rebase` under several known remote names
2. If not found, add `rocm-public` as a git remote pointing to `https://github.com/ROCm/rocm-libraries`
3. Fetch the `gfx950_mx_rebase` branch

No manual remote setup is needed.

---

## Recommended workflow

1. **Start from a clean worktree** on the `apply_all` branch
2. **Pick a strategy:**
   - **Option A — File by file:** Use `apply_single_conflict.sh` on individual files, resolve conflicts, then move to the next
   - **Option B — Bulk markers:** Run `apply_all_conflicts.sh`, then resolve all conflicts across files
   - **Option C — Accept theirs:** Run `accept_all_incoming.sh` to take all gfx950mx changes, then selectively revert files where our version is preferred
3. **Use the diff files** (`apply_all_vs_gfx950mx.diff` and `gfx950mx_vs_apply_all.diff`) as reference to understand what changed
4. **Use the table** (`conflicted_files_with_diff_lines.txt`) to prioritize — files with fewer diff lines are easier to resolve

---

## Notes

- **rocisa files are always skipped.** Their conflicts have been resolved separately and are compatible with `develop`.
- **3 files only exist in our branch** (`TensorDataMover.py`, `DataTypes_E5M3.hpp`, `DataTypes_E8.hpp`). These are skipped by `accept_all_incoming.sh` since there is no incoming version to accept.
- **Undo is safe.** Both `apply_all_conflicts.sh --undo` and `accept_all_incoming.sh --undo` restore files from the branch ref, not from a backup. This means undo always works regardless of how many times you've run the scripts.
