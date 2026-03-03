# Skill: Regenerate CK Tile Engine Operation Support Matrix

This document contains all instructions needed for a Claude Code instance to regenerate the `operation_support_matrix.md` file from scratch by scanning the composable_kernel source code. It is self-contained and does not depend on any external configuration.

**Prerequisite:** This skill must be run from within a composable_kernel source tree. All file paths below are relative to the **CK root** — the directory containing `include/ck_tile/`, `example/ck_tile/`, and `tile_engine/`. This works in two repo layouts:

- **Standalone repo** (`github.com/ROCm/composable_kernel`): the CK root is the repository root.
- **Super repo** (`github.com/ROCm/rocm-libraries`): the CK root is `projects/composablekernel/`.

Before starting, locate the CK root by finding the directory that contains `include/ck_tile/ops/`. All paths in this document are relative to that directory.

---

## 1. Purpose and Output Format

**Goal:** Produce a file called `operation_support_matrix.md` in the `tile_engine/` directory.

**Content:** A GitHub-flavored markdown document containing (in this exact order):
1. A brief introductory paragraph
2. A support matrix table
3. Numbered footnotes with kernel-specific notes (immediately after the table)
4. A legend
5. A layout codes explanation
6. A data type mapping table

### Table Structure

The table has 17 columns:

```
| Op | CK Tile Kernel | fp16 | fp8 | bf16 | bf8 | int8 | fp4 | fp6 | rcr | rrr | ccr | crr | 90a | 942 | 950 | 1201 |
```

- **Op** — Operation category (center-aligned)
- **CK Tile Kernel** — Multi-line cell (left-aligned), containing:
  - Line 1: kernel name, followed by a space then footnote references like `[1][2]` (if any)
  - Line 2 (if tile engine supports it): `engine: \`<tile_engine_ops_dir>/\``
  - Line 3 (if an example exists): `example: \`<example_dir>/\``
  - Lines are separated by `<br>` tags. Omit lines that don't apply.
  - **Formatting:** Wrap engine and example directory paths in backticks within the table cell. Include a space between the kernel name and the first footnote reference.
- **fp16 through fp6** — Data type support (center-aligned)
- **rcr through crr** — Layout support (center-aligned)
- **90a through 1201** — GPU target support, short names for gfx90a/gfx942/gfx950/gfx1201 (center-aligned)

### Three-State Cell Convention

Each indicator cell uses one of three states:

| Symbol | Meaning |
|--------|---------|
| ✅ | CK Tile implementation exists **and** tile engine supports this combination |
| ❌ | CK Tile implementation exists but tile engine does **not** support it |
| *(blank)* | No CK Tile implementation exists for this combination |

---

## 2. Data Collection Procedure

Scan the following locations to build the data needed for the matrix. All paths are relative to the repository root.

### 2.1 Enumerate CK Tile Operations

Scan `include/ck_tile/ops/` for subdirectories. Each directory represents a potential CK Tile operation. Exclude infrastructure directories using this procedure:

**Infrastructure detection:** A directory under `include/ck_tile/ops/` is infrastructure (not a standalone operation) if ALL of the following are true:
1. It has no `kernel/` subdirectory
2. It has no matching example directory in `example/ck_tile/` (no `NN_<dirname>/` or obvious name match)
3. Its contents are used as building blocks by another operation (e.g., `topk/` provides blocks for `topk_softmax/`) OR it contains only shared utility headers (e.g., `common/`, `epilogue/`)

**IMPORTANT:** Directories that lack a `kernel/` subdirectory are NOT automatically infrastructure. Many real operations only have `block/` subdirectories (e.g., `softmax/`, `norm_reduce/`). Only exclude a directory if ALL three criteria above are met. When in doubt, INCLUDE the directory as an operation. The bar for exclusion is high — you must be certain the directory is pure infrastructure.

**Exclude standalone `.hpp` files** (like `moe_flatmm.hpp`) — only directories represent operations.

The remaining directories are the complete list of CK Tile operations.

### 2.2 Enumerate CK Tile Examples

Scan `example/ck_tile/` for numbered directories (e.g., `01_fmha/`, `03_gemm/`). Each maps to a CK Tile operation. Record the mapping from example directory name to operation.

### 2.3 Enumerate Tile Engine Operations

Recursively search `tile_engine/ops/` for `*_instance_builder.py` files. Exclude any base class builders (files that define a parent class imported by other builders but are not invoked directly by CMake — identify these by checking the `CMakeLists.txt` in the same or parent directory; a base class builder will not appear in any `execute_process()` or `add_custom_command()` CMake call). Each remaining builder represents a tile engine-supported operation.

For each discovered builder:
1. Derive the operation name from the builder filename (e.g., `gemm_universal_instance_builder.py` → `gemm_universal`). **Exception:** if the builder's directory name differs from the kernel name (e.g., directory `gemm_streamk/` but kernel is `streamk_gemm`), check the generated C++ include path or the kernel class name in the builder to determine the correct kernel name.
2. Record the directory path relative to `tile_engine/ops/` as the engine directory.
3. Find the `CMakeLists.txt` in the same directory as the builder.
4. Search for `*_validation_utils.py` or `*_parameter.py` in the same directory or a shared parent directory (e.g., GEMM variant builders share `tile_engine/ops/gemm/gemm_validation_utils.py`).

### 2.4 Extract Tile Engine Support Parameters

For each tile engine operation discovered in 2.3, read the associated files to determine supported data types, layouts, and GPU targets:

**File 1 — CMakeLists.txt** (found in 2.3 step 3):
- Look for `set(<PREFIX>_DATATYPE "..." CACHE STRING ...)` — semicolon-separated default data types
- Look for `set(<PREFIX>_LAYOUT "..." CACHE STRING ...)` — semicolon-separated default layouts
- Look for `set(DESIRED_TARGETS "...")` — semicolon-separated GPU targets

**File 2 — Instance builder Python file** (found in 2.3):
- Look for `add_argument("--datatype"` and find the `choices=[...]` list — this is the maximum set of data types the builder can accept (may be broader than CMake defaults)
- Look for `add_argument("--layout"` and find the `choices=[...]` list — this is the maximum set of layouts the builder can accept

**File 3 — Validation utilities** (found in 2.3 step 4):
- For each `*_validation_utils.py`, look for dicts named `*_WARP_TILE_SUPPORTED_COMBINATIONS` or similar — keyed by GPU target, then by dtype combo string (e.g., `"fp16_fp16_fp16"`, `"fp8_fp8_fp16"`, `"int8_int8_int32"`)
- For each `*_parameter.py`, look for `TYPE_MAP` dicts or equivalent dtype mapping structures
- Note: multiple builders may share a single validation utils file (e.g., GEMM variants sharing a common validation module). Read the builder's imports to determine which validation dict applies to which operation (e.g., a preshuffle builder may use `GEMM_PRESHUFFLE_WARP_TILE_SUPPORTED_COMBINATIONS` rather than the general `GEMM_WARP_TILE_SUPPORTED_COMBINATIONS`).

---

## 3. Operation Discovery and Taxonomy

The matrix must reflect the **current state of the source code**, not a hardcoded list. Use dynamic discovery as the primary mechanism, with the known operations reference below for category assignment and validation.

### 3.1 Dynamic Discovery Steps

**Step A — Discover all CK Tile operations:**

1. List all subdirectories in `include/ck_tile/ops/`. Exclude infrastructure directories using the procedure from section 2.1 (directories with no `kernel/` subdirectory, no matching example, and whose contents serve as building blocks for other operations). Exclude standalone `.hpp` files (e.g., `moe_flatmm.hpp`).
2. **Operation naming:** For most directories, the operation name IS the directory name. For long names, use a shortened form if it's standard (e.g., `grouped_convolution` → `grouped_conv`). Each directory = one matrix row, UNLESS it contains multiple distinct operations (see step 3).
3. **Multi-operation directories:** Some directories contain multiple distinct operations, each with its own `*_kernel.hpp` header. Scan `kernel/` for files matching `*_kernel.hpp`. Group by base name — headers that share a common prefix (e.g., `multi_reduce2d_kernel.hpp`, `multi_reduce2d_multiblock_kernel.hpp`) are sub-variants of one operation; headers with distinct base names (e.g., `multi_reduce2d_*` vs `reduce2d_*`) are separate operations. Each distinct operation gets its own matrix row. Known multi-operation directories:
   - `gemm/kernel/`: contains many GEMM variants (`universal_gemm`, `batched_gemm`, `block_scale_gemm`, etc.) — each gets its own row. Note: some are in subdirectories (e.g., `streamk_gemm/`).
   - `gemm_quant/kernel/`: contains `gemm_quant` and `grouped_gemm_quant` — each gets its own row.
   - `reduce/kernel/`: contains `multi_reduce2d` and `reduce2d` — each gets its own row.
   - `flatmm/kernel/`: contains multiple flatmm variants (`flatmm`, `mx_flatmm`, `moe_flatmm`, etc.) — collapse into a single `flatmm` row because they represent the same operation with different precision modes.
   When discovering a new multi-kernel directory, use judgment: if the kernels serve fundamentally different problem types (like different GEMM flavors), expand to separate rows. If they're precision/scheduling variants of one operation, collapse to one row.

**Step B — Discover all examples:**

1. List all numbered directories in `example/ck_tile/` (e.g., `01_fmha/`, `03_gemm/`).
2. Match each example to an operation by name: strip the numeric prefix and match against discovered operation names.
   - **Exact match ONLY:** Strip the numeric prefix from each example directory name (e.g., `01_fmha/` → `fmha`, `38_block_scale_gemm/` → `block_scale_gemm`). An example maps to an operation ONLY if the stripped name exactly equals the operation name. `38_block_scale_gemm/` maps to `block_scale_gemm`, NOT to `gemm_quant` or any other operation.
   - **Known non-obvious mappings:** `04_img2col/` → `image_to_column`, `20_grouped_convolution/` → `grouped_conv`, `05_reduce/` → both `reduce2d` AND `multi_reduce2d` (shared example).
   - **Do NOT use substring matching.** An example must match the operation name specifically — `09_topk_softmax/` maps to `topk_softmax` only, NOT to `softmax`. Similarly, `05_reduce/` maps to `reduce2d` and `multi_reduce2d` (both reduce ops share one example), NOT to `norm_reduce`.
   - **GEMM variant mapping:** `03_gemm/` maps to `gemm_universal` only (the basic GEMM). Do NOT map it to other GEMM variants like `gemm_preshuffle`, `batched_gemm`, etc. Each GEMM variant must have its own dedicated example directory (e.g., `40_streamk_gemm/`, `16_batched_gemm/`) or no example at all.
   - **No-match rule:** If no example directory matches an operation, the operation has no example. Do NOT fabricate example directory names — only use directory names that actually exist in `example/ck_tile/`. Verify by listing the directory contents. Operations known to have no example: `gemm_quant`, `grouped_gemm_quant`, `gemm_preshuffle`, `softmax`, `norm_reduce`.
   - **Unmatched examples create new operations (selective):** After matching all examples to operations from step A, check for example directories whose stripped names did NOT match any discovered operation. These MAY represent additional operations with no dedicated `include/ck_tile/ops/` directory. Apply this filter: only add an unmatched example as a new operation if it contains GEMM-related kernel invocations and represents a distinct computation (e.g., `38_block_scale_gemm/` is a distinct GEMM variant using quantized block scaling). Do NOT add examples that are:
     - Infrastructure/utility examples (e.g., `13_moe_sorting/` — a sorting utility, not a standalone kernel)
     - Sub-components of another operation (e.g., `14_moe_smoothquant/` — a preprocessing step for MoE)
     - Layout-only variants (e.g., `37_transpose/` — a transpose utility)

**Step C — Discover all tile engine operations:**

1. Recursively search `tile_engine/ops/` for `*_instance_builder.py` files.
2. Each builder file corresponds to a tile engine-supported operation.
3. For each builder found, also locate the associated `CMakeLists.txt` in the same directory and any validation utility files.

**Step D — Discover dtype support for non-engine operations:**

For operations WITHOUT tile engine support, determine which data types CK Tile implements by scanning the operation's example source code:

1. Look for C++ type aliases in example `.cpp`/`.hpp` files: `ck_tile::half_t` (fp16), `ck_tile::bf16_t` (bf16), `ck_tile::fp8_t` (fp8), `ck_tile::bf8_t` (bf8), `ck_tile::int8_t` (int8), `ck_tile::fp4_t` or `ck_tile::f4_t` (fp4), `ck_tile::fp6_t` or `ck_tile::f6_t` (fp6).
2. Look for `--dtype` or `--data_type` argparse options in example Python wrapper scripts (if any) — the `choices=` list indicates supported types.
3. Check the ops kernel headers under `include/ck_tile/ops/<op>/` for template parameter constraints or supported type lists. Also check the sibling `.hpp` header (e.g., `include/ck_tile/ops/softmax.hpp`) for type references.
4. If a dtype is referenced anywhere in the example or kernel code for this operation, CK Tile supports it — mark as ❌ in the matrix. If no reference exists, leave blank.
5. **Fallback for template-generic operations:** If an operation's kernel code uses only generic templates with no explicit dtype references (common for simple operations like softmax, layernorm, etc.), check the operation's sibling top-level `.hpp` file and any example that uses this operation. If the operation appears to be type-generic (accepting any float type), mark fp16 and bf16 as ❌ (these are the standard float types used throughout CK Tile). Do NOT assume fp8/bf8/int8/fp4/fp6 support unless explicitly referenced.

### 3.2 Category Assignment

Assign each discovered operation to a category using these naming rules. Apply rules in order; first match wins:

   - If the kernel is under `include/ck_tile/ops/gemm/` or `include/ck_tile/ops/gemm_quant/`, or contains "gemm" in its name → **GEMM**
   - If the name contains "contraction" or "flatmm" → **GEMM** (these are GEMM variants that use GEMM pipelines internally)
   - If the name contains "reduce" and NOT "norm" → **Reduce**
   - If the name contains "attn" or "attention" or "fmha" → **Attention**
   - If the name contains "softmax" → **Activation**
   - If the name contains "conv" → **Conv**
   - If the name contains "transpose", "permute", "img2col", or "image_to_column" → **Data Move**
   - If the name contains "elementwise" → **Elementwise**
   - If the name contains "moe" → **MoE**
   - If the name contains "norm" or "layernorm" or "rmsnorm" → **Norm**
   - If the name contains "pool" → **Pooling**
   - If the name contains "quant" and not "gemm" → **Quant**
   - Otherwise → use the directory name as a new category, and add a `<!-- NEW CATEGORY -->` comment in the output

---

## 4. Classification Algorithm

For each (operation, column) cell in the matrix, apply these rules:

### Data Type Columns (fp16, fp8, bf16, bf8, int8, fp4, fp6)

```
IF operation has NO CK Tile implementation for this dtype:
    cell = blank
ELIF operation has tile engine support AND builder accepts this dtype
     AND validation utils have entries for this dtype:
    cell = ✅
ELSE:
    cell = ❌
```

**How to determine "CK Tile implementation exists for this dtype":**

For ALL operations (with or without tile engine), determine dtype support dynamically:

1. **Operations WITH tile engine support:** Check the operation's own validation utils dict (e.g., `WARP_TILE_SUPPORTED_COMBINATIONS` or `GEMM_WARP_TILE_SUPPORTED_COMBINATIONS`) for dtype keys. The key format is `<a_type>_<b_type>_<c_type>` (e.g., `fp16_fp16_fp16`, `fp8_fp8_fp16`, `int8_int8_int32`). If an entry exists for ANY GPU target, CK Tile has an implementation.
   - **Shared vs variant-specific validation dicts:** When multiple builders share a validation file, check which dict the builder actually uses (follow the builder's imports and function calls). Entries in a shared dict only count for a specific variant if that variant's builder references them. For example, if `gemm_multi_d` only has a fp16 example and builder, inherited shared dict entries for fp8/bf16/bf8 do NOT constitute gemm_multi_d-specific CK Tile support — mark those as **blank**.

2. **Operations WITHOUT tile engine support:** Use section 3.1 step D's dynamic scanning procedure:
   - Scan the operation's example source code (`.cpp`/`.hpp` files in `example/ck_tile/<NN>_<name>/`) for CK Tile type aliases: `ck_tile::half_t` (fp16), `ck_tile::bf16_t` (bf16), `ck_tile::fp8_t` (fp8), `ck_tile::bf8_t` (bf8), `ck_tile::int8_t` (int8), `ck_tile::fp4_t` or `ck_tile::f4_t` (fp4), `ck_tile::fp6_t` or `ck_tile::f6_t` (fp6).
   - Also scan `include/ck_tile/ops/<op>/` kernel headers for type references.
   - If a dtype is referenced anywhere in the example or kernel code, CK Tile supports it → mark as ❌. If not referenced → blank.

**Important:** For operations where CK Tile has *any* implementation at all (any dtype supported), the GPU columns should always be ❌ (not blank), because CK Tile kernels are architecturally GPU-generic.

**How to determine "tile engine supports this dtype":**
- The builder's argparse `choices=` for `--datatype` must include the dtype label.
- The operation's validation dict must have at least one entry for a dtype combo matching this label on at least one supported GPU.

**For tile engine ops where CK Tile supports more dtypes than the builder accepts:**
- If the builder does NOT accept the dtype but CK Tile does support it (e.g., multi_reduce2d with bf16, or gemm_universal with int8), mark as ❌ (not blank). Determine CK Tile support using the validation utils dicts, example source code, or config parameter files (e.g., `reduce_parameter.py` TYPE_MAP).

### Layout Columns (rcr, rrr, ccr, crr)

```
IF operation category is NOT "GEMM":
    cell = blank  (layouts do not apply to non-GEMM operations)
ELIF operation has NO CK Tile implementation for ANY dtype:
    cell = blank
ELIF operation has tile engine support AND builder/CMake accepts this layout:
    cell = ✅
ELSE:
    cell = ❌  (this includes ALL non-engine GEMM ops: gemm_quant, grouped_gemm_quant,
                block_scale_gemm, batched_gemm, flatmm, etc. — they all get ❌
                for all 4 layout columns because CK Tile GEMM kernels are layout-generic)
```

**How to determine layout support:**
- For tile engine ops: check the builder's `choices=` for `--layout` (NOT the CMake default). The builder `choices=` defines what the tile engine *can* support. The CMake default (e.g., `GEMM_STREAMK_LAYOUT "rcr"`) is just the default build configuration -- if the builder accepts a layout, the tile engine supports it. Mark as ✅.
- For non-tile-engine GEMM ops: ALL standard GEMM layouts (rcr, rrr, ccr, crr) should be marked ❌. This includes GEMM variants under `gemm_quant/` (e.g., `gemm_quant`, `grouped_gemm_quant`), `block_scale_gemm`, `flatmm`, `batched_gemm`, `batched_contraction`, `gemm_multi_abd`, `grouped_gemm`, etc. — any operation in the GEMM category that has a CK Tile implementation gets ❌ for all 4 layout columns because CK Tile GEMM kernels are layout-generic via template parameters.

### GPU Target Columns (90a, 942, 950, 1201)

```
IF operation has NO CK Tile implementation for ANY dtype:
    cell = blank
ELIF operation has tile engine support AND this GPU is in DESIRED_TARGETS:
    cell = ✅
ELSE:
    cell = ❌  (CK Tile kernels are arch-generic; ❌ means tile engine hasn't added target)
```

**Important:** All CK Tile kernels are architecturally generic -- they have no compile-time GPU guards. The GPU filtering in the tile engine is a validation/testing scope decision. So any operation with a CK Tile implementation gets ❌ (not blank) for GPUs not in the tile engine's DESIRED_TARGETS.

---

## 5. Edge Case Procedures

These procedural rules handle situations where the general classification algorithm needs refinement. Apply them by reading the relevant source files.

### 5.1 Shared Validation Dicts vs Per-Variant Support

Multiple tile engine GEMM builders may share a single validation utils file (e.g., `gemm_validation_utils.py`). The validation dict (e.g., `GEMM_WARP_TILE_SUPPORTED_COMBINATIONS`) may contain dtype entries that apply to some but not all variants.

**Procedure:** For each GEMM variant with tile engine support:
1. Read the builder's `choices=` for `--datatype` — this defines what the builder accepts.
2. Read the validation dict entries — this defines what tile configs exist.
3. A dtype is tile-engine-supported (✅) only if BOTH the builder accepts it AND the validation dict has entries for it.
4. If a dtype appears in the shared validation dict but NOT in the builder's `choices=`, determine whether CK Tile supports this dtype for this SPECIFIC variant:
   - If `gemm_universal`: CK Tile's universal GEMM kernel is type-generic → mark as ❌ (CK Tile supports, tile engine doesn't).
   - For other variants: check if there is a standalone CK Tile kernel implementation or example for this variant with this dtype. If not (the dtype only appears via shared validation code inheritance), mark as **blank** (no CK Tile implementation for this variant).

### 5.2 Layout Restrictions

**Procedure:** Read each tile engine builder's `choices=` for `--layout`:
- If the builder accepts fewer layouts than the standard set (rcr, rrr, ccr, crr), mark unsupported layouts according to the CK Tile implementation:
  - If the CK Tile kernel/example for this variant does NOT support the layout (e.g., preshuffle's data format requires specific A/B ordering) → **blank** (no CK Tile support).
  - If the CK Tile kernel supports it but the tile engine doesn't → **❌**.
- If the builder uses a different layout code format (e.g., 4-character codes for gemm_multi_d with a D tensor layout), map to the standard 3-character A/B/C format for display.

### 5.3 GPU Targets

**Procedure:** Read `DESIRED_TARGETS` from each tile engine operation's `CMakeLists.txt`. Some operations may not have a `--gpu_target` CLI flag — their GPU support comes solely from `DESIRED_TARGETS`. For all tile engine operations:
- GPUs IN `DESIRED_TARGETS` → ✅
- GPUs NOT in `DESIRED_TARGETS` → ❌ (CK Tile kernels are arch-generic; the tile engine just hasn't added this target)
- Check for TODO comments about pending GPU targets — include these in footnotes.

### 5.4 Non-GEMM Layout Columns

All non-GEMM operations do not use matrix layouts. Their layout columns (rcr, rrr, ccr, crr) should always be **blank**.

### 5.5 Exclusions

Infrastructure directories and standalone `.hpp` files are identified and excluded by the procedure in section 2.1. No additional hardcoded exclusion list is needed.

### 5.6 Builder Accepting Dtypes Without Configs

**Procedure:** If a builder's `choices=` includes a dtype but there are no default tile configurations for it (no entries in the validation dict and no default JSON config), mark as ❌ (CK Tile supports the dtype, but the tile engine has no configs). This indicates a coverage gap where configs could be added.

---

## 6. Composing the Output

### 6.1 Introductory Paragraph

Begin the file with:

```markdown
# CK Tile Engine Operation Support Matrix

This matrix shows all CK Tile operations with per-data-type, per-layout, and per-GPU support status. It uses a three-state convention: ✅ = supported by both CK Tile and tile engine, ❌ = supported by CK Tile but not yet in the tile engine, blank = not supported by CK Tile itself.
```

### 6.2 Table Formatting

- **Row ordering:** Within each category, list tile-engine-supported operations first (sorted alphabetically by kernel name), then non-tile-engine operations (sorted alphabetically by kernel name).
- **Category grouping:** The Op column repeats the category name for each row in that group.
- **Alignment row:** Op column center-aligned (`:--:`), CK Tile Kernel left-aligned (`---`), all indicator columns center-aligned (`:---:`).

### 6.3 Footnote System

Add numbered footnote references `[N]` in the CK Tile Kernel column of the table, adjacent to the kernel name. These link to notes below the table. Format: `kernel_name [N][M]` — note the space before the first bracket and NO space between consecutive brackets.

**When to generate a footnote:** For each tile engine operation, compare the builder's `choices=`, CMake defaults, validation utils, and `DESIRED_TARGETS`. Generate a footnote whenever any of these discrepancies are found:

1. **CMake default is a subset of builder capabilities:** The CMake `set(<PREFIX>_DATATYPE ...)` lists fewer dtypes than the builder's argparse `choices=`. Note which dtypes require a `-D` override to enable.
2. **CK Tile supports a dtype the builder does not accept:** A dtype key exists in the validation utils dict (e.g., `int8_int8_int32`) but the builder's `choices=` does not include it. Note the coverage gap.
3. **Builder accepts a dtype but has no tile configurations:** The builder accepts a dtype in `choices=` but there are no default tile configs (no JSON entries or no validation dict entries for that dtype). Note that building requires custom configs.
4. **Layout restrictions:** The builder accepts fewer layouts than other GEMM variants, or CMake defaults to a subset. Note any layout-specific constraints and their technical reason (e.g., data format requirements).
5. **GPU target restrictions:** `DESIRED_TARGETS` excludes GPUs that other similar operations support. Note pending targets, especially if there are TODO comments in CMakeLists.txt.
6. **Dtype gap between CK Tile and tile engine:** CK Tile example or kernel supports a dtype (e.g., bf16 in reduce) but the tile engine's config only covers a subset (e.g., fp16 only). Note the gap.
7. **Runtime adaptation notes:** If the kernel has notable runtime behavior (e.g., wave32/wave64 adaptation via `is_wave32()`), include this in the same footnote as the dtype gap for that operation.

Assign one or more consecutive footnote numbers per operation. Operations with multiple distinct discrepancies may have multiple footnotes. Combine related notes (e.g., dtype gap + runtime behavior) into a single footnote when they concern the same operation.

**Footnote format in the Notes section:**

```markdown
- [N] **kernel_name:** Description text with `inline code` as needed.
```

### 6.4 Legend

```markdown
**Legend:**
- **CK Tile Kernel column:** First line is the kernel name. Lines prefixed with "engine:" show the tile engine directory under `ops/`. Lines prefixed with "example:" show the CK Tile example directory under `example/ck_tile/`.
- **Green cell** (✅): CK Tile implementation exists **and** the tile engine supports it.
- **Red cell** (❌): CK Tile implementation exists **but** the tile engine does **not** support it.
- **Grey cell** (blank): No CK Tile implementation exists for this combination.
```

### 6.5 Layout Codes Explanation

Include a paragraph explaining layout codes: each character (r or c) specifies row-major or column-major for tensors A, B, and C. For gemm_multi_d, mention the 4-character code where the 4th character is the D tensor layout (always `r`).

### 6.6 Data Type Mapping Table

Include a reference table showing how each config label maps to actual tensor types (A, B, Acc, C). **Derive this table from the source code** by reading `populate_kernel_dtype_layout()` in `tile_engine/ops/gemm/gemm_instance_builder.py`:

1. The function shows: A type = B type = the config label's dtype. Accumulator = `"float"` (fp32) by default.
2. C type = same as the config label, EXCEPT when the function overrides it (e.g., `if self.datatype in ["fp8", "bf8"]: c_type = "fp16"`).
3. For int8: check `GEMM_WARP_TILE_SUPPORTED_COMBINATIONS` in `gemm_validation_utils.py` for the key format `int8_int8_int32` — this reveals A=int8, B=int8, Acc=int32, C=int32.
4. For fp4/fp6: these are NOT in the tile engine builder. Check `example/ck_tile/18_flatmm/` and `example/ck_tile/38_block_scale_gemm/` source code for their type mappings (fp4 uses mixed-precision with fp16/bf16 source A, fp4 B; fp6 uses fp6 for both A and B with fp32 accumulator and output).

Build the table with columns: Config Label, A (source), B (source), Acc, C (output). Include rows for all dtypes that appear in the matrix (fp16, bf16, int8, fp8, bf8, fp6, fp4). Order by descending precision: fp16, bf16, int8, fp8, bf8, fp6, fp4.

---

## 7. Validation Checklist

After generating the output, verify:

1. **Row count:** The table has one row per discovered operation. All directories from `include/ck_tile/ops/` (minus exclusions) and all GEMM kernel variants are represented.
2. **Tile engine ops have checkmarks:** Every operation with a tile engine directory has at least one ✅ in each support dimension (dtype, layout if GEMM, GPU).
3. **Non-tile-engine ops have no checkmarks:** Every operation without a tile engine directory has zero ✅ cells.
4. **Non-GEMM layout columns are blank:** All non-GEMM operations (including Reduce) have blank cells in all 4 layout columns.
5. **GPU columns:** Every operation with a CK Tile implementation has either ✅ or ❌ (not blank) in every GPU column.
6. **No duplicates:** No operation appears more than once.
7. **Footnote consistency:** Every `[N]` reference in the table has a matching `[N]` definition in the Notes section. Numbers are sequential.
8. **Category coverage:** Every operation should be assigned to a category using section 3.2's naming rules. No operation should have an uncategorized or `<!-- NEW CATEGORY -->` marker unless its name genuinely matches none of the existing patterns.
9. **Tile engine completeness:** Every `*_instance_builder.py` found in `tile_engine/ops/` should have a corresponding row in the matrix with ✅ marks based on the builder's capabilities.

---

## 8. Execution Summary

**The source code is always the single source of truth.** Whether generating from scratch or updating an existing matrix, the algorithm is the same — discover everything from the current source code and generate the complete matrix. The existence (or absence) of a prior `operation_support_matrix.md` is irrelevant to correctness; never rely on its contents for data.

To regenerate the matrix:

1. Read this entire SKILL.md document
2. Locate the CK root directory (section 1 prerequisite)
3. **Discover** all operations, examples, and tile engine support (section 3.1 steps A-D)
4. **Categorize** each discovered operation (section 3.2)
5. Follow section 2 to extract tile engine parameters (dtypes, layouts, GPU targets) from CMakeLists.txt, builders, and validation utils
6. **Classify** each cell using section 4's algorithm, with dynamic dtype scanning for non-engine ops
7. Apply section 5's edge cases
8. Format the output per section 6
9. Run section 7's validation checklist
10. Write the result to `tile_engine/operation_support_matrix.md`
