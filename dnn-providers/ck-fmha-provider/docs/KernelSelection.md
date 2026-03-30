# CK FMHA Plugin: Kernel Selection

## Kernel Inventory

The plugin ships **96 precompiled kernels** for gfx950, covering the hipDNN
SDPA forward feature surface.

### Cross-product breakdown

| Axis | Values | Count |
|------|--------|-------|
| hdim groups | h32, h64, h80x96, h96x128, h128, h160, h192x128, h192 | 8 |
| dtypes | fp16, bf16 | 2 |
| masks | no_mask, causal (top_left) | 2 |
| LSE | off, on | 2 |
| padding variants | pad_d only, pad_all | ~1.5 avg |

8 x 2 x 2 x 2 x ~1.5 = **~96 kernels**

### Tile configuration per hdim

Each hdim group uses exactly one tile -- the 100% win-rate best tile from
empirical benchmarking on gfx950 (see `tile_engine/ops/fmha/INSIGHTS.md`):

| hdim | Pipeline | Tile (m0 x n0 x k0) | Notes |
|------|----------|---------------------|-------|
| h32 | qr_async | 128 x 64 x 16 | Small hdim, small tile |
| h64 | qr_async | 128 x 64 x 32 | Sub-warp loads |
| h80x96 | qr_async | 128 x 128 x 16 | Irregular hdim, needs k0=16 |
| h96x128 | qr_async | 128 x 128 x 32 | Asymmetric hdim |
| h128 | qr_async | 128 x 128 x 32 | Primary target (Llama, Qwen, etc.) |
| h160 | qr_async | 128 x 128 x 32 | Mid-range hdim |
| h192x128 | qr_async | 128 x 128 x 32 | MLA (DeepSeek R1) |
| h192 | qr_async | 128 x 128 x 32 | Symmetric h192 |
| h256 | qr | 128 x 128 x 32 | Not yet shipped (uses qr, not qr_async) |

The pipeline rule is deterministic: `qr_async` for all hdim < 256, `qr` for
h256. The tile rule is deterministic: `128x128x32` for h96+, smaller tiles for
h32/h64/h80.

## Runtime Selection Path

When `graph.build(handle)` is called through the hipDNN API:

```
CkFmhaFwdPlanBuilder::isApplicable(handle, graph)
    CkFmhaParamParser::parseFwdGraph(graph)
        Extract from hipDNN SdpaAttributes:
          dtype      HALF -> "fp16", BFLOAT16 -> "bf16"
          hdim_q     from Q tensor dim[3]
          hdim_v     from V tensor dim[3]
          nhead_q    from Q tensor dim[1] (BHSD) or dim[2] (BSHD)
          nhead_k    from K tensor dim[1] or dim[2]
          seqlen_q   from Q tensor dim[2] or dim[1]
          seqlen_k   from K tensor dim[2] or dim[1]
          mask_type  causal_mask -> 1, causal_mask_bottom_right -> 2,
                     left_bound/right_bound -> 3, else -> 0
          bias_type  attn_mask_tensor_uid present -> 1, alibi_mask -> 2, else -> 0
          has_lse    generate_stats or stats_tensor_uid present
          has_dropout seed_tensor_uid + offset_tensor_uid present
          scale      attn_scale_value or 1/sqrt(hdim_q)
          layout     stride[1] > stride[2] -> BHSD, else BSHD

    FmhaProblemBuilder::build()
        Construct FmhaProblem with 25+ fields including all the above

    dispatcher->select_kernel(problem)
        FmhaDispatcher::select_first_fit(problem)
```

## select_first_fit: Seqtune-Aware Selection

The dispatcher iterates all registered kernels and picks the best using
a tuple-based scoring system. Lower score wins.

### Step 1: Filter by signature

For each kernel, `kernel.supports(problem)` checks:

- **`fmha_signature_matches`**: dtype, hdim_q, hdim_v, mask compatibility
  (causal top-left and bottom-right share one kernel), bias, lse, dropout,
  group_mode, paged_kv, fp8, logits_soft_cap all must match

- **`fmha_algorithm_supports`**: tile padding can handle actual seqlen/hdim,
  variable-seqlen constraints, max_seq_len_q alignment

Only kernels passing both checks are candidates.

### Step 2: Score by seqtune categories

Each surviving candidate is scored as `(category, selection_rank, tile_m0)`:

```
effective_max_sq = max_seqlen_q > 0 ? max_seqlen_q : seqlen_q
aligned = (tile_m0 > 0) && (effective_max_sq > 0) && (effective_max_sq % tile_m0 == 0)

Category 0: seqlen_q <= tile_m0 AND aligned
            Perfect single-tile fit. Smallest tile wins.
            Example: sq=64 with tile_m0=64 -> category 0

Category 1: tile_m0 == 64
            Unconditional fallback. The 64-wide tile handles everything.

Category 2: tile_m0 == max_tile_m0 (largest tile among all candidates)
            Catch-all for large sequences.

Category 3: aligned (effective_max_sq % tile_m0 == 0)
            No padding waste, but not a perfect single-tile fit.
            Example: sq=256 with tile_m0=128 -> category 3

Category 4: needs padding
            Last resort. Tile doesn't divide sequence length evenly.
            Example: sq=200 with tile_m0=128 -> category 4
```

### Step 3: Tiebreak

Within the same category, prefer lower `selection_rank` (kernel-level
priority), then smaller `tile_m0` (less padding waste).

### Step 4: Cache

The resolved `FmhaExecutionPlan` is cached using `canonical_key()` which
encodes all 25+ selection-relevant fields (api_family, dtype, arch, hdim_q,
hdim_v, group_mode, v_rowmajor, logits_soft_cap, lse, dropout, mask_type,
bias_type, qscale_type, rope_type, paged_kv, fp8_static_quant,
skip_min_seqlen_q, sink, dbias, store_randval, deterministic,
kv_memory_layout, kv_lookup_table, page_size). Cache is mutex-protected
for thread safety.

## Concrete Example

Shape: `B=4 Hq=32 Hk=8 Sq=2048 Sk=2048 D=128 fp16 no_mask no_lse`

1. **Parse**: dtype="fp16", hdim_q=128, hdim_v=128, mask_type=0, bias_type=0,
   has_lse=false, nhead_q=32, nhead_k=8 (GQA 4:1)

2. **Signature match**: Of 96 kernels, only h128 fp16 no_mask no_lse variants
   pass. That's ~2-4 candidates (different padding configs).

3. **Seqtune scoring**:
   - `effective_max_sq = 2048`
   - Candidate A: `128x128x32 pad(1,0,1,1)` -> aligned (2048 % 128 == 0) ->
     category 3, rank=0, tile_m0=128
   - Candidate B: `128x128x32 pad(1,1,1,1)` -> aligned but has extra padding ->
     category 4, rank=0, tile_m0=128
   - **Winner: Candidate A** (category 3 < category 4)

4. **Execution**: `dispatcher->run()` launches kernel on HIP stream

5. **Result**: **406 TFLOPS** on MI355X gfx950

## Heuristic Callback (Future)

The dispatcher supports a `FmhaHeuristicFunction` callback registered via
`set_heuristic()`:

```cpp
using FmhaHeuristicFunction = std::function<std::vector<std::string>(const FmhaProblem&)>;

dispatcher->set_heuristic([](const FmhaProblem& p) -> std::vector<std::string> {
    // Return ordered list of kernel IDs to try
    // Falls back to select_first_fit if none match
    return {"fmha_fwd_fp16_batch_h128_qr_async_...", ...};
});
```

This enables ML-driven kernel selection (Stage C2 in the RFC): train a
LightGBM model on benchmark data, deploy as a C++ callback that ranks
the 96 precompiled kernels in < 5ms.

## Comparison with AITER

AITER (`aiter/ops/mha.py`) uses **compile-time feature filtering** rather
than runtime kernel selection:

| Aspect | AITER | CK Dispatcher (this plugin) |
|--------|-------|---------------------------|
| Tile selection | Fixed per receipt | Seqtune-aware runtime scoring |
| Runtime kernel choice | None | `select_first_fit` + heuristic callback |
| Sequence-length awareness | None | Short-seq optimization (category 0) |
| Padding awareness | None | Prefers aligned over padded |
| Multi-kernel registry | One .so per feature combo | 96 kernels in single plugin |
| Extensibility | Hardcoded filter strings | `FmhaHeuristicFunction` callback |

AITER's advantage: v3 ASM pipeline kernels (gfx942-specific) that outperform
CK tile kernels for bf16 h128 seqlen > 128. These are not yet integrated
into the hipDNN plugin.

## JIT Fallback

When `CK_FMHA_ENABLE_JIT=1`, the plugin transparently JIT-compiles kernels
for shapes that have no precompiled match.

### Flow

```
isApplicable(handle, graph)
  select_kernel(problem) -> nullptr
  handle.jitAndLoad(problem)
    check CK_FMHA_ENABLE_JIT=1
    jit_compile_kernel(problem, arch)
      fork/execvp -> python3 -> setup_fmha_dispatcher
        generate_fmha_fallback.py -> hipcc -> .so
      capture stdout -> .so path
    load_jit_library(so_path, registry, arch)
      dlopen -> dlsym("ck_fmha_register_kernels") -> merge
    select_kernel(problem) -> non-null
  return true
```

### How JIT integrates with seqtune selection

The JIT-compiled kernel is registered into the same live `FmhaRegistry`.
After JIT, `select_first_fit()` picks it using the same seqtune scoring
as precompiled kernels. The JIT kernel participates in category/rank/tile
scoring identically -- it has no special priority or bypass.

### Timing

| Phase | Time |
|-------|------|
| First JIT (codegen + hipcc + link) | 16-31s |
| Cached JIT (.so exists on disk) | < 1s |
| Registry lookup after JIT | < 1ms |

### Environment variables

| Variable | Required | Default |
|----------|----------|---------|
| `CK_FMHA_ENABLE_JIT` | Yes (opt-in) | off |
| `CK_DISPATCHER_PYTHON_PATH` | Yes for JIT | -- |
| `CK_FMHA_JIT_CACHE_DIR` | No | `/tmp/ck_fmha_jit` |
