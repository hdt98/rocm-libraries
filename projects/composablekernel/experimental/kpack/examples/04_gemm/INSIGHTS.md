# Insights: GEMM through kpack — Validating the Schema

This example stress-tests the rocm_ck schema against a real 2D tiled kernel. GEMM is where complexity arrives: 3D tile geometry, hardware-constrained warp configurations, accumulator types independent of storage types, and fused epilogues that change both the computation and the ABI. Every insight here was earned by a failure or a design decision during implementation.

## 1. Bridge Pattern Extends to GEMM

The same bridge pattern from example 03 works for GEMM. A `.hip` variant file assembles CK Tile's 7-type template stack with `using` declarations, then wraps it in a thin `extern "C"` function:

```cpp
static constexpr rocm_ck::GemmKernel K = rocm_ck::make_kernel(
    rocm_ck::Signature{.dtype = rocm_ck::DataType::FP16, .ops = {rocm_ck::GemmOp{}}},
    rocm_ck::GemmAlgorithm{...});

extern "C" __global__ void gemm_fp16(rocm_ck::GemmArgs args) {
    rocm_ck::runGemm<K>(args);
}
```

The 7 CK Tile types (TileGemmShape, TileGemmTraits, PipelineProblem, GemmPipeline, TilePartitioner, CShuffleEpilogue, GemmKernel) are assembled in `gemm_dev.hpp`'s `runGemm<K>` from the `GemmKernel` NTTP fields. Each `.hip` file is ~15 lines. Host code never includes CK Tile headers.

> **Key difference from elementwise**: GEMM's bridge assembles 7 types (vs 4 for elementwise) and wires a 2D tile partitioner. But the pattern shape is identical — the bridge complexity scales with CK Tile's template depth, not with our schema.

## 2. Two-Axis Config: Signature × Algorithm

The `Signature` (what to compute) and `GemmAlgorithm` (how to compute it) are independent:

- **Signature**: data types, layouts, epilogue via operator composition
- **Algorithm**: `block_tile {M,N,K}`, `block_warps {M,N,K}`, `warp_tile {M,N,K}`

The same algorithm works across fp32, fp16, and bf16. The same type can use different tile configs (gemm_fp16 vs gemm_fp16_w32). This was proven by having 6 variants that mix types and tile shapes — no coupling between the axes.

> **Implication**: Tuning is purely an algorithm-space search. Given a signature, sweep tile configs and let `make_kernel` reject invalid combinations at compile time. The search space is defined by `is_valid_warp_gemm` × divisibility constraints.

## 3. Optional Dtype Hierarchy — Two Levels, Not Three

GEMM uses a two-level hierarchy (unlike elementwise's three):

```
dtype                 (kernel-level default)
├── a_dtype           (A input override)
├── b_dtype           (B input override)
├── c_dtype           (C output override)
└── acc_dtype         (accumulator, defaults to FP32 independently)
```

There is no `in_dtype` shared input default. GEMM's operands are asymmetric (A and B have different shapes and may have different types in mixed-precision training), so a shared input default would mislead rather than simplify.

> **Design principle**: Hierarchy depth is operation-dependent. Each operation defines the hierarchy shape that fits its semantics. Don't force a single hierarchy on all operations.

## 4. Accumulator Defaults Independently

`acc_dtype` defaults to FP32 regardless of `dtype`. Every practical GEMM accumulates in fp32 — even fp16×fp16 GEMM uses fp32 accumulation to avoid catastrophic precision loss over K iterations. CK Tile's `CShuffleEpilogue` handles the acc→output type conversion automatically.

This was flagged as medium risk during planning (would CShuffleEpilogue handle fp16 output from fp32 acc?) but worked on the first attempt across all three architectures.

## 5. Dim3: Tile Geometry as Structural Aggregates

`Dim3{m, n, k}` groups the natural M/N/K triplet:

```cpp
struct Dim3 { int m, n, k; };

GemmAlgorithm{
    .block_tile  = {128, 128, 32},
    .block_warps = {2, 2, 1},
    .warp_tile   = {16, 16, 16}
};
```

`Dim3` is a structural type — it works as an NTTP member. `K.block_tile.m` compiles cleanly in template arguments for `ck_tile::sequence<>`. Designated initializers make configs self-documenting.

> **Why not `ck_tile::sequence<M,N,K>`**: Our schema is CK-independent. The _dev header maps `Dim3` fields to `ck_tile::sequence<>` in one place. If CK Tile's API changes, only `gemm_dev.hpp` needs updating.

## 6. Hardware Constraints Flow Through consteval

The valid set of (dtype, warp_tile) combinations is hardware-specific and non-obvious. `is_valid_warp_gemm` is a consteval lookup table that mirrors CK Tile's `WarpGemmDispatcher` specializations for gfx9 (MFMA):

| dtype | 16×16 K values | 32×32 K values |
|-------|----------------|----------------|
| FP32  | 4, 8, 16       | 4, 8           |
| FP16  | 16, 32         | 8, 16          |
| BF16  | 16, 32         | 8, 16          |

`make_kernel` checks five constraints in sequence: positive dimensions, `block_warps.k == 1`, warp tile validity, tile divisibility, and derives `thread_block_size`. An invalid config throws a readable message at compile time.

> **Hard-won lesson**: The initial attempt used 256×256×64 block tile with 32×32×16 warp tile (copied from CK's `gemm_basic_invoker.hpp`). Failed on ALL architectures. Root causes: (1) no WarpGemmDispatcher for fp32 32×32×16, (2) 256×256 block tile requires 128KB LDS, exceeding 64KB limit. The fix — 128×128×32 with 16×16×16 — came from cross-referencing the dispatcher table with CK's own validated configs.

## 7. ABI Structs Per D-Tensor Count

Two fixed-size args structs, one per epilogue category:

| Struct | Size | D tensors | Fields |
|--------|------|-----------|--------|
| `GemmArgs` | 48 bytes | 0 | M, N, K, stride_A/B/C, a, b, c |
| `GemmArgs1D` | 64 bytes | 1 | M, N, K, stride_A/B/E, a, b, e, stride_D0, d0 |

Both have `static_assert` checks for size, alignment, and field offsets. `hipModuleLaunchKernel` passes args by value — variable-length structs would break.

`runGemm<K>` deduces `ArgsType` from the `extern "C"` function parameter. The `.hip` file picks the right args struct in its signature; the device code dispatches with `if constexpr(K.num_d_tensors == ...)`. Each branch is compiled away — no runtime overhead.

> **Design**: One struct per D-tensor count, not one struct with optional fields. This keeps the ABI contract explicit and verifiable with `offsetof`.

## 8. Epilogue Is Signature, Not Algorithm

Epilogue changes WHAT is computed (C = A×B vs E = A×B + D0 vs E = ReLU(A×B + D0)). It does NOT change HOW the matmul executes — CShuffleEpilogue handles all epilogue variants with the same shuffle strategy.

The operator-centric model expresses this naturally: operators in the signature's ops array compose the compute graph:

```cpp
// Plain GEMM
Signature{.dtype = FP16, .ops = {GemmOp{}}}

// GEMM + bias
Signature{.dtype = FP16,
          .ops = {GemmOp{.out = "C"},
                  AddOp{.lhs = "C", .rhs = "bias", .out = "D"}}}

// GEMM + bias + ReLU
Signature{.dtype = FP16,
          .ops = {GemmOp{.out = "C"},
                  AddOp{.lhs = "C", .rhs = "bias", .out = "D"},
                  ReluOp{.in = "D", .out = "E"}}}
```

`make_kernel` pattern-matches the ops array to determine epilogue configuration. Internally, this maps to `CombineOp` and `Activation` enums in the `GemmKernel` NTTP, which the device code uses to parameterize `ComposedCDEOp`.

> **Why operator composition over enums**: The old `CombineOp × Activation` enum model requires O(N×M) enum combinations for N combine ops and M activations. Operator composition is O(N+M) — add one operator struct, and it composes with everything. More importantly, users express what they want, not what the kernel builder anticipated.

## 9. CShuffleEpilogue Handles Everything

CK Tile's `CShuffleEpilogue` is the universal epilogue for GEMM:
- Handles acc→output type conversion (fp32 acc → fp16 output)
- Handles D tensor fusion (bias addition, scaling)
- Handles composed operations via a single functor parameter

`ComposedCDEOp<Combine, Act>` is a thin functor that converts to float, applies combine (fold over D tensor pack), applies activation (delegates to CK Tile's optimized implementations), and casts to output type. All epilogue variation is captured in one template — no `if constexpr` branching in `runGemm`.

## 10. What Generalized from Elementwise

| Pattern | Elementwise | GEMM | Verdict |
|---------|------------|------|---------|
| Config → consteval → Kernel | `ElementwiseConfig` → `VectorAddKernel` | `Signature + GemmAlgorithm` → `GemmKernel` | Generalizes |
| Optional dtype hierarchy | 3 levels (dtype→in_dtype→per-operand) | 2 levels (dtype→per-operand) | Shape varies by op |
| `_api` / `_dev` header split | Yes | Yes | Generalizes |
| `CkTypeMap` enum→type dispatch | Yes | Yes + `CkLayoutMap` | Generalizes (adds layout) |
| ABI struct with `void*` + offsets | 40 bytes, 6 asserts | 48/64 bytes, same pattern | Generalizes |
| consteval validation in `make_kernel` | 6 checks | 5 checks | Generalizes (different checks) |
| One `.hip` file per variant | ~15 lines each | ~15 lines each | Generalizes |
| `Signature` with operator composition | `AddOp{}` only | `GemmOp` + epilogue ops | Generalizes |

## 11. What Didn't Generalize

- **Dtype hierarchy depth**: Elementwise has a shared input default (`in_dtype`); GEMM does not. Each operation must define its own hierarchy shape.
- **Tile validation**: Elementwise validates 1D `kVectorM`; GEMM validates 3D tile divisibility + MFMA warp dispatch. The validation logic is operation-specific.
- **Scalar parameters**: Elementwise bakes `alpha`/`beta` into the args struct and kernel. GEMM's D-tensor fusion is data-driven (D tensor passed as a pointer, not a scalar). These are different mechanisms for different use cases.
- **Grid calculation**: Elementwise uses `ceil(N / block_tile)`; GEMM uses `ceil(M/M_tile) × ceil(N/N_tile)` flattened to 1D via `GemmTile1DPartitioner`. More complex but CK Tile handles it.

## 12. Open Design Questions

1. **Runtime epilogue parameters** — `ScaleOp` with a float scale can't be a consteval field. Needs a different args-passing mechanism (scalar in args struct? separate buffer?).
2. **Composed activations** — CK Tile doesn't have built-in Add+GELU composition. Current model requires custom functors for each combination.
3. **Broadcast D tensors** — 1×N bias via stride=0 works at the tensor descriptor level but is untested in CK Tile examples.
4. **Architecture-dependent tile configs** — Optimal configs differ across gfx90a/gfx942/gfx950. kpack archives can include multiple variants per arch, but the selection mechanism isn't defined yet.
5. **Pipeline selection** — All variants use `GemmPipelineAGmemBGmemCRegV1` (simplest). Production needs `GemmPipelineAGmemBGmemCRegV3` (prefetching) and possibly async pipelines. This is a future `GemmAlgorithm` field.
