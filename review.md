# Code Review: LDS Bank Swizzle Feature

**Branch:** current branch vs `origin/develop`
**Scope:** ~1161 lines added across 23 files. Adds `LDSBankSwizzleMode` kernel option, `ExpressionTransform` coordinate edge, GR/LR swizzle transforms in `LowerTile.cpp`, per-K-unroll chain separation in `AssignIndexExpressions.cpp`, client/rrperf integration, unit tests, and documentation.

---

## Findings

~~**1. `addStoreThreadTileCT` accepts `ldsSwizzle` parameter but never uses it.**
The parameter is threaded through to `addStoreThreadTileCT` at lines 1513, 2563, 2579 of `LowerTile.cpp`, but the function body unconditionally sets `grSwizzleNThrY = nThrY` (line ~1619) and never references `ldsSwizzle`. This is dead code -- the parameter should either be removed or the function should actually use it (if store-side swizzle is needed for via-VGPR path). Currently it creates a false impression that the store side applies swizzle.**~~

~~**2. Forward diff visitor for `ExpressionTransform` throws unconditionally but has dead code after the throw.**
In `CoordinateEdgeVisitor.hpp` (forward diff visitor, ~line 393), the `operator()(ExpressionTransform const& e)` throws immediately, then has an `if constexpr(false)` block with an alternative implementation. This dead code will never compile or run. Either remove it entirely or leave a `// TODO` comment explaining the intended future behavior. As-is, it's confusing and adds maintenance burden.**~~

~~**3. Reverse diff visitor for `ExpressionTransform` passes through delta unchanged -- correctness depends on unstated assumptions.**
In `CoordinateEdgeVisitor.hpp` (~line 558), the reverse diff visitor propagates the first destination's delta to all source tags and returns the reverse expression. The comment says "the actual transform only matters at codegen time" but doesn't explain why passing through the delta is safe. If the swizzle transform is non-affine (which it is -- XOR + rotation), the delta (stride) through it is not well-defined. This works only because the per-K-unroll chain separation in `AssignIndexExpressions.cpp` ensures the stride path never traverses the `ExpressionTransform` edge. Add a comment or assertion that documents this invariant, otherwise a future change that routes a stride through this edge will silently produce wrong results.**~~

~~**4. `LDSBankSwizzleMode` missing `fromString` specialization -- `enumStrings` template may not link.**
`gemm.cpp:1771` calls `enumStrings<LDSBankSwizzleMode>()` and `CLI_Utils.hpp` calls `fromString<LDSBankSwizzleMode>()`. These rely on the `CCountedEnum` template in `Utils.hpp`. `LDSBankSwizzleMode` has `Count` so the concept is satisfied, but the template instantiation needs a `toString(LDSBankSwizzleMode)` that's visible at instantiation time. Verify that the linker resolves this correctly -- if it doesn't, you'll get a link error only when the CLI path is exercised. (This likely works since it builds, but there's no explicit template instantiation like other enum types may have.)~~

~~**5. `rrperf` `GEMMSolution.ldsBankSwizzle` field is defined but never wired into `command()` output.**
In `problems.py:221`, `ldsBankSwizzle` is a field with default `"None"`. The `command()` method at line 319 auto-generates CLI args from dataclass fields via `argName()`. However, there's no entry in `specialNames` for `ldsBankSwizzle`, so it will be passed as `--ldsBankSwizzle=Swizzle`. The client CLI expects `--ldsBankSwizzle` (which matches). This appears to work by convention but is fragile -- verify the CLI flag name matches exactly. If the client flag is actually `--ldsBankSwizzle`, this is fine.**~~

**6. `LDSSwizzleParams::columnsPerBankRow` is hardcoded to 16 (GFX950) with no runtime validation against actual bank count.**
The `LowerTileVisitor` constructor asserts CDNA4, but `columnsPerBankRow = 16` is a compile-time constant in `LDSSwizzleParams`. If future CDNA architectures change the bank count, this will silently produce wrong swizzle patterns. Consider making this derived from `GPUArchitecture` properties or at minimum adding a comment that this must be updated for non-64-bank architectures.

**7. `buildLRSwizzleTransform` generates unnecessary ALU when `rowsPerBankRow == 1`.**
When `rowsPerBankRow == 1`, `logRows` is 0, so `bankRowIdx = row >> 0 = row`. The subsequent swap and rotation expressions still generate full ALU sequences. The `noConflicts()` check handles the `numColumns >= 16` case, but `rowsPerBankRow == 1` with `numColumns < 16` is impossible given `rowsPerBankRow = 16 / numColumns`, so this is not a bug -- but worth noting that the expressions could be simplified for small `rowsPerBankRow` values to reduce VALU pressure.

**8. Test `GPU_GEMM_FP4_MT256x256x128_LDSSwizzle` asserts `countSubstring("v_or_b32") == 0` as a proxy for swizzle correctness.**
This is a fragile assertion -- any future codegen change that introduces `v_or_b32` for unrelated purposes (e.g., register packing, address computation) will break this test. Consider a more targeted assertion, such as checking that the swizzled LDS offset expressions contain expected XOR/shift patterns, or at least add a comment acknowledging the fragility.

**9. `getInlineUnrollInfo` walks up Body edges with `parents[0]` -- assumes single parent.**
In `AssignIndexExpressions.cpp` (~line 1339), the upward walk uses `parents[0]` unconditionally. If a node has multiple Body parents (e.g., from control flow merges), this will only follow one path and may miss the relevant `SetCoordinate`. This seems safe for the current loop structure (unrolled K loop with sequential SetCoordinate -> Body -> LoadLDSTile), but the assumption should be documented.

**10. `ExpressionTransform` serialization doesn't round-trip test the expression trees.**
`Serialization/CoordinateGraph.hpp` adds `MappingTraits` for `ExpressionTransform` that serializes `forward` and `reverse` expression vectors. But there are no serialization round-trip tests. If `ExpressionPtr` serialization has issues with `PositionalArgument` nodes, the kernel graph save/load will silently corrupt the swizzle transforms.

**11. No negative test for `LDSBankSwizzleMode::Swizzle` on non-CDNA4 architectures.**
The `LowerTileVisitor` constructor asserts CDNA4, but there's no test that verifies this assertion fires on non-CDNA4. A simple death test or exception test would catch future regressions if someone removes the guard.

~~**12. Documentation references Tensile's `SubtileBasedKernel.py` but doesn't explain divergences.**
`LDSSwizzling.md` references Tensile's `_grSwizzleColIds` and `lraTileAssignment` but the implementation differs (e.g., no `columnInGroup` / `groupIndex` splitting mentioned in the GR transform builder, though the doc says it's "required for the Tile edge"). Ensure the doc matches the actual implementation. The doc section "GR ExpressionTransform" mentions group splitting but `buildGRSwizzleTransform` in the code doesn't appear to implement it -- it uses a simple rotation + XOR without group index logic.**~~

---

## Summary

The feature is well-structured overall: clean separation between GR (write) and LR (read) transforms, proper gating via `LDSBankSwizzleMode`, and the `ExpressionTransform` coordinate edge is a reasonable extension point. The per-K-unroll chain separation in `AssignIndexExpressions.cpp` is the most complex part and appears correct for the current use case.

Key risks:
- Finding 3 (reverse diff correctness relies on implicit invariant)
- Finding 1 (unused parameter creates confusion)
- Finding 8 (fragile test assertions)

Say "fix 1 4 5" (or any combination of numbers) to apply specific fixes.
