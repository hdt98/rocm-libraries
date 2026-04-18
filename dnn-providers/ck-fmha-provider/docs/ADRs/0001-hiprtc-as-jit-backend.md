# ADR 0001: hipRTC as the primary CK FMHA JIT backend

**Status:** Accepted (Phase 9 lands Rtc+Hipcc+Auto backend switch; Phase 10 makes Rtc the compile-time default once deployment validates it).

**Date:** 2026-04-17

**Context:**

The CK FMHA plugin's original JIT path (`CkFmhaJit.cpp::jit_compile_kernel`)
shells out to `python3 -> fmha_utils.setup_fmha_dispatcher -> hipcc -> .so -> dlopen`.
First-call latency is 20-40 s on a cold shape, the plugin carries a Python
runtime dependency, and the subprocess model interacts badly with
multi-threaded inference servers. The parent RFC
([ck_fmha_hipdnn_rfc.md](../../../ck_fmha_hipdnn_rfc.md)) Appendix B assessed
hipRTC as non-viable at the time because CK Tile headers required global-namespace
`uint32_t`/`size_t`, pulled `<iostream>`/`<variant>`/`<string>` transitively,
conflicted with hipRTC's built-in runtime when fed preprocessed source, and
produced 12 MB/375 K-line translation units.

**Decision:**

Adopt hipRTC as the runtime JIT backend using the MGX-derived
`ck_tile_headers_preprocessor.cpp` technique (`strip_host_bodies`) that rewrites
every `CK_TILE_HOST`-annotated function body into a harmless stub. This removes
the transitive stdlib dependency and the namespace-isolation issue because
the device-only TU no longer instantiates the host-side CK machinery.

**Architecture:**

```
ck_fmha_plugin (CkFmhaJit.cpp)
  pick_jit_backend() reads CK_FMHA_JIT_BACKEND env var
    rtc   -> jit::compile_rtc
    hipcc -> jit::compile_hipcc
    auto  -> compile_rtc, fall back to compile_hipcc on failure

  compile_rtc (Phase 9)
    map FmhaProblem -> ck::host::device_fmha_fwd::Problem
    call ck::host::fmha_rtc::compile_fwd(problem, solution, arch, params)
      pulls GetTileHeaders() (strip_host_bodies-processed)
      interpolates kernel source
      hipRTC -> code object -> hipModule_t + hipFunction_t
    wrap in RtcFmhaKernelInstance, register into FmhaRegistry::instance()
```

**Phase progression:**

| Phase | Outcome |
|-------|---------|
| 0     | Cherry-pick MGX `strip_host_bodies` + fwd wrapper + embed libraries |
| 1     | Promote `codegen/test/rtc/` to public `ck_rtc` + `ck::host::fmha_rtc::compile_fwd` facade |
| 2-3   | Expand fwd wrapper template surface: mask/bias enums, LSE, dropout, soft-cap, sink, group-mode, GQA, FP8 dtypes, QuantScaleEnum |
| 4     | Autogenerate per-(arch, dtype) tile tables from fmha_arch_specs.json |
| 5     | Scaffolding for pagedkv / splitkv(+combine) / appendkv / batch_prefill families |
| 6     | Scaffolding for bwd three-stage (dot_do_o / dq_dk_dv / convert_dq) |
| 7     | gfx950 unblocked: qr_async_trload + V3 + qr_async_trload_v3 pipelines |
| 8     | gfx11 + gfx12 unblocked: qr_nwarp_sshuffle pipeline |
| 9     | Plugin `CkFmhaJit.cpp` gains compile_rtc/compile_hipcc/compile_auto backends + `CK_FMHA_JIT_BACKEND` runtime env + `CK_FMHA_WITH_RTC` compile-time opt-in |
| 10    | `CK_FMHA_DEFAULT_BACKEND_RTC` compile-time flag to flip default from Auto to Rtc once validated |
| 11    | (this doc) Documentation update |

**Rollback:**

- Runtime: set `CK_FMHA_JIT_BACKEND=hipcc`. The Python/hipcc subprocess path is preserved.
- Build-time: CMake `-DCK_FMHA_WITH_RTC=OFF` removes the RTC code path entirely; the plugin reverts to hipcc-only.

**Consequences:**

Positive:
- No runtime Python dependency when `CK_FMHA_JIT_BACKEND=rtc`.
- No `fork/execvp` during inference.
- First-call latency expected to drop from 20-40 s to 5-15 s (hipRTC avoids the host-compile stage).
- Single `.so` deployment artefact, no on-disk kernel library scan required.
- Unified foundation with MGX; future CK changes benefit both consumers.

Negative / risks:
- Plugin `.so` grows to embed the `ck_tile` header tree. Mitigation: `Embed.cmake`'s `SANITIZE` flag compresses non-ASCII; further narrowing via file-glob filter if the 100 MB/arch budget is threatened.
- `strip_host_bodies` is a C++ tokeniser (not AST-aware). Future CK headers with unusual `CK_TILE_HOST` patterns may need the replacement table extended. Mitigation: CI test that compiles every embedded header through hipRTC standalone.
- Phase 5/6/7/8 scaffolding leaves `TODO(phaseN-followup)` markers where per-family Kargs composition, per-arch intrinsics, or pipeline-config filters are still incomplete. Correctness for those paths lands once live tests exercise each.

**References:**

- [Parent RFC §14.4](../../../ck_fmha_hipdnn_rfc.md#144-hiprtc-reassessment)
- [Sub-RFC: hiprtc migration](../../../ck_fmha_hiprtc_migration_rfc.md)
- [MGX commit series](https://github.com/ROCm/rocm-libraries) `users/music-dino/ck_tile_fmha_rtc_api_for_mgx`
- `codegen/src/ck_tile_headers_preprocessor.cpp` -- MIGraphX-derived lexer.
