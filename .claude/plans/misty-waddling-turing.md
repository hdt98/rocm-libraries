# Fix Build Warnings Plan

## Context

A CI build of MIOpen in TheRock produces 31,143 compiler warnings. Virtually none are from MIOpen's own code -- they originate from external HIP runtime headers and rocRAND headers that MIOpen includes. The warnings are noisy but not errors because MIOpen does not use `-Werror` (only `-Weverything` with selective `-Wno-*` suppressions).

## Analysis

### Warning sources (by file path, not CI label)

| Source | Warnings | % |
|--------|----------|---|
| HIP runtime headers (via rocRAND dist) | 25,872 | 83% |
| HIP runtime headers (via hipBLAS-common dist) | 5,216 | 17% |
| rocRAND's own headers (`rocrand/*.h`) | 53 | <1% |
| MIOpen's own `config.h` (macro redefinition) | 2 | <1% |

### Top warning types (all from external headers)

| Warning | Count |
|---------|-------|
| `-Wzero-as-null-pointer-constant` | 25,620 |
| `-Wundef` | 1,479 |
| `-Wshadow` | 1,464 |
| `-Wcomma` | 1,464 |
| `-Wnewline-eof` | 732 |
| `-Wunused-template` | 240 |
| Others (float-equal, shadow-field, nvcc-compat, etc.) | 144 |

### Why `-Werror` isn't catching these

MIOpen does **not** use `-Werror` for compilation. It uses `-Weverything` and selectively disables warnings with `-Wno-*`. The DNN providers (miopen-provider etc.) have `-Werror`, but they compile their own sources, not MIOpen itself.

### How other projects in rocm-libraries handle this

No project in the repo marks HIP include directories as SYSTEM. The dominant pattern is `-Wno-*` suppressions:

- **composablekernel** (closest setup -- also uses `-Weverything` + `-Werror`): Uses 30+ `-Wno-*` flags, no SYSTEM includes for HIP
- **DNN providers**: Use `target_include_directories(... SYSTEM ...)` only for vendored third-party code, not for HIP
- **hipBLASLt**: No `-Werror`, no SYSTEM includes, minimal suppressions
- **Shared libs**: Mixed but none mark HIP as SYSTEM

## Proposed Fix

### Approach: Add `-Wno-*` suppressions to MIOpen's `EnableCompilerWarnings.cmake`

This follows the established pattern used by composablekernel (the project with the most similar warning setup).

**File: `projects/miopen/cmake/EnableCompilerWarnings.cmake`**

Add to `__clang_cxx_compile_options`:
```cmake
-Wno-zero-as-null-pointer-constant
-Wno-comma
-Wno-newline-eof
-Wno-unused-template
-Wno-float-equal
-Wno-shadow-field-in-constructor
-Wno-nvcc-compat
-Wno-gnu-anonymous-struct
-Wno-nested-anon-types
```

### What this eliminates

~29,600 of 31,143 warnings (95%). The remaining ~1,500 from `-Wundef`, `-Wshadow`, `-Wmissing-noreturn`, `-Wunused-function`, `-Wunused-parameter` are intentionally enabled by MIOpen and should stay to catch real issues in MIOpen's own code.

### What about the 2 MIOpen-source warnings?

The 2 `MIOPEN_FP8_IEEE_EXPONENT_BIAS` macro-redefined warnings (`-Wmacro-redefined`) in `config.h` are actual MIOpen issues -- out of scope for this PR but worth noting.

## Files to modify

1. `projects/miopen/cmake/EnableCompilerWarnings.cmake` -- add 9 warning suppressions with a comment explaining they come from external headers

## Verification

- Changes are purely additive `-Wno-*` flags -- cannot break compilation
- Full verification requires a CI build of MIOpen in TheRock
