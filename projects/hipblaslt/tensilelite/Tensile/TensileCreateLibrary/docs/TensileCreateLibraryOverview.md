# TensileCreateLibrary Overview

## Purpose and Responsibility

`TensileCreateLibrary` is a standalone sub-package (also invokable as
`python -m Tensile.TensileCreateLibrary`) that takes a directory of library-logic YAML files
and produces deployable GPU kernel libraries. It is the packaging and compilation step that
follows library-logic analysis: it reads the winning-solution YAML files, code-generates and
assembles the kernels, links them into HIP code-object (`.co`) files, and writes the master
solution library manifests that the runtime uses to dispatch kernels at inference time.

This sub-package is separate from the benchmarking pipeline (`BenchmarkProblems`) so that
pre-tuned libraries can be rebuilt for a new compiler version without re-running benchmarks.

---

## Entry Points

- **CLI**: `python -m Tensile.TensileCreateLibrary <LogicPath> <OutputPath> HIP [options]`
- **Function**: `TensileCreateLibrary.Run.run()` — the `@profile`-decorated top-level function.
- **`__main__.py`** — calls `run()` when the package is executed as a script.

---

## Source Files and Their Roles

| File | Role |
|---|---|
| `Run.py` | All the substantive logic: argument handling, library loading, code generation, assembly, linking, manifest writing. |
| `ParseArguments.py` | `parseArguments()` — defines and parses the CLI argument schema. Returns a plain `dict`. |
| `__main__.py` | Entry point for `python -m` invocation; delegates entirely to `run()`. |
| `__init__.py` | Re-exports `run`, `libraryDir`, `writeSolutionsAndKernels`, `copyStaticFiles` for use by `BenchmarkProblems.py`. |

---

## Key Functions in `Run.py`

### `run()` — top-level orchestration

The main sequence executed by the CLI:

1. Parse arguments via `parseArguments()`.
2. Validate toolchain (compilers, bundler, assembler) via `validateToolchain()`.
3. Resolve target architectures and build `isaInfoMap` via `makeIsaInfoMap()`.
4. Glob for logic YAML files matching the `LogicPath` / `Architecture` / `LogicFilter` args;
   optionally filter out `Experimental` files.
5. Call `generateLogicDataAndSolutions()` to load YAMLs and build `masterLibraries`.
6. Call `generateKernelObjectsFromSolutions()` to deduplicate kernels.
7. Call `generateKernelHelperObjects()` to build auxiliary kernel writers.
8. Call `copyStaticFiles()` to copy static C++ headers.
9. Call `writeSolutionsAndKernelsTCL()` to code-generate, assemble, and link.
10. Call `passPostKernelInfoToLibrary()` to back-fill occupancy/PGR metadata into the library.
11. Write per-arch master library manifests and lazy-library mapping files via `LibraryIO.write()`.
12. Optionally clean up `build_tmp`.

### `generateLogicDataAndSolutions(logicFiles, args, assembler, isaInfoMap)`

Loads all logic YAML files in parallel via `LibraryIO.parseLibraryLogicFile()` (using
`ParallelMap2`). Merges per-architecture `MasterSolutionLibrary` objects. After all files are
merged, re-indexes solution indices to be globally monotonic and deterministic (sorted by
architecture name, then by source filename). Handles `"fallback"` architecture libraries:
merges them into every per-arch master, then calls `renameFallbacksPerArch()` to arch-suffix
their on-disk names to prevent overlay collisions.

Returns: `(solutions, masterLibraries, codeObjectFilesIndex)`.

### `writeSolutionsAndKernelsTCL(outputPath, asmToolchain, srcToolchain, solutions, kernels, ...)`

The streaming version of kernel compilation used by the CLI (a batch version
`writeSolutionsAndKernels()` is used by `BenchmarkProblems`). Steps:

1. Detect duplicate kernels by `getKernelFileBase()`; mark duplicates so they are only
   compiled once.
2. Code-generate each unique kernel with `processKernelSource()` in parallel
   (`ParallelMap2`).
3. Write each generated `.s` file to `assemblyTmpPath` with `writeAssembly()`.
4. Assemble each `.s` to a `.o` file using `asmToolchain.assembler`.
5. Link and bundle `.o` files into `.co` (HIP code object) files via
   `buildAssemblyCodeObjectFiles()`.
6. Write `Kernels.cpp` / `Kernels.h` helper kernels via `writeHelpers()`.
7. Compile and bundle `Kernels.cpp` with `buildSourceCodeObjectFiles()`.

Returns: `(numUniqueKernels, uniqueKernels, results)`.

### `processKernelSource(kernelWriterAssembly, data, outOptions, splitGSU, kernel) -> KernelCodeGenResult`

Drives one kernel through code generation:
1. Calls `kernelWriter.setRocIsa(data, outOptions)` to configure the `rocisa` backend.
2. Calls `kernelWriter.getSourceFileString(kernel)` to emit the assembly text.
3. Calls `kernelWriter.getHeaderFileString(kernel)` to emit the header.
4. Returns a `KernelCodeGenResult` named tuple.

### `KernelCodeGenResult` (NamedTuple)

Fields: `err`, `src`, `header`, `name`, `targetObjFilename`, `isa`, `wavefrontSize`,
`cuoccupancy`, `pgr`, `mathclk`.

### `generateKernelHelperObjects(solutions, cxxCompiler, isaInfoMap)`

Collects the minimal set of auxiliary kernel writers needed across all solutions (beta-only,
reduction, conversion, activation enum headers). Ensures activation enum header writers appear
first in the returned list (so `Kernels.h` has the enum before the activation function kernels
that reference it).

### `renameFallbacksPerArch(masterLibraries)`

Deep-copies each per-arch `MasterSolutionLibrary` that absorbed a fallback and arch-suffixes
both the `lazyLibraries` dict keys and the `PlaceholderLibrary.filenamePrefix` strings inside
the library tree. This prevents on-disk filename collisions between per-arch fallback shards in
overlay-style installs.

### `libraryDir(outputPath, archs) -> Path`

Returns the output directory for compiled library files:
- Single arch → `<outputPath>/library/<arch>/`
- Multiple or zero archs → `<outputPath>/library/`

### `copyStaticFiles(outputPath)`

Copies the six static C++ headers that all generated kernels depend on:
`TensileTypes.h`, `tensile_bfloat16.h`, `tensile_float8_bfloat8.h`, `KernelHeader.h`,
`ReductionTemplate.h`, `memory_gfx.h`.

---

## Key CLI Arguments (`ParseArguments.py`)

Positional:
- `LogicPath` — directory containing library logic YAML files.
- `OutputPath` — where compiled libraries are written.
- `RuntimeLanguage` — `HIP`, `OCL`, or `HSA`.

Notable optional flags:
- `--architecture` — target GPU architectures (e.g., `gfx942_gfx1100`); default `all`.
- `--code-object-version` — `4` or `5`.
- `--library-format` — `msgpack` (default) or `yaml`.
- `--logic-format` — `yaml` (default) or `json`.
- `--no-compress` — skip zlib compression of code objects.
- `--experimental` — include logic files under `Experimental/` directories.
- `--lazy-library-loading` — emit per-lazy-library manifest files.
- `--gen-sol-table` — write a `MatchTable` mapping solution index to source YAML.
- `--keep-build-tmp` — preserve intermediate `.s` / `.o` files.

---

## Output Directory Layout

```
<OutputPath>/
  library/
    <arch>/                      # when single arch
      TensileLibrary_<arch>      # master library manifest (.yaml or .msgpack)
      TensileLibrary_lazy_<arch> # lazy-loading master manifest
      TensileLiteLibrary_lazy_<arch>_Mapping  # lazy-loading index mapping
      <kernel_name>.co           # compiled HIP code objects
  Kernels.cpp                    # auxiliary HIP kernel source
  Kernels.h                      # auxiliary HIP kernel header
  TensileTypes.h                 # (and other static headers)
  build_tmp/                     # temporary assembly/object files (removed by default)
```

---

## Interaction with the Rest of Tensile

- `BenchmarkProblems.py` calls `writeSolutionsAndKernels()` (the non-TCL variant) during
  benchmarking to compile candidate kernels on the fly.
- `LibraryIO.parseLibraryLogicFile()` is the primary reader of logic YAML files.
- `SolutionLibrary.MasterSolutionLibrary.merge()` combines per-arch libraries.
- `Toolchain.Assembly` and `Toolchain.Source` provide the assembler, linker, and bundler.
