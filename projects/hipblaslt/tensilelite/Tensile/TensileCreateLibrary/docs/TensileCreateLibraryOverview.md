# TensileCreateLibrary Overview

`TensileCreateLibrary` is a sub-package of Tensile (also runnable as
`python -m Tensile.TensileCreateLibrary`) that takes a directory of library-logic YAML files
and produces deployable GPU kernel libraries. It reads the winning-solution YAML files,
generates and assembles the kernels, links them into HIP code-object (`.co`) files, and
writes the master library manifests the runtime uses to dispatch kernels.

This sub-package is separate from the benchmarking pipeline so pre-tuned libraries can be
rebuilt for a new compiler version without re-running benchmarks.

## Entry Points

- CLI: `python -m Tensile.TensileCreateLibrary <LogicPath> <OutputPath> <RuntimeLanguage> [options]`
- Function: `Tensile.TensileCreateLibrary.run()` — `@profile`-decorated.
- `__main__.py` calls `run()` when the package is executed as a script.
- `__init__.py` re-exports `run`, `libraryDir`, `writeSolutionsAndKernels`, and `copyStaticFiles`.

## Source Files

| File                | Role                                                                                                         |
|---------------------|--------------------------------------------------------------------------------------------------------------|
| `Run.py`            | All substantive logic: orchestration, library loading, code generation, assembly, linking, manifest writing. |
| `ParseArguments.py` | `parseArguments()` defines the CLI schema and returns a `Dict[str, Any]`.                                    |
| `__main__.py`       | Entry point for `python -m`; calls `run()`.                                                                  |
| `__init__.py`       | Re-exports the public API.                                                                                   |

## `run()` Orchestration

The sequence in `Run.py:run()`:

1. Parse arguments via `parseArguments()` and set verbosity from `PrintLevel`.
2. Resolve the output path and validate the toolchain via `validateToolchain()`, which
   returns the C++ compiler and offload bundler.
3. Split `Architecture` on `;` or `_`; expand `all` to `SUPPORTED_GFX`; pull predicate
   filters out via `splitArchsFromPredicates()`.
4. Map archs to ISAs with `gfxToIsa()`, build `isaInfoMap` via `makeIsaInfoMap()`, then
   `assignGlobalParameters()`.
5. Build `asmToolchain` via `makeAssemblyToolchain()` and `srcToolchain` via
   `makeSourceToolchain()`.
6. Glob `<LogicPath>/**/<LogicFilter>.<ext>` (extension is `.yaml` or `.json` per
   `--logic-format`). Filter to files whose embedded gfx arch matches the request, and
   drop any path containing an `experimental` segment unless `--experimental` is set.
   Apply predicate filters via `filterLogicFilesByPredicates()`.
7. Call `generateLogicDataAndSolutions()` to load the YAMLs in parallel, merge per-arch
   `MasterSolutionLibrary` objects, re-index solution indices deterministically, fold any
   `"fallback"` library into every per-arch master, then arch-suffix the fallback names
   via `renameFallbacksPerArch()`.
8. Call `generateKernelObjectsFromSolutions()` to collect the unique kernel objects.
9. Call `generateKernelHelperObjects(kernels, ...)` to collect helper-kernel writers.
10. Construct the `KernelWriterAssembly`.
11. Call `copyStaticFiles()` to copy the static C++ headers into `OutputPath`.
12. Call `writeSolutionsAndKernelsTCL()` to code-generate, write `.s` files, assemble to
    `.o`, link/bundle to `.co`, and emit `Kernels.cpp` / `Kernels.h`.
13. Call `passPostKernelInfoToLibrary()` to write per-kernel `CUOccupancy`,
    `MathClocksUnrolledLoop`, `PrefetchGlobalRead`, non-temporal flags, wave-separate
    flags, `UnrollLoopSwapGlobalReadOrder`, and `DirectToVgpr{A,B}` back into each
    solution's `sizeMapping`.
14. Build a `solDict` keyed by kernel name, then for each arch write a per-arch lazy-library
    mapping file `TensileLiteLibrary_lazy_<arch>_Mapping` (msgpack) when the mapping has
    entries.
15. For each arch in `masterLibraries`, write the master manifest (`TensileLibrary_lazy_<arch>`
    when `LazyLibraryLoading` is on, otherwise `TensileLibrary_<arch>`) via
    `LibraryIO.write()` in the format chosen by `--library-format`. Then write each lazy
    sub-library in parallel.
16. Unless `--keep-build-tmp` is set, remove `<OutputPath>/build_tmp` and the sibling
    `<OutputPath>/../library/build_tmp` if present.

## Key Functions in `Run.py`

### `generateLogicDataAndSolutions(logicFiles, args, assembler, isaInfoMap)`

Loads logic YAMLs in parallel via `LibraryIO.parseLibraryLogicFile()` (using
`ParallelMap2`). Merges into a per-arch `masterLibraries` dict. After merging, re-indexes
solution indices to be globally monotonic (sorted by arch name, then by lazy-library name,
with each lazy library's solutions sorted by source filename). When `--no-generate-solution-table`
is not set, writes a `MatchTable` mapping each solution index to `[srcName, libraryLogicIndex]`.
If a `"fallback"` arch is present, merges it into every other per-arch master, removes the
`"fallback"` key, then calls `renameFallbacksPerArch()`. Returns
`(solutions, masterLibraries, codeObjectFilesIndex)`.

### `writeSolutionsAndKernelsTCL(outputPath, asmToolchain, srcToolchain, solutions, kernels, kernelHelperObjs, kernelWriterAssembly, cmdlineArchs, disableAsmComments=False, compress=True, removeTemporaries=True)`

The variant called by the CLI. `splitGSU` is hardcoded `False`. Steps:

1. Filter kernels to assembly-only (`KernelLanguage == "Assembly"`).
2. Mark duplicates by `getKernelFileBase()`; keep only unique kernels for code generation.
3. For each unique kernel, run `processKernelSource()` → `writeAssembly()` → assemble
   (`asmToolchain.assembler`) as one composed function under `ParallelMap2`. When
   `removeTemporaries` is true, the `.s` file is unlinked after assembly.
4. Link and bundle all `.o` files into `.co` via `buildAssemblyCodeObjectFiles()`.
5. Write `Kernels.cpp` / `Kernels.h` via `writeHelpers()`.
6. Compile and bundle `Kernels.cpp` via `buildSourceCodeObjectFiles()`.

Returns `(numUniqueKernels, uniqueKernels, results)`.

`writeSolutionsAndKernels()` is a separate variant used by `BenchmarkProblems`; it has
the same shape but takes `splitGSU` as a parameter, supports `errorTolerant` and
`generateSourcesAndExit`, and runs validation/post-info passes against the solution list
during compilation.

### `processKernelSource(kernelWriterAssembly, data, outOptions, splitGSU, kernel, compress=False) -> KernelCodeGenResult`

Generates the source for one kernel: calls `kernelWriter.setRocIsa(data, outOptions)`,
then `getSourceFileString(kernel)`, then `getHeaderFileString(kernel)`. Returns a
`KernelCodeGenResult`. When `compress=True`, the source string is `zlib`-compressed via
`memCompress()`.

### `KernelCodeGenResult` (NamedTuple)

Fields: `err: int`, `src: Union[str, bytes]`, `header: Optional[str]`, `name: str`,
`targetObjFilename: str`, `isa: IsaVersion`, `wavefrontSize: int`, `cuoccupancy: int`,
`pgr: int`, `mathclk: int`.

### `generateKernelHelperObjects(solutions, cxxCompiler, isaInfoMap)`

Collects the unique helper-kernel writers needed across all solutions (beta-only,
reduction, conversion, activation, activation enum headers). Returns the list sorted so
that helpers whose name contains `"Enum"` come first, ensuring the activation enum is
emitted in `Kernels.h` before any kernel that references it. Note: `run()` passes the
deduplicated `kernels` list as the first argument.

### `renameFallbacksPerArch(masterLibraries)`

After fallback libraries have been merged into every per-arch master, deep-copies each
arch's master and arch-suffixes both the `lazyLibraries` dict keys and every
`PlaceholderLibrary.filenamePrefix` whose prefix contains `"_fallback"`. This breaks the
shared-alias problem so on-disk fallback shard filenames (`*_fallback_<arch>.dat`) do not
collide between archs in overlay-style installs, and so the per-arch Mapping filter on
`endswith("_<arch>")` matches fallback entries.

### `passPostKernelInfoToLibrary(results, kernels, masterLibraries, splitGSU)`

Builds a name → `KernelCodeGenResult` dict keyed by `getKernelFileBase()`, then walks each
master library (and its lazy sub-libraries) and writes occupancy, math-clock, prefetch,
non-temporal, wave-separate, swap-order, and DirectToVgpr fields onto each solution's
`sizeMapping`. Raises `KeyError` with diagnostic context if a kernel name is missing from
the result dict.

### `libraryDir(outputPath, archs) -> Path`

- Single arch → `<outputPath>/library/<arch>/`
- Zero or multiple archs → `<outputPath>/library/`

### `copyStaticFiles(outputPath)`

Copies six static headers from `Tensile/Source/`: `TensileTypes.h`, `tensile_bfloat16.h`,
`tensile_float8_bfloat8.h`, `KernelHeader.h`, `ReductionTemplate.h`, `memory_gfx.h`.

## CLI Arguments (`ParseArguments.py`)

Positional:
- `LogicPath` — directory containing library-logic YAML (or JSON) files.
- `OutputPath` — directory to write libraries into.
- `RuntimeLanguage` — one of `OCL`, `HIP`, `HSA`.

Optional (long form, dest, default):

| Flag                           | Dest                 | Default                             | Notes                                                              |
|--------------------------------|----------------------|-------------------------------------|--------------------------------------------------------------------|
| `--cxx-compiler`               | `CxxCompiler`        | `ToolchainDefaults.CXX_COMPILER`    |                                                                    |
| `--c-compiler`                 | `CCompiler`          | `ToolchainDefaults.C_COMPILER`      |                                                                    |
| `--cmake-cxx-compiler`         | `CmakeCxxCompiler`   | none                                | exported into `CMAKE_CXX_COMPILER` env var                         |
| `--offload-bundler`            | `OffloadBundler`     | `ToolchainDefaults.OFFLOAD_BUNDLER` |                                                                    |
| `--assembler`                  | `Assembler`          | `ToolchainDefaults.ASSEMBLER`       |                                                                    |
| `--code-object-version`        | `CodeObjectVersion`  | `4`                                 | choices `4`, `5`, `V4`, `V5`, `default`; mapped via `coVersionMap` |
| `--architecture`               | `Architecture`       | `all`                               | `_` or `;` separated, e.g. `gfx942_gfx1100`                        |
| `--no-compress`                | `NoCompress`         | false                               | inverted into `UseCompression`                                     |
| `--experimental`               | `Experimental`       | false                               | include logic files under `Experimental/`                          |
| `--no-enumerate`               | —                    | false                               | parsed but not consumed by `run()`                                 |
| `--version`                    | —                    | none                                | parsed but not consumed by `run()`                                 |
| `--logic-format`               | `LogicFormat`        | `yaml`                              | `yaml` or `json`                                                   |
| `--library-format`             | `LibraryFormat`      | `msgpack`                           | `msgpack` or `yaml`                                                |
| `--jobs` / `-j`                | `CpuThreads`         | `-1`                                |                                                                    |
| `--verbose` / `-v`             | `PrintLevel`         | `1`                                 |                                                                    |
| `--no-lazy-library-loading`    | `LazyLibraryLoading` | true                                | flag turns lazy loading off                                        |
| `--enable-marker`              | `EnableMarker`       | false                               |                                                                    |
| `--no-generate-solution-table` | `GenSolTable`        | true                                | flag turns the MatchTable off                                      |
| `--asm-debug`                  | `AsmDebug`           | false                               |                                                                    |
| `--build-id`                   | `BuildIdKind`        | `sha1`                              |                                                                    |
| `--address-sanitizer`          | `AsanBuild`          | false                               |                                                                    |
| `--keep-build-tmp`             | `KeepBuildTmp`       | false                               | preserves `.s` and `.o`                                            |
| `--logic-filter`               | `LogicFilter`        | `*`                                 | glob pattern joined into `<LogicPath>/**/<filter><ext>`            |
| `--disable-asm-comments`       | `DisableAsmComments` | false                               |                                                                    |

## Output Directory Layout

```
<OutputPath>/
  library/
    <arch>/                                  # when archs == 1
      TensileLibrary_<arch>                  # master manifest (.yaml or .msgpack)
      TensileLibrary_lazy_<arch>             # written instead when lazy loading is on
      TensileLiteLibrary_lazy_<arch>_Mapping # per-arch lazy mapping, msgpack
      <lazy_library_name>                    # individual lazy sub-libraries
      <kernel_name>.co                       # compiled HIP code objects
  Kernels.cpp                                # helper-kernel source
  Kernels.h                                  # helper-kernel header
  TensileTypes.h                             # plus the other five static headers
  build_tmp/<OUTPATHSTEM>/assembly/          # generated .s and .o (removed by default)
  build_tmp/<OUTPATHSTEM>/code_object_tmp/   # intermediate .hsaco
```

When two or more archs are requested, the `library/<arch>/` level is flattened to
`library/`.

## Interaction with the Rest of Tensile

- `BenchmarkProblems` calls the non-TCL `writeSolutionsAndKernels()` to compile candidate
  kernels during benchmarking.
- `LibraryIO.parseLibraryLogicFile()` reads the logic YAML/JSON files.
- `SolutionLibrary.MasterSolutionLibrary.merge()` combines per-arch libraries.
- `Toolchain.Assembly.makeAssemblyToolchain()` and `Toolchain.Source.makeSourceToolchain()`
  produce the assembler, linker, bundler, and source compiler.
