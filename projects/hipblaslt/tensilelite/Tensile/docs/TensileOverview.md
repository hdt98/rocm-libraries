# Tensile Module Overview

The `Tensile` package generates and tunes AMD GPU kernels for GEMM and related
contraction operations. It covers the full tuning lifecycle: enumerate candidate
kernels, benchmark them on hardware, analyze the results, and emit a deployable
library that maps each problem size to its best kernel.

`Tensile.py::executeStepsInConfig()` runs the workflow in three stages, driven
by keys in the user's YAML config:

1. `BenchmarkProblems` — `BenchmarkProblems.main()` enumerates fork permutations,
   compiles them, and benchmarks them on the GPU. Output goes to
   `1_BenchmarkProblems` and `2_BenchmarkData`.
2. `LibraryLogic` — `LibraryLogic.main()` reads the benchmark CSVs and writes
   logic YAML files to `3_LibraryLogic` mapping problem sizes to winning solutions.
3. `LibraryClient` — `ClientWriter.main()` writes the client config and the
   build/run scripts that drive the compiled benchmark client; output goes to
   `4_LibraryClient`.

The CLI entry point `bin/Tensile` calls `Tensile.main()`, which parses arguments
and dispatches to `executeStepsInConfig()`.

`TensileCreateLibrary` is a separate sub-package (not part of the three-stage
flow above). It compiles tuned solutions into the static library shipped with
hipBLASLt and is invoked by its own CLI entry point.

---

## Key Abstractions

### Problem and hardware description
- `Contractions.py` — `ProblemType`, `FreeIndex`, `BatchIndex`, `BoundIndex`
  describe a contraction by its index roles (free, batch, summation).
  `Solution`, `SizeMapping`, `ProblemPredicate`, and `TaskPredicate` live here too.
- `Properties.py` — base `Property` and `Predicate` classes used to express
  conditions in the solution-selection tree.
- `Hardware.py` — `HardwarePredicate`, plus helpers like `parseDeviceNameToHex`
  and the chip-id fallback graph that resolves GFX device names to ISAs.

### Solution selection
- `SolutionLibrary.py` — the library tree node types: `SingleSolutionLibrary`,
  `IndexSolutionLibrary`, `PlaceholderLibrary`, `MatchingLibrary`,
  `FreeSizeLibrary`, `PredictionLibrary`, `MLPClassificationLibrary`,
  `ProblemMapLibrary`, `PredicateLibrary`, and the top-level `MasterSolutionLibrary`.
  A library is a tree of predicate nodes that maps a (hardware, problem) pair
  to a solution index.
- `SolutionSelectionLibrary.py` — `analyzeSolutionSelection()` and supporting
  helpers that build the tile-aware selection structures from raw benchmark data.
- `LibraryLogic.py` — `analyzeProblemType()` and the `LogicAnalyzer` class read
  benchmark CSVs and emit the logic YAML; `generateLogic()` and `main()` drive
  the analysis from a config dict.

### Benchmarking pipeline
- `BenchmarkStructs.py` — `BenchmarkProcess` parses the YAML config into a
  benchmark plan. `constructForkPermutations` expands fork parameters into
  the cross product of solution candidates. `BenchmarkStep` represents one
  step of the plan.
- `BenchmarkProblems.py` — `main()` drives the benchmarking loop: it generates
  forked solutions (`_generateForkedSolutions`), writes the build files
  (`writeBenchmarkFiles`), compiles the kernels via the assembly and source
  toolchains, runs the client, and caches results between runs.
- `BenchmarkSplitter.py` — `BenchmarkSplitter` loads a benchmark YAML and
  splits it into smaller files keyed by problem and grouping, so large configs
  can be run as separate jobs.
- `ParallelExecution.py` — `detectAvailableGpus()` discovers visible devices;
  `runClientParallel()` shards a benchmark config across them and merges the
  per-GPU CSVs with `mergeResultsCsv()`.

### Kernel code generation
- `KernelWriterBase.py` — abstract `KernelWriterBase` defines the three
  abstract methods every writer implements: `getKernelName`,
  `getHeaderFileString`, and `getSourceFileString`.
- `KernelWriter.py` — `KernelWriter` (abstract, extends `KernelWriterBase`)
  holds the high-level codegen state. The dataclasses `MatrixInfo`,
  `ABMatrixInfo`, `StateValues`, `StateVgprs`, `CodeModules`, `ConstValues`,
  and `ExternClasses` (all defined at module scope) hold the per-kernel state
  it threads through code generation. It dispatches to `Component` instances
  for pluggable sub-routines.
- `KernelWriterAssembly.py` — `KernelWriterAssembly` extends `KernelWriter` and
  emits `.s` assembly for AMD GPUs by building `rocisa` IR modules. Helper
  dataclasses `TailOptParams`, `GprInfo`, and `GlobalReadGprRecord` live here.
- `KernelWriterBetaOnly.py`, `KernelWriterReduction.py`,
  `KernelWriterConversion.py`, `KernelWriterActivationFunction.py`,
  `KernelWriterActivationEnumHeader.py` — concrete `KernelWriterBase`
  subclasses for the auxiliary kernels (beta scaling, GSU reduction, datatype
  conversion, activation function bodies, and activation enum headers).
- `KernelWriterModules.py` — free functions used by `KernelWriterAssembly`
  for things like wait-count emission (`wait`, `tdmWait`, `syncThreads`) and
  acc-to-arch register mapping (`mapAcctoArchRegs`).
- `KernelHelperNaming.py` — `KernelHelperEnum` enumerates the helper kernel
  types; `initHelperKernelObjects()` and the per-type `init*KernelObjects`
  constructors build the writer instances for each helper kernel a solution
  needs.

### Component system
- `Component.py` — `Component` is a registry base class with metaclass
  `ComponentMeta`. Subclasses (`MAC`, `Signature`, `LocalRead`, `SumUnroll`,
  `ShiftVectorComponents`, `ComputeStoreVgprs`, `NotLocalFullTileElements`,
  `LraTileAssignment`, `PackData`, `SIA`, `GlobalWriteComponents`,
  `TensorDataMover`) define hook points; their concrete implementations
  register themselves and are dispatched at codegen time by
  `Component.<Subtype>.find(writer)`. `PartialMatch()` is the matcher that
  selects an implementation against the writer's state. See
  `ComponentSystem.md` for details.

### Assembly helpers
- `AsmAddressCalculation.py` — `AddrCalculation` computes element addresses
  for C/D/E/bias matrices during the store epilogue.
- `AsmStoreState.py` — `StoreState`, `VectorDataTypes`, and `VectorUnit`
  track VGPR allocation across batches of global writes.
- `AsmMemoryInstruction.py` — `MemoryInstruction` describes a load/store
  variant (width, modifiers, latency hints).
- `AsmMemoryHelpers.py` — small helpers like `dsLoad` and `dsStore` for
  emitting LDS access instructions.

### Activation
- `Activation.py` — `ActivationType` enum and `ActivationTypeRegister` table
  declare the supported activations (relu, gelu, sigmoid, tanh, silu, etc.).
  `ActivationModule` emits the inline assembly for a chosen activation.
  Module-level functions `CombineInstructions`, `FuseInstruction`,
  `ConvertCoeffToHex`, etc. post-process the emitted modules.

### IO and library packaging
- `LibraryIO.py` — read/write YAML, JSON, and msgpack for solution and logic
  files. `parseLibraryLogicFile`, `parseSolutionsFile`, `writeSolutions`,
  and `createLibraryLogic` are the main entry points; `MinimumRequiredVersion`
  is checked via `versionIsCompatible`.
- `CustomYamlLoader.py` — `load_yaml_stream()` and the per-key/per-index
  loaders parse logic YAMLs lazily.
- `ClientWriter.py` — `main()`, `writeClientConfig()`, and
  `writeClientConfigIni()` emit the client `.ini`; `runClient()` and
  `runNewClient()` invoke the compiled client. `getBuildClientLibraryScript()`
  and `writeRunScript()` produce the build/run shell scripts.
- `ClientExecutable.py` — `getClientExecutable()` builds (or locates) the
  benchmark client binary using a `CMakeEnvironment`.
- `TensileClientConfig.py` — top-level `TensileClientConfig()` script that
  parses a benchmark YAML and writes out a client `.ini`.
- `EmbeddedData.py` — `EmbeddedDataFile`, `Namespace`, and `Indent` write
  C++ headers that embed binary data inline.

### Tooling and utilities
- `Tensile.py` — orchestration; `Tensile()` is the script entry point.
- `TensileCreateLibrary/` — sub-package that compiles solutions and produces
  the static library.
- `TensileMergeLibrary.py` — merges multiple library logic YAMLs.
- `TensileUpdateLibrary.py`, `TensileRetuneLibrary.py` — update or retune an
  existing library with new benchmark data.
- `TensileLibLogicToYaml.py` — converts a binary library logic file back to YAML.
- `TensileBenchmarkCluster.py`, `TensileBenchmarkClusterScripts.py`,
  `TensileBenchmarkLibraryClient.py` — cluster benchmarking support.
- `GenerateSummations.py` — `GenerateSummations()` generates summation
  problem configs.
- `CustomKernels.py` — `getCustomKernelConfig()` and `isCustomKernelConfig()`
  load hand-written assembly kernels from `CustomKernels/` so they can be
  benchmarked alongside generated ones.
- `Configuration.py` — `Parameter`, `CallableParameter`, `ProjectConfig`,
  and `ExpressionEvaluator` underpin parameter validation and expression
  evaluation in benchmark configs.

---

## Source file to concept map

| Source file(s) | Concept |
|---|---|
| `Tensile.py` | Top-level workflow orchestration |
| `BenchmarkProblems.py`, `BenchmarkStructs.py`, `BenchmarkSplitter.py` | Benchmarking pipeline |
| `LibraryLogic.py`, `SolutionLibrary.py`, `SolutionSelectionLibrary.py` | Solution selection and library logic |
| `Contractions.py`, `Properties.py`, `Hardware.py` | Problem and hardware description |
| `KernelWriter.py`, `KernelWriterBase.py`, `KernelWriterAssembly.py`, `KernelWriterModules.py` | Main kernel code generation |
| `KernelWriterBetaOnly.py`, `KernelWriterReduction.py`, `KernelWriterConversion.py`, `KernelHelperNaming.py` | Auxiliary kernel generation |
| `KernelWriterActivationFunction.py`, `KernelWriterActivationEnumHeader.py`, `Activation.py` | Activation functions |
| `Component.py` | Pluggable component registry |
| `AsmAddressCalculation.py`, `AsmStoreState.py`, `AsmMemoryInstruction.py`, `AsmMemoryHelpers.py` | Assembly store/address helpers |
| `LibraryIO.py`, `CustomYamlLoader.py`, `EmbeddedData.py` | IO and serialization |
| `ClientWriter.py`, `ClientExecutable.py`, `TensileClientConfig.py` | Benchmark client |
| `ParallelExecution.py` | Multi-GPU benchmark execution |
| `CustomKernels.py` | Hand-written kernels |
| `TensileMergeLibrary.py`, `TensileUpdateLibrary.py`, `TensileRetuneLibrary.py`, `TensileLibLogicToYaml.py` | Library management utilities |

---

## Entry points

- CLI: `bin/Tensile` -> `Tensile.main()` -> `Tensile.Tensile()` ->
  `executeStepsInConfig()`.
- Benchmark stage: `BenchmarkProblems.main()`.
- Analysis stage: `LibraryLogic.main()`, which calls
  `LibraryLogic.analyzeProblemType()` per problem type.
- Client stage: `ClientWriter.main()`.
- Kernel writing: instantiate `KernelWriterAssembly` from a `Solution`, then
  call `getSourceFileString()` (and `getHeaderFileString()`) to emit assembly.
- Component dispatch: `Component.<Subtype>.find(writer)` from inside a writer.
- Custom kernels: `CustomKernels.getCustomKernelConfig()` during benchmark
  solution generation.
