# Tensile Module Overview

## Purpose and Responsibility

The `Tensile` package is the core Python layer of the TensileLite framework for generating
and tuning AMD GPU kernels that perform GEMM (General Matrix-Matrix Multiply) and related
contraction operations. Its responsibilities span the full lifecycle of a tuned kernel:
benchmark execution, performance data analysis, assembly kernel generation, and library
packaging.

The high-level workflow is orchestrated by `Tensile.py` through three sequential stages:

1. **BenchmarkProblems** — enumerate candidate solutions, compile them, and benchmark on hardware.
2. **LibraryLogic** — analyze benchmark results to select the best kernel per problem size.
3. **ClientWriter / TensileCreateLibrary** — package the winning kernels into a deployable library.

Entry point: `executeStepsInConfig()` in `Tensile.py`, called by the CLI in `bin/Tensile`.

---

## Key Abstractions and How They Relate

### Problem description layer
- `Contractions.py` — `ProblemType`, `FreeIndex`, `BatchIndex`, `BoundIndex`: describe a
  contraction mathematically (which indices are free, which are batch, which are summation).
- `Properties.py` — `Property`, `Predicate`, `Solution` predicates: describe conditions under
  which a library entry applies (e.g., hardware predicates, size ranges).
- `Hardware.py` — maps GFX device names and PCI chip IDs to ISA versions; handles fallback
  chains and GPU predicate objects used in solution dispatch.

### Solution selection layer
- `SolutionLibrary.py` — `SingleSolutionLibrary`, `IndexSolutionLibrary`, `PlaceholderLibrary`,
  and the composite tree structures built from `Properties.Predicate`. A library is a tree of
  predicate nodes that maps (hardware, problem size) to a specific solution index.
- `SolutionSelectionLibrary.py` — higher-level wrappers around `SolutionLibrary` that handle
  tile-aware selection and decision-tree dispatch.
- `LibraryLogic.py` — `analyzeProblemType()`: reads raw benchmark CSVs, runs regression or
  direct-lookup analysis, and writes the YAML logic files that become a deployed library.

### Benchmarking layer
- `BenchmarkStructs.py` — `BenchmarkProcess`, `constructForkPermutations()`: parse the YAML
  benchmark config into a list of parameter sweeps (fork permutations) and problem sizes.
- `BenchmarkProblems.py` — `main()`: drives the full benchmarking loop. Compiles kernels using
  the assembly and source toolchains, writes and runs the benchmark client, collects results.
- `BenchmarkSplitter.py` — splits large benchmark runs into parallel jobs.
- `ParallelExecution.py` — `detectAvailableGpus()`, `runClientParallel()`: discovers GPUs and
  distributes benchmark jobs across them.

### Kernel code-generation layer
- `KernelWriterBase.py` — `KernelWriterBase` (abstract): defines the interface `getKernelName()`,
  `getHeaderFileString()`, `getSourceFileString()` that all kernel writers must implement.
- `KernelWriter.py` — `KernelWriter` (concrete main writer): owns the register allocation state
  (`StateValues`, `MatrixInfo`, `ABMatrixInfo`) and high-level assembly logic. Works with
  `Component` objects for pluggable sub-routines.
- `KernelWriterAssembly.py` — `KernelWriterAssembly` (extends `KernelWriter`): produces `.s`
  assembly source for AMD GCN/CDNA architectures using `rocisa` IR.
- `KernelWriterBetaOnly.py`, `KernelWriterReduction.py`, `KernelWriterConversion.py` — specialist
  writers for auxiliary kernels (beta-scaling, GSU reduction, data-type conversion).
- `KernelWriterActivationFunction.py`, `KernelWriterActivationEnumHeader.py` — generate the
  activation function kernels and C++ enum headers.
- `KernelWriterModules.py` — shared module utilities used by all kernel writers.
- `KernelHelperNaming.py` — `KernelHelperEnum`, `initHelperKernelObjects()`: enumeration of
  helper kernel types (beta-only, reduction, conversion) and their construction.

### Component system
- `Component.py` — `Component` (metaclass-based registry), `ComponentMeta`, `PartialMatch()`:
  allow pluggable kernel sub-routines selected at runtime by hardware capabilities and kernel
  parameters. See `ComponentSystem.md` for details.

### Assembly helpers
- `AsmAddressCalculation.py` — `AddrCalculation`: computes element addresses for C/D/E/bias
  matrices during store epilogue.
- `AsmStoreState.py` — `StoreState`, `VectorDataTypes`, `VectorUnit`: tracks VGPR allocation
  state across batches of global writes (epilogue store state machine).
- `AsmMemoryInstruction.py`, `AsmMemoryHelpers.py` — wrappers and helpers for DS (LDS) and
  global memory instructions.

### Activation layer
- `Activation.py` — `ActivationType` enum, `Activation` class, `ActivationModule`: generates
  inline and out-of-line activation function assembly (ReLU, GELU, sigmoid, etc.).

### IO and library packaging layer
- `LibraryIO.py` — read/write YAML and msgpack logic files; version compatibility checks.
- `ClientWriter.py` — `runClient()`, `writeClientConfig()`: write benchmark client configuration,
  invoke the compiled client executable.
- `ClientExecutable.py`, `TensileClientConfig.py` — locate and configure the benchmark client binary.
- `EmbeddedData.py` — embed compiled kernel data into generated C++ headers.
- `CustomYamlLoader.py` — `load_yaml_stream()`: YAML loading with custom constructor support.

### Utility / tooling scripts
- `Tensile.py` — top-level orchestration.
- `TensileCreateLibrary/` — sub-package: compiles solutions, links static libraries.
- `TensileMergeLibrary.py` — merges multiple library logic YAML files.
- `TensileUpdateLibrary.py`, `TensileRetuneLibrary.py` — update/retune an existing library with new data.
- `TensileLibLogicToYaml.py` — converts binary library logic back to YAML.
- `TensileBenchmarkCluster.py`, `TensileBenchmarkClusterScripts.py`, `TensileBenchmarkLibraryClient.py` — cluster benchmarking support.
- `GenerateSummations.py` — generates summation (convolution) problem configs.
- `CustomKernels.py` — `getCustomKernelConfig()`, `isCustomKernelConfig()`: support for
  hand-written assembly kernels that bypass code generation.
- `Configuration.py` — configuration validation and defaults.

---

## Source File to Concept Map

| Source file(s) | Concept |
|---|---|
| `Tensile.py` | Top-level workflow orchestration |
| `BenchmarkProblems.py`, `BenchmarkStructs.py`, `BenchmarkSplitter.py` | Benchmarking pipeline |
| `LibraryLogic.py`, `SolutionLibrary.py`, `SolutionSelectionLibrary.py` | Solution selection and library logic |
| `Contractions.py`, `Properties.py`, `Hardware.py` | Problem and hardware description |
| `KernelWriter.py`, `KernelWriterBase.py`, `KernelWriterAssembly.py` | Assembly kernel generation |
| `KernelWriterBetaOnly.py`, `KernelWriterReduction.py`, `KernelWriterConversion.py` | Auxiliary kernel generation |
| `KernelWriterActivationFunction.py`, `KernelWriterActivationEnumHeader.py`, `Activation.py` | Activation functions |
| `Component.py` | Pluggable component system |
| `AsmAddressCalculation.py`, `AsmStoreState.py`, `AsmMemoryInstruction.py`, `AsmMemoryHelpers.py` | Assembly store/address helpers |
| `LibraryIO.py`, `CustomYamlLoader.py`, `EmbeddedData.py` | IO and serialization |
| `ClientWriter.py`, `ClientExecutable.py`, `TensileClientConfig.py` | Benchmark client |
| `ParallelExecution.py` | GPU parallel execution |
| `CustomKernels.py` | Custom hand-written kernels |
| `TensileMergeLibrary.py`, `TensileUpdateLibrary.py`, `TensileRetuneLibrary.py` | Library management utilities |

---

## Entry Points

- **Benchmarking a new problem**: `Tensile.py::executeStepsInConfig()` → `BenchmarkProblems.main()`
- **Library analysis**: `Tensile.py::executeStepsInConfig()` → `LibraryLogic.main()`
- **Kernel writing**: `KernelWriterAssembly` constructed from a `Solution` dict, then
  `getSourceFileString()` called to emit assembly text.
- **Component selection**: `Component.<SubType>.find(writer)` called from within a kernel writer.
- **Custom kernel lookup**: `CustomKernels.getCustomKernelConfig()` invoked during benchmarking
  when a solution is flagged as a custom kernel.
