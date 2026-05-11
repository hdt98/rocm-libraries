# Kernel Writer Hierarchy

## What It Is and Why It Exists

The kernel writer hierarchy is the code-generation layer of TensileLite. Given a fully
parameterized `Solution` object, a kernel writer is responsible for producing the assembly or
HIP C++ source text for one GPU kernel. Different kernel types (main GEMM kernel, beta-scaling,
GSU reduction, data-type conversion, activation functions) are served by different writer classes
that all share a common abstract base.

---

## Class Hierarchy

The kernel writer classes form two parallel hierarchies sharing the same `KernelWriterBase`
abstract interface for auxiliary writers, while the primary GEMM writer uses its own abstract
base:

```
KernelWriterBase  (abstract — KernelWriterBase.py)
├── KernelWriterBetaOnly  (KernelWriterBetaOnly.py)
├── KernelWriterReduction  (KernelWriterReduction.py)
├── KernelWriterConversion  (KernelWriterConversion.py)
├── KernelWriterActivationFunction  (KernelWriterActivationFunction.py)
└── KernelWriterActivationEnumHeader  (KernelWriterActivationEnumHeader.py)

KernelWriter  (abstract — KernelWriter.py, metaclass=abc.ABCMeta, independent of KernelWriterBase)
└── KernelWriterAssembly  (KernelWriterAssembly.py)
```

`KernelWriter` and `KernelWriterBase` are independent abstract classes. The auxiliary writers
(`BetaOnly`, `Reduction`, etc.) extend `KernelWriterBase` and emit HIP C++. The primary GEMM
writer (`KernelWriter` / `KernelWriterAssembly`) extends `KernelWriter` directly and emits
assembly.

---

## `KernelWriterBase` (`KernelWriterBase.py`)

The abstract interface for all kernel writers. Defines:

- `state` dict — holds kernel parameters (acts like a dict via `__getitem__` / `__setitem__`).
- Abstract methods:
  - `getKernelName() -> str` — return a unique kernel name string.
  - `getHeaderFileString() -> str` — return the C++ header text.
  - `getSourceFileString() -> str` — return the kernel source text (assembly or HIP C++).
- String constants for HIP/OpenCL intrinsics (`getGroupIdStr`, `sharedDeclStr`, etc.) —
  concrete writers override these for their target language.

Constants defined here:
- `KERNEL_HELPER_FILENAME_CPP = "Kernels.cpp"` — filename used when writing helper kernel sources.
- `KERNEL_HELPER_FILENAME_H = "Kernels.h"` — filename used for helper kernel headers.

---

## `KernelWriter` (`KernelWriter.py`)

The main writer for GPU GEMM kernels. Contains the core register allocation and code-generation
logic shared between the assembly and any future backends. Key responsibilities:

- **Register state tracking** — `StateValues` dataclass holds all `asmCaps`, `archCaps`,
  `kernel` parameters, register counts (bpeA, bpeB, bpeCinternal, bpeCexternal, etc.), and
  flags for the current code-generation context (e.g., `inTailLoop`, `invalidLSUCode`).
- **Matrix info** — `MatrixInfo`, `ABMatrixInfo` dataclasses record per-matrix VGPR/SGPR
  allocations: `numVgprValu`, `startVgprG2L`, `numSgprGlobalReadIncs`, etc.
- **`ConstValues`** — frozen dataclass of sentinel values used to initialise registers
  (`initLdsValue = 0xFFFFFFFF`, `initVgprValue = 0xFFFFFFFF`, `initSgprValue = 0x0`).
- **Component dispatch** — calls `Component.<SubType>.find(self)` to delegate assembly
  sub-routines (MAC, Signature, LocalRead, SIA, etc.) to their registered implementations.

---

## `KernelWriterAssembly` (`KernelWriterAssembly.py`)

Extends `KernelWriter` to produce AMD GCN/CDNA assembly via the `rocisa` IR library.
Key responsibilities:

- Emits the full assembly kernel body: kernel prologue, global reads, LDS writes, inner loop
  (MAC/MFMA instructions), LDS reads, store epilogue.
- Uses `rocisa` objects — `Module`, `KernelBody`, `StructuredModule`, `vgpr()`, `sgpr()`,
  `accvgpr()`, individual instruction classes — to build the instruction IR before serialising
  to text.
- Calls `customMainLoopSchedule()` (`Components/CustomSchedule.py`) for kernels that have a
  CMS (Custom Main Loop Schedule) override.
- Imports and uses `AsmMemoryHelpers.dsStore()`, `AsmMemoryHelpers.dsLoad()` for LDS operations
  and `AsmAddressCalculation.AddrCalculation` for computing epilogue addresses.

---

## Auxiliary Kernel Writers

### `KernelWriterBetaOnly` (`KernelWriterBetaOnly.py`)

Generates a HIP C++ kernel that applies the beta-scaling pass: `D = beta * C`. Used when the
main GEMM accumulates into a temporary buffer (GSU split-K mode) and the final beta-scale
write needs to be separated. Language: `"HIP"`. Derives `datatype` from
`ProblemType["ComputeDataType"]` and emits HIP global kernel source.

### `KernelWriterReduction` (`KernelWriterReduction.py`)

Generates a HIP C++ kernel that reduces the per-wave GSU partial sums into the final output.
Language: `"HIP"`. Derives `datatype` from `ProblemType["ComputeDataType"]`; for `int8`
with HPA the internal type is `int32`. Kernel body is currently a placeholder (`kernelBody`
returns an empty string — the logic lives in C++ templates).

### `KernelWriterConversion` (`KernelWriterConversion.py`)

Generates type-conversion epilogue kernels (e.g., converting from float32 accumulation to fp16
or bf16 output). Language: `"HIP"`.

### `KernelWriterActivationFunction` (`KernelWriterActivationFunction.py`)

Generates standalone activation function kernels that apply a fused activation (ReLU, GELU,
sigmoid, etc.) to the output tensor. Works with the `Activation` module to obtain
`ActivationType`-specific assembly.

### `KernelWriterActivationEnumHeader` (`KernelWriterActivationEnumHeader.py`)

Generates the C++ header containing the `ActivationType` enum used by the host-side dispatch
code. Produces a `.h` file rather than a kernel source file.

---

## Helper Kernel Naming (`KernelHelperNaming.py`)

`KernelHelperEnum` enumerates the types of helper kernels: `BetaOnly`, `Reduction`,
`Conversion`. `initHelperKernelObjects(solution)` constructs the appropriate writer objects
for a given solution and returns them indexed by `KernelHelperEnum`.

---

## Shared Modules (`KernelWriterModules.py`)

A module that `KernelWriter` and `KernelWriterAssembly` star-import to bring in shared
code-generation utilities. These are lower-level helpers that generate specific assembly
sequences reused across multiple code paths.

---

## Key Data Flow

1. A `Solution` dict (from `BenchmarkProblems` or library logic) is passed to
   `KernelWriterAssembly(solution)`.
2. `KernelWriterAssembly.__init__()` initialises `StateValues` (register sizes, ISA caps,
   kernel parameters).
3. `getSourceFileString()` is called, which calls into the main kernel body generation,
   dispatching sub-routines via `Component.find()`.
4. The resulting assembly text (and header) are written to `.s` / `.h` files by the build
   pipeline in `BenchmarkProblems.py` / `TensileCreateLibrary`.

---

## Source Files

| File | Class |
|---|---|
| `KernelWriterBase.py` | `KernelWriterBase` (abstract base) |
| `KernelWriter.py` | `KernelWriter`, `StateValues`, `MatrixInfo`, `ABMatrixInfo`, `ConstValues` |
| `KernelWriterAssembly.py` | `KernelWriterAssembly` |
| `KernelWriterBetaOnly.py` | `KernelWriterBetaOnly` |
| `KernelWriterReduction.py` | `KernelWriterReduction` |
| `KernelWriterConversion.py` | `KernelWriterConversion` |
| `KernelWriterActivationFunction.py` | `KernelWriterActivationFunction` |
| `KernelWriterActivationEnumHeader.py` | `KernelWriterActivationEnumHeader` |
| `KernelHelperNaming.py` | `KernelHelperEnum`, `initHelperKernelObjects()` |
| `KernelWriterModules.py` | shared utilities (star-imported) |
