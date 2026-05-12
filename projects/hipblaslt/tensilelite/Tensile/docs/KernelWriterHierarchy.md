# Kernel Writer Hierarchy

The kernel writer classes generate the assembly or HIP C++ source for one GPU kernel
from a parameterized `Solution`. Different kernel types (main GEMM, beta scaling, GSU
reduction, type conversion, activation) have different writer classes. They split into
two independent abstract hierarchies.

## Class Hierarchy

```
KernelWriterBase  (abstract, KernelWriterBase.py, subclasses ABC)
├── KernelWriterBetaOnly                (KernelWriterBetaOnly.py)
├── KernelWriterReduction               (KernelWriterReduction.py)
├── KernelWriterConversion              (KernelWriterConversion.py)
├── KernelWriterActivationFunction      (KernelWriterActivationFunction.py)
└── KernelWriterActivationEnumHeader    (KernelWriterActivationEnumHeader.py)

KernelWriter  (abstract, KernelWriter.py, metaclass=abc.ABCMeta)
└── KernelWriterAssembly                (KernelWriterAssembly.py)
```

`KernelWriter` and `KernelWriterBase` are independent — neither inherits from the
other. The auxiliary writers extend `KernelWriterBase` and emit HIP C++. The main
GEMM writer extends `KernelWriter` and emits AMD GCN/CDNA assembly through `rocisa`.

## `KernelWriterBase` (`KernelWriterBase.py`)

Common interface for the auxiliary writers. Subclasses `abc.ABC`.

The constructor sets `self.state = {}` plus a set of language-string attributes used
when emitting source: `getGroupIdStr`, `getNumGroupsStr`, `getLocalIdStr`,
`getGlobalIdStr`, `sharedDeclStr`, `macFStr`, `macDStr`, `int64Str`, `uint64Str`,
`atomicCasStr`, `deviceFunctionStr`, `endLine`, and `vectorComponents`. Subclasses
override these as needed.

Abstract methods:
- `getKernelName()`
- `getHeaderFileString()`
- `getSourceFileString()`

The class also implements `keys`, `__len__`, `__iter__`, `__getitem__`, `__setitem__`,
`__hash__`, and `__eq__` so instances behave like a dict over `state` and can be used
as keys.

Module-level constants: `KERNEL_HELPER_FILENAME_CPP = "Kernels.cpp"` and
`KERNEL_HELPER_FILENAME_H = "Kernels.h"`.

## `KernelWriter` (`KernelWriter.py`)

The base class for the assembly GEMM writer. Constructor signature is
`__init__(self, assembler: Assembler, debugConfig: DebugConfig)` — it does not take a
`Solution`. The kernel data is supplied later when generating a specific kernel, at
which point `self.states = StateValues(version, kernel, kernelName)` is built.

`KernelWriter` declares a large number of `@abc.abstractmethod` hooks that
`KernelWriterAssembly` implements. Examples include `functionSignature`,
`localReadAddresses`, `localWriteAddresses`, `defineAndResources`, `checkResources`,
`graWorkGroup`, `graTileAssignment`, `graUnrollAssignment`, `graTileOffsets`,
`graUnrollOffsets`, `graShift`, `graFinalOffsets`, `graAddresses`, `graIncrements`.
There are several dozen in total covering global-read addressing, local-read/write
addressing, the unroll loop, and the store epilogue.

The class dispatches to pluggable code-generation components via
`Component.<SubType>.find(self)`. The codebase contains calls such as
`Component.SIA.find(self)`, `Component.GSU.find(self)`, `Component.LSU.find(self)`,
`Component.StreamK.find(self)`, and `Component.PersistentLoop.find(self)`. The custom
main-loop schedule is invoked from `KernelWriter.kernelBody` (`KernelWriter.py` line
~4180) via `customMainLoopSchedule()` from `Components/CustomSchedule.py`.

## `KernelWriterAssembly` (`KernelWriterAssembly.py`)

Concrete subclass of `KernelWriter` that emits AMD GCN/CDNA assembly using the
`rocisa` IR library. It implements the abstract methods declared on `KernelWriter`
and adds an internal `GlobalReadGprRecord`. The constructor takes the same
`(assembler, debugConfig)` signature as the base class.

Source emission goes through `rocisa` types (`Module`, `KernelBody`,
`StructuredModule`, `RegSet`, the instruction classes) and `rocisa.container.vgpr`,
`sgpr`, `accvgpr`. LDS load/store helpers come from
`AsmMemoryHelpers.dsStore` and `AsmMemoryHelpers.dsLoad`. Store-side state lives in
`AsmStoreState.StoreState`; the per-element address calculation
`AsmAddressCalculation.AddrCalculation` is constructed there and inside
`Components/GSU.py`, not directly in `KernelWriterAssembly`.

## State Data Classes (defined in `KernelWriter.py`)

`ConstValues` — frozen dataclass of init sentinels. Fields: `initLdsValue =
0xFFFFFFFF`, `initSgprValue = 0x0`, `initVgprValue = 0xFFFFFFFF`, `maxOccupancy =
10`, `ldsOOB = 0xF00000`.

`MatrixInfo` — VGPR/SGPR allocation record common to every operand. Fields:
`numVgprValu`, `numVgprValuPack`, `startVgprValu`, `startVgprValuPack`,
`startVgprValuPackTemp`, `numSgprStrides`, and a `tileInfo` object.

`ABMatrixInfo(MatrixInfo)` — adds A/B-side fields including `numVgprValuPerBlock`,
`numVgprG2L`, `numVgprG2LAllocated`, `numVgprG2LTailLoopAllocated`, `startVgprG2L`,
`numVgprLocalReadAddr`, `startVgprLocalReadAddr`, `numVgprLocalWriteAddr`,
`numVgprGlobalReadOffsets`, `startVgprGlobalReadOffset`, the swap-address pairs,
`numSgprGlobalReadIncs`, `useConstSgprGlobalReadIncs`, `numPackCvt`, the transpose
and Direct32X-emulation flags, and several swizzle-related fields.

`StateValues` — the main per-kernel state container. Required fields are `version`,
`kernel`, and `kernelName`; `language` defaults to `"ASM"`. `__init__=False` fields
filled in later include `asmCaps`, `archCaps`, `regCaps`, `laneSGPRCount`, `bpeA`,
`bpeB`, `bpeE`, `bpeCexternalGSU1`, `bpeCexternal`, `bpeCinternal`, `srdShiftLeft`,
and `indexChars`. It also carries scheduling flags (`scheduleGlobalRead`,
`scheduleLocalWrite`, `scheduleIterAlg`), loop-state flags (`inTailLoop`,
`invalidLSUCode`, `overflowedResources`), the LRVW/coalesced-read counts for A, B,
MXSA, MXSB and Metadata, and operand sub-records: `a`, `b`, `mxsa`, `mxsb`, `m` are
`ABMatrixInfo`; `c`, `d`, `e`, `bias` are `MatrixInfo`.

Two more dataclasses live alongside `StateValues`: `StateVgprs` (VGPR slots used
by the store epilogue, e.g. `coord0`, `coord1`, `cinRowPtr`, `addrD`, plus a
`globalReadRegisters` dict) and `CodeModules` (per-region `Module` references such
as `accVgprRead`, `localWriteA`, `globalReadA`, `unrollLoopHeader`,
`perIterGlobalRead`). `ExternClasses` holds the `ActivationModule` and an optional
`Component.SumUnroll`.

## Auxiliary Kernel Writers

All auxiliary writers extend `KernelWriterBase`, set `self.language = "HIP"`, and
emit HIP C++ source.

`KernelWriterBetaOnly` (`KernelWriterBetaOnly.py`) — HIP kernel that applies the
beta-scale pass `D = beta * C` (with optional bias add). Used in GSU and StreamK
atomic paths. Derives `self.datatype` from `ProblemType["ComputeDataType"]`.

`KernelWriterReduction` (`KernelWriterReduction.py`) — declares the reduction
kernel signatures used to combine GSU partial sums into the bias output. Used only
when both `Gradient` and `UseBias` are set. `self.datatype` is the compute type, or
`int32` for an int8 problem with HPA + single-precision compute. The Python side
emits the header (`getHeaderFileString`) with `extern "C"` wrappers; the body
(`kernelBody`) is empty — the actual reduction lives in C++ templates dispatched
by name.

`KernelWriterConversion` (`KernelWriterConversion.py`) — HIP epilogue that converts
the GSU/PGR working buffer into the destination type (alpha/beta scale, optional
bias add, optional fused activation). Constructor takes
`(state, vw, isaInfoMap)`.

`KernelWriterActivationFunction` (`KernelWriterActivationFunction.py`) — emits the
activation-function kernels used by the host dispatcher when
`ProblemType["ActivationType"]` is `'all'` or `'hipblaslt_all'`. Works with the
`Activation` module to obtain `ActivationType`-specific code. Constructor takes
`(state, cxxCompiler, isaInfoList)`.

`KernelWriterActivationEnumHeader` (`KernelWriterActivationEnumHeader.py`) — emits
the `ActivationType` enum header (`.h`) used by the host-side dispatch code.

## Helper Kernel Selection (`KernelHelperNaming.py`)

`KernelHelperEnum` is an `IntEnum` with members `BetaOnly`, `Conversion`,
`ActivationEnumHeader`, `ActivationFunction`, `Reduction`, `All`.

`initHelperKernelObjects(solution, kernelHelperType, cxxCompiler, isaInfoMap)`
returns the writer instances needed for `solution`, filtered by `kernelHelperType`.
Conditions: beta-only kernels are emitted when `GlobalSplitU > 1`, `GlobalSplitU ==
-1`, or `StreamK > 0` with `StreamKAtomic == 1`; reduction kernels require both
`Gradient` and `UseBias`; activation enum/function kernels require
`ActivationType in ('all', 'hipblaslt_all')`. The returned list is sorted so enum
headers come first.

The module also exposes per-type name lookups (`betaOnlyKernelNames`,
`conversionKernelNames`, `reductionKernelNames`, `activationFunctionNames`,
`activationEnumHeaderNames`) used during library generation.

## `KernelWriterModules.py`

Star-imported by both `KernelWriter` and `KernelWriterAssembly`. Contains shared
assembly-generation utilities (e.g. `tdmWait` and the wait-count helpers) built on
`rocisa` instruction primitives.

## Source Files

| File                                  | Defines                                                                                                                  |
|---------------------------------------|--------------------------------------------------------------------------------------------------------------------------|
| `KernelWriterBase.py`                 | `KernelWriterBase`, `KERNEL_HELPER_FILENAME_*`                                                                           |
| `KernelWriter.py`                     | `KernelWriter`, `StateValues`, `MatrixInfo`, `ABMatrixInfo`, `ConstValues`, `StateVgprs`, `CodeModules`, `ExternClasses` |
| `KernelWriterAssembly.py`             | `KernelWriterAssembly`, `GlobalReadGprRecord`, `GprInfo`, `TailOptParams`                                                |
| `KernelWriterBetaOnly.py`             | `KernelWriterBetaOnly`                                                                                                   |
| `KernelWriterReduction.py`            | `KernelWriterReduction`                                                                                                  |
| `KernelWriterConversion.py`           | `KernelWriterConversion`                                                                                                 |
| `KernelWriterActivationFunction.py`   | `KernelWriterActivationFunction`                                                                                         |
| `KernelWriterActivationEnumHeader.py` | `KernelWriterActivationEnumHeader`                                                                                       |
| `KernelHelperNaming.py`               | `KernelHelperEnum`, `initHelperKernelObjects` and per-type name/object factories                                         |
| `KernelWriterModules.py`              | shared codegen helpers (star-imported)                                                                                   |
