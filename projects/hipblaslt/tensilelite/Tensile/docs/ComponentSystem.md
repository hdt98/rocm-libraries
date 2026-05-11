# Component System

## What It Is and Why It Exists

The Component system is a capability-based plug-in registry for kernel sub-routines. Rather
than selecting code paths with large `if arch == ...` chains, each sub-routine (e.g., a
multiply-accumulate block or a local-read block) is packaged as an independent `Component`
subclass that declares the hardware and kernel conditions under which it applies. At kernel
generation time the writer calls `Component.<SubType>.find(writer)` and gets back the single
matching implementation — or `None` if no component matches, allowing a graceful fallback to
legacy code.

This pattern makes it possible to add a new architecture-specific implementation without
touching existing code: just create a new class in the `Components/` directory, declare its
matching conditions, and it is automatically discovered.

---

## Key Classes

### `ComponentMeta` (`Component.py`)

A metaclass (inherits `abc.ABCMeta`) that intercepts class creation. Every time a new
`Component` subclass is defined, `ComponentMeta.__init__()` registers it in the `implementations`
dict of each of its parent classes. This builds the hierarchical registry automatically at
import time.

### `Component` (`Component.py`)

Base class for all pluggable sub-routines. Key class-level attributes on concrete subclasses:

- `asmCaps` (dict) — required assembly capabilities (e.g., `{"v_fma_f16": True}`). Matched
  against `writer.states.asmCaps`.
- `archCaps` (dict) — required architecture capabilities. Matched against `writer.states.archCaps`.
- `kernel` (dict) — required kernel parameter values. Matched against `writer.states.kernel`.
  Values may be plain values or callables for complex predicates.
- `versions` (list, optional) — restrict to specific ISA versions (tuples); matched against
  `writer.version`.

Key class methods:

- `Component.matches(cls, writer, debug=False) -> bool` — checks all declared attributes
  against the writer state using `PartialMatch()`. Returns `True` if this implementation
  applies.
- `Component.findAll(cls, writer, debug=False) -> list` — recursively walks the
  `implementations` registry, returning all matching concrete implementations.
- `Component.find(cls, writer, debug=False) -> Component | None` — calls `findAll()` and
  returns the single match. Raises `RuntimeError` if more than one implementation matches
  (ambiguity is a configuration error).
- `Component.componentPath(cls) -> list[str]` — returns the class hierarchy as a list of
  names, used for comment headers in generated code.

Abstract method: `__call__(self)` — concrete subclasses must implement this to emit the actual
assembly code.

### `PartialMatch(pattern, obj) -> bool` (`Component.py`)

A recursive matcher used internally by `Component.matches()`. It handles three cases:
- If `pattern` is callable, call it with `obj` and return its boolean result.
- If both `pattern` and `obj` are `Mapping` instances, recurse on each key-value pair.
- Otherwise, use plain equality `pattern == obj`.

This is what allows lambda predicates in `asmCaps`, `archCaps`, and `kernel` dicts.

### `LraTileProperties` (`Component.py`)

A frozen dataclass that carries properties of the LRA (Local Read Address) tile assignment.
Used as a return value by `LraTileAssignment` components.

---

## Concrete Component Types

All of these are abstract base classes (themselves subclasses of `Component`) whose concrete
implementations live in the `Components/` subdirectory:

| Class | Role |
|---|---|
| `MAC` | Multiply-accumulate instruction block (MFMA, WMMA, FMA, etc.) |
| `Signature` | Kernel function signature |
| `LocalRead` | LDS read block; provides `_emitLdsRead()` helper with memory token tracking |
| `SumUnroll` | Inner summation unroll loop; abstract methods: `initSumUnroll()`, `loopSum()`, `storeSumLDS()` |
| `ShiftVectorComponents` | Tail-loop vector shift |
| `ComputeStoreVgprs` | Compute VGPR assignments for the store epilogue |
| `NotLocalFullTileElements` | Predicate for partial tile edge handling |
| `LraTileAssignment` | Local-read address tile mapping |
| `PackData` | Data packing (e.g., fp16 packing) |
| `SIA` | Schedule-Iter-Algorithm; abstract method: `schedIntoIteration()` |
| `GlobalWriteComponents` | Global write (epilogue store) helper |
| `TensorDataMover` (`TDM`) | Moves tensor data between global memory and registers |

---

## How Components Are Registered

At the bottom of `Component.py`:

```python
from .Components import *  # noqa
```

This wildcard import loads every module listed in `Components/__init__.__all__`. Each module
defines one or more concrete `Component` subclasses. The `ComponentMeta` metaclass registers
each subclass automatically as the class body is executed, so no explicit registration call
is needed.

---

## Usage Pattern in Kernel Writers

```python
# In KernelWriter or KernelWriterAssembly:
component = Component.MAC.find(self)
if component:
    return component(self, m, innerUnroll)
# else fall back to older code path
```

The `find()` call searches the entire `MAC.implementations` tree for an implementation whose
`asmCaps`, `archCaps`, and `kernel` conditions all match `self.states`. The returned object
is a fresh instance (constructed via `found[0]()`), so `__call__` is invoked on it with
the kernel writer and any extra arguments.

---

## Interaction with Other Concepts

- **`KernelWriter` / `KernelWriterAssembly`** — the `writer` argument passed to `find()` and
  `matches()`. Its `states` attribute (a `StateValues` dataclass) holds `asmCaps`, `archCaps`,
  `kernel`, and `version`.
- **`Components/` subdirectory** — contains all concrete implementations. Adding a new
  implementation only requires creating a class there; no changes to `Component.py` itself.
- **`rocisa`** — the assembly IR library used by most component implementations to emit
  instructions.

---

## Source Files

- `Tensile/Component.py` — base classes and registry metaclass.
- `Tensile/Components/` — all concrete component implementations.
