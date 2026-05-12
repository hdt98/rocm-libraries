# Component System

A capability-based registry for kernel sub-routines. Each sub-routine (a
multiply-accumulate block, a local-read block, an epilogue store helper, etc.) is a
`Component` subclass that declares the hardware and kernel conditions under which it
applies. The kernel writer asks `Component.<SubType>.find(writer)` for the matching
implementation; if none matches it gets `None` and falls back to legacy code. New
architecture-specific implementations can be added by dropping a class into the
`Components/` directory and registering its module in `Components/__init__.py`.

## Source layout

- `Tensile/Component.py` — base class, metaclass, partial-match helper, and the
  abstract category bases that ship out-of-the-box.
- `Tensile/Components/` — concrete implementations and additional category bases
  (`GSU`, `LSU`, `Priority`, `PersistentLoop`, `StreamK`, `XCCMapping`,
  `StreamKMemoryOrdering`).
- `Tensile/Components/__init__.py` — `__all__` lists every module that the wildcard
  import at the bottom of `Component.py` will pull in.

## Key classes

### `ComponentMeta`

Metaclass derived from `abc.ABCMeta`. When a class is created, `ComponentMeta.__init__`
adds the new class to each base's `implementations` dict (and as a named attribute on
the base) so that `Component.MAC` resolves to the `MAC` class and
`Component.MAC.implementations` lists every concrete `MAC` subclass. If the class is
itself abstract, it gets a fresh empty `implementations` dict of its own. Registration
runs at import time, which is why merely importing the module is enough to register
the class.

### `Component`

Base class for every pluggable sub-routine. Declared with
`metaclass=ComponentMeta`. Concrete subclasses use class-level attributes to declare
their match conditions:

- `asmCaps` (dict) — required assembly capabilities, matched against
  `writer.states.asmCaps`.
- `archCaps` (dict) — required architecture capabilities, matched against
  `writer.states.archCaps`.
- `kernel` (dict) — required kernel parameter values, matched against
  `writer.states.kernel`. Values may be plain values or callables for predicates
  that need more than equality.
- `versions` (optional) — restricts to specific ISA versions; matched against
  `writer.version` (not `writer.states.version`).

Class methods:

- `matches(writer, debug=False) -> bool` — checks `versions` if present, then runs
  `PartialMatch` for each declared dict against the corresponding attribute on
  `writer.states`. Missing attributes are skipped, so a class that omits `archCaps`
  is treated as having no architecture constraint.
- `findAll(writer, debug=False, *args, **kwargs) -> list` — walks the
  `implementations` registry. Abstract subclasses are recursed into; concrete
  subclasses are tested with `matches` and collected when they pass.
- `find(writer, debug=False, *args, **kwargs) -> Component | None` — returns
  `None` for zero matches, raises `RuntimeError` for more than one (ambiguity is a
  configuration error), and otherwise returns a fresh instance via `found[0]()`.
- `componentPath(path=None, bases=None) -> list[str]` — walks `__bases__[0]` up to
  `Component` and returns the chain of class names.

Instance methods:

- `__call__` is `@abc.abstractmethod`; concrete subclasses must implement it. This is
  what `find` calls (after instantiation) to emit the actual code.
- `commentHeader()` returns `'.'.join(self.componentPath())`, used as a header in
  generated assembly so a reader can see which component produced a block.

### `PartialMatch(pattern, obj, debug=False, level=0) -> bool`

Module-level function. Three cases, in order:

1. If `pattern` is callable, return `pattern(obj)` coerced to bool.
2. If both `pattern` and `obj` are `Mapping`, recurse on each key in `pattern`. A
   key missing from `obj` fails the match. Keys present in `obj` but not in
   `pattern` are ignored — that is what makes the match partial.
3. Otherwise return `pattern == obj`.

This is what allows lambda predicates inside `asmCaps`, `archCaps`, and `kernel`
dicts.

### `LraTileProperties`

Plain `@dataclass` (not frozen) declared in `Component.py` as a placeholder for
properties carried by `LraTileAssignment` implementations.

## Category base classes in `Component.py`

Each is a direct subclass of `Component` and is the search root for `find`:

- `MAC` — multiply-accumulate block (FMA, MFMA, dot, mad-mix, etc.).
- `Signature` — kernel function signature.
- `LocalRead` — LDS read block. Provides `_getLdsReadMemToken` and `_emitLdsRead`
  helpers that wrap an LDS read instruction with a memory token from
  `writer.states.ldsReadTokenIdx`.
- `SumUnroll` — inner summation unroll. Abstract methods: `initSumUnroll`,
  `loopSum`, `storeSumLDS`.
- `ShiftVectorComponents` — tail-loop vector shift.
- `ComputeStoreVgprs` — VGPR assignment for the store epilogue.
- `NotLocalFullTileElements` — partial tile / edge handling.
- `LraTileAssignment` — local-read address tile mapping.
- `PackData` — data packing (fp16, bf16, fp8, int8, etc.).
- `SIA` — schedule-iter-algorithm. Abstract method: `schedIntoIteration`.
- `GlobalWriteComponents` — epilogue / global store; concrete implementation lives
  in `Components/GlobalWriteBatch.py` as `GlobalWriteBatchComponent`.
- `TensorDataMover` — moves tensor data between global memory and registers.

Additional category bases declared in `Components/`:

- `GSU` (`GSU.py`) — global split-U.
- `LSU` (`LSU.py`) — local split-U.
- `Priority` (`Priority.py`) — workgroup priority control.
- `PersistentLoop` (`PersistentLoop.py`) — persistent kernel loop.
- `StreamK`, `XCCMapping`, `StreamKMemoryOrdering` (`StreamK.py`) — Stream-K
  scheduling support.

## Registration

The last line of `Component.py` is:

```python
from .Components import *  # noqa
```

The wildcard pulls in every module named in `Components/__init__.__all__`. As each
module's class bodies execute, `ComponentMeta.__init__` registers the classes in
their parents' `implementations` dicts. Adding a new module to the directory means
also adding its name to `__all__`; otherwise the wildcard import will not see it
and the class never registers.

## Usage in kernel writers

```python
component = Component.MAC.find(self)
if component:
    return component(self, m, innerUnroll)
# fall back to existing code
```

`find` returns an already-instantiated component; the call after the assignment
invokes `__call__` with the writer and any extra arguments the category expects.
For categories whose abstract base declares additional methods (`SIA`, `SumUnroll`,
`GSU`, etc.), the writer calls those methods directly on the returned instance,
e.g. `siaComponent.schedIntoIteration(self, kernel, ...)` in
`KernelWriter.py`.

The `writer` argument is a `KernelWriter` (or `KernelWriterAssembly`). Its
`states` attribute (a `StateValues` dataclass) carries `asmCaps`, `archCaps`, and
`kernel`; the ISA version is on `writer.version` directly.
