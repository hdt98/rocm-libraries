# HipUtils.hpp — `CopyTensorVoid` issue tracker

Source: `include/Tensile/hip/HipUtils.hpp:153-203`
Sole call site: `client/src/DataInitialization.cpp:773` (`copyBadInputBuffers`)

## Issues

### #1 — Critical: `srcBytes` reads from `dst` instead of `src`
- **Location:** `HipUtils.hpp:198-199`
- **Defect:**
  ```cpp
  uint8_t* dstBytes = (uint8_t*)dst + bytesOffset;
  uint8_t* srcBytes = (uint8_t*)dst + bytesOffset;   // should be src
  ```
  Every `hipMemcpyAsync` copies a region onto itself; the `src` argument is silently ignored.
- **Compare:** templated `CopyTensor` at lines 251-252 does it correctly.
- **Impact:** Whenever `src != dst` the function is a no-op masquerading as a tensor copy. The earlier bulk `hipMemcpy` in `copyBadInputBuffers` may mask it for the byte range it covers, but "bad input" data past that range is never propagated.
- **Secondary:** the cast strips `const` from `void const* src`. Fix should use `uint8_t const*`.
- **Status:** open

### #2 — UB / `std::out_of_range` when `contiguousDimensions == 0`
- **Location:** `HipUtils.hpp:184-186`
- **Defect:** If `strides[0] > 1` (transposed view, strided slice, column slice of a row-major matrix), the coalescing loop breaks at `i = 0` leaving `contiguousDimensions = 0`. Then:
  - `std::max_element` over an empty range returns `end()`; dereferencing is **UB**.
  - `sizes.at(contiguousDimensions - 1)` evaluates `sizes.at((size_t)-1)` → throws.
- **Note:** templated `CopyTensor` (lines 237-239) has the same bug. Fix both.
- **Suggested fix:** special-case `contiguousDimensions == 0` (e.g., bump to 1 and copy per-element).
- **Status:** open

### #3 — `maxStride = std::max_element(...)` is the wrong abstraction
- **Location:** `HipUtils.hpp:184-186`
- **Defect:** Intent is "size of contiguous slab in elements" — that is `strides[contiguousDimensions-1] * sizes[contiguousDimensions-1]`. Using `max_element` only happens to work because the contiguity check at lines 173-179 implicitly forces monotonically non-decreasing strides on the contiguous prefix. Any future loosening of the contiguity rule (e.g., broadcast dims with stride 0) silently picks the wrong stride.
- **Suggested fix:** replace with `strides[contiguousDimensions - 1]`.
- **Status:** open

### #4 — Sub-byte alignment hazard via `multiplyElementSize`
- **Location:** `HipUtils.hpp:197` ↔ `Utils.hpp:199-214`
- **Defect:** `multiplyElementSize(beginOffset, 0.5f)` is `beginOffset >> 1` (rounds down). For FP4/INT4 tensors with an odd `beginOffset` the resulting `bytesOffset` aliases the wrong nibble — copy starts a half-byte early and loses the trailing half-byte. Same hazard for `0.75` (3:4 packed).
- **Suggested fix:** assert that `beginOffset * elementSize` is integral, or reject sub-byte tensors at this entry point.
- **Status:** open

### #5 — Coalescing accepts `strides[i] < expectedStride` (overlap)
- **Location:** `HipUtils.hpp:173`
- **Defect:**
  ```cpp
  if(strides[i] > expectedStride) break;
  ```
  Only breaks on gaps, not on overlap (e.g., broadcast with stride 0, or aliased dims). When overlapping, the code extends `contiguousDimensions` and computes `copyBytes` as if dense — copying overlapped memory and over-counting bytes.
- **Suggested fix:** `if(strides[i] != expectedStride) break;`.
- **Status:** open

### #6 — Performance: serialized null-stream enqueue
- **Location:** `HipUtils.hpp:157, 201`
- **Issue:** `stream` parameter defaults to `0` (legacy null stream); call site at `DataInitialization.cpp:773` does not pass one. With large `copyCount`, each iteration synchronizes against all other work — the loop becomes effectively serial.
- **Suggested fix:** require an explicit stream from the caller, and/or batch into a single `hipMemcpy2DAsync` when `contiguousDimensions == desc.dimensions() - 1`.
- **Status:** open (not a correctness bug)

## Suggested minimal patch (covers #1, #2, partially #3)

```cpp
if(contiguousDimensions == 0)
    contiguousDimensions = 1;            // fall back to per-element copies

size_t copyBytes = multiplyElementSize(
    strides[contiguousDimensions - 1] * sizes[contiguousDimensions - 1],
    desc.elementBytes());

for(size_t idx = 0; idx < copyCount; idx++)
{
    CoordNumbered(idx,
                  coord.begin() + contiguousDimensions, coord.end(),
                  sizes.begin()  + contiguousDimensions, sizes.end());

    auto           beginOffset = desc.index(coord);
    size_t         bytesOffset = multiplyElementSize(beginOffset, desc.elementBytes());
    uint8_t*       dstBytes    = static_cast<uint8_t*>(dst)       + bytesOffset;
    uint8_t const* srcBytes    = static_cast<uint8_t const*>(src) + bytesOffset;

    HIP_CHECK_EXC(hipMemcpyAsync(dstBytes, srcBytes, copyBytes, direction, stream));
}
```

## Priority

1. #1 — only defect that misbehaves on the current call path.
2. #2, #5 — latent; trigger as soon as a non-row-major or strided descriptor reaches this code.
3. #3, #4, #6 — quality / robustness; address while in the file.
