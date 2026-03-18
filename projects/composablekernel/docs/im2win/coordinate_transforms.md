# CK Tile Tensor Coordinate Transforms — Algebraic Reference

All transforms are implemented in
`include/ck_tile/core/algorithm/coordinate_transform.hpp`.

Each transform maps **upper indices** (logical/visible) to **lower indices**
(physical/memory-side).  Transforms are composed by
`transform_tensor_descriptor`, which chains them into a single logical tensor
view without copying any data.  At kernel launch time the composed chain
converts a tile's logical `[i₀, i₁, …]` directly to a flat memory offset.

---

## 1. `make_pass_through_transform(length)`

**Shape**: 1 upper dim → 1 lower dim

### Index mapping

```
i_low = i_up
```

### Validity

Always valid.

### Effect on strides

None — the stride of the lower dimension passes through unchanged.

### Example

```cpp
// N dimension with physical stride = C * Hi * Wi
make_pass_through_transform(N_)
// physical contribution: i_up_N * (C * Hi * Wi)
```

---

## 2. `make_pad_transform(low_length, left_pad, right_pad)`

**Shape**: 1 upper dim → 1 lower dim

Upper length = `low_length + left_pad + right_pad`

### Index mapping

```
i_low = i_up - left_pad
```

### Validity predicate

```
valid  iff  left_pad ≤ i_up < low_length + left_pad
```

Elements in the padding zones produce `i_low < 0` or `i_low ≥ low_length`
and are invalid.  The kernel's `kPadK = true` tile window automatically
zeroes reads to invalid addresses.

### Effect on strides

None — the stride is unchanged; the index origin shifts by `−left_pad`.

### Example

```
Hi = 198,  LPH = 1,  RPH = 1  →  Hi_pad = 200

make_pad_transform(Hi_, InLeftPadH_, InRightPadH_)

i_up ∈ [0, 200)   →   i_low = i_up − 1  ∈ [−1, 199)

i_up =   0  →  i_low = −1  →  INVALID  (top padding, read as 0)
i_up =   1  →  i_low =  0  →  valid    (first real row)
i_up = 199  →  i_low = 198 →  valid    (last real row)
i_up = 199  →  i_low = 199 →  INVALID  (would be bottom padding if RPH > 0 made it so)
```

---

## 3. `make_embed_transform(up_lengths, coefficients)`

**Shape**: N upper dims → 1 lower dim

### Index mapping

Dot product of upper indices with coefficients:

```
i_low = Σ  i_up[k] · coefficients[k]   for k = 0 … N−1
```

### Validity

Always valid — no bounds checking (assumes upper indices are in range).

### Effect on strides

Each coefficient **becomes** the stride of its upper dimension in the
physical address space:

```
physical_stride_of_up[k]  =  coefficients[k] × physical_stride_of_low
```

This is the core mechanism for the sliding-window convolution mapping.

### Example — height windowing `(Y, Ho) → Hi_pad`

```cpp
make_embed_transform(make_tuple(Y_, Ho_),
                     make_tuple(ConvDilationH_, ConvStrideH_))

// i_low = i_Y * ConvDilationH  +  i_Ho * ConvStrideH

// Strides of the two new upper dimensions:
//   stride(Y)  = ConvDilationH * HiStride   [= Dy * Wi * G * C * sizeof(elem)]
//   stride(Ho) = ConvStrideH   * HiStride   [= Sy * Wi * G * C * sizeof(elem)]

// Unit stride / dilation (Sy = Dy = 1):
//   i_low = y + ho     →  physical row = filter_row + output_row  ✓

// Stride 2, dilation 1 (Sy = 2, Dy = 1):
//   i_low = y * 1 + ho * 2
//   →  strided convolution: every 2 input rows per output row
```

### Example — width windowing `(X, Wo) → Wi_pad`

```cpp
make_embed_transform(make_tuple(X_, Wo_),
                     make_tuple(ConvDilationW_, ConvStrideW_))

// i_low = i_X * ConvDilationW  +  i_Wo * ConvStrideW

// Strides:
//   stride(X)  = ConvDilationW * WiStride   [= Dx * G * C * sizeof(elem)]
//   stride(Wo) = ConvStrideW   * WiStride   [= Sx * G * C * sizeof(elem)]
```

---

## 4. `make_merge_transform(low_lengths)`

**Shape**: N lower dims → 1 upper dim

Upper length = `L₀ × L₁ × … × Lₙ₋₁`

### Index mapping — forward (lower → upper, i.e. flattening)

```
i_up = i_low[0] · (L₁ · L₂ · … · Lₙ₋₁)
     + i_low[1] · (L₂ · … · Lₙ₋₁)
     + …
     + i_low[N−2] · Lₙ₋₁
     + i_low[N−1]
```

This is standard **row-major** flattening with `i_low[0]` as the slowest
(outermost) dimension and `i_low[N−1]` as the fastest (innermost).

### Index mapping — inverse (upper → lower, i.e. recovery)

Implemented with **magic-number integer division** for efficiency on GPU:

```
tmp = i_up
for k = N−1 down to 1:
    tmp2      = tmp / Lₖ          (magic division, no integer divide instruction)
    i_low[k]  = tmp − tmp2 · Lₖ  (= tmp mod Lₖ)
    tmp       = tmp2
i_low[0] = tmp
```

### Validity

Always valid — any flat index in `[0, product)` maps to valid lower indices.

### Effect on strides

The upper dimension inherits the stride of `i_low[N−1]`, the **last /
innermost** lower dimension.  Vectorisation targets this innermost dimension.

### Example — merge `[N, Ho, Wo] → M = N×Ho×Wo`

```cpp
make_merge_transform(make_tuple(N_, Ho_, Wo_))

// i_up → (i_N, i_Ho, i_Wo) via modular arithmetic
//
// Stride of the merged M dimension = stride of Wo
//   NHWGK output: stride(Wo) = G * K * sizeof(elem)
//   → vectorised stores along K (K unit-stride, innermost)
```

### Example — merge `[Y, X, C] → K_gemm = Y×X×C`

```cpp
make_merge_transform(make_tuple(Y_, X_, C_))

// i_up → (i_Y, i_X, i_C) via modular arithmetic
//
// Stride of K_gemm = stride of C = 1  (C innermost in NHWGC)
// → vectorised loads of size VectorSizeA along C
```

---

## 5. `make_xor_transform(low_lengths)`

**Shape**: 2 upper dims → 2 lower dims  (shape-preserving, bijective)

Both upper and lower have the same shape `[L₀, L₁]`.

### Index mapping

```
i_low[0] = i_up[0]
i_low[1] = i_up[1]  XOR  (i_up[0] mod L₁)
```

### Validity

Always valid — XOR is a bijection, no element goes out of bounds.

### Effect on strides

Strides pass through unchanged.  The XOR permutes **which element** within
dimension 1 is accessed based on the value in dimension 0; it does not change
the physical memory layout.

### Purpose in CK im2win group-merge

Used in the output (C) descriptor when `NumGroupsToMerge = Gm > 1`.
The M dimension of the GEMM tile contains `K × Gm` elements (K output
channels from Gm merged groups).  The XOR shifts which spatial N-positions
each GEMM warp writes, so that valid output elements fall on a diagonal
pattern and warps do not write to the same output location:

```
// Conceptually (simplified for Gm = 4, K = 4, so M_tile = 16):
//
//  GEMM row   group  channel    XOR shift applied to N
//    0         g=0    k=0          N_start XOR 0
//    1         g=0    k=1          N_start XOR 1
//    2         g=0    k=2          N_start XOR 2
//    3         g=0    k=3          N_start XOR 3
//    4         g=1    k=0          N_start XOR 4   ← different spatial position
//    ...
//
// Only the diagonal entries (group g writes to its own spatial block)
// are valid outputs.  All other MFMA results are discarded.
```

---

## How strides compose through a descriptor chain

Consider a physical NHWGC input `I[N, Hi, Wi, G, C]` with packed strides:

```
stride(N)  = Hi * Wi * G * C
stride(Hi) = Wi * G * C
stride(Wi) = G * C
stride(G)  = C
stride(C)  = 1
```

After calling `make_pad_transform(Hi_, LPH, RPH)` on the Hi dimension:

```
stride(Hi_pad) = stride(Hi)   [unchanged — only origin shifts]
upper length   = Hi + LPH + RPH
```

After calling `make_embed_transform((Y_, Ho_), (Dy, Sy))` on the Hi_pad dimension:

```
stride(Y)  = Dy * stride(Hi)  =  Dy * Wi * G * C
stride(Ho) = Sy * stride(Hi)  =  Sy * Wi * G * C
```

After calling `make_merge_transform((N_, Ho_, Wo_))`:

```
stride(M = N×Ho×Wo) = stride(Wo) = Sx * Wi * G * C
```

At kernel runtime, the tile window computes the byte offset for element
`(i_M, i_K)` in the A matrix purely through integer arithmetic — no
intermediate tensor is ever materialised in memory.

---

## Summary table

| Transform | Upper dims | Lower dims | Index formula | Validity check | Stride of upper dim |
|---|---|---|---|---|---|
| `pass_through` | 1 | 1 | `i_low = i_up` | always | = stride of lower |
| `pad` | 1 | 1 | `i_low = i_up − left_pad` | `left_pad ≤ i_up < left_pad + length` | = stride of lower |
| `embed` | N | 1 | `i_low = Σ i_up[k] · coeff[k]` | always | `coeff[k] × stride(lower)` |
| `merge` | 1 | N | modular decomposition of `i_up` | always | = stride of `low[N−1]` |
| `xor` | 2 | 2 | `[i_up[0], i_up[1] ^ (i_up[0] % L₁)]` | always | = stride of lower (pass-through) |
