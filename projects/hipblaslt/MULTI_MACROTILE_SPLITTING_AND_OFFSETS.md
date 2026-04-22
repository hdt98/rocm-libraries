# Multi-MacroTile: Matrix Splitting and Offset Calculation

## Overview

The GEMM operation computes `D = α·op(A)·op(B) + β·C`, where `op()` is either identity (N) or transpose (T). Multi-macrotile splits this single GEMM into independent sub-GEMMs along **one** axis:

- **M-split** (strategies 17, 19, 21, 23): Partition M-rows into chunks. Each sub-problem has `(m_sub, N, K)`. **B is shared** across all sub-problems.
- **N-split** (strategies 18, 20, 22, 24): Partition N-columns into chunks. Each sub-problem has `(M, n_sub, K)`. **A is shared** across all sub-problems.

**K is never split** — every sub-problem multiplies against the full K dimension.

## Physical Storage Convention

All matrices use **column-major** storage (standard BLAS convention). For a matrix with `rows` rows and `cols` columns with leading dimension `ld`, element `(i, j)` lives at byte address:

```
base + (i + j * ld) * elem_size
```

The physical storage shape depends on the transpose flag:

| Matrix | transA/B = N | transA/B = T |
|--------|-------------|-------------|
| **A** | `[M × K]`, `lda ≥ M` | `[K × M]`, `lda ≥ K` |
| **B** | `[K × N]`, `ldb ≥ K` | `[N × K]`, `ldb ≥ N` |
| **C, D** | Always `[M × N]`, `ldc ≥ M` | Always `[M × N]`, `ldd ≥ M` |

This is enforced in `client.cpp`:

```cpp
int64_t min_lda = arg.transA == 'N' ? arg.M[i] : arg.K[i];
int64_t min_ldb = arg.transB == 'N' ? arg.K[i] : arg.N[i];
int64_t min_ldc = arg.M[i];
int64_t min_ldd = arg.M[i];
```

## The Offset Functions

Sub-problems are **windows** into the same parent buffer. The leading dimensions stay the same (they come from the full problem). Only the base pointer shifts by a byte offset.

### `calculateOffsetA` — depends on `transA`

```cpp
inline size_t calculateOffsetA(int64_t m_off, int64_t k_off, int64_t lda,
                                hipblasOperation_t transA, hipDataType dt)
{
    size_t e = getDataTypeSize(dt);
    return (transA == HIPBLAS_OP_N) ? m_off * e : (m_off + k_off * lda) * e;
}
```

Since K is never split, `k_off` is always 0 in practice:

- **`transA = N`**: A is `[M × K]`. M runs along rows. Skipping `m_off` rows = skipping `m_off` elements from the start of each column. Offset = **`m_off × elem_size`**. The same `lda` still strides between columns correctly.

- **`transA = T`**: A is `[K × M]`. M runs along columns. With `k_off = 0`, the formula simplifies to `m_off × elem_size`. **Note:** to skip `m_off` columns in a `[K × M]` matrix, you would normally need `m_off × lda × elem_size`. The current formula produces `m_off × elem_size`, which advances by `m_off` physical rows (the K axis). This appears to be a latent issue for the `transA = T` + M-split combination — in practice it works when the common case is `transA = T` with N-split (where `offset_A_bytes = 0`, i.e., A is shared and the formula is not exercised).

### `calculateOffsetB` — depends on `transB`

```cpp
inline size_t calculateOffsetB(int64_t n_off, int64_t k_off, int64_t ldb,
                                hipblasOperation_t transB, hipDataType dt)
{
    size_t e = getDataTypeSize(dt);
    return (transB == HIPBLAS_OP_N) ? n_off * ldb * e : n_off * e;
}
```

Again `k_off = 0`:

- **`transB = N`**: B is `[K × N]`. N runs along columns. Skipping `n_off` columns = `n_off × ldb` elements. Offset = **`n_off × ldb × elem_size`**.

- **`transB = T`**: B is `[N × K]`. N runs along rows. Skipping `n_off` rows = `n_off` elements from the start. Offset = **`n_off × elem_size`**.

### `calculateOffsetCD` — always column-major `[M × N]`

```cpp
inline size_t calculateOffsetCD(int64_t m_off, int64_t n_off, int64_t ld,
                                 hipDataType dt)
{
    return (m_off + n_off * ld) * getDataTypeSize(dt);
}
```

C and D are never transposed — always `[M × N]` column-major. This is the standard column-major element formula `(row + col × ld)`:

- **M-split**: skip `m_off` rows → `(m_off + 0 × ld) × e = m_off × e`
- **N-split**: skip `n_off` cols → `(0 + n_off × ld) × e = n_off × ld × e`

### `calculateOffsetBias` — 1D vector along M

```cpp
inline size_t calculateOffsetBias(int64_t m_off, hipDataType dt)
{
    return m_off * getDataTypeSize(dt);
}
```

Bias is a 1D vector of length M, so only M-split needs an offset.

## The Split Loop

```cpp
int64_t offset = 0;
for (int i = 0; i < num_splits; i++)
{
    GemmSubProblem s{};
    int64_t sz = split_sizes[i];

    if (is_m_split)
    {
        s.m_size = sz;        s.n_size = N;   s.k_size = K;
        s.m_offset = offset;  s.n_offset = 0;
        s.offset_A_bytes    = calculateOffsetA(offset, 0, lda, transA, a_type);
        s.offset_B_bytes    = 0;
        s.offset_C_bytes    = calculateOffsetCD(offset, 0, ldc, c_type);
        s.offset_D_bytes    = calculateOffsetCD(offset, 0, ldd, d_type);
        s.offset_E_bytes    = calculateOffsetCD(offset, 0, lde, aux_type);
        s.offset_bias_bytes = calculateOffsetBias(offset, bias_type);
    }
    else
    {
        s.m_size = M;         s.n_size = sz;  s.k_size = K;
        s.m_offset = 0;       s.n_offset = offset;
        s.offset_A_bytes    = 0;
        s.offset_B_bytes    = calculateOffsetB(offset, 0, ldb, transB, b_type);
        s.offset_C_bytes    = calculateOffsetCD(0, offset, ldc, c_type);
        s.offset_D_bytes    = calculateOffsetCD(0, offset, ldd, d_type);
        s.offset_E_bytes    = calculateOffsetCD(0, offset, lde, aux_type);
        s.offset_bias_bytes = 0;
    }
    offset += sz;
}
```

The running `offset` accumulates as `0, m0, m0+m1, ...` (for M-split) or `0, n0, n0+n1, ...` (for N-split).

### Summary Table

| | M-split | N-split |
|---|---------|---------|
| **A** | Offset advances with `m_off` | Shared (`offset = 0`) |
| **B** | Shared (`offset = 0`) | Offset advances with `n_off` |
| **C, D, E** | Advance along M | Advance along N |
| **Bias** | Advance along M | Stays at 0 |

## How Layouts Are Created for Sub-problems

In `testing_matmul.hpp`, the hipBLASLt layout for each sub-problem uses the **sub-problem's logical size** but the **parent's leading dimension**:

```cpp
int64_t matA_rows = transA == HIPBLAS_OP_N ? sub.m_size : sub.k_size;
int64_t matA_cols = transA == HIPBLAS_OP_N ? sub.k_size : sub.m_size;
hipblasLtMatrixLayoutCreate(&ctx.matA, arg.a_type, matA_rows, matA_cols, lda[0]);

int64_t matB_rows = transB == HIPBLAS_OP_N ? sub.k_size : sub.n_size;
int64_t matB_cols = transB == HIPBLAS_OP_N ? sub.n_size : sub.k_size;
hipblasLtMatrixLayoutCreate(&ctx.matB, arg.b_type, matB_rows, matB_cols, ldb[0]);

hipblasLtMatrixLayoutCreate(&ctx.matC, arg.c_type, sub.m_size, sub.n_size, ldc[0]);
hipblasLtMatrixLayoutCreate(&ctx.matD, arg.d_type, sub.m_size, sub.n_size, ldd[0]);
```

`lda[0]` (the full-problem ld) is used because each sub-matrix is a **strided window** into the parent allocation, not a separately packed buffer. The reduced rows/cols tell hipBLASLt the logical extent of the sub-problem, while the ld tells it the physical stride between columns. The pointer is then shifted:

```cpp
ctx.A_ptr = static_cast<char*>(dA[0].buf()) + sub.offset_A_bytes;
ctx.B_ptr = static_cast<char*>(dB[0].buf()) + sub.offset_B_bytes;
ctx.C_ptr = static_cast<char*>(dC[0].buf()) + sub.offset_C_bytes;
ctx.D_ptr = static_cast<char*>((*dDp)[0].buf()) + sub.offset_D_bytes;
```

## Visual Summary

For a 2-way M-split of `D = op(A) × op(B)`:

```
              K                     N                        N
        ┌───────────┐        ┌───────────┐            ┌───────────┐
   m0   │  A_sub0   │        │           │       m0   │  D_sub0   │
        ├───────────┤   ×  K │     B     │   =        ├───────────┤
   m1   │  A_sub1   │        │  (shared) │       m1   │  D_sub1   │
        └───────────┘        └───────────┘            └───────────┘

        A offset moves              B offset = 0          D offset moves
        with m_off                                        with m_off
```

For a 2-way N-split:

```
              K                     n0    n1                n0    n1
        ┌───────────┐        ┌─────┬─────┐          ┌─────┬─────┐
        │           │        │     │     │          │     │     │
      M │     A     │   ×  K │ B0  │ B1  │   =    M │ D0  │ D1  │
        │  (shared) │        │     │     │          │     │     │
        └───────────┘        └─────┴─────┘          └─────┴─────┘

        A offset = 0          B offset moves          D offset moves
                               with n_off              with n_off
```

## Strategy Reference

| Strategy | Split Axis | Method |
|----------|-----------|--------|
| 17 | M | Origami analytical search |
| 18 | N | Origami analytical search |
| 19 | M | Brute-force search |
| 20 | N | Brute-force search |
| 21 | M | 3-way split |
| 22 | N | 3-way split |
| 23 | M | XCD-aware (L2 cache optimized) |
| 24 | N | XCD-aware (L2 cache optimized) |
