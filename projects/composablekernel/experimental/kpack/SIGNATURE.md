# Kernel Signatures

A kernel **signature** is the mathematical contract: what a kernel computes. It describes
typed tensors, scalar parameters, and the operation that relates them.

The signature says nothing about *how* the kernel computes — tile geometry, pipeline
strategy, and scheduling belong to the **algorithm**. Together, a signature and an
algorithm fully specify a compiled kernel variant.

## Tensors

A tensor has compile-time properties (fixed when the kernel is compiled) and runtime
properties (provided at each launch).

**Compile-time:**

| Property | Description |
|----------|-------------|
| **name** | Semantic role: `"A"`, `"bias"`, `"query"`, `"output"` |
| **dtype** | Element type: FP32, FP16, BF16, FP8, INT8, ... |
| **rank** | Number of dimensions |
| **layout** | Memory layout: `Row`, `Col`, `Contiguous`, or strides |

**Runtime:**

| Property | Description |
|----------|-------------|
| **pointer** | Device memory address |
| **sizes** | Extent of each dimension |
| **strides** | Memory stride of each dimension |

### Layout

Layout describes how logical dimensions map to memory. The canonical representation is
**strides** — one integer per dimension giving the element offset between consecutive
indices.

For rank-2 tensors, layout enums provide convenient shorthand:

```
Row-major [M, K]:  strides = [K, 1]    (rows are contiguous)
Col-major [M, K]:  strides = [1, M]    (columns are contiguous)
```

For higher-rank tensors, layout tags name common stride patterns:

```
NHWC [N, H, W, C]:  strides = [H*W*C, W*C, C, 1]
NCHW [N, C, H, W]:  strides = [C*H*W, H*W, W, 1]
```

Layout enums are **input sugar** — they resolve to strides. Strides are the internal
representation because they handle every case: transposes, padding, non-contiguous views,
and layouts we haven't named yet.

### Type resolution

Signatures use an optional cascade for ergonomic type specification:

```cpp
// Set everything to FP16 (accumulator defaults to FP32):
{ .dtype = FP16 }

// Mixed precision — FP16 inputs, FP32 output:
{ .dtype = FP16, .c_dtype = FP32 }

// Fully explicit:
{ .a_dtype = FP16, .b_dtype = BF16, .c_dtype = FP32, .acc_dtype = FP32 }
```

Each operation defines its own cascade rules. The common pattern:
- `dtype` sets the default for all tensors
- Per-tensor overrides (`a_dtype`, `b_dtype`, ...) take precedence
- `acc_dtype` defaults to FP32 when not specified

Resolution is `consteval` — unresolvable types are compile errors, not runtime surprises.

## Scalars

Scalar parameters have a name, type, optional default, and a runtime value.

```
alpha     : float, default 1.0
scale     : float                   (required, no default)
groups    : int32                   (convolution group count)
causal    : bool, default false     (FMHA causal masking)
```

**Problem-size scalars** (M, N, K, sequence lengths) describe the extent of tensor
dimensions. They are always runtime values but may have compile-time constraints
(e.g., "K must be a multiple of 32 for this variant").

## Operations

An operation defines which tensors participate, which are optional, and what shape
constraints hold between them.

### GEMM

```
C[B, M, N] = sum_k( A[B, M, K] * B[B, K, N] )
```

| Tensor | Rank | Role | Required |
|--------|------|------|----------|
| A | 2-3 | In | yes |
| B | 2-3 | In | yes |
| C | 2-3 | Out | yes |

Batch is a first-class dimension. An unbatched GEMM is simply batch size 1 (or rank 2).

Constraints: `A.size[-2] == C.size[-2]`, `A.size[-1] == B.size[-2]`, `B.size[-1] == C.size[-1]`

### Convolution

```
Y[N, K, oH, oW] = sum( X[N, C, iH, iW] * W[K, C/g, kH, kW] )
```

| Tensor | Rank | Role | Required |
|--------|------|------|----------|
| X | 4+ | In | yes |
| W | 4+ | In | yes |
| Y | 4+ | Out | yes |

Parameters: `pad`, `stride`, `dilation`, `groups`

### FMHA (Flash Attention)

```
O = softmax( Q @ K^T / scale ) @ V
```

| Tensor | Rank | Role | Required |
|--------|------|------|----------|
| Q | 4 | In | yes |
| K | 4 | In | yes |
| V | 4 | In | yes |
| O | 4 | Out | yes |
| bias | 4 | In | no |

Parameters: `scale`, `causal`

### Elementwise

```
C[i] = alpha * A[i] + beta * B[i]
```

| Tensor | Rank | Role | Required |
|--------|------|------|----------|
| A | 1+ | In | yes |
| B | 1+ | In | yes |
| C | 1+ | Out | yes |

Parameters: `alpha`, `beta`

## Fusion

Fusion adds tensors and pointwise operations to a base computation. Rather than
enumerating every combination, we describe fusion as an **epilogue chain** applied
to the base result:

```
Base:       acc[M, N] = A * B                  (GEMM in accumulator type)
Epilogue:   acc       = acc + D0[M, N]         (bias addition)
            acc       = relu(acc)              (activation)
            C[M, N]   = cast(acc)              (output in storage type)
```

The fused signature is the base signature plus the epilogue tensors and operations.
Each epilogue tensor (D0, D1, ...) has its own dtype and layout. The epilogue operation
describes what to do with them.

The epilogue model should be as general as possible — at least as flexible as the best
parts of CK Tile's epilogue system. The current enum (`None`, `Add`, `Multiply`) is a
starting point; the target is a composable graph of typed operations that can express
arbitrary epilogue fusions.

## Compile-Time / Runtime Boundary

Every kernel has two distinct phases:

**Compile time** — the signature and algorithm are validated, and a kernel binary is
produced. This determines:
- Tensor dtypes, ranks, layouts
- Layout patterns
- Operation type and epilogue structure
- Accumulator type
- Tile geometry and pipeline (from the algorithm)

**Runtime** — the kernel is launched with concrete arguments:
- Tensor pointers, sizes, strides
- Scalar values (problem sizes, alpha/beta, etc.)

The same compiled binary handles different problem sizes. Validation at compile time
(`consteval make_kernel`) catches configuration errors as compiler diagnostics, not
runtime crashes.

The runtime arguments are a **flat POD struct** with stable ABI — no constructors,
no virtual functions, no heap allocation. Field order and alignment are documented
and asserted with `static_assert`.

## Design Decisions

**Batch dimensions are explicit.** Batch, head count, and other semantic dimensions are
first-class in the signature, not implicit extra tensor dimensions. Keep semantic
information in the signature — it enables better validation and clearer code.

**Epilogue composition is general.** The target is a composable graph of typed epilogue
operations, at least as flexible as CK Tile's best epilogue patterns. We start simple
(enum + D tensors) and grow toward full generality. A more general solution is often
simpler than accumulating special cases.

**Split-K is algorithmic.** Parallelism strategies like split-K, workspace allocation,
and tuning parameters belong in the algorithm, not the signature. The signature describes
*what* is computed; the caller should not need to know *how* partial results are managed.

**Problem-size constraints live in the algorithm.** Constraints like "K must be a multiple
of 32" arise from tile geometry choices, not from the mathematical operation. The algorithm
declares its constraints; `make_kernel` cross-validates them against the signature. Runtime
argument validation checks actual values against the compiled variant's requirements.
