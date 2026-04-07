# hipconv: Architecture and 3D Extension Plan

## Notation

Throughout this document the following names are used for tensor dimensions:

| Symbol | Meaning |
|--------|---------|
| N | batch size |
| C | number of input channels |
| K | number of output channels |
| Hi, Wi | input height and width |
| Ho, Wo | output height and width |
| Y, X | filter height and width (2D); Z, Y, X = filter depth, height, width (3D) |
| G | number of groups; C_g = C/G channels per group |

Filter size is always 3×3 (2D) or 3×3×3 (3D), so Y = X = 3 (and Z = 3 for 3D).

**Implementation note — name mapping:**
The source code uses a mix of conventions across layers. The table below maps design-doc
names to the names found in the implementation:

| Design doc | `Conv2dParams` struct | Kernel function parameters |
|------------|-----------------------|---------------------------|
| Hi, Wi     | `h`, `w`              | `hi`, `wi`                |
| Ho, Wo     | `p`, `q`              | `ho`, `wo`                |
| Y, X       | `kh`, `kw`            | `fy`, `fx`                |
| pad_h, pad_w | `pad_h`, `pad_w`    | `py`, `px`                |
| stride_h, stride_w | `stride_h`, `stride_w` | `sy`, `sx`      |
| dilation_h, dilation_w | `dilation_h`, `dilation_w` | `dy`, `dx` |

Inside the kernel body the filter row index is named `R` (0 … Y−1) and the filter column
index is named `S` (0 … X−1). The output row coordinate is named `p_out` and its
accumulator slot index is named `p_idx`.

---

## Part 1 — Architecture of the Current 2D Implementation

### 1.1 Overview

hipconv is a direct-convolution library written in HIP, targeting AMD CDNA 4 (gfx950 / MI355X).
It supports grouped 2D convolution (Fprop and Dgrad) in fp16, NHWC layout, with a 3×3 filter
(Y = X = 3), unit stride, unit dilation, and the constraint `C_g = K_g` (same channel count per
group in and out). The number of channels per group (C_g) is fixed at 4, 8, 16, or 32 depending
on the kernel variant.

There is no im2col or Winograd transform step. The kernel is a fused direct convolution that
loads input tiles from global memory into LDS (shared memory), and then drives MFMA instructions
directly from LDS-staged registers.

---

### 1.2 File Structure

```
hipconv/
  include/hipconv/
    hipconv.hpp          — public API: get_valid_configs, launch
    conv2d_params.hpp    — Conv2dParams struct, SizeView helper
  src/
    registry.cpp         — algorithm enumeration and dispatch
    algo_config.h        — AlgoConfig (algorithm id + config index)
    algo_entry.h         — AlgorithmEntry (vtable for an algorithm)
    kernel_variant.h     — KernelVariant (vtable for one kernel variant)
    launch_params.h      — LaunchParams (grid, block_size)
    detail.h             — static_for<N>, dispatch<N> (compile-time loops)
    types.h              — fp16x4_t, fp32x4_t, etc.
    memory.h             — wait_vmcnt inline asm
    mathutil.h           — divup, maximum
    matrix_layout.h      — MatrixLayout<M,K,Batch,T> (lane → MFMA coordinate)
    swizzle.h            — SwizzleT<C>: LDS bank-conflict-free address mapping
    transpose_4x4.h      — 4×4 matrix transpose via wave shuffle (Dgrad)
    transpose_16x16.h    — 16×16 row-major load with transpose
    grouped/
      grouped_conv.hpp/cpp — variant table and dispatch
      grouped_4c_fp16.h    — 4-channel kernel  (MFMA 4×4×4 batch 16)
      grouped_8c_fp16.h    — 8-channel kernel  (MFMA 16×16×32)
      grouped_8c_transforms.h — GT / D block-matrix documentation
      grouped_16c_fp16.h   — 16-channel kernel (MFMA 16×16×16)
      grouped_32c_fp16.h   — 32-channel kernel (MFMA 16×16×32)
```

---

### 1.3 Dispatch Hierarchy

```
hipconv::launch()           (public API, registry.cpp)
  └─ algorithms[algo_id]    (AlgorithmEntry vtable)
       └─ grouped::launch() (grouped_conv.cpp)
            └─ variants[variant_id]   (KernelVariant vtable)
                 └─ grouped_Xc::launch() → GPU kernel<<<grid, block>>>
```

Each variant holds function pointers for `is_applicable`, `config_is_compatible`,
`get_launch_params`, and `launch`. The caller iterates the variant table, selects the first
applicable one that has a compatible configuration, then launches it.

---

### 1.4 Data Layouts

| Tensor  | Global layout      | Unit             |
|---------|--------------------|------------------|
| Input   | N × Hi × Wi × C   | uint4 = 8 fp16   |
| Weights | K × Y  × X  × C   | fp16             |
| Output  | N × Ho × Wo × K   | uint4 = 8 fp16   |

Output dimensions: `Ho = (Hi + 2·pad_h − Y) / stride_h + 1`,
`Wo = (Wi + 2·pad_w − X) / stride_w + 1`.

For Dgrad the roles of input and output are swapped: the kernel receives the output-gradient
in the "input" position and writes the input-gradient to the "output" position.

**Implementation note:** the `Conv2dParams` struct stores input size as `h`, `w` and output
size as `p`, `q` (see the Notation table above).

---

### 1.5 Grid and Block Decomposition (4-channel kernel as reference)

The workgroup tile covers:

- **Wo-dimension (output width):** `block_wo = waves_q4 × 4` output columns
  *(named `block_q` in the source; `waves_q4` is the number of waves along the output-width dimension)*
- **C-dimension (channels):** `block_c = waves_c64 × 64` channels
  *(`waves_c64` is the number of waves along the channel dimension)*
- **N-dimension (batch):** handled through `n_fold` folding

Grid dimensions:

```
grid.x = ceil(Wo / block_wo) × n_fold
grid.y = ceil(C / block_c)
grid.z = ceil(N / n_fold)
```

Each block contains `waves_c64 × waves_q4 × 64` threads.

---

### 1.6 Global Memory → LDS: Buffer Load to LDS

Global → LDS transfers use the AMD-specific intrinsic:

```cpp
__builtin_amdgcn_raw_ptr_buffer_load_lds(rsrc, lds_ptr, 16, voffset, 0, 0, 0);
```

This is a 128-bit (16-byte = 8 × fp16) asynchronous load that writes directly into LDS without
going through VGPRs. It honours the active EXEC mask, so out-of-bounds threads are simply
suppressed.

A buffer resource descriptor is constructed with the full tensor size as the bounds, so
out-of-range accesses return zero automatically (padding handling).

One row of input is loaded per iteration: `BLOCK_W × BLOCK_C8` vectors of uint4, where
`BLOCK_W = block_wo + (X − 1)` (output columns plus the left/right halo needed by the 3×3
filter, i.e. the input width covered by the tile). `BLOCK_C8 = block_c / 8` (number of
uint4 vectors covering the channel block).

A **double buffer** (two LDS slots, toggled by `tic`/`toc`) allows the next row's load to
overlap with the MFMA computation on the current row.

---

### 1.7 LDS Swizzle (`swizzle.h`)

LDS has 32 banks of 4 bytes each. Naïve NHWC storage would cause bank conflicts when different
threads in a wave access the same channel offset for different spatial positions. SwizzleT
applies a permutation:

```
offset_uint2(x, c4) = x*C4  +  ((x + c8) % C8)*2  +  c4%2
```

where `x` is the spatial position along the width dimension (0 … BLOCK_W−1), `c4` is the
channel index in units of 4 fp16 values, `C4 = C_g/4` and `C8 = C_g/8`.

The XOR-like term `(x + c8) % C8` rotates the channel dimension by the spatial position,
spreading accesses across different LDS banks for threads that differ in `x`. The inverse
functions `x(offset)` and `c8(offset)` let global-load addresses be computed from a flat thread
index, while MFMA-read addresses use `(x, c4)` coordinates.

---

### 1.8 Weight Loading (Prologue)

The weight tensor is loaded **once** from global memory into LDS at the start of the kernel,
then read from LDS into registers before the main loop begins.

**4-channel kernel (C_g = 4):**
- Weights for one group: `Y × X` tiles of `C_g = 4` fp16 values = `Y × X` uint2 words.
- All `block_groups` groups are loaded cooperatively, with threads striding over
  `WEIGHT_LDS_SIZE_UINT4` vectors.
- After synchronisation, each thread reads its `weights_reg[Y*X]` slice from LDS.
- For Dgrad: `transpose_4x4_batch16` flips the 4×4 weight matrix in-place so that the same
  forward kernel computes the backward filter-flip.

**8-channel kernel (C_g = 8) — GT transformation:**
- Weights are loaded into the GT block-matrix structure (see §1.9) at this stage.

Weights stay in registers for the entire duration of the main loop — this is the key to avoiding
repeated loads.

---

### 1.9 The 8-Channel GT/Toeplitz Trick (`grouped_8c_transforms.h`)

The 8-channel kernel maps a 1D convolution with a 3-tap filter onto the MFMA 16×16×32
instruction by exploiting a Toeplitz structure.

**Scalar F(2,3):** convolving a 4-element signal `d` with a 3-tap filter `g` yields 2 outputs:

```
q = d · G,    G =  [ g0  0  ]
                    [ g1  g0 ]
                    [ g2  g1 ]
                    [ 0   g2 ]
```

25% of G is zeros, so MFMA utilisation is 75%.

**Block generalisation to C_g = 8 channels per group:**
- Each scalar in `d` becomes a row-vector of 8 input channels.
- Each scalar in `g` becomes an 8×8 weight matrix.
- G becomes a 32×16 block matrix **GT** (transposed for MFMA B-matrix column-major convention).
- 16 overlapping 4-element windows are stacked into D (16×32), giving the full 16×16×32 matmul.

This structure is computed once in the prologue; the main loop then drives MFMA with
pre-staged registers.

---

### 1.10 Main Compute Loop

The main loop iterates over input rows `y_base` in steps of `Y` (= kh = 3).

`Y_LOCAL` (0 … Y−1) is the local row index within the current step, always a compile-time
constant via `static_for`. `R` (0 … Y−1) is the filter row index; `S` (0 … X−1) is the
filter column index.

```
for y_base in 0 .. Hi-Y step Y:
    static_for<Y>(Y_LOCAL):            // unrolled, compile-time Y_LOCAL
        wait for previous load
        issue async load for row y+1
        static_for<X>(S):              // unrolled, compile-time S
            read input tile from LDS
            static_for<Y>(R):          // unrolled, compile-time R
                p_idx = (Y_LOCAL - R + Y) % Y   // constexpr → register index
                acc[p_idx] += MFMA(weights_reg[R*X+S], input_reg)
        swap tic/toc
        if output row p_out is valid: flush acc[p_flush] → LDS → global
        acc[p_flush] = 0
```

Key properties enabled by `static_for`:
- All loop indices (`Y_LOCAL`, `S`, `R`) are compile-time constants.
- `p_idx` is therefore a compile-time constant → `acc[p_idx]` is a fixed register, not an
  array access through a pointer.
- The accumulator array `acc[Y]` lives entirely in AGPR registers (MFMA accumulator file).

**Circular accumulator buffer:** `acc[Y]` has Y = 3 slots, indexed by
`p_idx = (Y_LOCAL − R + Y) % Y`. Slot `p` accumulates contributions from all `(R, S)` filter
taps for output row `p`. When all contributions for a given output row have been received, the
slot is flushed to global memory and reset to zero.

The output row coordinate is `p_out = y_base + Y_LOCAL − (Y − 1)`. Flushing happens when
`p_out` is a valid output row (0 … Ho−1).

---

### 1.11 Output Writing

After each row's MFMA contributions are accumulated, the completed output row is written:

1. Convert `fp32x4_t acc` → `fp16x4_t` via `__float22half2_rn`.
2. Write to LDS at the swizzled address `output_lds_offset`.
3. `__syncthreads()` to ensure the full tile is written.
4. Threads assigned to stores read back `uint4` words from LDS and write to global memory
   in NHWC layout.

The output LDS buffer reuses the same `output_lds[]` array that was used for weight staging
in the prologue (the weights have been moved to registers by this point).

---

### 1.12 Remainder and Flush

- **Remainder loop:** handles the last `Hi % Y` input rows using the same `static_for`
  structure (with a runtime guard `if Y_LOCAL >= Hi % Y: return`).
- **Flush loop:** output rows whose last contributing input row falls outside `[0, Hi)` are
  never flushed by the main loop. A final sequential loop over these rows writes them out.

---

### 1.13 MFMA Instructions Used

| Kernel (C_g) | MFMA instruction              | Output tile | K-reduction | Batch |
|--------------|-------------------------------|-------------|-------------|-------|
| 4            | `mfma_f32_4x4x4f16`           | 4×4         | 4           | 16    |
| 8            | `mfma_f32_16x16x32_f16`       | 16×16       | 32          | —     |
| 16           | `mfma_f32_16x16x16f16`        | 16×16       | 16          | —     |
| 32           | `mfma_f32_16x16x32_f16`       | 16×16       | 32          | —     |

Accumulators are always fp32; conversion to fp16 happens only at the output-write stage.

---

### 1.14 Synchronisation Pattern

```
prologue:  issue weight loads
           issue first input row load
           s_waitcnt vmcnt(1)   — weights done, first input still in flight
           __syncthreads()
           read weights from LDS → registers

main loop: s_waitcnt vmcnt(0)   — previous input load done
           __syncthreads()
           issue next input load (async)
           MFMA using input from LDS
           ...
           store output to LDS
           __syncthreads()      — LDS output tile complete
           write LDS → global
```

The double-buffer tic/toc toggle ensures that the new load targets a different LDS slot from
the one being consumed by MFMA.

---

## Part 2 — Plan for 3D Convolution Extension

### 2.1 Target Specification

Extend hipconv to support **3D forward convolution** with:

- Filter: **3×3×3** (Z × Y × X = depth × height × width)
- Stride: **1×1×1**
- Dilation: **1×1×1**
- Groups: **1** (standard convolution, not grouped)
- Layout: **NDHWC** input, **KZYXC** weights, **NOPQK** output
  *(Di, Hi, Wi = input depth/height/width; Do, Ho, Wo = output depth/height/width)*
- Data type: **fp16**, accumulation in fp32
- Target architecture: CDNA 4 (gfx950)

Channel shapes of interest:

| C  | K   | Notes |
|----|-----|-------|
| 3  | 96  | Few input channels |
| 96 | 3   | Few output channels |
| 16 | 384 | Moderate C, large K |

---

### 2.2 Conceptual Mapping: 2D → 3D

The 2D kernel treats the convolution as a direct loop over filter rows (`R = 0..Y-1`, height)
with an inner loop over filter columns (`S = 0..X-1`, width). The input is staged row by row
(one 2D spatial row = one Hi-plane slice).

For 3D, a new **depth dimension** `T` (filter depth index, 0 … Z−1) is added. The input is
an `N × Di × Hi × Wi × C` tensor, and the output is `N × Do × Ho × Wo × K` where
`Do = Di − Z + 1` (unit stride, no padding in depth).

The key insight is that the **depth dimension is structurally identical to the height dimension
in 2D**: it introduces another level of the sliding-window accumulation pattern. The 2D kernel
loops over `R ∈ [0, Y)` for the height contribution; the 3D kernel will additionally loop
over `T ∈ [0, Z)` for the depth contribution.

---

### 2.3 Algorithm Design

#### 2.3.1 Outer Structure

The 3D kernel will follow the same structure as the 4c/8c 2D kernels, with the following
changes:

```
Prologue:
  load weights[Z × Y × X × C_g] → LDS → registers

Main loop (over input depth slices d_base in steps of Z):
  Inner loop (over input rows y_base in steps of Y):
    static_for<Y>(Y_LOCAL):
      load input row (d_cur, y_cur, ...) → LDS (double-buffered)
      static_for<X>(S):
        read input tile from LDS
        static_for<Z>(T):
          static_for<Y>(R):
            p_idx_2d = (Y_LOCAL - R + Y) % Y
            p_idx_3d = (D_LOCAL - T + Z) % Z
            p_idx = p_idx_3d * Y + p_idx_2d
            acc[p_idx] += MFMA(weights_reg[T*Y*X + R*X + S], input_reg)
```

`D_LOCAL` (0 … Z−1) is the local depth index within the current depth step, a compile-time
constant via an outer `static_for<Z>`. The accumulator buffer grows from `Y` slots to
`Z × Y = 3 × 3 = 9` slots.

#### 2.3.2 Accumulator Indexing

In 2D, the circular buffer uses:
```cpp
constexpr int p_idx = (Y_LOCAL - R + Y) % Y;
```

In 3D, with a 2D circular buffer over (T, R):
```cpp
constexpr int p_idx = ((D_LOCAL - T + Z) % Z) * Y
                    + ((Y_LOCAL - R + Y) % Y);
```

Since `D_LOCAL`, `T`, `Y_LOCAL`, and `R` are all compile-time constants (via nested
`static_for`), `p_idx` is a compile-time constant, keeping `acc[p_idx]` as a fixed AGPR
register address — exactly as in 2D.

#### 2.3.3 Flushing

An output voxel `(od, oh)` is complete when its last contributing input is at
`(d, h) = (od + Z − 1, oh + Y − 1)`. Flushing happens as soon as the last contribution
is computed. The flush condition becomes a 2D check:

- After processing input row `(d, h)`, flush output row `(d − Z + 1, h − Y + 1)` if that
  output is in bounds.

The final flush loop handles voxels whose last contribution falls outside the tensor boundary,
analogous to the 2D case.

---

### 2.4 LDS Layout for 3D

In 2D, one LDS buffer holds one row: `BLOCK_W × C_g` channels.
In 3D, the same approach applies: load one (Hi, Wi) row of the Di-dimension at a time.
The LDS double-buffer scheme remains identical; the row address computation gains a depth offset:

```cpp
input_voffset = base
              + d * stride_d           // stride_d = Hi * Wi * C8 (in uint4 units)
              + h * stride_h
              + global_col * stride_w;
```

No change to LDS size or swizzle is needed — the depth dimension is handled by the outer loop,
not by the LDS tile.

---

### 2.5 Weight Layout

Weights are `K × Z × Y × X × C` (KZYXC layout). For a single group with `C_g` channels per
group, the register array grows from `Y × X` to `Z × Y × X` entries:

```cpp
fp16x4_t weights_reg[Z * Y * X];   // 27 entries for 3×3×3
```

The prologue loads these through LDS exactly as before, now spanning `Z × Y × X` filter
positions per group.

---

### 2.6 Channel Mapping and MFMA Reuse

The MFMA instruction operates on the channel dimension — it is completely orthogonal to the
spatial dimensions (Di, Hi, Wi). Therefore the MFMA instruction and its operand layouts are
**unchanged**. The additional depth loop simply adds more MFMA calls per output voxel:

- 2D (3×3):   Y × X = 9 MFMA calls per output row per channel group
- 3D (3×3×3): Z × Y × X = 27 MFMA calls per output voxel per channel group

The ratio of arithmetic to memory access improves (more MFMA reuse per loaded input element),
which is beneficial for roofline performance.

---

### 2.7 Parameter Struct Extensions

Add depth parameters to `Conv2dParams` or create a new `Conv3dParams`:

```cpp
struct Conv3dParams {
    DataType   in_type, wei_type, out_type;
    TensorOrder order;
    Direction  direction;
    int n, c, k;
    int di, hi, wi;    // input  depth/height/width
    int do_, ho, wo;   // output depth/height/width  (do_ avoids C++ keyword clash)
    int kz, ky, kx;    // filter depth/height/width  (Z, Y, X)
    int stride_d, stride_h, stride_w;
    int dilation_d, dilation_h, dilation_w;
    int pad_d, pad_h, pad_w;
    int groups;
};
```

---

### 2.8 Grid Decomposition for 3D

Add an output-depth dimension to the grid. The simplest decomposition maps:

```
grid.x = ceil(Wo / block_wo) × n_fold   // output width tiles × batch fold
grid.y = ceil(C / block_c)              // channel tiles
grid.z = ceil(N / n_fold) × Do          // batch fold × output depth slices
```

Alternatively, compress (Do, batch) into grid.z as the 2D kernel compresses (n_fold, batch)
into grid.x and grid.z. The depth tile size can start at 1 (one output depth slice per block)
since the depth loop is already the outermost spatial loop inside the kernel.

---

### 2.9 Differences from 2D — Summary Table

| Aspect                    | 2D Kernel                     | 3D Extension                         |
|---------------------------|-------------------------------|--------------------------------------|
| Filter shape              | Y × X (3×3)                   | Z × Y × X (3×3×3)                   |
| Accumulator slots         | Y = 3                         | Z × Y = 9                           |
| Accumulator index         | `(Y_LOCAL - R + Y) % Y`       | 2D index over (D_LOCAL, Y_LOCAL)     |
| Weights in registers      | Y × X entries                 | Z × Y × X entries                   |
| Main loop depth           | over Hi only                  | nested over Di and Hi                |
| LDS per buffer            | BLOCK_W × C_g channels        | same (one Hi-row at a time)          |
| LDS swizzle               | SwizzleT<C_g>                 | unchanged                            |
| Global address stride     | row stride (Hi×Wi×C)          | add depth stride (Hi×Wi×C per Di-step) |
| MFMA instruction          | unchanged                     | unchanged                            |
| Grid dimension            | (Wo-tiles × n_fold, C-tiles, N-tiles) | add Do dimension to grid.z  |
| Flush logic               | 1D circular over Y            | 2D circular over Z × Y              |

---

### 2.10 Implementation Steps

1. **Add `Conv3dParams`** (or extend `Conv2dParams` with depth fields). Add a new
   `SizeView3d` helper analogous to `SizeView`.

2. **Add a new algorithm namespace** `grouped_3d` following the pattern in
   `hipconv/src/grouped/`.

3. **Implement `grouped_3d_Xc_fp16.h`** starting from the 4c kernel as the simplest baseline:
   - Add `Z` (= kd = 3) as a compile-time template/config parameter.
   - Change `weights_reg` from `[Y*X]` to `[Z*Y*X]`.
   - Change `acc` from `[Y]` to `[Z*Y]`.
   - Add an outer `static_for<Z>(D_LOCAL)` around the existing `static_for<Y>(Y_LOCAL)`.
   - Compute `p_idx` as a 2D circular index over (D_LOCAL, Y_LOCAL).
   - Update global address computation to include the depth stride.
   - Update flush logic for the 2D circular buffer.

4. **Register the new algorithm** in `registry.cpp` and `algo_entry.h`.

5. **Add launch parameters** to `grouped_3d::get_launch_params` including the Do grid
   dimension.

6. **Update MIOpen integration** (`conv_hipconv.cpp`) to handle the 3D applicability check
   (5D tensor shape, `Z == 3`, etc.) and invoke the 3D launch path.

7. **Test and tune:** verify correctness against a reference implementation, then tune
   `waves_c64`, `waves_q4`, and `n_fold` for the target problem sizes.

---

### 2.11 Open Questions and Trade-offs

| Question | Options |
|----------|---------|
| Should weights be 100% in registers for 3D? | Z×Y×X = 27 × `fp16x4_t` entries = 216 bytes per thread — still fits in VGPR budget for the 4c variant. |
| Double-buffer over depth? | The depth loop is slower to iterate than height (each step loads a full Hi×Wi plane slice). A second level of double buffering over depth would reduce stalls but increase LDS usage. |
| Tile size over Do? | Starting with Do-tile = 1 (one output depth per block) is simplest; could later tile Do to amortise weight loads across multiple depth outputs. |
| 8c/16c/32c variants for 3D? | Follow the same GT-transform approach; the depth dimension is transparent to the MFMA operand layout since it is handled by the outer loop. |
| Padding for depth? | Use the same buffer-descriptor out-of-bounds-zeroing trick: set `input_voffset = input_bytes` for out-of-bounds depth/height/width positions. |
