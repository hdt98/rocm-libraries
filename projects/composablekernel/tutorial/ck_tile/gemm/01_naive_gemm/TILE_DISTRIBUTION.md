# Tile Distribution: Mapping Threads to Data

## Overview

**Tile Distribution** describes how each thread in a thread block maps to elements of a block tile.
It defines the hierarchical pattern of data distribution across threads, warps, and thread blocks,
enabling the compiler to compute register counts, load addresses, and store addresses entirely
at compile time.

## The Problem

Given a block tile of size `MPerBlock × KPerBlock` (e.g., 256×32), we need to determine:
- Which threads load which elements.
- How the threads are organized into warps.
- The number of times each warp repeats its pattern to cover the full tile.
- The number of elements each thread can load in a single vector instruction.

---

## Bottom-Up Construction Approach

The derivation starts from hardware constraints (vector load width, warp size, block size)
and works upward toward the tile dimensions.

### Step 1: Determine K Dimension Layout

**Start with the innermost dimension (K) for vectorized loads:**

```cpp
constexpr index_t K1 = 16 / sizeof(ADataType);  // Elements per 128-bit load
constexpr index_t K0 = kKPerBlock / K1;          // Groups of K1 needed to cover K
```

**Example (fp16, kKPerBlock=32):**
- `K1 = 16 / 2 = 8` → Each thread loads 8 fp16 elements per `global_load_dwordx4` instruction
- `K0 = 32 / 8 = 4` → We need 4 groups along K to cover the full 32-element K dimension

Each thread loads K1=8 consecutive elements. K0=4 threads within a warp cover all 32 K slots.

**Visual:**
```
K dimension (32 elements):
Thread (K0=0): [K 0-7]   Thread (K0=1): [K 8-15]   Thread (K0=2): [K 16-23]   Thread (K0=3): [K 24-31]
     K1=8 elements              K1=8 elements              K1=8 elements              K1=8 elements
├───────────────────────────────────────────────────────────────────────────────────────────────────┤
                                     K0=4 groups covering all of kKPerBlock=32
```

---

### Step 2: Determine M Dimension Layout

**Partition the M dimension hierarchically across threads and warps:**

#### Level 1: Threads per Warp in M (M2)

```cpp
constexpr index_t M2 = get_warp_size() / K0;
```

- Warp size = 64 threads
- K dimension already uses `K0 = 4` threads per warp row
- `M2 = 64 / 4 = 16` → Each warp covers 16 M rows simultaneously

**Visual (single warp, 64 threads arranged as M2×K0 = 16×4):**
```
         K dimension (K0=4 threads per row)
      ┌─────┬─────┬─────┬─────┐
   0  │ T0  │ T1  │ T2  │ T3  │
   1  │ T4  │ T5  │ T6  │ T7  │
   2  │ T8  │ T9  │ T10 │ T11 │
M  3  │ T12 │ T13 │ T14 │ T15 │  ← M2=16 rows
   ...│ ... │ ... │ ... │ ... │
  15  │ T60 │ T61 │ T62 │ T63 │
      └─────┴─────┴─────┴─────┘
     One warp = 64 threads (M2×K0 = 16×4)
```

#### Level 2: Warps per Block (M1)

```cpp
constexpr index_t M1 = kBlockSize / get_warp_size();
```

- `kBlockSize = 256` threads per block
- `M1 = 256 / 64 = 4` → 4 warps per block, each covering M2=16 rows

**Visual (4 warps in a block, stacked along M):**
```
       Warp 0 (rows 0-15)
       Warp 1 (rows 16-31)
       Warp 2 (rows 32-47)
       Warp 3 (rows 48-63)
       ↑
    M1 = 4 warps cover 64 rows total (M1 × M2 = 4 × 16 = 64)
```

#### Level 3: Repetitions (M0)

```cpp
constexpr index_t M0 = kMPerBlock / (M2 * M1);
```

- `kMPerBlock = 256` rows to cover in total
- `M2 * M1 = 16 * 4 = 64` rows covered by all warps in one pass
- `M0 = 256 / 64 = 4` → Each warp must repeat its 16-row pattern 4 times

**Visual (complete block, M0=4 iterations):**
```
┌──────────────┐
│ Iteration 0  │ ← Warp 0: rows 0-15,    Warp 1: rows 16-31,   Warp 2: rows 32-47,   Warp 3: rows 48-63
│ (rows 0-63)  │
├──────────────┤
│ Iteration 1  │ ← Warp 0: rows 64-79,   Warp 1: rows 80-95,   Warp 2: rows 96-111,  Warp 3: rows 112-127
│(rows 64-127) │
├──────────────┤
│ Iteration 2  │ ← Warp 0: rows 128-143, Warp 1: rows 144-159, Warp 2: rows 160-175, Warp 3: rows 176-191
│(rows 128-191)│
├──────────────┤
│ Iteration 3  │ ← Warp 0: rows 192-207, Warp 1: rows 208-223, Warp 2: rows 224-239, Warp 3: rows 240-255
│(rows 192-255)│
└──────────────┘
  M0 = 4 iterations
```

---

## The Tile Distribution Encoding

With all dimensions derived, we construct the distribution:

```cpp
tile_distribution_encoding<
    sequence<1>,                                      // [1] Replication
    tuple<sequence<M0, M1, M2>, sequence<K0, K1>>,   // [2] Hierarchy
    tuple<sequence<1>, sequence<1, 2>>,               // [3] Thread ID → hierarchy
    tuple<sequence<1>, sequence<2, 0>>,               // [3] Thread ID → hierarchy
    sequence<1, 2>,                                   // [4] Per-thread elements (Y dimensions)
    sequence<0, 1>                                    // [4] Per-thread elements (Y dimensions)
>
```

### [1] Replication: `sequence<1>`

Controls how many thread blocks replicate the same access pattern:
- `1` = No replication — each block is independent, accessing different tile regions.
- A value > 1 would mean multiple blocks share the same access pattern (useful in some reduction kernels, not needed here).

---

### [2] Hierarchy: The Multi-Level Structure

```cpp
tuple<sequence<M0, M1, M2>, sequence<K0, K1>>
     └───────────────────┘  └──────────────┘
       M dimension levels      K dimension levels
```

**Concrete values (fp16, kMPerBlock=256, kKPerBlock=32, kBlockSize=256):**
- M hierarchy: `sequence<4, 4, 16>` = (M0=4 repetitions, M1=4 warps, M2=16 threads/warp in M)
- K hierarchy: `sequence<4, 8>` = (K0=4 threads in K, K1=8 elements/thread)

**Sanity check:**
- M: M0 × M1 × M2 = 4 × 4 × 16 = 256 = kMPerBlock ✓
- K: K0 × K1 = 4 × 8 = 32 = kKPerBlock ✓
- Threads: M1 × (M2 × K0) = 4 × (16 × 4) = 256 = kBlockSize ✓
- Elements per thread: M0 × K1 = 4 × 8 = 32 (four 128-bit vector loads per thread) ✓

---

### [3] Thread ID → Hierarchy Mapping

```cpp
tuple<sequence<1>, sequence<1, 2>>   // which hierarchy levels are indexed by thread id (row)
tuple<sequence<1>, sequence<2, 0>>   // which hierarchy indices map to which thread-id components (col)
```

These two paired rows encode how the thread ID (composed of warp ID and lane ID) maps into
the M and K hierarchy levels.

**Read column-by-column:**

**Column 1 (M dimension):**
- Row 1: `sequence<1>` — M1 (warp index) is indexed by thread-id component 1 (= warp ID)
- Row 2: `sequence<1>` — thread-id component 1 addresses hierarchy level 1 (M1)

**Column 2 (K dimension):**
- Row 1: `sequence<1, 2>` — K0 and K1 are addressed by thread-id components 1 and 2
- Row 2: `sequence<2, 0>` — component 2 (lane / M2 index) addresses K0; component 0 addresses K1

In short, the warp ID selects the M1 strip, and the lane index within the warp is split
between M2 (which M row within the warp) and K0 (which K group within the row).

---

### [4] Per-Thread Element Sequences (Yield)

```cpp
sequence<1, 2>   // which hierarchy levels produce per-thread elements (Y0, Y1)
sequence<0, 1>   // which indices within those levels map to Y0, Y1
```

These sequences describe which dimensions of the hierarchy belong to each thread's "private"
portion — i.e., the elements the thread owns in its registers.

**Reading column-by-column:**

- Column 1: Y0 is at hierarchy[1][0] = M0 (repetitions along M)
- Column 2: Y1 is at hierarchy[2][1] = K1 (elements per vector load)

Result: each thread owns a 2D block indexed by (M0, K1) = (4, 8) = 32 elements in VGPRs.
The compiler allocates exactly 32 fp16 registers per thread for one A or B block tile.

---

## Complete Example: Thread 25 in Warp 0

Let's trace where **Thread 25** in **Warp 0** reads data from the A block tile.

### Step 1: Decompose thread 25 within the warp (lane ID = 25)

The warp's 64 threads are arranged as M2×K0 = 16×4:
```
M2 row index = 25 / K0 = 25 / 4 = 6
K0 col index = 25 % K0 = 25 % 4 = 1
```

### Step 2: M Position

```
M0 iteration: 0   (first iteration, M0 index = 0)
M1 warp:      0   (Warp 0, M1 index = 0)
M2 thread:    6   (6th row within the warp)
→ M position = M0_idx * (M1 * M2) + M1_idx * M2 + M2_idx
             = 0 * 64 + 0 * 16 + 6 = 6
```

### Step 3: K Position

```
K0 thread: 1   (column group 1, covering K1 elements starting at K0*K1 = 1*8 = 8)
K1 elements: 8  (loads elements 8 through 15 in one 128-bit vector)
→ K position = K0_idx * K1 = 1 * 8 = 8  →  loads A[6, 8:15]
```

**Result:** Thread 25 in Warp 0 loads **row 6, columns 8–15** (8 fp16 elements) in its first
M0 iteration. In M0 iterations 1–3 it loads rows 70, 134, 198 respectively (each `+64` from
the previous).

---

## Why This Matters

### 1. Memory Coalescing

Within a single wavefront instruction, all 64 threads in a warp issue loads simultaneously.
Consider K0=4 threads in a row, each loading K1=8 consecutive fp16 elements:
- 4 × 8 = 32 fp16 = 64 bytes = 1 cache line
- All 32 elements are contiguous in memory (K is the contiguous dimension)
- The 16 M2 threads each load from a different cache line (different M rows)
- No gaps, no wasted transactions — fully coalesced

### 2. Vectorization

Each thread issues 4 × 128-bit `global_load_dwordx4` instructions (one per M0 iteration),
each loading 8 fp16 values. Maximum vector width, minimum instruction count.

### 3. Warp Efficiency

All 64 threads in a warp are utilized: M2 × K0 = 16 × 4 = 64.

### 4. Register Allocation

The compiler knows each thread holds exactly M0 × K1 = 4 × 8 = 32 elements. Register
allocation is determined at compile time with no guesswork.

---

## Summary Table

| Parameter | Value | Meaning |
|-----------|-------|---------|
| **K1** | 8 | fp16 elements per 128-bit vector load |
| **K0** | 4 | Thread groups in K per warp row |
| **M2** | 16 | Threads covering M per warp per pass |
| **M1** | 4 | Warps per block |
| **M0** | 4 | Warp repetitions over M |
| **Total Threads** | 256 | M1 × (M2 × K0) = 4 × 64 |
| **Total Elements** | 8192 | kMPerBlock × kKPerBlock = 256 × 32 |
| **Elements/Thread** | 32 | M0 × K1 = 4 × 8 (four 128-bit loads) |

---

## Visualization: Complete Thread Block

```
Block Tile: 256 rows × 32 cols (kMPerBlock × kKPerBlock)

      K dimension (32 elements, K0=4 thread groups × K1=8 elems each)
      ├──────────────────────────────────────────────────────────────┤
  0   ┌─────────────────────────────────────────────────────────────┐  ┐
  16  │  Warp 0 (M2=16 rows per warp)                               │  │
  32  │  Warp 1                                                      │  │ Iteration 0 (M0=0)
  48  │  Warp 2                                                      │  │ rows 0–63
  64  │  Warp 3                                                      │  ┘
  80  ├─────────────────────────────────────────────────────────────┤  ┐
  96  │  Warp 0                                                      │  │
 112  │  Warp 1                                                      │  │ Iteration 1 (M0=1)
 128  │  Warp 2                                                      │  │ rows 64–127
 144  │  Warp 3                                                      │  ┘
 160  ├─────────────────────────────────────────────────────────────┤  ┐
 176  │  Warp 0                                                      │  │
 192  │  Warp 1                                                      │  │ Iteration 2 (M0=2)
 208  │  Warp 2                                                      │  │ rows 128–191
 224  │  Warp 3                                                      │  ┘
 240  ├─────────────────────────────────────────────────────────────┤  ┐
 256  │  Warp 0                                                      │  │
      │  Warp 1                                                      │  │ Iteration 3 (M0=3)
      │  Warp 2                                                      │  │ rows 192–255
      │  Warp 3                                                      │  ┘
      └─────────────────────────────────────────────────────────────┘

Each warp covers 16 rows × 32 cols = 512 elements per iteration.
Each iteration covers 64 rows × 32 cols = 2048 elements.
Total: 4 iterations × 2048 = 8192 elements = 256 × 32 ✓
```

---

## Key Takeaways

1. **Bottom-up construction**: Start from vector width (K1=8), build up through K0, M2, M1, M0
   to satisfy tile dimensions, thread count, and alignment simultaneously.
2. **All parameters determined at compile time**: The distribution encoding is a type, not a
   runtime object. The compiler fully unrolls loads, stores, and index computations.
3. **Same distribution backs DRAM load and LDS store**: Thread T loads element (m,k) from DRAM
   and stores the same element to LDS — the physical layout in LDS matches what MFMA expects.
4. **Coalescing and vectorization are co-designed**: K0 × K1 = 32 bytes per warp row → 1 cache
   line per instruction, zero wasted bandwidth.
