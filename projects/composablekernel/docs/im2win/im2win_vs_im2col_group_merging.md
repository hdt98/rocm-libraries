# Im2win vs Im2col: Group Merging Asymmetry

## Why im2col group merging gains ~2× TFLOPs but im2win group merging does not

---

## The core asymmetry: which dimension holds K×Gm

Both algorithms merge Gm groups to get an effective dimension of `K×Gm = 4×32 = 128`.
The question is **which GEMM axis it lands on** and what that implies for tile shapes.

### Im2col NHWGC + Gm=32

```
M      = N×Ho×Wo×Gm  = large (spatial, many M-tiles)
N_gemm = K×Gm = 128   ← K×Gm fills the N dimension
K_gemm = C×Y×X = 36
```

With 4 warps (256 threads), the natural split is **4 N-warps × 1 M-warp**:

```
N_Tile = N_Warp × N_WT = 4 × 32 = 128   ← exactly K×Gm = 128  (100% N utilisation ✓)
M_Tile = M_Warp × M_WT = 1 × 32 = 32    (spatial, fully packed)
```

K×Gm falls into N_gemm, matched by 4 N-warps → **N_Tile=128 = K×Gm → zero N waste**.

---

### Im2win NHWGC + Gm=32

```
M      = K×Gm = 128         ← K×Gm fills the M dimension
N_gemm = N×Ho×Wo×Gm = 40.96M  (spatial, very large)
K_gemm = C×Y×X = 36
```

To fill M_Tile=128 with 32×32 MFMA tiles you need **4 M-warps**.
With a fixed 4-warp budget (256 threads):

```
M_Warp = 4  →  M_Tile = 4 × 32 = 128   ✓  (fills K×Gm)
N_Warp = 1  →  N_Tile = 1 × 32 = 32    ← forced small
```

K×Gm lands in M, consuming all 4 M-warps, leaving only 1 N-warp →
**N_Tile=32 is small**, and the large spatial N_gemm dimension is covered
by many small tiles.

---

## The MFMA budget constraint — visualised

```
Budget: 4 warps × 64 threads = 256 threads

Im2col NHWGC + Gm=32:                 Im2win NHWGC + Gm=32:
  K×Gm=128 → N_Tile = 128               K×Gm=128 → M_Tile = 128
  spatial  → M_Tile = 32                spatial  → N_Tile = 32

  4 N-warps × 1 M-warp                  1 N-warp × 4 M-warps
  ┌──────────────────────┐               ┌──────┐
  │   N_Tile = 128       │               │ N=32 │
  │   = K×Gm            │  M=128        │      │
  │   100% N utilised ✓  │  = K×Gm      │      │
  │                      │  100% M ✓    │      │
  └──────────────────────┘               └──────┘
```

---

## Consequences

| | Im2col + Gm=32 | Im2win + Gm=32 |
|---|---|---|
| Filled dimension | N_Tile=128 = K×Gm ✓ | M_Tile=128 = K×Gm ✓ |
| Other dimension tile | M_Tile=32 (spatial) | **N_Tile=32 (spatial)** |
| N utilisation | **100%** (vs 25% without merge) | N_Tile=32 for 40.96M spatial → many tiny tiles |
| Grid size | ~320K blocks | ~1.28M blocks **(4× more!)** |
| K_gemm iterations/block | 1 (K_gemm=36 < K_Tile=64) | 1 — same |
| Pipeline | **ComputeV3** ✓ | Memory pipeline (ComputeV3 blocked for small M) |
| Valid outputs/block | M_Tile × K = 32 × 4 = 128 | N_Tile × K = 32 × 4 = 128 — same! |
| Performance gain | **~4× MFMA utilisation** | Marginal or negative |

---

## XOR diagonal waste — the same in both

Both algorithms use the XOR-diagonal trick, and both waste the same fraction:

```
Valid fraction = 1/Gm = 1/32 ≈ 3%
```

The waste is symmetric — this is NOT the cause of the difference.

---

## The intrinsic reason

**Im2col** group merging makes `N_gemm = K×Gm` the expanded dimension.
This maps perfectly onto **N-warps** in the warp layout:

- 4 N-warps → N_Tile = 128 = K×Gm → **100% N utilisation**
- ComputeV3 pipeline handles large-M × medium-N efficiently
- Before merge: 4/16 = 25% N utilisation; after merge: 128/128 = 100% → **4× improvement**

**Im2win** group merging makes `M = K×Gm` the expanded dimension.
This maps onto **M-warps**, consuming the full warp budget:

- 4 M-warps needed → N_Warp = 1 → **N_Tile = 32 only**
- N_Tile=32 is too small for the huge spatial N_gemm = 40.96M → 1.28M blocks
- Memory pipeline required (ComputeV3 cannot handle M_Tile=4 before merge)
- Net result: overhead exceeds the MFMA utilisation gain

**In one sentence:**
The warp layout that fills `K×Gm` in N_gemm (im2col) leaves room for a wide
N_Tile and works with ComputeV3, while the warp layout that fills `K×Gm` in M
(im2win) forces a narrow N_Tile and requires the slower Memory pipeline.

---

## Im2col vs Im2win: are they fundamentally different?

No — for all practical purposes:

1. **Same physical memory accesses**: both access `I[n, c, ho·Sy+fy·Dy-LPH, wo·Sx+fx·Dx-LPW]`
   in the same pattern. `MakeIm2winDescriptor` (the explicit I' step) is just a descriptor
   decomposition, not a different memory layout.

2. **Same weight access**: weight accessed as `W[k, c, fy, fx]` identically.

3. **The only real difference is the GEMM shape** (M and N swapped).
   This affects which MFMA instruction can be used for small K:
   - Im2col: K=4 → N=4, needs N_Tile=16 min → 75% N waste
   - Im2win: K=4 → M=4, can use 4×64×16 MFMA → 0% M waste
   - *But*: the 4×64×16 MFMA requires the Memory pipeline, partially offsetting the gain.

4. **Group merging helps im2col more** because the im2col GEMM shape is
   naturally suited to the K×Gm expansion (lands in N_gemm, matched by N-warps).
   Im2win's expansion lands in M, starving the N dimension of warps.

---

## Practical conclusions

| Problem | Best approach | Why |
|---|---|---|
| Small K (K=4), small C, large G | Im2col NHWGC + group merging | K×Gm fills N_Tile, ComputeV3, 4× gain |
| Small K (K=4), no group merging | Im2win GNCHW or NHWGC | 4×64×16 MFMA, but Memory pipeline |
| Large K (K=2376), large C=256 | Either (equivalent) | Standard large GEMM, shape doesn't matter |
| Small K + group merging via im2win | Not beneficial | MFMA budget mismatch, tiny N_Tile |
