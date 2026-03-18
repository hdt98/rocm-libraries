# Im2win Group Merging: 32×32×8 MFMA + 512 Threads

## Status: WarpGemmMfmaF16F16F32M32N32K8 exists ✓

The `Dispatcher<half_t, half_t, float, 32, 32, 8>` specialisation is defined in
`warp_gemm_dispatcher.hpp` (line 45), so M_Warp_Tile=32, N_Warp_Tile=32, K_Warp_Tile=8
is a valid combination for fp16 and bf16.

---

## Problem recap

```
K=4, Gm=32  →  M = K×Gm = 128  (fills M_Tile perfectly)
N×Ho×Wo = 32×200×200 = 1,280,000 spatial positions
K_gemm = C×Y×X = 36
```

---

## Why the current 256-thread design is limited

With 256 threads (4 warps) and M_Tile=128=K×Gm requiring 4 M-warps:

```
M_Warp=4, N_Warp=1  →  M_Tile=128 ✓   N_Tile=32 (only 1 N-warp left)
```

N_Tile=32 covers only 32 spatial positions per block. This forces 1.28M blocks.

---

## New design: 32×32×8 MFMA + 512 threads (8 warps)

With 512 threads (8 warps) and 4 M-warps × 2 N-warps:

```
M_Warp=4, N_Warp=2  →  M_Tile = 4×32 = 128 ✓   N_Tile = 2×32 = 64
BlockSize = 4×2×64 = 512 threads
```

### Direct comparison

| Config           | Threads | M_Tile | N_Tile | Warps    | Blocks  | Valid/block | MFMA pipeline |
|-----------------|---------|--------|--------|----------|---------|-------------|---------------|
| Merge_Gm32_K16  |   256   |  128   |   32   |  4M×1N   | 1,280K  |    128      | Memory        |
| **NEW_K8_512**  | **512** |**128** | **64** |**4M×2N** | **640K**| **256**     | TBD           |

Key improvements:
- **2× larger N_Tile** (64 vs 32) → covers 2 spatial positions per group per block
- **2× fewer blocks** (640K vs 1.28M) → less dispatch overhead
- **2× more valid outputs per block** (256 vs 128)
- **K_WT=8** → 8 MFMA instructions per K_Tile step instead of 4 → better instruction-level parallelism

---

## Comparison with im2col (the "baseline double-performance" case)

Im2col NHWGC + Gm=32 with 512 threads (best case):

```
M_Warp=2, N_Warp=4  →  M_Tile=64, N_Tile=128=K×Gm (100% N utilisation)
Valid/block = M_Tile × K = 64 × 4 = 256
```

| Config           | Threads | M_Tile | N_Tile   | N fill     | Valid/block |
|-----------------|---------|--------|----------|------------|-------------|
| Im2col + Gm=32  |   512   |   64   | **128**  | **100%** ✓ |   256       |
| Im2win + Gm=32  |   512   | **128**|   64     |  5%        |   256       |

**Valid outputs per block are now equal (256)!**

However, the N_Tile fill rates still differ:
- Im2col: N_Tile=128 = K×Gm → **100% N fill** (no padding waste)
- Im2win: N_Tile=64 for 1.28M spatial → **~5% N fill** (Gm-XOR valid fraction = 1/32 ≈ 3%)

The **remaining gap** after this improvement:

1. Im2col's N_Tile=128 fills perfectly (K×Gm), no N-tile padding loss.
   Im2win's N_Tile=64 still wastes ~97% of MFMA compute (XOR diagonal: 1/Gm valid).

2. Im2col can use ComputeV3 pipeline; im2win may still need Memory pipeline
   (ComputeV3 constraints with 512-thread blocks need checking).

---

## Can the remaining gap be closed?

The XOR diagonal waste (1/Gm = 3%) is fundamental and the same for both algorithms.
What differs is which dimension fills the tile:

```
Im2col:  K×Gm → N_Tile  →  tile fills perfectly (100%)    → waste only from XOR
Im2win:  K×Gm → M_Tile  →  tile fills perfectly (100%)    → but N_Tile is only 5% full
```

The XOR-diagonal waste affects WRITES (C descriptor), not READS (A and B loads).
So both algorithms do the same amount of wasted MFMA computation (1/Gm = 3%).

The remaining advantage of im2col is that its N_Tile=K×Gm=128 is exactly filled by
the K-channel dimension, meaning the epilogue writes K×M_Tile = 128×64 = 8192 useful
elements per block vs im2win's K×N_Tile = 4×64 = 256 useful writes.

Wait — that's actually the same:

```
Im2col valid writes = M_Tile × K = 64 × 4 = 256   (after XOR, only K channels from 1/Gm of M)
Im2win valid writes = K × N_Tile_valid = 4 × 64 = 256  (K channels from 64 spatial)
```

At 512 threads, both produce 256 valid outputs per block. The remaining performance
difference comes entirely from **pipeline quality** (ComputeV3 vs Memory) and
**K_gemm efficiency** (K_gemm=36 is small relative to K_Tile=64 → ~1 K-loop iteration).

---

## Proposed new config for im2win group merging

```cpp
// Merge_Gm32_M128N64K64_512T (32×32×8 MFMA, 512 threads)
template <typename PrecType>
struct Im2winConfig_Merge_Gm32_M128N64K64_512T : public Im2winConvConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 64;
    static constexpr ck_tile::index_t K_Tile = 64;   // 8 MFMA steps of K_WT=8

    static constexpr ck_tile::index_t M_Warp = 4;   // 4×32 = 128 = K×Gm ✓
    static constexpr ck_tile::index_t N_Warp = 2;   // 2×32 = 64
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 32;  // 32×32×8 MFMA
    static constexpr ck_tile::index_t N_Warp_Tile = 32;
    static constexpr ck_tile::index_t K_Warp_Tile = 8;   // ← new (was 16)

    // BlockSize = 4×2×64 = 512 threads
    static constexpr ck_tile::GemmPipeline Pipeline = ck_tile::GemmPipeline::MEMORY;
    static constexpr ck_tile::GemmPipelineScheduler Scheduler =
        ck_tile::GemmPipelineScheduler::Intrawave;
    static constexpr ck_tile::index_t NumGroupsToMerge = 32;
};
```

### Expected improvement over Merge_Gm32_M128N32K64 (256 threads)

| Metric             | 256 threads (current) | 512 threads (proposed) | Improvement |
|-------------------|----------------------|------------------------|-------------|
| N_Tile            |         32           |        **64**          |     2×      |
| Blocks            |     1,280,000        |      **640,000**       |    0.5×     |
| Valid/block       |        128           |       **256**          |     2×      |
| K_WT              |         16           |         **8**          | 2× MFMA ILP |
| Expected speedup  |        —             |       ~1.5–2×          |      ?      |

---

## LDS usage concern (512 threads)

With K_Tile=64, K_WT=8 (8 MFMA steps per K_Tile):

```
A LDS tile: M_Tile × K_Tile × sizeof(fp16) = 128 × 64 × 2 = 16 KB
B LDS tile: N_Tile × K_Tile × sizeof(fp16) =  64 × 64 × 2 =  8 KB
Total (single buffer): 24 KB
Total (double buffer): 48 KB   ← 512 threads × 2 doubles → may limit occupancy
```

gfx950 LDS per CU: 64 KB
- Single buffer: 24 KB → ~2.6 blocks/CU → reasonable
- Double buffer: 48 KB → ~1.3 blocks/CU → tight occupancy

**Recommendation**: use single buffer for the 512-thread config to maintain
reasonable occupancy (kBlockPerCu=1 with 24 KB LDS allows ~2 blocks/CU).

---

## Summary

| Approach | Threads | N_Tile | Valid/blk | Remaining gap vs im2col |
|---------|---------|--------|-----------|------------------------|
| Im2win 256T current | 256 | 32 | 128 | 2× gap |
| **Im2win 512T new** | **512** | **64** | **256** | **Pipeline quality only** |
| Im2col 512T best | 512 | 128 | 256 | — (baseline) |

The 32×32×8 MFMA + 512 threads eliminates the valid-output-count gap with im2col.
The remaining performance difference, if any, comes from:
1. Pipeline (ComputeV3 for im2col vs Memory for im2win)
2. N_Tile fill rate (100% for im2col, ~5% for im2win due to XOR diagonal)
   → But this does NOT cause wasted READS — A and B are fully read regardless
   → It only causes wasted WRITES (XOR suppresses them), same for both
