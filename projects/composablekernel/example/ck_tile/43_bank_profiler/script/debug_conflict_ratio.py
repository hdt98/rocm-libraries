#!/usr/bin/env python3
"""
Debug script: Analyze the GEMM bank conflict ratio with correct XOR swizzle
LDS layout and gfx942 phase structure.

Key corrections from original analysis:
1. Uses UniversalGemmPipelineAgBgCrPolicy (XOR swizzle), NOT DefaultPolicy (+1 padding)
2. Actual kernel uses GemmPipelineAgBgCrCompV4 with ds_read_b128, NOT ds_read_b64
3. Phase structure from hardware probing on gfx942
"""

import os


NUM_BANKS = 32
BANK_WIDTH = 4  # bytes per bank
def xor_swizzle_byte_offset(m, k, MPerBlock, KPerBlock, KPack=8, MLdsLayer=None):
    """
    Compute the LDS byte offset for logical index (m, k) in the
    UniversalGemmPipelineAgBgCrPolicy XOR swizzle layout (RowMajor A, fp16).
    """
    if MLdsLayer is None:
        # MLdsLayer = max(1, NBanks * dwords_per_128b / KPerBlock / sizeof(fp16))
        MLdsLayer = max(1, NUM_BANKS * 4 // KPerBlock // 2)

    M_outer = m // MLdsLayer
    M_layer = m % MLdsLayer
    K_outer = k // KPack
    K_inner = k % KPack

    K_chunk_permuted = M_layer * (KPerBlock // KPack) + K_outer

    xor_range = KPerBlock // KPack * MLdsLayer
    K_chunk_base = K_chunk_permuted ^ (M_outer % xor_range)

    element_offset = K_chunk_base * KPack + M_outer * (KPerBlock * MLdsLayer) + K_inner
    return element_offset * 2  # fp16 = 2 bytes


def get_bank(byte_offset):
    return (byte_offset // BANK_WIDTH) % NUM_BANKS


def get_banks_for_access(byte_offset, access_bytes):
    start_bank = get_bank(byte_offset)
    num_banks = access_bytes // BANK_WIDTH
    return [(start_bank + i) % NUM_BANKS for i in range(num_banks)]


# gfx942 phase structures (from hardware probing)
# ds_read_b64: 4 phases x 16 threads
GFX942_PHASES_READ_B64 = [
    list(range(0, 4)) + list(range(12, 16)) + list(range(20, 28)),
    list(range(4, 12)) + list(range(16, 20)) + list(range(28, 32)),
    list(range(32, 36)) + list(range(44, 48)) + list(range(52, 60)),
    list(range(36, 44)) + list(range(48, 52)) + list(range(60, 64)),
]

# ds_read_b128: phase structure TBD from probing.
# Using same grouping assumption as b64 for now (4 phases x 16 threads).
# The user should verify this.
GFX942_PHASES_READ_B128 = GFX942_PHASES_READ_B64


def analyze_a_read_conflicts(MPerBlock, KPerBlock, M_Warp_Tile, K_Warp_Tile,
                             read_bytes, phases, label=""):
    """
    Analyze bank conflicts for A tile LDS reads during MFMA.

    MFMA 32x32x8 fp16 thread mapping:
    - Thread t reads A[t % 32, (t // 32) * (read_bytes // 2) : ...]
    """
    KPack = 8
    MLdsLayer = max(1, NUM_BANKS * 4 // KPerBlock // 2)
    mfma_k = 8  # MFMA K dimension for fp16
    read_elems = read_bytes // 2  # fp16 elements per read

    print(f"\n{'='*70}")
    print(f"A Read Conflict Analysis: {label}")
    print(f"{'='*70}")
    print(f"Tile: {MPerBlock}x{KPerBlock}, M_Warp_Tile={M_Warp_Tile}, K_Warp_Tile={K_Warp_Tile}")
    print(f"LDS read: ds_read_b{read_bytes*8} ({read_bytes} bytes = {read_bytes//BANK_WIDTH} banks)")
    print(f"XOR swizzle: MLdsLayer={MLdsLayer}, KPack={KPack}")
    print(f"Phases: {len(phases)} x {len(phases[0])} threads")

    banks_per_read = read_bytes // BANK_WIDTH

    # For ds_read_b128: each thread reads 16 bytes = 8 fp16 = 4 banks
    # MFMA 32x32x8 fp16: thread t contributes 4 fp16 to A
    # With ds_read_b128: thread reads 8 fp16, covering 2 MFMA K-steps at once
    # Thread t accesses: m = t%32, k_start = some offset based on t//32

    # For ds_read_b128 with MFMA 32x32x8:
    # Threads 0-31: read A[t, 0:8] (one ds_read_b128 covers full k=8)
    # Threads 32-63: read A[t-32, 0:8] (SAME m indices, SAME k!)
    # Wait - that means threads 0 and 32 read the same (m, k). But they'd
    # get the same data, which is the MFMA operand duplication.
    # Actually for ds_read_b128: thread reads 128 bits = 4 floats or 8 fp16.
    # For MFMA 32x32x8 fp16, each thread needs 4 fp16 per MFMA call.
    # With K_Warp_Tile=16, there are 2 MFMA calls.
    # ds_read_b128 reads 8 fp16 = enough for 2 MFMA calls.
    # So 1 ds_read_b128 per thread for A, serving both MFMA iterations.

    # Thread mapping for the read:
    # Thread t reads from m = t%32, k = k_base (8 consecutive fp16 via ds_read_b128)
    # Threads 0-31 and 32-63 both read m = t%32, same k range
    # → threads t and t+32 access the SAME LDS address!

    total_conflicts_all = 0
    for k_base in range(0, K_Warp_Tile, mfma_k):
        print(f"\n--- K range [{k_base}, {k_base + mfma_k}) ---")

        for phase_idx, phase_threads in enumerate(phases):
            bank_to_threads = {}

            for t in phase_threads:
                m = t % 32
                # For ds_read_b128: reads 8 fp16 starting at (m, k_base)
                # For ds_read_b64: reads 4 fp16 starting at (m, k_base + (t//32)*4)
                if read_bytes == 16:  # ds_read_b128
                    k = k_base
                else:  # ds_read_b64
                    k = k_base + (t // 32) * (read_bytes // 2)

                byte_off = xor_swizzle_byte_offset(m, k, MPerBlock, KPerBlock)
                banks = get_banks_for_access(byte_off, read_bytes)
                bank_key = tuple(banks)

                if bank_key not in bank_to_threads:
                    bank_to_threads[bank_key] = []
                bank_to_threads[bank_key].append((t, m, k))

            conflicts = 0
            conflict_details = []
            for bank_set, threads in sorted(bank_to_threads.items()):
                if len(threads) > 1:
                    conflicts += len(threads) - 1
                    thread_list = ", ".join(f"t{t}(m={m})" for t, m, k in threads)
                    conflict_details.append(
                        f"  Banks {list(bank_set)}: {thread_list} -> {len(threads)}-way")

            if conflicts == 0:
                print(f"  Phase {phase_idx}: NO conflicts")
            else:
                print(f"  Phase {phase_idx}: {conflicts} extra cycles")
                for d in conflict_details[:4]:  # show first 4
                    print(d)
                if len(conflict_details) > 4:
                    print(f"  ... and {len(conflict_details)-4} more conflict groups")

            total_conflicts_all += conflicts

    print(f"\nTotal A read conflicts per wavefront instruction: {total_conflicts_all}")
    return total_conflicts_all


def compute_expected_ratio(MPerBlock, NPerBlock, KPerBlock,
                           M_Warp, N_Warp, K_Warp_Tile,
                           read_bytes, phases,
                           M, N, K):
    """Estimate expected conflict ratio given the analysis."""
    num_blocks = (M // MPerBlock) * (N // NPerBlock)
    num_warps = M_Warp * N_Warp
    num_k_loops = K // KPerBlock
    BlockSize = num_warps * 64

    mfma_k = 8
    mfma_iters = K_Warp_Tile // mfma_k

    print(f"\n{'='*70}")
    print(f"Expected Ratio Estimation")
    print(f"{'='*70}")
    print(f"Config: {MPerBlock}x{NPerBlock}x{KPerBlock}, {M_Warp}x{N_Warp} warps")
    print(f"Problem: {M}x{N}x{K}")
    print(f"Grid: {num_blocks} blocks, {num_warps} warps/block, {num_k_loops} K loops")

    # ds_read_b128 reads: 1 per thread for A (covers both MFMA iters), 1 for B
    # So per warp per K iteration: 1 A read + 1 B read = 2 reads (if ds_read_b128)
    # Actually it depends on K_Warp_Tile and read size...
    if read_bytes == 16:  # ds_read_b128 reads 8 fp16
        reads_a_per_warp = mfma_iters  # Each reads 8 fp16 for one MFMA K-step
        reads_b_per_warp = mfma_iters
    else:
        reads_a_per_warp = mfma_iters * 2  # ds_read_b64: 4 fp16, need 2 per MFMA
        reads_b_per_warp = mfma_iters * 2

    # From ISA: ds_write_b16 for stores
    a_elements = MPerBlock * KPerBlock
    b_elements = NPerBlock * KPerBlock
    total_write_elements = a_elements + b_elements
    writes_per_thread = total_write_elements // BlockSize  # fp16 elements per thread
    # ds_write_b16: 1 element per write instruction
    writes_per_warp = writes_per_thread  # each thread issues this many writes

    reads_per_warp = reads_a_per_warp + reads_b_per_warp
    total_lds_per_warp_per_k = reads_per_warp + writes_per_warp

    total_lds = total_lds_per_warp_per_k * num_warps * num_k_loops * num_blocks

    print(f"\nPer warp per K iteration:")
    print(f"  A reads:  {reads_a_per_warp} (ds_read_b{read_bytes*8})")
    print(f"  B reads:  {reads_b_per_warp} (ds_read_b{read_bytes*8})")
    print(f"  Writes:   {writes_per_warp} (ds_write_b16)")
    print(f"  Total:    {total_lds_per_warp_per_k}")
    print(f"\nEstimated total SQ_INSTS_LDS: {total_lds}")

    return total_lds


def main():
    print("=" * 70)
    print("GEMM Bank Conflict Debug Analysis")
    print("Real kernel: GemmPipelineAgBgCrCompV4, fp16, 256x256x32")
    print("Real LDS instructions: ds_read_b128 (reads), ds_write_b16 (writes)")
    print("=" * 70)

    # Measured values from rocprofv3
    measured_lds = 67584
    measured_conflicts = 131072
    measured_ratio = measured_conflicts / measured_lds
    print(f"\nMeasured (256x256x32 tile, 1024x1024x1024):")
    print(f"  SQ_INSTS_LDS        = {measured_lds:,}")
    print(f"  SQ_LDS_BANK_CONFLICT = {measured_conflicts:,}")
    print(f"  Ratio                = {measured_ratio:.4f}")
    print(f"  Grid: 16 blocks, 256 threads/block, 4 warps/block")

    import math
    if measured_conflicts == 2 ** round(math.log2(measured_conflicts)):
        print(f"  Note: {measured_conflicts:,} = 2^{round(math.log2(measured_conflicts))}")

    # Analyze with ds_read_b128 (the actual instruction)
    analyze_a_read_conflicts(
        MPerBlock=256, KPerBlock=32,
        M_Warp_Tile=32, K_Warp_Tile=16,
        read_bytes=16,  # ds_read_b128
        phases=GFX942_PHASES_READ_B128,
        label="ds_read_b128, 256x256x32"
    )

    # Also show ds_read_b64 for comparison
    analyze_a_read_conflicts(
        MPerBlock=256, KPerBlock=32,
        M_Warp_Tile=32, K_Warp_Tile=16,
        read_bytes=8,  # ds_read_b64
        phases=GFX942_PHASES_READ_B64,
        label="ds_read_b64 (for comparison), 256x256x32"
    )

    # Estimate expected SQ_INSTS_LDS
    compute_expected_ratio(
        MPerBlock=256, NPerBlock=256, KPerBlock=32,
        M_Warp=2, N_Warp=2, K_Warp_Tile=16,
        read_bytes=16,
        phases=GFX942_PHASES_READ_B128,
        M=1024, N=1024, K=1024
    )

    print(f"\n{'='*70}")
    print("KEY FINDINGS")
    print(f"{'='*70}")
    print(f"""
1. Actual GEMM kernel uses GemmPipelineAgBgCrCompV4 (NOT V1)
2. LDS reads use ds_read_b128 (16 bytes = 4 banks per thread)
3. LDS writes use ds_write_b16 (individual fp16 elements!)
4. With ds_read_b128, threads 0 and 32 both read m=0 at same k
   → they access the SAME LDS address → potential 2-way conflict
5. ds_write_b16 writes touch only 1 bank per thread per instruction
   → unlikely to cause bank conflicts (spread across all 32 banks)
6. The conflict ratio of {measured_ratio:.2f} likely comes primarily from
   ds_read_b128 bank conflicts in the XOR-swizzled LDS layout.
""")


if __name__ == "__main__":
    main()
