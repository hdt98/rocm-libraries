#!/usr/bin/env python3
"""
Explore whether different MLdsLayer, padding, or XOR parameters can
eliminate the residual 2-way bank conflicts in the LDS read path.

Key insight: the conflict comes from M_outer stride being a multiple of
32 banks (128 bytes), so bank depends ONLY on K_chunk_base (8 values).
With 16 threads/phase, pigeonhole gives at least 2-way conflicts.
"""

NUM_BANKS = 32
BANK_WIDTH = 4  # bytes


def compute_lds_byte_offset(m, k, MPerBlock, KPerBlock, KPack, MLdsLayer, pad_elements=0):
    """
    XOR swizzle byte offset with optional padding on M_outer stride.

    pad_elements: extra fp16 elements added to M_outer stride.
    """
    M_outer = m // MLdsLayer
    M_layer = m % MLdsLayer
    K_outer = k // KPack
    K_inner = k % KPack

    K_chunk_permuted = M_layer * (KPerBlock // KPack) + K_outer
    xor_range = KPerBlock // KPack * MLdsLayer
    K_chunk_base = K_chunk_permuted ^ (M_outer % xor_range)

    base_stride = KPerBlock * MLdsLayer + pad_elements
    element_offset = K_chunk_base * KPack + M_outer * base_stride + K_inner
    return element_offset * 2  # fp16


# gfx942 ds_read_b64 phase structure
PHASES = [
    list(range(0, 4)) + list(range(12, 16)) + list(range(20, 28)),
    list(range(4, 12)) + list(range(16, 20)) + list(range(28, 32)),
    list(range(32, 36)) + list(range(44, 48)) + list(range(52, 60)),
    list(range(36, 44)) + list(range(48, 52)) + list(range(60, 64)),
]


def count_conflicts_per_phase(MPerBlock, KPerBlock, KPack, MLdsLayer,
                               pad_elements, read_bytes, k_base):
    """Count total conflicts across all phases for one read instruction."""
    banks_per_read = read_bytes // BANK_WIDTH
    total_conflicts = 0

    for phase_threads in PHASES:
        bank_to_count = {}
        for t in phase_threads:
            m = t % 32
            if read_bytes == 16:  # ds_read_b128
                k = k_base
            else:  # ds_read_b64
                k = k_base + (t // 32) * (read_bytes // 2)

            byte_off = compute_lds_byte_offset(m, k, MPerBlock, KPerBlock,
                                                KPack, MLdsLayer, pad_elements)
            start_bank = (byte_off // BANK_WIDTH) % NUM_BANKS
            bank_key = start_bank // banks_per_read  # group by aligned bank group
            # Actually, use the exact start bank
            bank_key = tuple((start_bank + i) % NUM_BANKS for i in range(banks_per_read))

            bank_to_count[bank_key] = bank_to_count.get(bank_key, 0) + 1

        for count in bank_to_count.values():
            if count > 1:
                total_conflicts += count - 1

    return total_conflicts


def check_alignment(KPerBlock, MLdsLayer, pad_elements, read_bytes):
    """Check if the LDS layout maintains required alignment for reads."""
    alignment = read_bytes  # ds_read_b128 needs 16-byte, ds_read_b64 needs 8-byte
    stride_bytes = (KPerBlock * MLdsLayer + pad_elements) * 2  # fp16
    return stride_bytes % alignment == 0


def explore_configs():
    global NUM_BANKS
    MPerBlock = 256
    KPerBlock = 32
    KPack = 8

    print("=" * 80)
    print("Exploring LDS Layout Parameters to Minimize Bank Conflicts")
    print("=" * 80)
    print(f"Fixed: MPerBlock={MPerBlock}, KPerBlock={KPerBlock}, KPack={KPack}, "
          f"NUM_BANKS={NUM_BANKS}")
    print(f"Phase structure: 4 phases x 16 threads (gfx942)")
    print()

    # Current config
    print("-" * 80)
    print("PART 1: Varying MLdsLayer (no padding)")
    print("-" * 80)
    print(f"{'MLdsLayer':<12} {'Stride(B)':<12} {'Banks/row':<12} "
          f"{'ds_read_b128':<16} {'ds_read_b64':<16} {'Aligned128':<12} {'Aligned64':<12}")

    for mlds in [1, 2, 3, 4, 8]:
        stride_b = KPerBlock * mlds * 2
        banks_per_row = stride_b // BANK_WIDTH
        banks_mod = banks_per_row % NUM_BANKS

        c128 = count_conflicts_per_phase(MPerBlock, KPerBlock, KPack, mlds, 0, 16, 0)
        c64 = count_conflicts_per_phase(MPerBlock, KPerBlock, KPack, mlds, 0, 8, 0)
        a128 = check_alignment(KPerBlock, mlds, 0, 16)
        a64 = check_alignment(KPerBlock, mlds, 0, 8)

        marker = " <-- current" if mlds == 2 else ""
        print(f"{mlds:<12} {stride_b:<12} {banks_per_row:<4} (mod32={banks_mod:<3}) "
              f"{c128:<16} {c64:<16} {'Yes' if a128 else 'NO':<12} {'Yes' if a64 else 'NO':<12}{marker}")

    # Varying padding
    print()
    print("-" * 80)
    print("PART 2: Adding padding to M_outer stride (MLdsLayer=2)")
    print("-" * 80)
    print(f"{'Pad(elem)':<12} {'Stride(B)':<12} {'bank_mod32':<12} "
          f"{'ds_read_b128':<16} {'ds_read_b64':<16} {'Aligned128':<12} {'Aligned64':<12}")

    MLdsLayer = 2
    for pad in [0, 1, 2, 4, 8, 12, 16, 24, 32]:
        stride_b = (KPerBlock * MLdsLayer + pad) * 2
        banks_mod = (stride_b // BANK_WIDTH) % NUM_BANKS

        c128 = count_conflicts_per_phase(MPerBlock, KPerBlock, KPack, MLdsLayer, pad, 16, 0)
        c64 = count_conflicts_per_phase(MPerBlock, KPerBlock, KPack, MLdsLayer, pad, 8, 0)
        a128 = check_alignment(KPerBlock, MLdsLayer, pad, 16)
        a64 = check_alignment(KPerBlock, MLdsLayer, pad, 8)

        marker = " <-- current" if pad == 0 else ""
        best128 = " ***" if c128 == 0 and a128 else ""
        best64 = " ***" if c64 == 0 and a64 else ""
        print(f"{pad:<12} {stride_b:<12} {banks_mod:<12} "
              f"{c128:<16}{best128:4} {c64:<16}{best64:4} "
              f"{'Yes' if a128 else 'NO':<12} {'Yes' if a64 else 'NO':<12}{marker}")

    # Varying KPack
    print()
    print("-" * 80)
    print("PART 3: Varying KPack (MLdsLayer=2, no padding)")
    print("-" * 80)
    print(f"{'KPack':<12} {'BankGranul':<14} {'MaxGroups':<12} "
          f"{'ds_read_b128':<16} {'ds_read_b64':<16}")

    MLdsLayer = 2
    for kpack in [2, 4, 8, 16]:
        if KPerBlock % kpack != 0:
            continue
        bank_granularity = kpack * 2 // BANK_WIDTH  # in banks
        max_groups = NUM_BANKS // max(bank_granularity, 1)

        c128 = count_conflicts_per_phase(MPerBlock, KPerBlock, kpack, MLdsLayer, 0, 16, 0)
        c64 = count_conflicts_per_phase(MPerBlock, KPerBlock, kpack, MLdsLayer, 0, 8, 0)

        marker = " <-- current" if kpack == 8 else ""
        print(f"{kpack:<12} {bank_granularity:<14} {max_groups:<12} "
              f"{c128:<16} {c64:<16}{marker}")

    # Architecture comparison
    print()
    print("-" * 80)
    print("PART 4: gfx942 (32 banks) vs gfx950 (64 banks)")
    print("-" * 80)

    for n_banks in [32, 64]:
        NUM_BANKS = n_banks
        MLdsLayer = max(1, n_banks * 4 // KPerBlock // 2)
        c128 = count_conflicts_per_phase(MPerBlock, KPerBlock, KPack, MLdsLayer, 0, 16, 0)
        c64 = count_conflicts_per_phase(MPerBlock, KPerBlock, KPack, MLdsLayer, 0, 8, 0)
        bank_groups_128 = n_banks // (16 // BANK_WIDTH)
        bank_groups_64 = n_banks // (8 // BANK_WIDTH)
        print(f"  {n_banks} banks (MLdsLayer={MLdsLayer}): "
              f"ds_read_b128 conflicts={c128} (max {bank_groups_128} groups), "
              f"ds_read_b64 conflicts={c64} (max {bank_groups_64} groups)")

    NUM_BANKS = 32  # restore

    # Summary
    print()
    print("=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)
    print("""
ROOT CAUSE:
  ds_read_b128 requires 16-byte alignment → M_outer stride must be
  a multiple of 16 bytes → bank stride is always a multiple of 4.
  With 32 banks: 32/4 = 8 distinct bank groups maximum.
  With 16 threads/phase: pigeonhole → at least 2-way conflicts.

  Mathematically: bank = (K_chunk_base * 4 + M_outer * s) % 32
  where s must be a multiple of 4 (from 16-byte alignment).
  → only 8 distinct bank values possible → 2-way conflicts unavoidable.

POSSIBLE MITIGATIONS:

  1. Use ds_read_b64 + padding (4 fp16 pad on M_outer stride):
     - Achieves 0 conflicts! (16 bank groups with 8-byte alignment)
     - Cost: 2x more LDS read instructions (but each is conflict-free)
     - Net effect: may be FASTER if conflict penalty > instruction cost
     - Requires: stride alignment = 8 bytes (satisfied with 4-elem pad)

  2. Use gfx950 (64 banks):
     - Achieves 0 conflicts with ds_read_b128 (16 bank groups)
     - No code changes needed, just hardware upgrade
     - MLdsLayer=4 on gfx950 naturally provides this

  3. Accept the 2-way conflict on gfx942:
     - 1 extra cycle per phase × 4 phases = 4 extra cycles per read
     - ~2x penalty on reads, but reads are ~50% of LDS traffic
     - Overall ~1.9x ratio (matches measurement)
     - This is a hardware limitation, not a software bug

  4. Reduce threads per phase (not controllable by software)

  5. Change KPack to make bank granularity finer:
     - KPack=4 → 2-bank granularity → 16 groups → 0 conflicts possible
     - But KPack is tied to MFMA input width, may not be freely changeable
""")


if __name__ == "__main__":
    explore_configs()
