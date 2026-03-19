#!/usr/bin/env python3
"""Print A, B, C macro tiles for 256x256x256 mxfp4 GEMM with 16x16x128 MFMA.

Shows which lane reads each K-chunk for rocRoller's LR pattern.
rocRoller LR: lane (SIMDIndex=s, laneInSIMD=l) reads K-chunk s, M-row l.
"""

import polars as pl

# MFMA: 16x16x128 fp4
mfma_m, mfma_n, mfma_k = 16, 16, 128  # fp4 elements
k_chunk = 32  # fp4 per ds_read_b128 (16 bytes)
chunks_per_mfma = mfma_k // k_chunk  # 4

# Macro tile
macro_m, macro_n, macro_k = 256, 256, 256  # fp4 elements for K

# K-chunks in the macro tile
n_k_chunks = macro_k // k_chunk  # 8
n_mfma_m = macro_m // mfma_m     # 16
n_mfma_n = macro_n // mfma_n     # 16

# Wave config: 2x2
# Wave A-partition: wv_id & 1 -> waves 0,2 get M0-M7; waves 1,3 get M8-M15
# Each wave's LR: SIMDIndex (0-3) selects K-chunk, laneInSIMD (0-15) selects M-row

def lane_range(simd):
    return f"L{simd*16}-{simd*16+15}"

# A macro tile at K-chunk granularity (16 M-tiles x 8 K-chunks)
# For each (M-tile, K-chunk): which SIMD (= lane range) reads it
# rocRoller LR: K-chunk = SIMDIndex for first MFMA, SIMDIndex+4 for second MFMA
a_rows = []
for m in range(n_mfma_m):
    wave = "w0,2" if m < 8 else "w1,3"
    row = {"M": f"M{m} ({m*mfma_m}-{(m+1)*mfma_m-1})"}
    for kc in range(n_k_chunks):
        simd = kc % chunks_per_mfma  # 0-3
        mfma_idx = kc // chunks_per_mfma  # 0 or 1
        row[f"K{kc}"] = f"{wave} {lane_range(simd)}"
    a_rows.append(row)
a_df = pl.DataFrame(a_rows)

# B macro tile at K-chunk granularity (8 K-chunks x 16 N-tiles)
# Wave B-partition: wv_id >> 1 -> waves 0,1 get N0-N7; waves 2,3 get N8-N15
b_rows = []
for kc in range(n_k_chunks):
    simd = kc % chunks_per_mfma
    row = {"K": f"K{kc} ({kc*k_chunk}-{(kc+1)*k_chunk-1})"}
    for n in range(n_mfma_n):
        wave = "w0,1" if n < 8 else "w2,3"
        row[f"N{n}"] = f"{wave} {lane_range(simd)}"
    b_rows.append(row)
b_df = pl.DataFrame(b_rows)

# C macro tile at MFMA granularity (16 M x 16 N)
# Each wave computes its 128x128 quadrant
# C tile with MFMA iteration numbers
# Loop order from gemm.cpp mainloop:
#   4 quadrants: (a0<4,b0<4), (a0>=4,b0<4), (a0<4,b0>=4), (a0>=4,b0>=4)
#   Within each: for b0 in range(4): for a0 in range(4): for km in range(2): emit_mfma
# a0 = M-subtile (0-7 per wave), b0 = N-subtile (0-7 per wave)
# subtile_size_0 = 16 M-rows

def mfma_iter(a0, b0):
    """Return (wave, iter_k0, iter_k1) for subtile (a0, b0) within a wave."""
    a_half = a0 // 4  # 0 or 1
    b_half = b0 // 4  # 0 or 1
    a_local = a0 % 4
    b_local = b0 % 4
    quadrant = a_half * 2 + b_half  # 0,1,2,3 but order is (0,0),(1,0),(0,1),(1,1)
    # Quadrant order: Q0=(a<4,b<4), Q1=(a>=4,b<4), Q2=(a<4,b>=4), Q3=(a>=4,b>=4)
    q_base = [0, 32, 64, 96][a_half + b_half * 2]  # note: b_half*2 because Q2 is (a<4,b>=4)
    # Wait, let me re-derive from the code order:
    # Q1: a0=[0..3], b0=[0..3]  -> base 0
    # Q2: a0=[4..7], b0=[0..3]  -> base 32
    # Q3: a0=[0..3], b0=[4..7]  -> base 64
    # Q4: a0=[4..7], b0=[4..7]  -> base 96
    q_base = (a_half + b_half * 2) * 32  # a_half=0,b_half=0->0; a_half=1,b_half=0->32; etc.
    offset = b_local * 8 + a_local * 2
    return q_base + offset, q_base + offset + 1

c_rows = []
for m_row in range(macro_m):
    row = {"M-row": str(m_row)}
    # Which wave and subtile?
    if m_row < 128:
        wave_m = "w0"  # or w2 for N>=128
        a0 = m_row // 16  # 0-7 subtile within wave's 128-row range
    else:
        wave_m = "w1"
        a0 = (m_row - 128) // 16
    for n in range(n_mfma_n):
        if n < 8:
            w = wave_m.replace("w0", "w0").replace("w1", "w1")
            b0 = n
        else:
            w = wave_m.replace("w0", "w2").replace("w1", "w3")
            b0 = n - 8
        i0, i1 = mfma_iter(a0, b0)
        row[f"N{n}"] = f"{w} i{i0},{i1}"
    c_rows.append(row)
c_df = pl.DataFrame(c_rows)

pl.Config.set_tbl_cols(20)
pl.Config.set_tbl_rows(-1)  # show all rows (we truncate manually)
pl.Config.set_tbl_width_chars(300)

def truncate(df: pl.DataFrame, head: int = 20, tail: int = 3) -> pl.DataFrame:
    """Show first `head` rows, ellipsis row, then last `tail` rows."""
    if len(df) <= head + tail:
        return df
    ellipsis_row = {col: "..." for col in df.columns}
    return pl.concat([
        df.head(head),
        pl.DataFrame(ellipsis_row),
        df.tail(tail),
    ])

print("=" * 80)
print("LDS READ (ds_read_b128) -- which lanes read which data")
print("=" * 80)

# Rebuild at per-M-row granularity
# LR: lane (SIMDIndex=s, laneInSIMD=l) reads K-chunk s from M-row l
# SIMDIndex = lane // 16, laneInSIMD = lane % 16
# For K-chunks 0-3: first MFMA in K. K-chunks 4-7: second MFMA.
# Wave partition for A: wv_id & 1 -> waves 0,2 = partition 0, waves 1,3 = partition 1
# Each partition handles 128 M-rows (0-127 or 128-255)

a_rows = []
for m_row in range(macro_m):
    wave_part = "w0,2" if m_row < 128 else "w1,3"
    row = {"M-row": str(m_row)}
    for kc in range(n_k_chunks):
        simd = kc % chunks_per_mfma  # 0-3
        # laneInSIMD = m_row % 16 (which of the 16 lanes within the SIMD)
        lane_in_simd = m_row % 16
        lane_start = simd * 16  # SIMD base lane
        lane = lane_start + lane_in_simd
        row[f"K{kc}"] = f"{wave_part} L{lane}"
    a_rows.append(row)
a_df = pl.DataFrame(a_rows)

b_rows = []
for n_row in range(macro_n):
    wave_part = "w0,1" if n_row < 128 else "w2,3"
    row = {"N-row": str(n_row)}
    for kc in range(n_k_chunks):
        simd = kc % chunks_per_mfma
        lane_in_simd = n_row % 16
        lane = simd * 16 + lane_in_simd
        row[f"K{kc}"] = f"{wave_part} L{lane}"
    b_rows.append(row)
b_df = pl.DataFrame(b_rows)

print("\nLR A tile (M x K): 256 M-rows x 8 K-chunks")
print("Cell = which wave partition + lane reads that (M-row, K-chunk) via ds_read_b128")
print(truncate(a_df))
print()

print("LR B tile (N x K): 256 N-rows x 8 K-chunks")
print(truncate(b_df))
print()

print("C macro tile (M x N): 256 x 256 = 16 x 16 MFMA output tiles")
print("Cell = which wave computes that output tile")
print(truncate(c_df))
print()

# =====================================================================
# GR: Global Read to LDS -- rocRoller (no half-wave split, sequential cols)
# =====================================================================
print("\n" + "=" * 80)
print("GLOBAL READ TO LDS (buffer_load_dwordx4 ... lds) -- rocRoller")
print("=" * 80)

# rocRoller: col = lane % 8, row = lane // 8 (8 contiguous rows per wave)
# No half-wave split. Instruction i, wave w loads rows (i*32 + w*8) .. (i*32 + w*8 + 7)

print("\n--- GR Global A tile -- rocRoller (256 M-rows x 8 K-chunks) ---")
print("Cell = wave.lane that loads from global memory")
print("No half-wave split: 8 contiguous rows per wave, all lanes in one block")

rr_gr_rows = []
for m_row in range(macro_m):
    instr = m_row // 32
    wave = (m_row % 32) // 8
    row_in_wave = m_row % 8
    row = {"M-row": str(m_row)}
    for c in range(8):
        lane = row_in_wave * 8 + c
        row[f"K{c}"] = f"w{wave}.L{lane}"
    rr_gr_rows.append(row)

rr_gr_df = pl.DataFrame(rr_gr_rows)
print(truncate(rr_gr_df))

# =====================================================================
# GR: Global Read to LDS -- No-swizzle (half-wave split, sequential cols)
# =====================================================================
print("\n" + "=" * 80)
print("GLOBAL READ TO LDS (buffer_load_dwordx4 ... lds) -- No-swizzle")
print("=" * 80)

# No-swizzle GR: same half-wave split as swizzled, but col = lane % 8 (sequential)
# newserial = (lane % 32) + wave * 32
# col = newserial % 8 = lane % 8
# row_in_half = (lane % 32) // 8  (0-3)
# half_wave = lane // 32  (0 or 1)
# All waves have identical col pattern (no rotation)

def noswizzle_gr_mapping():
    """Build (m_row, k_chunk) -> wave.lane mapping for no-swizzle kernel."""
    mapping = {}
    for instr in range(8):
        for wave in range(4):
            for lane in range(64):
                col = lane % 8  # sequential, no swizzle
                row_in_half = (lane % 32) // 8  # 0-3
                half_wave = lane // 32  # 0 or 1
                m_row_in_subtile = wave * 4 + row_in_half
                m_row_base = instr * 16
                if half_wave == 0:
                    m_row = m_row_base + m_row_in_subtile
                else:
                    m_row = m_row_base + m_row_in_subtile + macro_m // 2
                if m_row < macro_m:
                    mapping[(m_row, col)] = f"w{wave}.L{lane}"
    return mapping

ns_gr = noswizzle_gr_mapping()

print("\n--- GR Global A tile -- No-swizzle (256 M-rows x 8 K-chunks) ---")
print("Cell = wave.lane that loads from global memory")
print("Half-wave split: lanes 0-31 -> top half, lanes 32-63 -> bottom half (macM/2)")

ns_gr_rows = []
for m_row in range(macro_m):
    row = {"M-row": str(m_row)}
    for c in range(8):
        row[f"K{c}"] = ns_gr.get((m_row, c), "?")
    ns_gr_rows.append(row)

ns_gr_df = pl.DataFrame(ns_gr_rows)
print(truncate(ns_gr_df))

# =====================================================================
# GR: Swizzled kernel
# =====================================================================
print("\n" + "=" * 80)
print("GLOBAL READ TO LDS (buffer_load_dwordx4 ... lds) -- Swizzled")
print("=" * 80)

# Build inverse mapping: for each (M-row, K-chunk), which wave.lane loads it?
# Swizzled GR:
#   newserial = (lane % 32) + wave * 32
#   col_base = newserial % 8
#   lds_row_id = newserial >> 4
#   perp_group = lds_row_id & 1
#   col_xor = col_base ^ 1 if perp_group == 0 else col_base
#   rotation = (8 - (lds_row_id // 2) * 2) % 8
#   col = (col_xor + rotation) % 8
#   row_in_half = (lane % 32) // 8  (0-3)
#   half_wave = lane // 32  (0 or 1)
#
# Per subtile instruction i:
#   M-row = i * 16 + wave_row_offset + row_in_half  (half-wave 0)
#   M-row = i * 16 + wave_row_offset + row_in_half + macM/2  (half-wave 1)
#   where wave_row_offset comes from the 4 rows that this wave handles

block_size = 8
perp_group_size = 2  # lds_row_bank_size / depthu_k_bytes = 256 / 128

def swizzled_col(lane, wave):
    """Compute the K-chunk (col) fetched by this lane in this wave."""
    ns = (lane % 32) + wave * 32
    col_base = ns % block_size
    lds_row_id = ns >> 4
    perp = lds_row_id & 1
    col_xor = col_base ^ 1 if perp == 0 else col_base
    rot = (block_size - (lds_row_id // 2) * 2) % block_size
    return (col_xor + rot) % block_size

# Build mapping: iterate over all waves, lanes, subtile instructions
# For one subtile instruction: 4 waves x 64 lanes = 256 loads
# Covers 16 M-rows (half-wave 0) + 16 M-rows (half-wave 1) = 32 M-rows
# 8 subtile instructions cover 256 M-rows

sw_gr_a = {}  # (m_row, k_chunk) -> "w{wave}.L{lane}"

for instr in range(8):  # 8 subtile instructions
    for wave in range(4):
        for lane in range(64):
            ns = (lane % 32) + wave * 32
            row_in_half = (lane % 32) // 8  # 0-3
            half_wave = lane // 32  # 0 or 1
            col = swizzled_col(lane, wave)

            # M-row for this subtile instruction
            # Each instruction covers 16 rows for half-wave 0 and 16 for half-wave 1
            # Within the 16 rows: 4 waves x 4 rows = 16
            # Wave w handles rows: (w * 4 + row_in_half) within the 16
            m_row_base = instr * 16
            m_row_in_subtile = wave * 4 + row_in_half
            if half_wave == 0:
                m_row = m_row_base + m_row_in_subtile
            else:
                m_row = m_row_base + m_row_in_subtile + macro_m // 2  # + 128

            if m_row < macro_m:
                key = (m_row, col)
                sw_gr_a[key] = f"w{wave}.L{lane}"

print("\n--- GR Global A tile -- Swizzled (256 M-rows x 8 K-chunks) ---")
print("Cell = wave.lane that loads from global memory")
print("Note: half-wave split puts lanes 0-31 in top half, 32-63 in bottom half (macM/2)")

sw_gr_a_rows = []
for m_row in range(macro_m):
    row = {"M-row": str(m_row)}
    for c in range(8):
        row[f"K{c}"] = sw_gr_a.get((m_row, c), "?")
    sw_gr_a_rows.append(row)

sw_gr_a_df = pl.DataFrame(sw_gr_a_rows)
print(truncate(sw_gr_a_df))
