#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Example: using the Origami all-gather + GEMM analytical model.

Demonstrates three usage levels:
  1. High-level OrigamiAllGatherMatmulSelector (requires torch + GPU)
  2. Mid-level select_config via the C++ bindings (no GPU needed)
  3. Low-level predict_latency / rank_configs for manual exploration

Run:
    python all_gather_gemm_example.py
"""

import origami
from origami.algorithms import all_gather_gemm


# ═══════════════════════════════════════════════════════════════════════════
#  Helper: create hardware + problem without a real GPU
# ═══════════════════════════════════════════════════════════════════════════

def make_hardware(arch="gfx942"):
    """Create a hardware descriptor without a live device."""
    archs = {
        "gfx942": (origami.architecture_t.gfx942, 228, 64 * 1024, 24 * 1024 * 1024, 1700000),
        "gfx950": (origami.architecture_t.gfx950, 304, 64 * 1024, 32 * 1024 * 1024, 2100000),
    }
    return origami.get_hardware_for_arch(*archs[arch])


def make_problem(M, N, K, world_size=8, link_bw=50.0, dtype="f16"):
    """Create a fused all-gather GEMM problem descriptor."""
    prob = all_gather_gemm.all_gather_matmul_problem_t()
    prob.size = origami.dim3_t(M, N, K)
    dt = origami.string_to_datatype(dtype)
    prob.a_dtype = dt
    prob.b_dtype = dt
    prob.c_dtype = dt
    prob.d_dtype = dt
    prob.mi_dtype = dt
    prob.a_transpose = origami.transpose_t.T
    prob.b_transpose = origami.transpose_t.N
    prob.world_size = world_size
    prob.link_bw_gbps = link_bw
    return prob


def print_result(result, label=""):
    """Pretty-print a prediction result."""
    cfg = result.config
    if label:
        print(f"\n{'─' * 60}")
        print(f"  {label}")
        print(f"{'─' * 60}")

    print(f"  Tile sizes        : {cfg.mt.m}×{cfg.mt.n}×{cfg.mt.k}")
    print(f"  num_fetch_sms     : {cfg.num_fetch_sms}")
    print(f"  k_per_flag        : {cfg.k_per_flag}")
    print(f"  num_fetch_stages  : {cfg.num_fetch_stages}")
    print(f"  num_warps         : {cfg.num_warps}")
    print(f"  group_size_m      : {cfg.group_size_m}")
    print()
    print(f"  est_kernel_ms     : {result.est_kernel_ms:.4f}")
    print(f"  compute_time_ms   : {result.compute_time_ms:.4f}  (standalone GEMM)")
    print(f"  comm_time_ms      : {result.comm_time_ms:.4f}")
    print(f"  pipeline_ms       : {result.pipeline_ms:.4f}")
    print(f"  comm/compute ratio: {result.comm_compute_ratio:.3f}")
    print(f"  effective TFLOPS  : {result.roofline_tflops:.1f}")
    print(f"  grid_size         : {result.grid_size}  "
          f"({result.total_gemm_wgs} gemm + {result.total_fetch_wgs} fetch)")


# ═══════════════════════════════════════════════════════════════════════════
#  Example 1: select_config  (auto-select best parameters)
# ═══════════════════════════════════════════════════════════════════════════

def example_select_config():
    print("\n" + "═" * 60)
    print("  Example 1: select_config (automatic parameter selection)")
    print("═" * 60)

    hw      = make_hardware("gfx942")
    network = all_gather_gemm.network_t.for_architecture(hw.arch, 50.0, 8)
    prob    = make_problem(M=131072, N=2048, K=16384, world_size=8)

    result = all_gather_gemm.select_config(prob, hw, network)
    print_result(result, "M=131072  N=2048  K=16384  ws=8  fp16")


# ═══════════════════════════════════════════════════════════════════════════
#  Example 2: predict_latency  (evaluate a specific config)
# ═══════════════════════════════════════════════════════════════════════════

def example_predict_latency():
    print("\n" + "═" * 60)
    print("  Example 2: predict_latency (manually specified config)")
    print("═" * 60)

    hw      = make_hardware("gfx942")
    network = all_gather_gemm.network_t.for_architecture(hw.arch, 50.0, 8)
    prob    = make_problem(M=131072, N=2048, K=16384)

    mi = hw.get_recommended_matrix_instruction(origami.string_to_datatype("f16"))

    cfg = all_gather_gemm.all_gather_matmul_config_t()
    cfg.mt = origami.dim3_t(256, 256, 64)
    cfg.mi = mi
    cfg.occupancy          = 1
    cfg.num_fetch_sms      = 64
    cfg.k_per_flag         = 32
    cfg.num_fetch_stages   = 8
    cfg.first_stage_fetch_sms = hw.N_CU
    cfg.num_warps          = 8
    cfg.group_size_m       = 4

    result = all_gather_gemm.predict_latency(prob, hw, network, cfg)
    print_result(result, "Manual config: 256×256×64, 64 fetch SMs, 8 stages")


# ═══════════════════════════════════════════════════════════════════════════
#  Example 3: rank_configs  (compare multiple configs)
# ═══════════════════════════════════════════════════════════════════════════

def example_rank_configs():
    print("\n" + "═" * 60)
    print("  Example 3: rank_configs (compare tile sizes)")
    print("═" * 60)

    hw      = make_hardware("gfx942")
    network = all_gather_gemm.network_t.for_architecture(hw.arch, 50.0, 8)
    prob    = make_problem(M=131072, N=2048, K=16384)

    mi = hw.get_recommended_matrix_instruction(origami.string_to_datatype("f16"))

    configs = []
    for bm, bn in [(128, 128), (128, 256), (256, 128), (256, 256)]:
        cfg = all_gather_gemm.all_gather_matmul_config_t()
        cfg.mt = origami.dim3_t(bm, bn, 64)
        cfg.mi = mi
        cfg.occupancy          = 1
        cfg.num_fetch_sms      = 64
        cfg.k_per_flag         = 32
        cfg.num_fetch_stages   = 8
        cfg.first_stage_fetch_sms = hw.N_CU
        cfg.num_warps          = 8 if bm * bn >= 256 * 256 else 4
        cfg.group_size_m       = 4
        configs.append(cfg)

    ranked = all_gather_gemm.rank_configs(prob, hw, network, configs)

    print(f"\n  {'Rank':<6}{'Tile':>12}{'Kernel (ms)':>14}{'TFLOPS':>10}")
    print(f"  {'─' * 42}")
    for i, r in enumerate(ranked):
        tile = f"{r.config.mt.m}×{r.config.mt.n}×{r.config.mt.k}"
        print(f"  {i+1:<6}{tile:>12}{r.est_kernel_ms:>14.4f}{r.roofline_tflops:>10.1f}")


# ═══════════════════════════════════════════════════════════════════════════
#  Example 4: sweep world sizes / link bandwidths
# ═══════════════════════════════════════════════════════════════════════════

def example_sweep():
    print("\n" + "═" * 60)
    print("  Example 4: sweep world_size and link_bw")
    print("═" * 60)

    hw = make_hardware("gfx942")
    M, N = 131072, 2048

    print(f"\n  {'ws':<4}{'link_bw':>10}{'K':>8}{'kernel_ms':>12}"
          f"{'comm_ms':>10}{'compute_ms':>12}{'ratio':>8}")
    print(f"  {'─' * 64}")

    for ws in [2, 4, 8]:
        K = 2048 * ws
        for link_bw in [25.0, 50.0]:
            network = all_gather_gemm.network_t.for_architecture(hw.arch, link_bw, ws)
            prob    = make_problem(M, N, K, world_size=ws, link_bw=link_bw)
            result  = all_gather_gemm.select_config(prob, hw, network)

            print(f"  {ws:<4}{link_bw:>10.1f}{K:>8}"
                  f"{result.est_kernel_ms:>12.4f}"
                  f"{result.comm_time_ms:>10.4f}"
                  f"{result.compute_time_ms:>12.4f}"
                  f"{result.comm_compute_ratio:>8.3f}")


# ═══════════════════════════════════════════════════════════════════════════
#  Example 5: high-level selector (requires torch + GPU)
# ═══════════════════════════════════════════════════════════════════════════

def example_high_level_selector():
    print("\n" + "═" * 60)
    print("  Example 5: OrigamiAllGatherMatmulSelector (torch + GPU)")
    print("═" * 60)

    try:
        import torch
        if not torch.cuda.is_available():
            print("\n  [skipped — no GPU available]")
            return

        from origami import OrigamiAllGatherMatmulSelector

        sel = OrigamiAllGatherMatmulSelector(
            m=131072, n=2048, k=16384,
            a_dtype=torch.float16,
            b_dtype=torch.float16,
            out_dtype=torch.float16,
            device=torch.device("cuda:0"),
            world_size=8,
            link_bw=50.0,
        )

        print(f"\n  block_m           : {sel.block_m}")
        print(f"  block_n           : {sel.block_n}")
        print(f"  block_k           : {sel.block_k}")
        print(f"  num_fetch_sms     : {sel.num_fetch_sms}")
        print(f"  k_per_flag        : {sel.k_per_flag}")
        print(f"  num_fetch_stages  : {sel.num_fetch_stages}")
        print(f"  num_warps         : {sel.num_warps}")
        print(f"  est_kernel_ms     : {sel.est_kernel_ms:.4f}")
        print(f"  comm/compute ratio: {sel.comm_compute_ratio:.3f}")
        print(f"  effective TFLOPS  : {sel.roofline_tflops:.1f}")

    except ImportError:
        print("\n  [skipped — torch not installed]")


# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    example_select_config()
    example_predict_latency()
    example_rank_configs()
    example_sweep()
    example_high_level_selector()
    print()
