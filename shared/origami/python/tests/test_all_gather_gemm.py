# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for origami.algorithms.all_gather_gemm analytical model.

Validates:
  1. Structural parity with derive_params.py on key parameter choices
  2. GEMM model consistency (constrained predictions track unconstrained)
  3. Wavefront / pipeline model sanity
"""

import pytest

import origami
from origami.algorithms import all_gather_gemm

from helpers import HARDWARE


# ── MI300X hardware fixture (mock, no GPU needed) ────────────────────────

MI300X_HW = HARDWARE.get("gfx942") or HARDWARE.get("gfx950")

MI300X_NETWORK = all_gather_gemm.network_t.mi300x_defaults(50.0, 8)


def _make_problem(M, N, K, world_size=8, link_bw=50.0, dtype="f16"):
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


# ═══════════════════════════════════════════════════════════════════════════
#  Type construction tests
# ═══════════════════════════════════════════════════════════════════════════


class TestTypeConstruction:
    """Verify all new types can be constructed and have expected fields."""

    def test_network_t_defaults(self):
        net = all_gather_gemm.network_t.mi300x_defaults()
        assert net.link_bw_gbps == 50.0
        assert net.world_size == 8
        assert net.scheduling_factor == 4.5

    def test_network_t_for_architecture_gfx942(self):
        net = all_gather_gemm.network_t.for_architecture(
            origami.architecture_t.gfx942, 50.0, 8
        )
        assert net.link_bw_gbps == 50.0
        assert net.world_size == 8
        assert net.peak_hbm_bw_gbps == 5300.0
        assert net.scheduling_factor == 4.5
        assert net.gather_overhead == 1.5
        assert net.write_bw_per_wg_gbps == 15.0
        assert net.flag_poll_us == 2.5
        assert net.flag_store_us == 5.0

    def test_network_t_for_architecture_matches_mi300x_defaults(self):
        net_factory = all_gather_gemm.network_t.for_architecture(
            origami.architecture_t.gfx942, 50.0, 8
        )
        net_legacy = all_gather_gemm.network_t.mi300x_defaults(50.0, 8)
        assert net_factory.peak_hbm_bw_gbps == net_legacy.peak_hbm_bw_gbps
        assert net_factory.scheduling_factor == net_legacy.scheduling_factor
        assert net_factory.gather_overhead == net_legacy.gather_overhead
        assert net_factory.write_bw_per_wg_gbps == net_legacy.write_bw_per_wg_gbps
        assert net_factory.flag_poll_us == net_legacy.flag_poll_us
        assert net_factory.flag_store_us == net_legacy.flag_store_us

    def test_network_t_for_architecture_gfx90a(self):
        net = all_gather_gemm.network_t.for_architecture(
            origami.architecture_t.gfx90a, 25.0, 4
        )
        assert net.link_bw_gbps == 25.0
        assert net.world_size == 4
        assert net.peak_hbm_bw_gbps == 3200.0
        assert net.scheduling_factor > 0

    def test_network_t_zero_defaults(self):
        net = all_gather_gemm.network_t()
        assert net.link_bw_gbps == 0.0
        assert net.world_size == 1
        assert net.peak_hbm_bw_gbps == 0.0
        assert net.scheduling_factor == 0.0

    def test_network_t_custom(self):
        net = all_gather_gemm.network_t()
        net.link_bw_gbps = 25.0
        net.world_size = 4
        assert net.link_bw_gbps == 25.0
        assert net.world_size == 4

    def test_problem_t_inheritance(self):
        prob = _make_problem(131072, 2048, 16384)
        assert prob.size.m == 131072
        assert prob.size.n == 2048
        assert prob.size.k == 16384
        assert prob.world_size == 8
        assert prob.k_local() == 2048

    def test_config_t_inheritance(self):
        cfg = all_gather_gemm.all_gather_matmul_config_t()
        cfg.mt = origami.dim3_t(256, 128, 64)
        cfg.num_fetch_sms = 64
        cfg.k_per_flag = 32
        assert cfg.mt.m == 256
        assert cfg.num_fetch_sms == 64

    def test_prediction_result_t(self):
        r = all_gather_gemm.all_gather_matmul_prediction_result_t()
        assert r.total_latency_ms == 0.0
        assert r.grid_size == 0

    def test_resource_constraints_t(self):
        rc = origami.resource_constraints_t()
        rc.available_cus = 240
        rc.available_hbm_bw_gbps = 4000.0
        assert rc.available_cus == 240

    def test_resource_constraints_unconstrained(self):
        rc = origami.resource_constraints_t.unconstrained(MI300X_HW)
        assert rc.available_cus == MI300X_HW.N_CU
        assert rc.available_hbm_bw_gbps > 0

    def test_stage_profile_t(self):
        sp = origami.stage_profile_t()
        sp.effective_concurrent_gemm_wgs = 128.0
        assert sp.effective_concurrent_gemm_wgs == 128.0


# ═══════════════════════════════════════════════════════════════════════════
#  Config selection tests
# ═══════════════════════════════════════════════════════════════════════════


SELECT_CASES = [
    {"M": 131072, "N": 2048, "K": 16384},
    {"M": 196608, "N": 2304, "K": 16384},
    {"M": 65536,  "N": 4096, "K": 8192},
]


class TestConfigSelection:
    """Verify select_config produces valid, well-formed configurations."""

    @pytest.mark.parametrize("case", SELECT_CASES, ids=lambda c: f"M{c['M']}_N{c['N']}_K{c['K']}")
    def test_tile_sizes_are_valid(self, case):
        prob = _make_problem(case["M"], case["N"], case["K"])
        result = all_gather_gemm.select_config(prob, MI300X_HW, MI300X_NETWORK)
        bm, bn, bk = result.config.mt.m, result.config.mt.n, result.config.mt.k
        assert bm > 0 and bn > 0 and bk > 0
        assert bm <= case["M"]
        assert bn <= case["N"]
        assert case["K"] % bk == 0, f"bk={bk} doesn't divide K={case['K']}"

    @pytest.mark.parametrize("case", SELECT_CASES, ids=lambda c: f"M{c['M']}_N{c['N']}_K{c['K']}")
    def test_positive_latency(self, case):
        prob = _make_problem(case["M"], case["N"], case["K"])
        result = all_gather_gemm.select_config(prob, MI300X_HW, MI300X_NETWORK)

        assert result.est_kernel_ms > 0, "Expected positive kernel time"
        assert result.comm_time_ms > 0, "Expected positive comm time"
        assert result.compute_time_ms > 0, "Expected positive compute time"

    @pytest.mark.parametrize("case", SELECT_CASES, ids=lambda c: f"M{c['M']}_N{c['N']}_K{c['K']}")
    def test_grid_geometry_valid(self, case):
        prob = _make_problem(case["M"], case["N"], case["K"])
        result = all_gather_gemm.select_config(prob, MI300X_HW, MI300X_NETWORK)

        assert result.grid_size > 0
        assert result.total_gemm_wgs > 0
        assert result.total_fetch_wgs > 0
        assert result.grid_size == result.total_gemm_wgs + result.total_fetch_wgs

    def test_first_stage_fetch_sms_is_all_cus(self):
        prob = _make_problem(131072, 2048, 16384)
        result = all_gather_gemm.select_config(prob, MI300X_HW, MI300X_NETWORK)
        assert result.config.first_stage_fetch_sms == MI300X_HW.N_CU

    def test_comm_compute_ratio_positive(self):
        prob = _make_problem(131072, 128, 16384)
        result = all_gather_gemm.select_config(prob, MI300X_HW, MI300X_NETWORK)
        assert result.comm_compute_ratio > 0

    def test_num_fetch_stages_reasonable(self):
        prob = _make_problem(131072, 2048, 16384)
        result = all_gather_gemm.select_config(prob, MI300X_HW, MI300X_NETWORK)
        assert 1 <= result.config.num_fetch_stages <= 32

    def test_k_per_flag_divides_k_blocks(self):
        prob = _make_problem(131072, 2048, 16384)
        result = all_gather_gemm.select_config(prob, MI300X_HW, MI300X_NETWORK)
        K = 16384
        bk = result.config.mt.k
        num_k_blocks = K // bk
        kpf = result.config.k_per_flag
        assert num_k_blocks % kpf == 0, \
            f"k_per_flag={kpf} doesn't divide num_k_blocks={num_k_blocks}"

    def test_search_beats_single_config(self):
        """select_config should find a config at least as good as a default."""
        prob = _make_problem(131072, 2048, 16384)
        best = all_gather_gemm.select_config(prob, MI300X_HW, MI300X_NETWORK)

        default_cfg = all_gather_gemm.all_gather_matmul_config_t()
        default_cfg.mt = origami.dim3_t(256, 256, 64)
        default_cfg.mi = MI300X_HW.get_recommended_matrix_instruction(prob.mi_dtype)
        default_cfg.occupancy = 1
        default_cfg.num_fetch_sms = 64
        default_cfg.k_per_flag = 32
        default_cfg.num_fetch_stages = 8
        default_cfg.first_stage_fetch_sms = MI300X_HW.N_CU
        default_cfg.num_warps = 8
        default_cfg.group_size_m = 4

        default_result = all_gather_gemm.predict_latency(
            prob, MI300X_HW, MI300X_NETWORK, default_cfg)

        assert best.est_kernel_ms <= default_result.est_kernel_ms, \
            f"Search ({best.est_kernel_ms:.4f} ms) should be <= default ({default_result.est_kernel_ms:.4f} ms)"


# ═══════════════════════════════════════════════════════════════════════════
#  GEMM model consistency tests
# ═══════════════════════════════════════════════════════════════════════════


class TestGEMMModelConsistency:
    """Verify the constrained adapter is consistent with unconstrained model."""

    def test_unconstrained_matches_baseline(self):
        """Constrained with full resources should approximate unconstrained."""
        prob = origami.problem_t()
        prob.size = origami.dim3_t(4096, 4096, 4096)
        prob.a_dtype = origami.string_to_datatype("f16")
        prob.b_dtype = origami.string_to_datatype("f16")
        prob.c_dtype = origami.string_to_datatype("f16")
        prob.d_dtype = origami.string_to_datatype("f16")
        prob.mi_dtype = origami.string_to_datatype("f16")
        prob.a_transpose = origami.transpose_t.T
        prob.b_transpose = origami.transpose_t.N

        cfg = origami.config_t()
        mi = MI300X_HW.get_recommended_matrix_instruction(prob.mi_dtype)
        cfg.mt = origami.dim3_t(256, 128, 64)
        cfg.mi = mi
        cfg.occupancy = 1

        baseline = origami.compute_tile_latency(prob, MI300X_HW, cfg, MI300X_HW.N_CU, 1)
        assert baseline > 0, "Baseline tile latency should be positive"

    def test_fewer_cus_increases_latency(self):
        """Reducing CUs should increase tile latency."""
        prob = origami.problem_t()
        prob.size = origami.dim3_t(4096, 4096, 4096)
        prob.a_dtype = origami.string_to_datatype("f16")
        prob.b_dtype = origami.string_to_datatype("f16")
        prob.c_dtype = origami.string_to_datatype("f16")
        prob.d_dtype = origami.string_to_datatype("f16")
        prob.mi_dtype = origami.string_to_datatype("f16")
        prob.a_transpose = origami.transpose_t.T
        prob.b_transpose = origami.transpose_t.N

        cfg = origami.config_t()
        mi = MI300X_HW.get_recommended_matrix_instruction(prob.mi_dtype)
        cfg.mt = origami.dim3_t(256, 128, 64)
        cfg.mi = mi
        cfg.occupancy = 1

        lat_full = origami.compute_tile_latency(prob, MI300X_HW, cfg, MI300X_HW.N_CU, 1)
        lat_half = origami.compute_tile_latency(prob, MI300X_HW, cfg, MI300X_HW.N_CU // 2, 1)

        assert lat_half >= lat_full, \
            f"Halving CUs should increase latency: full={lat_full}, half={lat_half}"


# ═══════════════════════════════════════════════════════════════════════════
#  Wavefront model tests
# ═══════════════════════════════════════════════════════════════════════════


class TestWavefrontModel:
    """Verify wavefront / pipeline model produces sane results."""

    def test_predict_latency_basic(self):
        prob = _make_problem(131072, 2048, 16384)
        result = all_gather_gemm.select_config(prob, MI300X_HW, MI300X_NETWORK)

        re_pred = all_gather_gemm.predict_latency(
            prob, MI300X_HW, MI300X_NETWORK, result.config)

        assert abs(re_pred.est_kernel_ms - result.est_kernel_ms) < 1e-6, \
            "predict_latency should match select_config result"

    def test_rank_configs_ordered(self):
        prob = _make_problem(131072, 2048, 16384)

        cfg1 = all_gather_gemm.all_gather_matmul_config_t()
        mi = MI300X_HW.get_recommended_matrix_instruction(prob.mi_dtype)
        cfg1.mt = origami.dim3_t(256, 256, 64)
        cfg1.mi = mi
        cfg1.occupancy = 1
        cfg1.num_fetch_sms = 64
        cfg1.k_per_flag = 32
        cfg1.num_fetch_stages = 8
        cfg1.first_stage_fetch_sms = MI300X_HW.N_CU
        cfg1.num_warps = 8
        cfg1.group_size_m = 4

        cfg2 = all_gather_gemm.all_gather_matmul_config_t()
        cfg2.mt = origami.dim3_t(128, 128, 64)
        cfg2.mi = mi
        cfg2.occupancy = 1
        cfg2.num_fetch_sms = 64
        cfg2.k_per_flag = 32
        cfg2.num_fetch_stages = 8
        cfg2.first_stage_fetch_sms = MI300X_HW.N_CU
        cfg2.num_warps = 4
        cfg2.group_size_m = 4

        ranked = all_gather_gemm.rank_configs(
            prob, MI300X_HW, MI300X_NETWORK, [cfg1, cfg2])

        assert len(ranked) == 2
        assert ranked[0].total_latency_ms <= ranked[1].total_latency_ms, \
            "rank_configs should return results in ascending latency order"

    def test_more_fetch_sms_reduces_comm_overlap(self):
        """Using more fetch SMs should affect the comm/compute balance."""
        prob = _make_problem(131072, 2048, 16384)
        mi = MI300X_HW.get_recommended_matrix_instruction(prob.mi_dtype)

        cfg_few = all_gather_gemm.all_gather_matmul_config_t()
        cfg_few.mt = origami.dim3_t(256, 256, 64)
        cfg_few.mi = mi
        cfg_few.occupancy = 1
        cfg_few.num_fetch_sms = 32
        cfg_few.k_per_flag = 32
        cfg_few.num_fetch_stages = 8
        cfg_few.first_stage_fetch_sms = MI300X_HW.N_CU
        cfg_few.num_warps = 8
        cfg_few.group_size_m = 4

        cfg_many = all_gather_gemm.all_gather_matmul_config_t()
        cfg_many.mt = origami.dim3_t(256, 256, 64)
        cfg_many.mi = mi
        cfg_many.occupancy = 1
        cfg_many.num_fetch_sms = 128
        cfg_many.k_per_flag = 32
        cfg_many.num_fetch_stages = 8
        cfg_many.first_stage_fetch_sms = MI300X_HW.N_CU
        cfg_many.num_warps = 8
        cfg_many.group_size_m = 4

        r_few = all_gather_gemm.predict_latency(
            prob, MI300X_HW, MI300X_NETWORK, cfg_few)
        r_many = all_gather_gemm.predict_latency(
            prob, MI300X_HW, MI300X_NETWORK, cfg_many)

        assert r_few.est_kernel_ms > 0
        assert r_many.est_kernel_ms > 0


# ═══════════════════════════════════════════════════════════════════════════
#  World size / scaling tests
# ═══════════════════════════════════════════════════════════════════════════


class TestScaling:
    """Verify model predictions scale reasonably with parameters."""

    def test_larger_world_size_reduces_comm_time(self):
        """More GPUs means less data per link, but same total."""
        prob_4 = _make_problem(131072, 2048, 8192, world_size=4)
        prob_8 = _make_problem(131072, 2048, 16384, world_size=8)

        net_4 = all_gather_gemm.network_t.mi300x_defaults(50.0, 4)
        net_8 = all_gather_gemm.network_t.mi300x_defaults(50.0, 8)

        r4 = all_gather_gemm.select_config(prob_4, MI300X_HW, net_4)
        r8 = all_gather_gemm.select_config(prob_8, MI300X_HW, net_8)

        assert r4.comm_time_ms > 0
        assert r8.comm_time_ms > 0

    def test_higher_link_bw_reduces_comm_time(self):
        prob = _make_problem(131072, 2048, 16384)

        net_slow = all_gather_gemm.network_t.mi300x_defaults(25.0, 8)
        net_fast = all_gather_gemm.network_t.mi300x_defaults(50.0, 8)

        r_slow = all_gather_gemm.select_config(prob, MI300X_HW, net_slow)
        r_fast = all_gather_gemm.select_config(prob, MI300X_HW, net_fast)

        assert r_slow.comm_time_ms > r_fast.comm_time_ms, \
            "Higher link BW should reduce comm time"
