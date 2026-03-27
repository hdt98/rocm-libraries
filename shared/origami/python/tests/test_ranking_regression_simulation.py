# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Ranking Regression Tests for Origami — Simulation (Formocast) Mode

These tests verify that simulation mode config rankings remain stable across
code changes.  Rankings are compared against golden baseline files to detect
unintended changes from PRs.

Usage:
    # Run tests (compares against baseline)
    pytest test_ranking_regression_simulation.py -v

    # Generate new baseline files (run from develop branch)
    pytest test_ranking_regression_simulation.py -v --generate-baseline

    # Update baseline for specific architecture
    pytest test_ranking_regression_simulation.py -v --generate-baseline -k gfx942
"""

from pathlib import Path

import pytest

import origami
from helpers import (
    HARDWARE,
    BaselineStore,
    compare_rankings,
    create_config_list,
    create_problem,
    is_dtype_supported,
    load_problem_sizes,
    result_to_config_tuple,
    transpose_key,
)


BASELINE_DIR = Path(__file__).parent / "baselines" / "rankings_simulation"
SUPPORTED_DTYPES = ["f16", "bf16", "f32", "f8"]
TRANSPOSE_VALUES = [origami.transpose_t.T, origami.transpose_t.N]
SIMULATION_ARCHS = ["gfx942", "gfx950", "gfx1201"]
TOP_K = 3

# Every 4th problem to keep simulation runtime practical.
TEST_PROBLEM_SIZES = load_problem_sizes(stride=4)

baselines = BaselineStore(BASELINE_DIR)


def generate_rankings(
    arch_name: str,
    dtype: str,
    transA: origami.transpose_t = origami.transpose_t.T,
    transB: origami.transpose_t = origami.transpose_t.N,
) -> dict[str, list[list[int]]]:
    """Generate simulation mode rankings for test problem sizes."""
    hardware = HARDWARE[arch_name]
    # Reduced config space vs estimation tests: Formocast simulation is ~100x
    # slower per config, so we use fewer tile sizes/depths to keep CI runtime
    # practical (~2.5 min) while still covering representative configurations.
    configs = create_config_list(
        hardware, dtype,
        mt_sizes=[64, 128, 256],
        depth_unroll=[32, 64],
        occupancy_values=[1, 2],
        wgm_values=[1],
    )

    if not configs:
        return {}

    for cfg in configs:
        cfg.prediction_mode = origami.prediction_modes_t.simulation
        cfg.tensile()

    rankings = {}
    for m, n, k, batch in TEST_PROBLEM_SIZES:
        problem = create_problem(m, n, k, dtype, batch, transA, transB)
        try:
            ranked = origami.select_topk_configs(problem, hardware, configs, TOP_K)
            if ranked:
                key = f"{m}x{n}x{k}x{batch}"
                rankings[key] = [result_to_config_tuple(r) for r in ranked]
        except Exception:
            pass

    return rankings


@pytest.mark.regression
class TestRankingRegressionSimulation:
    """Test suite for simulation mode ranking regression tests."""

    @pytest.mark.parametrize("arch_name", SIMULATION_ARCHS)
    @pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
    @pytest.mark.parametrize("transA", TRANSPOSE_VALUES)
    @pytest.mark.parametrize("transB", TRANSPOSE_VALUES)
    def test_ranking_stability(
        self,
        arch_name: str,
        dtype: str,
        transA: origami.transpose_t,
        transB: origami.transpose_t,
        generate_baseline: bool,
    ):
        """Test that simulation mode rankings remain stable compared to baseline."""
        trans_str = transpose_key(transA, transB)

        if not is_dtype_supported(arch_name, dtype):
            pytest.skip(f"No {dtype} support for {arch_name}")

        current = generate_rankings(arch_name, dtype, transA, transB)

        if not current:
            pytest.skip(f"No valid configs generated for {arch_name}/{dtype}/{trans_str}")

        if generate_baseline:
            baselines.save(arch_name, dtype, transA, transB, current)
            pytest.skip(f"Generated baseline for {arch_name}/{dtype}/{trans_str}")

        baseline = baselines.load(arch_name, dtype, transA, transB)
        if baseline is None:
            pytest.fail(
                f"No baseline file found for {arch_name}/{dtype}/{trans_str}. "
                f"Run with --generate-baseline to create it."
            )

        differences = compare_rankings(current, baseline)

        if differences:
            diff_summary = "\n".join(differences[:10])
            if len(differences) > 10:
                diff_summary += f"\n... and {len(differences) - 10} more differences"
            pytest.fail(
                f"Simulation ranking regression detected for {arch_name}/{dtype}/{trans_str}:\n{diff_summary}"
            )
