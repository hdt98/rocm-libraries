# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Shared test utilities for origami tests."""

import csv
from pathlib import Path

import yaml

import origami

try:
    from yaml import CSafeDumper as SafeDumper, CSafeLoader as SafeLoader
except ImportError:
    from yaml import SafeDumper, SafeLoader

PROBLEM_DATA_FILE = Path(__file__).parent / "data" / "problem_data.csv"


# Hardware objects for supported architectures
HARDWARE = {
    "gfx90a": origami.get_hardware_for_arch(
        origami.architecture_t.gfx90a, 110, 64 * 1024, 8 * 1024 * 1024, 1700000
    ),
    "gfx942": origami.get_hardware_for_arch(
        origami.architecture_t.gfx942, 228, 64 * 1024, 24 * 1024 * 1024, 1700000
    ),
    "gfx950": origami.get_hardware_for_arch(
        origami.architecture_t.gfx950, 304, 64 * 1024, 32 * 1024 * 1024, 2100000
    ),
    "gfx1100": origami.get_hardware_for_arch(
        origami.architecture_t.gfx1100, 96, 64 * 1024, 6 * 1024 * 1024, 2500000
    ),
    "gfx1201": origami.get_hardware_for_arch(
        origami.architecture_t.gfx1201, 60, 128 * 1024, 6 * 1024 * 1024, 2500000
    ),
}


def get_matrix_instructions(
    hardware: origami.hardware_t, dtype: str
) -> list[tuple[int, int, int]]:
    """Get valid matrix instructions from hardware for the given dtype."""
    dtype_enum = origami.string_to_datatype(dtype)
    instructions = hardware.get_valid_matrix_instructions(dtype_enum)
    return [(mi.m, mi.n, mi.k) for mi in instructions]


def _generate_mt_pairs(
    mi: tuple[int, int, int],
    mt_sizes: list[int] | None,
    waves: list[list[int]],
    max_mt: int,
) -> list[tuple[int, int]]:
    """Generate (mt_m, mt_n) pairs for a given matrix instruction."""
    if mt_sizes is not None:
        return [(mt_m, mt_n) for mt_m in mt_sizes for mt_n in mt_sizes]

    # Wave-based: MT = MI × wave_tile × wave
    mi_m, mi_n, _ = mi
    min_mt = 16
    pairs = []

    for wave in waves:
        wave_tile_m = 0
        while True:
            wave_tile_m += 1
            mt_m = mi_m * wave_tile_m * wave[0]
            if mt_m < min_mt:
                continue
            if mt_m > max_mt:
                break

            wave_tile_n = 0
            while True:
                wave_tile_n += 1
                mt_n = mi_n * wave_tile_n * wave[1]
                if mt_n < min_mt:
                    continue
                if mt_n > max_mt:
                    break
                pairs.append((mt_m, mt_n))

    return pairs


def create_config_list(
    hardware: origami.hardware_t,
    dtype: str,
    *,
    mt_sizes: list[int] | None = None,
    depth_unroll: list[int] | None = None,
    occupancy_values: list[int] | None = None,
    wgm_values: list[int] | None = None,
    waves: list[list[int]] | None = None,
    max_mt: int = 512,
) -> list[origami.config_t]:
    """Create a list of configurations for testing using dynamic MI discovery."""
    mi_list = get_matrix_instructions(hardware, dtype)
    if not mi_list:
        return []

    if depth_unroll is None:
        depth_unroll = [16, 32, 64, 128, 256, 512, 1024]
    if occupancy_values is None:
        occupancy_values = [1]
    if wgm_values is None:
        wgm_values = [6]
    if waves is None:
        waves = [[4, 1], [2, 2], [1, 4], [1, 2], [2, 1], [1, 1]]

    configs = []
    for mi in mi_list:
        mi_m, mi_n, mi_k = mi
        mt_pairs = _generate_mt_pairs(mi, mt_sizes, waves, max_mt)

        for mt_m, mt_n in mt_pairs:
            for mt_k in depth_unroll:
                for occ in occupancy_values:
                    for wgm in wgm_values:
                        config = origami.config_t()
                        config.mt = origami.dim3_t(mt_m, mt_n, mt_k)
                        config.mi = origami.dim3_t(mi_m, mi_n, mi_k)
                        config.occupancy = occ
                        config.workgroup_mapping = wgm
                        configs.append(config)

    return configs


def load_problem_sizes(stride: int = 1) -> list[tuple[int, int, int, int]]:
    """Load problem sizes from CSV file.

    Args:
        stride: Take every Nth problem (1 = all, 4 = every 4th).
    """
    problems = []
    with open(PROBLEM_DATA_FILE, "r") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i % stride != 0:
                continue
            problems.append((
                int(row["m"]), int(row["n"]),
                int(row["k"]), int(row["batch_count"]),
            ))
    return problems


def is_dtype_supported(arch_name: str, dtype: str) -> bool:
    """Check if a dtype is supported for the given architecture."""
    return len(get_matrix_instructions(HARDWARE[arch_name], dtype)) > 0


def create_problem(
    m: int,
    n: int,
    k: int,
    dtype: str,
    batch: int = 1,
    transA: origami.transpose_t = origami.transpose_t.T,
    transB: origami.transpose_t = origami.transpose_t.N,
) -> origami.problem_t:
    """Create a problem specification."""
    problem = origami.problem_t()
    problem.size = origami.dim3_t(m, n, k)
    problem.batch = batch
    problem.a_transpose = transA
    problem.b_transpose = transB
    problem.a_dtype = origami.string_to_datatype(dtype)
    problem.b_dtype = origami.string_to_datatype(dtype)
    problem.d_dtype = origami.string_to_datatype(dtype)
    problem.c_dtype = problem.d_dtype
    problem.mi_dtype = problem.a_dtype
    problem.a_mx_block_size = 0
    problem.b_mx_block_size = 0
    return problem


def result_to_config_tuple(result: origami.prediction_result_t) -> list[int]:
    """Convert prediction result to [mt_m, mt_n, mt_k, mi_m, mi_n, mi_k, occ, wgm]."""
    cfg = result.config
    return [
        cfg.mt.m, cfg.mt.n, cfg.mt.k,
        cfg.mi.m, cfg.mi.n, cfg.mi.k,
        cfg.occupancy, cfg.workgroup_mapping,
    ]


def transpose_key(transA: origami.transpose_t, transB: origami.transpose_t) -> str:
    """Convert transpose values to a string key like 'TN'."""
    a = "T" if transA == origami.transpose_t.T else "N"
    b = "T" if transB == origami.transpose_t.T else "N"
    return f"{a}{b}"


def compare_rankings(
    current: dict[str, list[list[int]]], baseline: dict[str, list[list[int]]]
) -> list[str]:
    """Compare current rankings against baseline. Returns list of differences."""
    differences = []

    for problem_key, base_configs in baseline.items():
        if problem_key not in current:
            differences.append(f"Missing problem: {problem_key}")
            continue

        curr_configs = current[problem_key]

        if len(curr_configs) != len(base_configs):
            differences.append(
                f"{problem_key}: Different rank count "
                f"(curr={len(curr_configs)}, base={len(base_configs)})"
            )

        for rank_idx, (curr_cfg, base_cfg) in enumerate(zip(curr_configs, base_configs)):
            if curr_cfg != base_cfg:
                differences.append(
                    f"{problem_key} rank {rank_idx}: Config mismatch\n"
                    f"  Current:  MT={tuple(curr_cfg[0:3])}, MI={tuple(curr_cfg[3:6])}, "
                    f"occ={curr_cfg[6]}, wgm={curr_cfg[7]}\n"
                    f"  Baseline: MT={tuple(base_cfg[0:3])}, MI={tuple(base_cfg[3:6])}, "
                    f"occ={base_cfg[6]}, wgm={base_cfg[7]}"
                )

    for problem_key in current:
        if problem_key not in baseline:
            differences.append(f"New problem not in baseline: {problem_key}")

    return differences


class BaselineStore:
    """Manages golden baseline YAML files for ranking regression tests."""

    def __init__(self, baseline_dir: Path):
        self._dir = baseline_dir
        self._cache: dict[str, dict | None] = {}

    def get_path(self, arch_name: str) -> Path:
        return self._dir / f"{arch_name}.yaml"

    def _load_arch(self, arch_name: str) -> dict | None:
        if arch_name not in self._cache:
            path = self.get_path(arch_name)
            if not path.exists():
                self._cache[arch_name] = None
            else:
                with open(path, "r") as f:
                    self._cache[arch_name] = yaml.load(f, Loader=SafeLoader)
        return self._cache[arch_name]

    def load(
        self, arch_name: str, dtype: str,
        transA: origami.transpose_t, transB: origami.transpose_t,
    ) -> dict[str, list[list[int]]] | None:
        """Load baseline rankings for a specific arch/dtype/transpose combo."""
        baseline = self._load_arch(arch_name)
        if baseline is None:
            return None
        try:
            return baseline[dtype][transpose_key(transA, transB)]
        except KeyError:
            return None

    def save(
        self, arch_name: str, dtype: str,
        transA: origami.transpose_t, transB: origami.transpose_t,
        rankings: dict[str, list[list[int]]],
    ) -> None:
        """Save rankings to the architecture's YAML baseline file."""
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self.get_path(arch_name)

        if path.exists():
            with open(path, "r") as f:
                baseline = yaml.load(f, Loader=SafeLoader) or {}
        else:
            baseline = {}

        if dtype not in baseline:
            baseline[dtype] = {}
        baseline[dtype][transpose_key(transA, transB)] = rankings

        class CompactDumper(SafeDumper):
            pass

        def represent_problem_configs(dumper, data):
            return dumper.represent_sequence(
                'tag:yaml.org,2002:seq', data, flow_style=True)

        CompactDumper.add_representer(list, represent_problem_configs)

        with open(path, "w") as f:
            yaml.dump(
                baseline, f, Dumper=CompactDumper,
                default_flow_style=False, sort_keys=True, width=1000,
            )

        self._cache.pop(arch_name, None)
