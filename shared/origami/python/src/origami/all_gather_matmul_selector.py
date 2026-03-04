# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Analytical parameter selector for fused all-gather + GEMM kernels.

Uses the Origami C++ analytical model (origami::algorithms::all_gather_gemm)
to predict optimal kernel parameters, mirroring the TritonBLAS
OrigamiMatmulSelector pattern.
"""

import torch

import origami
from origami.algorithms import all_gather_gemm


class OrigamiAllGatherMatmulSelector:
    """Analytical parameter selector for fused all-gather + GEMM kernels.

    Construct with a problem description (M, N, K, dtypes, device, world_size,
    link bandwidth) and read the recommended kernel parameters from properties.

    The selector internally calls the Origami C++ model which:
      1. Chooses tile sizes (block_m/n/k) using MFMA-aware heuristics
      2. Derives pipeline staging parameters (k_per_flag, num_fetch_stages)
      3. Optimises fetcher CU allocation (num_fetch_sms) using a wavefront
         concurrency model that balances communication and compute
      4. Predicts total kernel time using CU-work-queue + pipeline models
    """

    dtype_to_str = {
        torch.float32: "f32",
        torch.float64: "f64",
        torch.float16: "f16",
        torch.bfloat16: "bf16",
        torch.int8: "i8",
    }
    if hasattr(torch, "float8_e5m2fnuz"):
        dtype_to_str[torch.float8_e5m2fnuz] = "f8"
    if hasattr(torch, "float8_e4m3fnuz"):
        dtype_to_str[torch.float8_e4m3fnuz] = "f8"
    if hasattr(torch, "float8_e5m2"):
        dtype_to_str[torch.float8_e5m2] = "f8"
    if hasattr(torch, "float8_e4m3fn"):
        dtype_to_str[torch.float8_e4m3fn] = "f8"

    def __init__(
        self,
        m: int,
        n: int,
        k: int,
        a_dtype: torch.dtype,
        b_dtype: torch.dtype,
        out_dtype: torch.dtype,
        device: torch.device,
        world_size: int = 8,
        link_bw: float = 50.0,
        scheduling_factor: float = 4.5,
    ):
        self._m = m
        self._n = n
        self._k = k

        self._hardware = origami.get_hardware_for_device(device.index)

        self._network = all_gather_gemm.network_t.for_architecture(
            self._hardware.arch, link_bw, world_size
        )
        self._network.scheduling_factor = scheduling_factor

        self._problem = self._make_problem(
            m, n, k, a_dtype, b_dtype, out_dtype, world_size, link_bw
        )

        self._result = all_gather_gemm.select_config(
            self._problem, self._hardware, self._network
        )

    def _make_problem(
        self, m, n, k, a_dtype, b_dtype, out_dtype, world_size, link_bw
    ):
        a_str = self.dtype_to_str.get(a_dtype, "f16")
        b_str = self.dtype_to_str.get(b_dtype, "f16")
        out_str = self.dtype_to_str.get(out_dtype, "f16")

        prob = all_gather_gemm.all_gather_matmul_problem_t()
        prob.size = origami.dim3_t(m, n, k)
        prob.a_dtype = origami.string_to_datatype(a_str)
        prob.b_dtype = origami.string_to_datatype(b_str)
        prob.c_dtype = origami.string_to_datatype(out_str)
        prob.d_dtype = origami.string_to_datatype(out_str)
        prob.mi_dtype = origami.string_to_datatype(out_str)
        prob.a_transpose = origami.transpose_t.T
        prob.b_transpose = origami.transpose_t.N
        prob.world_size = world_size
        prob.link_bw_gbps = link_bw
        return prob

    # ── Kernel parameters ────────────────────────────────────────────────

    @property
    def block_m(self) -> int:
        return self._result.config.mt.m

    @property
    def block_n(self) -> int:
        return self._result.config.mt.n

    @property
    def block_k(self) -> int:
        return self._result.config.mt.k

    @property
    def group_size_m(self) -> int:
        return self._result.config.group_size_m

    @property
    def num_fetch_sms(self) -> int:
        return self._result.config.num_fetch_sms

    @property
    def k_per_flag(self) -> int:
        return self._result.config.k_per_flag

    @property
    def num_fetch_stages(self) -> int:
        return self._result.config.num_fetch_stages

    @property
    def first_stage_fetch_sms(self) -> int:
        return self._result.config.first_stage_fetch_sms

    @property
    def num_warps(self) -> int:
        return self._result.config.num_warps

    @property
    def grid_size(self) -> int:
        return self._result.grid_size

    # ── Performance estimates ────────────────────────────────────────────

    @property
    def est_kernel_ms(self) -> float:
        return self._result.est_kernel_ms

    @property
    def comm_time_ms(self) -> float:
        return self._result.comm_time_ms

    @property
    def compute_time_ms(self) -> float:
        return self._result.compute_time_ms

    @property
    def roofline_tflops(self) -> float:
        return self._result.roofline_tflops

    @property
    def comm_compute_ratio(self) -> float:
        return self._result.comm_compute_ratio

    @property
    def pipeline_ms(self) -> float:
        return self._result.pipeline_ms

    @property
    def total_fetch_wgs(self) -> int:
        return self._result.total_fetch_wgs

    @property
    def total_gemm_wgs(self) -> int:
        return self._result.total_gemm_wgs
