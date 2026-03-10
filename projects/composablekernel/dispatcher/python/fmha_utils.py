#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
FMHA Dispatcher Python Utilities

Provides Python wrappers for FMHA dispatcher kernels via ctypes.
Mirrors ctypes_utils.py (GEMM) and grouped_conv_utils.py (Conv).

Usage:
    from fmha_utils import FmhaDispatcherLib, FmhaRunner, FmhaProblem, cpu_attention_fwd

    runner = FmhaRunner.from_prebuilt()
    result = runner.run(Q, K, V, problem)
"""

import ctypes
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


# =============================================================================
# Utility helpers
# =============================================================================


def get_dispatcher_root() -> Path:
    return Path(__file__).parent.parent


def detect_gpu_arch() -> str:
    try:
        out = subprocess.check_output(
            ["rocminfo"], text=True, stderr=subprocess.DEVNULL
        )
        for line in out.splitlines():
            if "Name:" in line and "gfx" in line:
                return line.split()[-1].strip()
    except Exception:
        pass
    return "gfx950"


# =============================================================================
# Data types
# =============================================================================


@dataclass
class FmhaResult:
    success: bool
    output: Optional[np.ndarray] = None
    time_ms: float = 0.0
    tflops: float = 0.0
    error: str = ""


@dataclass
class FmhaProblem:
    batch: int = 2
    nhead_q: int = 8
    nhead_k: int = 8
    seqlen_q: int = 128
    seqlen_k: int = 128
    hdim_q: int = 128
    hdim_v: int = 128

    @property
    def scale(self) -> float:
        return 1.0 / (self.hdim_q**0.5)

    @property
    def num_ops(self) -> int:
        sq, sk = self.seqlen_q, self.seqlen_k
        return 2 * self.batch * self.nhead_q * sq * sk * (self.hdim_q + self.hdim_v)

    def q_shape(self):
        return (self.batch, self.nhead_q, self.seqlen_q, self.hdim_q)

    def k_shape(self):
        return (self.batch, self.nhead_k, self.seqlen_k, self.hdim_q)

    def v_shape(self):
        return (self.batch, self.nhead_k, self.seqlen_k, self.hdim_v)

    def o_shape(self):
        return (self.batch, self.nhead_q, self.seqlen_q, self.hdim_v)


@dataclass
class FmhaKernelConfig:
    """Complete kernel configuration for FMHA.

    All tile/wave/warp dimensions are explicitly named to match the
    GEMM pattern (tile_m, tile_n, tile_k) but extended for FMHA's
    two-stage computation (Q*K^T stage 0, Attn*V stage 1).
    """

    # -- Signature: what operation --
    family: str = "fwd"
    data_type: str = "fp16"
    mode: str = "batch"
    vlayout: str = "r"
    hdim_q: int = 128
    hdim_v: int = 128
    gfx_arch: str = "gfx950"

    # -- Algorithm: tile shape --
    # Stage 0 (Q * K^T): seqlen_q x seqlen_k x hdim_q
    tile_m0: int = 128  # seqlen_q tile
    tile_n0: int = 128  # seqlen_k tile
    tile_k0: int = 32  # hdim_q tile
    # Stage 1 (Attn * V): seqlen_q x hdim_v x seqlen_k
    tile_n1: int = 128  # hdim_v tile
    tile_k1: int = 32  # seqlen_k tile
    tile_k0max: int = 128  # max k0 (alignment)

    # -- Algorithm: wave config (warps per block) --
    wave_m0: int = 4
    wave_n0: int = 1
    wave_k0: int = 1
    wave_m1: int = 4
    wave_n1: int = 1
    wave_k1: int = 1
    wave_m2: int = 1
    wave_n2: int = 1
    wave_k2: int = 1

    # -- Algorithm: warp tile (elements per warp) --
    warp_m0: int = 32
    warp_n0: int = 32
    warp_k0: int = 16
    warp_m1: int = 32
    warp_n1: int = 32
    warp_k1: int = 16
    warp_m2: int = 16
    warp_n2: int = 16
    warp_k2: int = 16

    # -- Algorithm: padding --
    pad_s: bool = True  # pad seqlen_q
    pad_sk: bool = True  # pad seqlen_k
    pad_d: bool = True  # pad hdim_q
    pad_dv: bool = True  # pad hdim_v

    # -- Algorithm: pipeline --
    pipeline: str = "qr_async"
    block_per_cu: int = 1
    num_wave_groups: int = 1

    # -- Signature: features --
    mask: str = "no"
    bias: str = "no"
    lse: bool = False
    dropout: bool = False
    qscale: str = "no"
    rope: str = "none"
    logits: bool = False
    paged_kv: bool = False
    sink: bool = False

    @property
    def tile(self) -> Tuple[int, ...]:
        return (
            self.tile_m0,
            self.tile_n0,
            self.tile_k0,
            self.tile_n1,
            self.tile_k1,
            self.tile_k0max,
        )

    @property
    def wave(self) -> Tuple[int, ...]:
        return (
            self.wave_m0,
            self.wave_n0,
            self.wave_k0,
            self.wave_m1,
            self.wave_n1,
            self.wave_k1,
            self.wave_m2,
            self.wave_n2,
            self.wave_k2,
        )

    @property
    def warp(self) -> Tuple[int, ...]:
        return (
            self.warp_m0,
            self.warp_n0,
            self.warp_k0,
            self.warp_m1,
            self.warp_n1,
            self.warp_k1,
            self.warp_m2,
            self.warp_n2,
            self.warp_k2,
        )

    @property
    def padding(self) -> Tuple[bool, ...]:
        return (self.pad_s, self.pad_sk, self.pad_d, self.pad_dv)

    @property
    def name(self) -> str:
        return (
            f"fmha_{self.family}_{self.data_type}_h{self.hdim_q}"
            f"_{self.pipeline}_{self.tile_m0}x{self.tile_n0}x{self.tile_k0}"
        )

    def to_codegen_json(self) -> str:
        return json.dumps(
            {
                "arch": self.gfx_arch,
                "signature": {
                    "family": self.family,
                    "data_type": self.data_type,
                    "mode": self.mode,
                    "vlayout": self.vlayout,
                    "hdim_q": self.hdim_q,
                    "hdim_v": self.hdim_v,
                    "mask": self.mask,
                    "bias": self.bias,
                    "lse": self.lse,
                    "dropout": self.dropout,
                    "qscale": self.qscale,
                    "rope": self.rope,
                    "logits": self.logits,
                    "paged_kv": self.paged_kv,
                    "fp8_static_quant": False,
                    "skip_min_seqlen_q": False,
                    "sink": self.sink,
                    "dbias": False,
                    "store_randval": False,
                    "deterministic": False,
                    "kv_memory_layout": "vectorized",
                    "kv_lookup_table": "sglang",
                    "page_size": 1,
                },
                "algorithm": {
                    "pipeline": self.pipeline,
                    "tile": list(self.tile),
                    "wave": list(self.wave),
                    "warp": list(self.warp),
                    "padding": list(self.padding),
                    "block_per_cu": self.block_per_cu,
                    "num_wave_groups": self.num_wave_groups,
                    "max_splits_log2": 0,
                    "max_seq_len_q": 0,
                },
            }
        )


# =============================================================================
# CPU reference
# =============================================================================


def cpu_attention_fwd(
    Q: np.ndarray, K: np.ndarray, V: np.ndarray, scale: float
) -> np.ndarray:
    """CPU reference: scaled dot-product attention (supports GQA).

    Args:
        Q: [batch, nhead_q, seqlen_q, hdim_q]  float32
        K: [batch, nhead_k, seqlen_k, hdim_q]  float32
        V: [batch, nhead_k, seqlen_k, hdim_v]  float32

    Returns:
        O: [batch, nhead_q, seqlen_q, hdim_v]  float32
    """
    nhead_q = Q.shape[1]
    nhead_k = K.shape[1]
    if nhead_q != nhead_k:
        ratio = nhead_q // nhead_k
        K = np.repeat(K, ratio, axis=1)
        V = np.repeat(V, ratio, axis=1)
    S = np.matmul(Q, K.transpose(0, 1, 3, 2)) * scale
    S_max = S.max(axis=-1, keepdims=True)
    S_exp = np.exp(S - S_max)
    P = S_exp / S_exp.sum(axis=-1, keepdims=True)
    return np.matmul(P, V)


# =============================================================================
# Low-level ctypes wrapper
# =============================================================================


class FmhaDispatcherLib:
    """Wrapper for the FMHA dispatcher shared library (libdispatcher_fmha_lib.so)."""

    SEARCH_PATHS = [
        "build/examples/libdispatcher_fmha_lib.so",
        "build/libdispatcher_fmha_lib.so",
        "build/lib/libdispatcher_fmha_lib.so",
    ]

    def __init__(self, lib: ctypes.CDLL, path: Path):
        self._lib = lib
        self.path = path
        self._setup()

    def _setup(self):
        lib = self._lib
        lib.fmha_dispatcher_initialize.argtypes = [ctypes.c_char_p]
        lib.fmha_dispatcher_initialize.restype = ctypes.c_int
        lib.fmha_dispatcher_run_fwd.argtypes = [
            ctypes.c_void_p,  # q
            ctypes.c_void_p,  # k
            ctypes.c_void_p,  # v
            ctypes.c_void_p,  # o
            ctypes.c_int,  # batch
            ctypes.c_int,  # nhead_q
            ctypes.c_int,  # nhead_k
            ctypes.c_int,  # seqlen_q
            ctypes.c_int,  # seqlen_k
            ctypes.c_int,  # hdim_q
            ctypes.c_int,  # hdim_v
            ctypes.c_float,  # scale
            ctypes.c_int,  # mask_type
            ctypes.c_int,  # bias_type
            ctypes.c_int,  # has_lse
            ctypes.c_int,  # has_dropout
            ctypes.c_int,  # traits_hdim_q (0=same as hdim_q)
            ctypes.c_int,  # traits_hdim_v (0=same as hdim_v)
            ctypes.POINTER(ctypes.c_float),  # time_ms_out
        ]
        lib.fmha_dispatcher_run_fwd.restype = ctypes.c_int
        lib.fmha_dispatcher_run_bwd.argtypes = [
            ctypes.c_void_p,  # q
            ctypes.c_void_p,  # k
            ctypes.c_void_p,  # v
            ctypes.c_void_p,  # o
            ctypes.c_void_p,  # lse
            ctypes.c_void_p,  # do
            ctypes.c_void_p,  # dq
            ctypes.c_void_p,  # dk
            ctypes.c_void_p,  # dv
            ctypes.c_int,  # batch
            ctypes.c_int,  # nhead_q
            ctypes.c_int,  # nhead_k
            ctypes.c_int,  # seqlen_q
            ctypes.c_int,  # seqlen_k
            ctypes.c_int,  # hdim_q
            ctypes.c_int,  # hdim_v
            ctypes.c_float,  # scale
            ctypes.POINTER(ctypes.c_float),  # time_ms_out
        ]
        lib.fmha_dispatcher_run_bwd.restype = ctypes.c_int
        lib.fmha_dispatcher_kernel_count.argtypes = []
        lib.fmha_dispatcher_kernel_count.restype = ctypes.c_int
        lib.fmha_dispatcher_cleanup.argtypes = []
        lib.fmha_dispatcher_cleanup.restype = None

    @classmethod
    def find(cls) -> Optional["FmhaDispatcherLib"]:
        root = get_dispatcher_root()
        for rel in cls.SEARCH_PATHS:
            path = root / rel
            if path.exists():
                try:
                    lib = ctypes.CDLL(str(path))
                    return cls(lib, path)
                except OSError:
                    continue
        return None

    @classmethod
    def load(cls, path: str) -> "FmhaDispatcherLib":
        lib = ctypes.CDLL(path)
        return cls(lib, Path(path))

    def initialize(self, arch: str = "gfx950") -> bool:
        return self._lib.fmha_dispatcher_initialize(arch.encode()) == 0

    def run_fwd(
        self,
        q: ctypes.c_void_p,
        k: ctypes.c_void_p,
        v: ctypes.c_void_p,
        o: ctypes.c_void_p,
        prob: FmhaProblem,
        mask_type: int = 0,
        bias_type: int = 0,
        has_lse: int = 0,
        has_dropout: int = 0,
        traits_hdim_q: int = 0,
        traits_hdim_v: int = 0,
    ) -> Tuple[int, float]:
        time_ms = ctypes.c_float(0.0)
        rc = self._lib.fmha_dispatcher_run_fwd(
            q,
            k,
            v,
            o,
            prob.batch,
            prob.nhead_q,
            prob.nhead_k,
            prob.seqlen_q,
            prob.seqlen_k,
            prob.hdim_q,
            prob.hdim_v,
            prob.scale,
            mask_type,
            bias_type,
            has_lse,
            has_dropout,
            traits_hdim_q,
            traits_hdim_v,
            ctypes.byref(time_ms),
        )
        return rc, time_ms.value

    def run_bwd(
        self,
        q: ctypes.c_void_p,
        k: ctypes.c_void_p,
        v: ctypes.c_void_p,
        o: ctypes.c_void_p,
        lse: ctypes.c_void_p,
        do_grad: ctypes.c_void_p,
        dq: ctypes.c_void_p,
        dk: ctypes.c_void_p,
        dv: ctypes.c_void_p,
        prob: FmhaProblem,
    ) -> Tuple[int, float]:
        time_ms = ctypes.c_float(0.0)
        rc = self._lib.fmha_dispatcher_run_bwd(
            q,
            k,
            v,
            o,
            lse,
            do_grad,
            dq,
            dk,
            dv,
            prob.batch,
            prob.nhead_q,
            prob.nhead_k,
            prob.seqlen_q,
            prob.seqlen_k,
            prob.hdim_q,
            prob.hdim_v,
            prob.scale,
            ctypes.byref(time_ms),
        )
        return rc, time_ms.value

    def kernel_count(self) -> int:
        return self._lib.fmha_dispatcher_kernel_count()

    def cleanup(self):
        self._lib.fmha_dispatcher_cleanup()


# =============================================================================
# High-level GPU runner (mirrors GpuGroupedConvRunner)
# =============================================================================


class FmhaRunner:
    """High-level FMHA runner with NumPy interface and HIP memory management."""

    HIP_MEMCPY_H2D = 1
    HIP_MEMCPY_D2H = 2

    def __init__(self, dispatch_lib: FmhaDispatcherLib, arch: str = "gfx950"):
        self._lib = dispatch_lib
        self._arch = arch
        self._hip = None
        self._load_hip()
        if not dispatch_lib.initialize(arch):
            raise RuntimeError("Failed to initialize FMHA dispatcher")

    def _load_hip(self):
        for name in ["libamdhip64.so", "libamdhip64.so.6"]:
            try:
                self._hip = ctypes.CDLL(name)
                self._hip.hipMalloc.argtypes = [
                    ctypes.POINTER(ctypes.c_void_p),
                    ctypes.c_size_t,
                ]
                self._hip.hipMalloc.restype = ctypes.c_int
                self._hip.hipFree.argtypes = [ctypes.c_void_p]
                self._hip.hipFree.restype = ctypes.c_int
                self._hip.hipMemcpy.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_size_t,
                    ctypes.c_int,
                ]
                self._hip.hipMemcpy.restype = ctypes.c_int
                self._hip.hipMemset.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_int,
                    ctypes.c_size_t,
                ]
                self._hip.hipMemset.restype = ctypes.c_int
                return
            except OSError:
                continue
        raise RuntimeError("Could not load libamdhip64.so")

    @classmethod
    def from_prebuilt(cls, arch: Optional[str] = None) -> "FmhaRunner":
        arch = arch or detect_gpu_arch()
        lib = FmhaDispatcherLib.find()
        if lib is None:
            raise RuntimeError(
                "FMHA dispatcher library not found. Build with:\n"
                "  cd dispatcher/build && cmake .. -DBUILD_DISPATCHER_EXAMPLES=ON && make dispatcher_fmha_lib"
            )
        return cls(lib, arch)

    @classmethod
    def from_library(cls, path: str, arch: Optional[str] = None) -> "FmhaRunner":
        arch = arch or detect_gpu_arch()
        return cls(FmhaDispatcherLib.load(path), arch)

    def run(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        prob: FmhaProblem,
        mask_type: int = 0,
        bias_type: int = 0,
        has_lse: int = 0,
        has_dropout: int = 0,
    ) -> FmhaResult:
        """Run FMHA forward on GPU with automatic HIP memory management.

        Args:
            Q: [batch, nhead_q, seqlen_q, hdim_q]  float16
            K: [batch, nhead_k, seqlen_k, hdim_q]  float16
            V: [batch, nhead_k, seqlen_k, hdim_v]  float16

        Returns:
            FmhaResult with output array, timing, TFLOPS
        """
        Q_c = np.ascontiguousarray(Q.astype(np.float16))
        K_c = np.ascontiguousarray(K.astype(np.float16))
        V_c = np.ascontiguousarray(V.astype(np.float16))
        O_c = np.zeros(prob.o_shape(), dtype=np.float16)

        d_q, d_k, d_v, d_o = (ctypes.c_void_p() for _ in range(4))

        try:
            self._hip.hipMalloc(ctypes.byref(d_q), Q_c.nbytes)
            self._hip.hipMalloc(ctypes.byref(d_k), K_c.nbytes)
            self._hip.hipMalloc(ctypes.byref(d_v), V_c.nbytes)
            self._hip.hipMalloc(ctypes.byref(d_o), O_c.nbytes)

            self._hip.hipMemcpy(d_q, Q_c.ctypes.data, Q_c.nbytes, self.HIP_MEMCPY_H2D)
            self._hip.hipMemcpy(d_k, K_c.ctypes.data, K_c.nbytes, self.HIP_MEMCPY_H2D)
            self._hip.hipMemcpy(d_v, V_c.ctypes.data, V_c.nbytes, self.HIP_MEMCPY_H2D)
            self._hip.hipMemset(d_o, 0, O_c.nbytes)

            time_ms = ctypes.c_float(0.0)
            rc = self._lib._lib.fmha_dispatcher_run_fwd(
                d_q,
                d_k,
                d_v,
                d_o,
                prob.batch,
                prob.nhead_q,
                prob.nhead_k,
                prob.seqlen_q,
                prob.seqlen_k,
                prob.hdim_q,
                prob.hdim_v,
                prob.scale,
                mask_type,
                bias_type,
                has_lse,
                has_dropout,
                0,
                0,  # traits_hdim_q, traits_hdim_v (0 = same as hdim)
                ctypes.byref(time_ms),
            )

            if rc != 0:
                return FmhaResult(success=False, error=f"Kernel failed (rc={rc})")

            self._hip.hipMemcpy(O_c.ctypes.data, d_o, O_c.nbytes, self.HIP_MEMCPY_D2H)

            tflops = (
                prob.num_ops / (time_ms.value * 1e-3) / 1e12
                if time_ms.value > 0
                else 0.0
            )
            return FmhaResult(
                success=True, output=O_c, time_ms=time_ms.value, tflops=tflops
            )

        finally:
            for d in [d_q, d_k, d_v, d_o]:
                if d.value:
                    self._hip.hipFree(d)

    @property
    def kernel_count(self) -> int:
        return self._lib.kernel_count()

    @property
    def library_path(self) -> str:
        return str(self._lib.path)

    def cleanup(self):
        self._lib.cleanup()


# =============================================================================
# JIT Build Support (mirrors setup_multiple_gemm_dispatchers)
# =============================================================================


@dataclass
class FmhaSetupResult:
    success: bool
    config: Optional[FmhaKernelConfig] = None
    runner: Optional[FmhaRunner] = None
    library_path: str = ""
    error: str = ""
    build_time_s: float = 0.0


def _find_static_lib() -> Optional[Path]:
    root = get_dispatcher_root()
    for rel in ["build/libck_tile_dispatcher.a", "build/lib/libck_tile_dispatcher.a"]:
        p = root / rel
        if p.exists():
            return p
    return None


def _find_hipcc() -> str:
    for path in ["/opt/rocm/bin/hipcc", "/usr/bin/hipcc"]:
        if os.path.exists(path):
            return path
    return "hipcc"


def setup_fmha_dispatcher(
    config: FmhaKernelConfig,
    output_dir: Optional[Path] = None,
    verbose: bool = False,
) -> FmhaSetupResult:
    """JIT-compile a single FMHA kernel and return a runner.

    Steps:
      1. Run unified_fmha_codegen.py to generate kernel header + wrapper
      2. Run generate_fmha_fallback.py to create dispatch header
      3. Compile kernel .cpp into .o
      4. Compile fmha_ctypes_lib.cpp with -include dispatch header
      5. Link into .so
    """
    import time

    t0 = time.perf_counter()

    root = get_dispatcher_root()
    codegen_dir = root / "codegen"
    ctypes_src = root / "bindings" / "ctypes" / "fmha_ctypes_lib.cpp"
    static_lib = _find_static_lib()
    hipcc = _find_hipcc()

    if output_dir is None:
        output_dir = root / "build" / "examples" / f"fmha_jit_{config.name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    lib_name = f"libdispatcher_fmha_{config.name}.so"
    lib_path = output_dir / lib_name

    if not static_lib:
        return FmhaSetupResult(
            success=False, config=config, error="libck_tile_dispatcher.a not found"
        )
    if not ctypes_src.exists():
        return FmhaSetupResult(
            success=False, config=config, error="fmha_ctypes_lib.cpp not found"
        )

    # Step 1: Generate kernel
    gen_cmd = [
        sys.executable,
        str(codegen_dir / "generate_fmha_fallback.py"),
        "--output-dir",
        str(output_dir),
        "--gpu-target",
        config.gfx_arch,
        "--config-json",
        config.to_codegen_json(),
    ]
    r = subprocess.run(gen_cmd, capture_output=True, text=True, cwd=str(codegen_dir))
    if r.returncode != 0:
        return FmhaSetupResult(
            success=False, config=config, error=f"Codegen failed: {r.stderr[:500]}"
        )

    dispatch_header = output_dir / "fmha_python_dispatch.hpp"
    if not dispatch_header.exists():
        return FmhaSetupResult(
            success=False, config=config, error="Dispatch header not generated"
        )

    # Step 2: Compile kernel .cpp
    kernel_cpps = list(output_dir.glob("fmha_*.cpp"))
    kernel_objs = []
    include_dirs = [
        str(root.parent / "include"),
        str(root / "include"),
        str(root.parent),
    ]
    inc_flags = [f"-I{d}" for d in include_dirs]

    for cpp in kernel_cpps:
        obj = cpp.with_suffix(".o")
        compile_cmd = [
            hipcc,
            "-c",
            "-fPIC",
            "-O3",
            f"--offload-arch={config.gfx_arch}",
            "-std=c++17",
            *inc_flags,
            "-mllvm",
            "-enable-noalias-to-md-conversion=0",
            "-Wno-undefined-func-template",
            "-Wno-float-equal",
            "--offload-compress",
            str(cpp),
            "-o",
            str(obj),
        ]
        r = subprocess.run(compile_cmd, capture_output=True, text=True)
        if r.returncode != 0:
            return FmhaSetupResult(
                success=False,
                config=config,
                error=f"Kernel compile failed: {r.stderr[:500]}",
            )
        kernel_objs.append(str(obj))

    # Step 3: Compile fmha_ctypes_lib.cpp
    ctypes_obj = output_dir / "fmha_ctypes_lib.o"
    compile_cmd = [
        hipcc,
        "-c",
        "-fPIC",
        "-O3",
        f"--offload-arch={config.gfx_arch}",
        "-std=c++17",
        *inc_flags,
        f"-I{output_dir}",
        f"-I{output_dir / 'dispatcher_wrappers'}",
        f"-include{dispatch_header}",
        f'-DGFX_ARCH="{config.gfx_arch}"',
        "-mllvm",
        "-enable-noalias-to-md-conversion=0",
        "-Wno-undefined-func-template",
        "-Wno-float-equal",
        "--offload-compress",
        str(ctypes_src),
        "-o",
        str(ctypes_obj),
    ]
    r = subprocess.run(compile_cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return FmhaSetupResult(
            success=False,
            config=config,
            error=f"ctypes compile failed: {r.stderr[:500]}",
        )

    # Step 4: Link shared library
    link_cmd = [
        hipcc,
        "-shared",
        "-fPIC",
        str(ctypes_obj),
        *kernel_objs,
        str(static_lib),
        "-o",
        str(lib_path),
    ]
    r = subprocess.run(link_cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return FmhaSetupResult(
            success=False, config=config, error=f"Link failed: {r.stderr[:500]}"
        )

    # Step 5: Load and return runner
    try:
        runner = FmhaRunner.from_library(str(lib_path), config.gfx_arch)
    except Exception as e:
        return FmhaSetupResult(success=False, config=config, error=f"Load failed: {e}")

    elapsed = time.perf_counter() - t0
    return FmhaSetupResult(
        success=True,
        config=config,
        runner=runner,
        library_path=str(lib_path),
        build_time_s=elapsed,
    )


def setup_multiple_fmha_dispatchers(
    configs: List[FmhaKernelConfig],
    verbose: bool = False,
    max_workers: Optional[int] = None,
) -> List[FmhaSetupResult]:
    """Parallel JIT compile multiple FMHA kernels."""
    if not configs:
        return []

    workers = max_workers or min(len(configs), os.cpu_count() or 4)
    results: List[Optional[FmhaSetupResult]] = [None] * len(configs)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for i, cfg in enumerate(configs):
            f = pool.submit(setup_fmha_dispatcher, cfg, verbose=verbose)
            futures[f] = i
        for f in as_completed(futures):
            idx = futures[f]
            try:
                results[idx] = f.result()
            except Exception as e:
                results[idx] = FmhaSetupResult(
                    success=False, config=configs[idx], error=str(e)
                )

    return [r for r in results if r is not None]


# =============================================================================
# Registry (mirrors ctypes_utils.Registry)
# =============================================================================


class FmhaRegistry:
    """Kernel registry with parallel JIT build support."""

    def __init__(self, name: str = "fmha"):
        self._name = name
        self._kernels: List[FmhaKernelConfig] = []

    def register_kernel(self, config: FmhaKernelConfig):
        self._kernels.append(config)

    def __len__(self):
        return len(self._kernels)

    def build(
        self,
        verbose: bool = False,
        max_workers: Optional[int] = None,
    ) -> List[FmhaSetupResult]:
        return setup_multiple_fmha_dispatchers(
            self._kernels,
            verbose=verbose,
            max_workers=max_workers,
        )


# =============================================================================
# Cleanup / reset (mirrors ctypes_utils.cleanup_gemm / reset_for_example)
# =============================================================================

_active_runners: List[FmhaRunner] = []


def cleanup_fmha():
    """Clean up all active FMHA runners."""
    for r in _active_runners:
        try:
            r.cleanup()
        except Exception:
            pass
    _active_runners.clear()


def reset_for_example():
    """Reset state between examples."""
    cleanup_fmha()


# =============================================================================
# Validator (mirrors ctypes_utils.Validator)
# =============================================================================


class FmhaValidator:
    """Validates FMHA GPU output against a reference.

    Usage:
        validator = FmhaValidator(rtol=1e-2, atol=1e-2)
        ok, max_abs, max_rel = validator.check(gpu_output, cpu_reference)
    """

    def __init__(self, rtol: float = 1e-2, atol: float = 1e-2):
        self.rtol = rtol
        self.atol = atol

    def check(
        self, output: np.ndarray, reference: np.ndarray
    ) -> Tuple[bool, float, float]:
        """Check output against reference.

        Returns:
            (is_valid, max_abs_error, max_rel_error)
        """
        out_f32 = output.astype(np.float32)
        ref_f32 = reference.astype(np.float32)
        diff = np.abs(out_f32 - ref_f32)
        max_abs = float(diff.max())
        max_rel = float((diff / (np.abs(ref_f32) + 1e-6)).max())
        ok = bool(np.allclose(out_f32, ref_f32, atol=self.atol, rtol=self.rtol))
        return ok, max_abs, max_rel


# =============================================================================
# KernelSpec + spec_to_config (mirrors ctypes_utils.KernelSpec)
# =============================================================================


@dataclass
class FmhaKernelSpec:
    """High-level kernel specification for easy declaration.

    Mirrors GEMM's KernelSpec: specify name + key dimensions, get a
    full FmhaKernelConfig via spec_to_config().
    """

    name: str
    hdim: int = 128
    pipeline: str = "qr_async"
    # Stage 0 tile (Q*K^T)
    tile_m0: int = 128
    tile_n0: int = 128
    tile_k0: int = 32


def spec_to_config(
    spec: FmhaKernelSpec, dtype: str = "fp16", arch: str = "gfx950"
) -> FmhaKernelConfig:
    """Convert a high-level FmhaKernelSpec to a full FmhaKernelConfig."""
    hdim = spec.hdim
    return FmhaKernelConfig(
        data_type=dtype,
        hdim_q=hdim,
        hdim_v=hdim,
        pipeline=spec.pipeline,
        tile_m0=spec.tile_m0,
        tile_n0=spec.tile_n0,
        tile_k0=spec.tile_k0,
        tile_n1=hdim,
        tile_k1=spec.tile_k0,
        tile_k0max=hdim,
        gfx_arch=arch,
    )


# =============================================================================
# Split-K heuristic (from fmhaarch.md Section 9.5)
# =============================================================================


def num_splits_heuristic_ck(
    batch: int,
    nheads: int,
    seqlen_q: int,
    tile_m0: int = 128,
    num_cus: int = 304,
    min_util_rate: float = 0.85,
) -> int:
    """Recommend num_splits for split-KV, matching CK's heuristic.

    Args:
        batch: batch size
        nheads: number of Q heads
        seqlen_q: query sequence length
        tile_m0: tile size in seqlen_q dimension
        num_cus: number of compute units on GPU (gfx950: 304)
        min_util_rate: minimum CU utilization threshold

    Returns:
        Recommended num_splits (1 means no split)
    """
    import math

    m_blocks = math.ceil(seqlen_q / tile_m0) if tile_m0 > 0 else 1
    batch_nheads_mblocks = batch * nheads * m_blocks

    if batch_nheads_mblocks >= num_cus * min_util_rate:
        return 1

    for splits in [2, 4, 8, 16, 32]:
        if batch_nheads_mblocks * splits >= num_cus * min_util_rate:
            return splits

    return 1
