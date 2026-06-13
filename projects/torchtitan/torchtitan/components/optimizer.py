# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
import json
import math
import os
import re
import time
from collections import defaultdict
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, Generic, Literal, overload, TypeVar

import torch
import torch.distributed.tensor
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl
from torch.distributed.checkpoint.state_dict import (
    get_optimizer_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.tensor import DTensor, Replicate
from torch.optim import Optimizer
from torchtitan.config import Configurable
from torchtitan.distributed import ParallelDims
from torchtitan.tools.logging import logger

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - Triton is optional on non-ROCm hosts.
    triton = None
    tl = None

__all__ = [
    "Muon",
    "OptimizersContainer",
    "OptimizersInBackwardContainer",
    "ParamGroupConfig",
    "register_moe_load_balancing_hook",
]


def _optimizer_env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _optimizer_profile_path() -> str | None:
    path = (
        os.environ.get("TORCHTITAN_OPTIMIZER_PROFILE_PATH")
        or os.environ.get("CANARY_OPTIMIZER_PROFILE_PATH")
        or ""
    ).strip()
    if not path:
        return None
    rank = os.environ.get("RANK", "unknown")
    local_rank = os.environ.get("LOCAL_RANK", "unknown")
    pid = os.getpid()
    path = path.format(rank=rank, local_rank=local_rank, pid=pid)
    if path.endswith(".jsonl"):
        return path
    return os.path.join(path, f"optimizer_profile_rank{rank}_local{local_rank}_pid{pid}.jsonl")


def _optimizer_to_local_tensor(t: torch.Tensor) -> torch.Tensor:
    if isinstance(t, DTensor):
        return t.to_local()
    return t


def _tensor_nbytes(t: torch.Tensor) -> int:
    local = _optimizer_to_local_tensor(t)
    return int(local.numel() * local.element_size())


def _walk_tensor_state(value: Any, seen: set[int]) -> int:
    if isinstance(value, torch.Tensor):
        key = id(value)
        if key in seen:
            return 0
        seen.add(key)
        return _tensor_nbytes(value)
    if isinstance(value, dict):
        return sum(_walk_tensor_state(v, seen) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_walk_tensor_state(v, seen) for v in value)
    return 0


def _optimizer_state_nbytes(optimizer: Optimizer) -> int:
    seen: set[int] = set()
    total = _walk_tensor_state(getattr(optimizer, "state", {}), seen)
    native_muon = getattr(optimizer, "_native_muon", None)
    if native_muon is not None:
        total += _walk_tensor_state(getattr(native_muon, "state", {}), seen)
    return int(total)


def _optimizer_param_nbytes(optimizer: Optimizer) -> int:
    seen: set[int] = set()
    total = 0
    for group in optimizer.param_groups:
        for param in group["params"]:
            key = id(param)
            if key in seen:
                continue
            seen.add(key)
            total += _tensor_nbytes(param)
    return int(total)


def _optimizer_grad_nbytes(optimizer: Optimizer) -> int:
    seen: set[int] = set()
    total = 0
    for group in optimizer.param_groups:
        for param in group["params"]:
            grad = param.grad
            if grad is None:
                continue
            key = id(grad)
            if key in seen:
                continue
            seen.add(key)
            total += _tensor_nbytes(grad)
    return int(total)


def _optimizer_memory_snapshot() -> dict[str, int]:
    if not torch.cuda.is_available():
        return {}
    return {
        "cuda_allocated_bytes": int(torch.cuda.memory_allocated()),
        "cuda_reserved_bytes": int(torch.cuda.memory_reserved()),
        "cuda_max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "cuda_max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def _append_optimizer_profile(record: dict[str, Any]) -> None:
    path = _optimizer_profile_path()
    if path is None:
        return
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def _profiled_optimizer_step(
    optimizer: Optimizer,
    *,
    context: dict[str, Any] | None = None,
) -> None:
    profile_path = _optimizer_profile_path()
    if profile_path is None:
        optimizer.step()
        return

    reset_peak = _optimizer_env_flag("TORCHTITAN_OPTIMIZER_PROFILE_RESET_PEAK")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        if reset_peak:
            torch.cuda.reset_peak_memory_stats()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    else:
        start_event = None
        end_event = None
    before_mem = _optimizer_memory_snapshot()
    before_state_bytes = _optimizer_state_nbytes(optimizer)
    start_wall = time.perf_counter()
    optimizer.step()
    wall_ms = (time.perf_counter() - start_wall) * 1000.0
    if torch.cuda.is_available():
        assert start_event is not None and end_event is not None
        end_event.record()
        end_event.synchronize()
        gpu_ms = float(start_event.elapsed_time(end_event))
    else:
        gpu_ms = None
    after_mem = _optimizer_memory_snapshot()
    after_state_bytes = _optimizer_state_nbytes(optimizer)
    native_muon = getattr(optimizer, "_native_muon", None)
    record = {
        "kind": "optimizer_step",
        "optimizer_type": type(optimizer).__name__,
        "native_optimizer_type": (
            type(native_muon).__name__ if native_muon is not None else None
        ),
        "rank": int(os.environ.get("RANK", "-1")),
        "local_rank": int(os.environ.get("LOCAL_RANK", "-1")),
        "pid": os.getpid(),
        "wall_ms": wall_ms,
        "gpu_ms": gpu_ms,
        "param_bytes": _optimizer_param_nbytes(optimizer),
        "grad_bytes": _optimizer_grad_nbytes(optimizer),
        "state_bytes_before": before_state_bytes,
        "state_bytes_after": after_state_bytes,
        "state_bytes_delta": after_state_bytes - before_state_bytes,
        "memory_before": before_mem,
        "memory_after": after_mem,
    }
    if context:
        record.update(context)
    if hasattr(optimizer, "_batched_view_mode"):
        record.update(
            {
                "muon_batched_view_mode": optimizer._batched_view_mode,
                "muon_source_param_count": sum(
                    len(group["params"]) for group in optimizer.param_groups
                ),
                "muon_batched_source_param_count": optimizer._num_batched_source_params,
                "muon_dtensor_local_source_param_count": (
                    optimizer._num_dtensor_local_source_params
                ),
                "muon_native_rank2_param_count": len(optimizer._flat_params),
            }
        )
    _append_optimizer_profile(record)


def _optimizer_profile_in_backward_enabled() -> bool:
    return _optimizer_env_flag("TORCHTITAN_OPTIMIZER_PROFILE_IN_BACKWARD")


def _optimizer_dtype_from_name(raw: str, *, default: torch.dtype) -> torch.dtype:
    value = raw.strip().lower()
    if value == "":
        return default
    if value in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if value in {"float32", "fp32"}:
        return torch.float32
    if value in {"param", "parameter"}:
        raise ValueError("'param' dtype must be resolved with a concrete parameter")
    raise ValueError(f"Unsupported optimizer dtype {raw!r}")


if triton is not None and tl is not None:

    @triton.jit
    def _muon_source_update_kernel(
        grad,
        buf,
        out,
        total: tl.constexpr,
        rows: tl.constexpr,
        cols: tl.constexpr,
        grad_stride_b: tl.constexpr,
        grad_stride_r: tl.constexpr,
        grad_stride_c: tl.constexpr,
        buf_stride_b: tl.constexpr,
        buf_stride_r: tl.constexpr,
        buf_stride_c: tl.constexpr,
        momentum: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < total
        col = offsets % cols
        tmp = offsets // cols
        row = tmp % rows
        batch = tmp // rows
        grad_vals = tl.load(
            grad
            + batch * grad_stride_b
            + row * grad_stride_r
            + col * grad_stride_c,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        buf_vals = tl.load(
            buf
            + batch * buf_stride_b
            + row * buf_stride_r
            + col * buf_stride_c,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        tl.store(out + offsets, grad_vals * (1.0 - momentum) + buf_vals * momentum, mask=mask)

    @triton.jit
    def _muon_momentum_source_update_kernel(
        grad,
        buf,
        out,
        total: tl.constexpr,
        rows: tl.constexpr,
        cols: tl.constexpr,
        grad_stride_b: tl.constexpr,
        grad_stride_r: tl.constexpr,
        grad_stride_c: tl.constexpr,
        buf_stride_b: tl.constexpr,
        buf_stride_r: tl.constexpr,
        buf_stride_c: tl.constexpr,
        momentum: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < total
        col = offsets % cols
        tmp = offsets // cols
        row = tmp % rows
        batch = tmp // rows
        grad_offsets = (
            batch * grad_stride_b
            + row * grad_stride_r
            + col * grad_stride_c
        )
        buf_offsets = (
            batch * buf_stride_b
            + row * buf_stride_r
            + col * buf_stride_c
        )
        grad_vals = tl.load(grad + grad_offsets, mask=mask, other=0.0).to(tl.float32)
        old_buf = tl.load(buf + buf_offsets, mask=mask, other=0.0).to(tl.float32)
        new_buf = old_buf * momentum + grad_vals * (1.0 - momentum)
        source_update = grad_vals * (1.0 - momentum) + new_buf * momentum
        tl.store(buf + buf_offsets, new_buf, mask=mask)
        tl.store(out + offsets, source_update, mask=mask)

    @triton.jit
    def _muon_param_add_kernel(
        param,
        update,
        total: tl.constexpr,
        rows: tl.constexpr,
        cols: tl.constexpr,
        param_stride_b: tl.constexpr,
        param_stride_r: tl.constexpr,
        param_stride_c: tl.constexpr,
        update_stride_b: tl.constexpr,
        update_stride_r: tl.constexpr,
        update_stride_c: tl.constexpr,
        alpha: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < total
        col = offsets % cols
        tmp = offsets // cols
        row = tmp % rows
        batch = tmp // rows
        param_offsets = (
            batch * param_stride_b
            + row * param_stride_r
            + col * param_stride_c
        )
        update_offsets = (
            batch * update_stride_b
            + row * update_stride_r
            + col * update_stride_c
        )
        param_vals = tl.load(param + param_offsets, mask=mask, other=0.0).to(tl.float32)
        update_vals = tl.load(update + update_offsets, mask=mask, other=0.0).to(tl.float32)
        tl.store(param + param_offsets, param_vals + alpha * update_vals, mask=mask)


def _muon_triton_source_update_(
    grad_source: torch.Tensor,
    buf_source: torch.Tensor,
    matrix: torch.Tensor,
    *,
    momentum: float,
) -> bool:
    if triton is None or tl is None or matrix.ndim != 3:
        return False
    if not matrix.is_cuda or not grad_source.is_cuda or not buf_source.is_cuda:
        return False
    if not matrix.is_contiguous():
        return False
    batch, rows, cols = matrix.shape
    total = int(batch * rows * cols)
    if total == 0:
        return True
    block = 256
    _muon_source_update_kernel[(triton.cdiv(total, block),)](
        grad_source,
        buf_source,
        matrix,
        total,
        rows,
        cols,
        grad_source.stride(0),
        grad_source.stride(1),
        grad_source.stride(2),
        buf_source.stride(0),
        buf_source.stride(1),
        buf_source.stride(2),
        float(momentum),
        BLOCK=block,
    )
    return True


def _muon_triton_momentum_source_update_(
    grad_source: torch.Tensor,
    buf_source: torch.Tensor,
    matrix: torch.Tensor,
    *,
    momentum: float,
) -> bool:
    if triton is None or tl is None or matrix.ndim != 3:
        return False
    if not matrix.is_cuda or not grad_source.is_cuda or not buf_source.is_cuda:
        return False
    if not matrix.is_contiguous():
        return False
    if grad_source.dtype != buf_source.dtype or matrix.dtype != grad_source.dtype:
        return False
    batch, rows, cols = matrix.shape
    if grad_source.shape != (batch, rows, cols):
        return False
    if buf_source.shape != (batch, rows, cols):
        return False
    total = int(batch * rows * cols)
    if total == 0:
        return True
    block = 256
    _muon_momentum_source_update_kernel[(triton.cdiv(total, block),)](
        grad_source,
        buf_source,
        matrix,
        total,
        rows,
        cols,
        grad_source.stride(0),
        grad_source.stride(1),
        grad_source.stride(2),
        buf_source.stride(0),
        buf_source.stride(1),
        buf_source.stride(2),
        float(momentum),
        BLOCK=block,
    )
    return True


def _muon_triton_param_add_(
    param_chunk: torch.Tensor,
    update_view: torch.Tensor,
    *,
    alpha: float,
) -> bool:
    if triton is None or tl is None or param_chunk.ndim != 3:
        return False
    if not param_chunk.is_cuda or not update_view.is_cuda:
        return False
    if param_chunk.shape != update_view.shape:
        return False
    batch, rows, cols = param_chunk.shape
    total = int(batch * rows * cols)
    if total == 0:
        return True
    block = 256
    _muon_param_add_kernel[(triton.cdiv(total, block),)](
        param_chunk,
        update_view,
        total,
        rows,
        cols,
        param_chunk.stride(0),
        param_chunk.stride(1),
        param_chunk.stride(2),
        update_view.stride(0),
        update_view.stride(1),
        update_view.stride(2),
        float(alpha),
        BLOCK=block,
    )
    return True


class Muon(Optimizer):
    """Slow PyTorch reference Muon optimizer for training bring-up.

    This optimizer is intentionally small and checkpointable. It is not the final
    distributed Muon path for FSDP-sharded matrices: with sharded parameters it
    orthogonalizes the local shard. Use it to validate parameter routing,
    optimizer state save/load, and short correctness canaries before replacing it
    with a distributed Muon update.
    """

    def __init__(
        self,
        params,
        *,
        lr: float = 3e-4,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.0,
        ns_steps: int = 5,
        eps: float = 1e-7,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if ns_steps < 1:
            raise ValueError(f"ns_steps must be >= 1, got {ns_steps}")
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
            eps=eps,
        )
        super().__init__(params, defaults)
        state_dtype_name = os.environ.get(
            "TORCHTITAN_OPTIMIZER_MUON_STATE_DTYPE",
            os.environ.get("CANARY_OPTIMIZER_MUON_STATE_DTYPE", "bfloat16"),
        )
        self._state_dtype_name = state_dtype_name.strip().lower()
        self._chunk_matrices = int(
            os.environ.get(
                "TORCHTITAN_OPTIMIZER_MUON_CHUNK_MATRICES",
                os.environ.get("CANARY_OPTIMIZER_MUON_CHUNK_MATRICES", "1"),
            )
        )
        if self._chunk_matrices <= 0:
            raise ValueError(
                "TORCHTITAN_OPTIMIZER_MUON_CHUNK_MATRICES must be positive, "
                f"got {self._chunk_matrices}"
            )

    def _state_dtype(self, local_param: torch.Tensor) -> torch.dtype:
        if self._state_dtype_name in {"param", "parameter"}:
            return local_param.dtype
        return _optimizer_dtype_from_name(self._state_dtype_name, default=torch.bfloat16)

    @staticmethod
    def _orthogonalize_chunk(
        matrix: torch.Tensor,
        *,
        ns_steps: int,
        eps: float,
    ) -> torch.Tensor:
        matrix = matrix / (matrix.norm(dim=(-2, -1), keepdim=True) + eps)

        # Quintic Newton-Schulz coefficients commonly used by Muon references.
        a, b, c = 3.4445, -4.7750, 2.0315
        for _ in range(ns_steps):
            gram = matrix @ matrix.transpose(-2, -1)
            matrix = a * matrix + b * (gram @ matrix) + c * (gram @ gram @ matrix)
        return matrix

    @classmethod
    def _orthogonalize(
        cls,
        update: torch.Tensor,
        ns_steps: int,
        eps: float,
        *,
        chunk_matrices: int,
    ) -> torch.Tensor:
        if update.ndim < 2:
            raise ValueError("Muon requires matrix-like parameters with ndim >= 2")

        original_shape = update.shape
        matrix = update.float().reshape(-1, original_shape[-2], original_shape[-1])
        transposed = matrix.size(-2) > matrix.size(-1)
        if transposed:
            matrix = matrix.transpose(-2, -1)

        if matrix.shape[0] <= chunk_matrices:
            matrix = cls._orthogonalize_chunk(matrix, ns_steps=ns_steps, eps=eps)
        else:
            output = torch.empty_like(matrix)
            for start in range(0, matrix.shape[0], chunk_matrices):
                stop = min(start + chunk_matrices, matrix.shape[0])
                output[start:stop] = cls._orthogonalize_chunk(
                    matrix[start:stop], ns_steps=ns_steps, eps=eps
                )
            matrix = output

        if transposed:
            matrix = matrix.transpose(-2, -1)
        return matrix.reshape(original_shape)

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            weight_decay = group["weight_decay"]
            ns_steps = group["ns_steps"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                local_param = _optimizer_to_local_tensor(p)
                local_grad = _optimizer_to_local_tensor(p.grad)
                if local_grad.is_sparse:
                    raise RuntimeError("Muon does not support sparse gradients")
                if local_param.ndim < 2:
                    raise RuntimeError(
                        "Muon param groups must contain only matrix-like params; "
                        f"got shape {tuple(local_param.shape)}"
                    )

                grad = local_grad.float()
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(
                        local_param,
                        dtype=self._state_dtype(local_param),
                        memory_format=torch.preserve_format,
                    )
                buf = state["momentum_buffer"].float()
                buf.mul_(momentum).add_(grad, alpha=1.0 - momentum)
                state["momentum_buffer"].copy_(
                    buf.to(dtype=state["momentum_buffer"].dtype)
                )
                update = grad.add(buf, alpha=momentum) if nesterov else buf
                update = self._orthogonalize(
                    update,
                    ns_steps,
                    eps,
                    chunk_matrices=self._chunk_matrices,
                ).to(dtype=local_param.dtype)

                if weight_decay != 0.0:
                    local_param.mul_(1.0 - lr * weight_decay)
                local_param.add_(update, alpha=-lr)

        return loss


class NativeMuonWithBatchedParams(Optimizer):
    """Native ``torch.optim.Muon`` bridge for batched expert matrices.

    PyTorch's native Muon currently accepts only rank-2 parameters in the
    retained ROCm image. TorchTitan MoE expert weights are batched rank-3
    tensors ``[E, out, in]``. This wrapper keeps the public optimizer groups on
    the original parameters, but gives native Muon rank-2 leaf views that share
    storage with the expert tensor. The default view is per expert matrix;
    flattening the whole expert bank is kept only as an opt-in diagnostic.
    """

    def __init__(self, params, **kwargs) -> None:
        native_muon = getattr(torch.optim, "Muon", None)
        if native_muon is None:
            raise RuntimeError("torch.optim.Muon is not available in this PyTorch build")

        defaults = dict(kwargs)
        super().__init__(params, defaults)

        self._batched_view_mode = (
            os.environ.get("TORCHTITAN_OPTIMIZER_MUON_BATCHED_VIEW_MODE")
            or os.environ.get("CANARY_OPTIMIZER_MUON_BATCHED_VIEW_MODE")
            or "per_expert"
        ).strip().lower()
        if self._batched_view_mode == "":
            self._batched_view_mode = "per_expert"
        if self._batched_view_mode not in {"flatten", "per_expert"}:
            raise ValueError(
                "TORCHTITAN_OPTIMIZER_MUON_BATCHED_VIEW_MODE must be "
                f"'flatten' or 'per_expert', got {self._batched_view_mode!r}"
            )

        self._flat_entries_by_param: dict[
            torch.Tensor, list[tuple[torch.Tensor, int | None, bool]]
        ] = {}
        self._flat_params: list[torch.Tensor] = []
        self._num_batched_source_params = 0
        self._num_dtensor_local_source_params = 0
        flat_groups: list[dict[str, Any]] = []
        for group in self.param_groups:
            flat_group = {k: v for k, v in group.items() if k != "params"}
            flat_group_params = []
            for param in group["params"]:
                if param.ndim < 2:
                    raise RuntimeError(
                        "Muon param groups must contain only matrix-like params; "
                        f"got shape {tuple(param.shape)}"
                    )
                if param.ndim == 2:
                    flat_entries = [(param, None, False)]
                else:
                    source_param = param
                    source_is_local = False
                    if self._batched_view_mode == "per_expert" and isinstance(
                        param, DTensor
                    ):
                        # Do not index the global DTensor. DTensor slicing may
                        # redistribute to replicated placement and all-gather
                        # the expert bank during optimizer construction. Native
                        # Muon should operate on this rank's local shard views.
                        source_param = param.to_local()
                        source_is_local = True
                        self._num_dtensor_local_source_params += 1
                    if not source_param.is_contiguous():
                        raise RuntimeError(
                            "Batched native Muon bridge requires contiguous "
                            f"parameters, got shape={tuple(source_param.shape)} "
                            f"stride={tuple(source_param.stride())}"
                        )
                    self._num_batched_source_params += 1
                    if self._batched_view_mode == "flatten":
                        flat_entries = [
                            (
                                nn.Parameter(
                                    source_param.view(-1, source_param.shape[-1]),
                                    requires_grad=param.requires_grad,
                                ),
                                None,
                                source_is_local,
                            )
                        ]
                    else:
                        matrix_bank = source_param.view(
                            -1, source_param.shape[-2], source_param.shape[-1]
                        )
                        flat_entries = [
                            (
                                nn.Parameter(
                                    matrix_bank[i],
                                    requires_grad=param.requires_grad,
                                ),
                                i,
                                source_is_local,
                            )
                            for i in range(matrix_bank.shape[0])
                        ]
                self._flat_entries_by_param[param] = flat_entries
                for flat_param, _, _ in flat_entries:
                    self._flat_params.append(flat_param)
                    flat_group_params.append(flat_param)
            flat_group["params"] = flat_group_params
            flat_groups.append(flat_group)

        self._native_muon = native_muon(flat_groups)
        logger.info(
            "NativeMuonWithBatchedParams initialized "
            f"(batched_view_mode={self._batched_view_mode}, "
            f"source_params={sum(len(g['params']) for g in self.param_groups)}, "
            f"batched_source_params={self._num_batched_source_params}, "
            f"dtensor_local_source_params={self._num_dtensor_local_source_params}, "
            f"native_rank2_params={len(self._flat_params)})"
        )

    def _sync_flat_param_groups(self) -> None:
        for original_group, flat_group in zip(
            self.param_groups, self._native_muon.param_groups
        ):
            for key, value in original_group.items():
                if key != "params":
                    flat_group[key] = value

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        self._sync_flat_param_groups()
        for group in self.param_groups:
            for param in group["params"]:
                flat_entries = self._flat_entries_by_param[param]
                if param.grad is None:
                    for flat_param, _, _ in flat_entries:
                        flat_param.grad = None
                elif param.ndim == 2:
                    flat_entries[0][0].grad = param.grad
                elif self._batched_view_mode == "flatten":
                    flat_entries[0][0].grad = param.grad.reshape_as(
                        flat_entries[0][0]
                    )
                else:
                    source_grad = _optimizer_to_local_tensor(param.grad)
                    grad_bank = source_grad.view(
                        -1, source_grad.shape[-2], source_grad.shape[-1]
                    )
                    for flat_param, matrix_idx, _ in flat_entries:
                        assert matrix_idx is not None
                        flat_param.grad = grad_bank[matrix_idx]
        return self._native_muon.step(closure)

    def zero_grad(self, set_to_none: bool = True) -> None:
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                if set_to_none:
                    param.grad = None
                else:
                    param.grad.detach_()
                    param.grad.zero_()
        for flat_param in self._flat_params:
            flat_param.grad = None

    def state_dict(self) -> dict[str, Any]:
        self._sync_flat_param_groups()
        return self._native_muon.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._native_muon.load_state_dict(state_dict)


class BatchedMuonExpertParams(Optimizer):
    """Batched Muon update for local expert-bank tensors.

    This is an experiment-only bridge for TorchTitan DSv4 expert tensors shaped
    ``[E, out, in]``. It keeps Muon state on the original rank-3 parameter and
    applies Newton-Schulz over a bounded batch of expert matrices, avoiding the
    native PyTorch wrapper's 1-Parameter-per-expert construction.
    """

    def __init__(
        self,
        params,
        *,
        lr: float = 3e-4,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.0,
        ns_steps: int = 5,
        eps: float = 1e-7,
    ) -> None:
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
            eps=eps,
        )
        super().__init__(params, defaults)
        self._chunk_matrices = int(
            os.environ.get(
                "TORCHTITAN_OPTIMIZER_MUON_CHUNK_MATRICES",
                os.environ.get("CANARY_OPTIMIZER_MUON_CHUNK_MATRICES", "32"),
            )
        )
        if self._chunk_matrices <= 0:
            raise ValueError(
                "TORCHTITAN_OPTIMIZER_MUON_CHUNK_MATRICES must be positive, "
                f"got {self._chunk_matrices}"
            )
        self._state_dtype_name = (
            os.environ.get(
                "TORCHTITAN_OPTIMIZER_MUON_STATE_DTYPE",
                os.environ.get("CANARY_OPTIMIZER_MUON_STATE_DTYPE", "bfloat16"),
            )
            .strip()
            .lower()
        )
        self._ns_compute_dtype_name = (
            os.environ.get(
                "TORCHTITAN_OPTIMIZER_MUON_NS_COMPUTE_DTYPE",
                os.environ.get("CANARY_OPTIMIZER_MUON_NS_COMPUTE_DTYPE", "bfloat16"),
            )
            .strip()
            .lower()
        )
        if self._ns_compute_dtype_name not in {"bfloat16", "bf16", "float32", "fp32"}:
            raise ValueError(
                "TORCHTITAN_OPTIMIZER_MUON_NS_COMPUTE_DTYPE must be 'bfloat16' "
                f"or 'float32', got {self._ns_compute_dtype_name!r}"
            )
        self._workspace_mode = (
            os.environ.get(
                "TORCHTITAN_OPTIMIZER_MUON_WORKSPACE_MODE",
                os.environ.get("CANARY_OPTIMIZER_MUON_WORKSPACE_MODE", "none"),
            )
            .strip()
            .lower()
        )
        if self._workspace_mode not in {"none", "prealloc"}:
            raise ValueError(
                "TORCHTITAN_OPTIMIZER_MUON_WORKSPACE_MODE must be 'none' "
                f"or 'prealloc', got {self._workspace_mode!r}"
            )
        self._norm_mode = (
            os.environ.get(
                "TORCHTITAN_OPTIMIZER_MUON_NORM_MODE",
                os.environ.get("CANARY_OPTIMIZER_MUON_NORM_MODE", "vector_norm"),
            )
            .strip()
            .lower()
        )
        if self._norm_mode not in {"vector_norm", "gram_trace"}:
            raise ValueError(
                "TORCHTITAN_OPTIMIZER_MUON_NORM_MODE must be 'vector_norm' "
                f"or 'gram_trace', got {self._norm_mode!r}"
            )
        self._elementwise_backend = (
            os.environ.get(
                "TORCHTITAN_OPTIMIZER_MUON_ELEMENTWISE_BACKEND",
                os.environ.get(
                    "CANARY_OPTIMIZER_MUON_ELEMENTWISE_BACKEND",
                    "torch",
                ),
            )
            .strip()
            .lower()
        )
        if self._elementwise_backend not in {"torch", "triton", "triton_fused"}:
            raise ValueError(
                "TORCHTITAN_OPTIMIZER_MUON_ELEMENTWISE_BACKEND must be "
                f"'torch', 'triton', or 'triton_fused', got "
                f"{self._elementwise_backend!r}"
            )
        self._workspace_cache: dict[
            tuple[torch.device, torch.dtype, torch.dtype, int, int, int],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        ] = {}
        self._muon_impl = "batched"

    def _state_dtype(self, local_param: torch.Tensor) -> torch.dtype:
        if self._state_dtype_name in {"param", "parameter"}:
            return local_param.dtype
        return _optimizer_dtype_from_name(self._state_dtype_name, default=torch.bfloat16)

    def _ns_compute_dtype(self) -> torch.dtype:
        if self._ns_compute_dtype_name in {"float32", "fp32"}:
            return torch.float32
        return torch.bfloat16

    @staticmethod
    def _adjust_lr(lr: float, param_shape: torch.Size) -> float:
        rows, cols = param_shape[-2], param_shape[-1]
        return lr * math.sqrt(max(1.0, rows / cols))

    @staticmethod
    def _native_style_ns(
        update: torch.Tensor,
        *,
        ns_steps: int,
        eps: float,
        compute_dtype: torch.dtype,
    ) -> torch.Tensor:
        if update.ndim != 3:
            raise ValueError(
                "BatchedMuonExpertParams expects a rank-3 matrix bank chunk, "
                f"got shape={tuple(update.shape)}"
            )
        a, b, c = 3.4445, -4.7750, 2.0315
        matrix = update.to(dtype=compute_dtype)
        transposed = matrix.size(-2) > matrix.size(-1)
        if transposed:
            matrix = matrix.transpose(-2, -1)

        # Native PyTorch Muon normalizes after casting to BF16. On this ROCm
        # image that path can FPE on real DSv4 gradients, so compute the norm in
        # FP32 and cast the normalized matrix back to the requested NS dtype.
        norm = matrix.float().norm(dim=(-2, -1), keepdim=True).clamp(min=eps)
        matrix = (matrix / norm).to(dtype=compute_dtype)
        for _ in range(ns_steps):
            gram = matrix @ matrix.transpose(-2, -1)
            gram_update = torch.add(
                gram.mul(b),
                gram @ gram,
                alpha=c,
            )
            matrix = torch.baddbmm(matrix, gram_update, matrix, beta=a)

        if transposed:
            matrix = matrix.transpose(-2, -1)
        return matrix.to(dtype=update.dtype)

    def _get_workspace(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        source_dtype: torch.dtype,
        chunk_matrices: int,
        rows: int,
        cols: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        key = (device, dtype, source_dtype, chunk_matrices, rows, cols)
        workspace = self._workspace_cache.get(key)
        if workspace is None:
            matrix = torch.empty(
                (chunk_matrices, rows, cols),
                device=device,
                dtype=dtype,
            )
            next_matrix = torch.empty_like(matrix)
            gram = torch.empty(
                (chunk_matrices, rows, rows),
                device=device,
                dtype=dtype,
            )
            gram_update = torch.empty_like(gram)
            source_update = (
                matrix
                if source_dtype == dtype
                else torch.empty(
                    (chunk_matrices, rows, cols),
                    device=device,
                    dtype=source_dtype,
                )
            )
            workspace = (matrix, next_matrix, gram, gram_update, source_update)
            self._workspace_cache[key] = workspace
        return workspace

    def _apply_native_style_ns_update_prealloc_(
        self,
        *,
        param_chunk: torch.Tensor,
        grad_chunk: torch.Tensor,
        buf_chunk: torch.Tensor,
        momentum: float,
        nesterov: bool,
        ns_steps: int,
        eps: float,
        compute_dtype: torch.dtype,
        adjusted_lr: float,
        buf_already_updated: bool = True,
    ) -> None:
        if grad_chunk.ndim != 3:
            raise ValueError(
                "BatchedMuonExpertParams expects a rank-3 matrix bank chunk, "
                f"got shape={tuple(grad_chunk.shape)}"
            )
        a, b, c = 3.4445, -4.7750, 2.0315
        batch = int(grad_chunk.shape[0])
        source_rows = int(grad_chunk.shape[-2])
        source_cols = int(grad_chunk.shape[-1])
        transposed = source_rows > source_cols
        rows = min(source_rows, source_cols)
        cols = max(source_rows, source_cols)
        (
            matrix_buf,
            next_buf,
            gram_buf,
            gram_update_buf,
            source_update_buf,
        ) = self._get_workspace(
            device=grad_chunk.device,
            dtype=compute_dtype,
            source_dtype=grad_chunk.dtype,
            chunk_matrices=self._chunk_matrices,
            rows=rows,
            cols=cols,
        )
        matrix = matrix_buf[:batch]
        next_matrix = next_buf[:batch]
        gram = gram_buf[:batch]
        gram_update = gram_update_buf[:batch]
        source_update = source_update_buf[:batch]
        source_update_needs_copy = source_update_buf is not matrix_buf

        if transposed:
            grad_source = grad_chunk.transpose(-2, -1)
            buf_source = buf_chunk.transpose(-2, -1)
        else:
            grad_source = grad_chunk
            buf_source = buf_chunk
        triton_elementwise = self._elementwise_backend in {"triton", "triton_fused"}
        triton_fused_source = self._elementwise_backend == "triton_fused"
        if nesterov:
            if (
                triton_fused_source
                and not buf_already_updated
                and _muon_triton_momentum_source_update_(
                    grad_source,
                    buf_source,
                    matrix,
                    momentum=momentum,
                )
            ):
                pass
            else:
                if not buf_already_updated:
                    buf_source.lerp_(grad_source, 1.0 - momentum)
                if triton_elementwise and _muon_triton_source_update_(
                    grad_source,
                    buf_source,
                    matrix,
                    momentum=momentum,
                ):
                    pass
                else:
                    source_update.copy_(grad_source)
                    source_update.lerp_(buf_source, momentum)
                    if source_update_needs_copy:
                        matrix.copy_(source_update)
        else:
            if not buf_already_updated:
                buf_source.lerp_(grad_source, 1.0 - momentum)
            matrix.copy_(buf_source)
        if matrix.dtype != compute_dtype:
            matrix = matrix.to(dtype=compute_dtype)

        first_gram_ready = False
        if self._norm_mode == "gram_trace":
            torch.bmm(matrix, matrix.transpose(-2, -1), out=gram)
            norm_sq = (
                gram.diagonal(dim1=-2, dim2=-1)
                .float()
                .sum(dim=-1, keepdim=True)
                .clamp_(min=eps * eps)
                .view(batch, 1, 1)
            )
            inv_norm = torch.rsqrt(norm_sq).to(dtype=compute_dtype)
            matrix.mul_(inv_norm)
            gram.mul_(inv_norm * inv_norm)
            first_gram_ready = True
        else:
            norm = torch.linalg.vector_norm(
                matrix,
                dim=(-2, -1),
                keepdim=True,
                dtype=torch.float32,
            ).clamp_(min=eps)
            matrix.div_(norm.to(dtype=compute_dtype))
        for _ in range(ns_steps):
            if first_gram_ready:
                first_gram_ready = False
            else:
                torch.bmm(matrix, matrix.transpose(-2, -1), out=gram)
            torch.bmm(gram, gram, out=gram_update)
            gram_update.mul_(c).add_(gram, alpha=b)
            torch.baddbmm(matrix, gram_update, matrix, beta=a, out=next_matrix)
            matrix, next_matrix = next_matrix, matrix

        update_view = matrix.transpose(-2, -1) if transposed else matrix
        if update_view.dtype != param_chunk.dtype:
            update_view = update_view.to(dtype=param_chunk.dtype)
        if not (
            triton_elementwise
            and _muon_triton_param_add_(
                param_chunk,
                update_view,
                alpha=-adjusted_lr,
            )
        ):
            param_chunk.add_(update_view, alpha=-adjusted_lr)

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        compute_dtype = self._ns_compute_dtype()
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            weight_decay = group["weight_decay"]
            ns_steps = group["ns_steps"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                local_param = _optimizer_to_local_tensor(p)
                local_grad = _optimizer_to_local_tensor(p.grad)
                if local_grad.is_sparse:
                    raise RuntimeError("Muon does not support sparse gradients")
                if local_param.ndim < 2:
                    raise RuntimeError(
                        "Muon param groups must contain only matrix-like params; "
                        f"got shape {tuple(local_param.shape)}"
                    )

                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(
                        local_param,
                        dtype=self._state_dtype(local_param),
                        memory_format=torch.preserve_format,
                    )
                buf = state["momentum_buffer"]
                if weight_decay != 0.0:
                    local_param.mul_(1.0 - lr * weight_decay)
                adjusted_lr = self._adjust_lr(lr, local_param.shape)

                param_bank = local_param.reshape(
                    -1, local_param.shape[-2], local_param.shape[-1]
                )
                grad_bank = local_grad.reshape(
                    -1, local_grad.shape[-2], local_grad.shape[-1]
                )
                buf_bank = buf.reshape(-1, buf.shape[-2], buf.shape[-1])
                for start in range(0, grad_bank.shape[0], self._chunk_matrices):
                    stop = min(start + self._chunk_matrices, grad_bank.shape[0])
                    grad_chunk = grad_bank[start:stop]
                    if (
                        self._workspace_mode == "prealloc"
                        and self._elementwise_backend == "triton_fused"
                        and buf.dtype == local_grad.dtype
                    ):
                        self._apply_native_style_ns_update_prealloc_(
                            param_chunk=param_bank[start:stop],
                            grad_chunk=grad_chunk,
                            buf_chunk=buf_bank[start:stop],
                            momentum=momentum,
                            nesterov=nesterov,
                            ns_steps=ns_steps,
                            eps=eps,
                            compute_dtype=compute_dtype,
                            adjusted_lr=adjusted_lr,
                            buf_already_updated=False,
                        )
                        continue
                    if buf.dtype != local_grad.dtype:
                        buf_chunk = buf_bank[start:stop].to(dtype=local_grad.dtype)
                        buf_chunk.lerp_(grad_chunk, 1.0 - momentum)
                        buf_bank[start:stop].copy_(buf_chunk.to(dtype=buf.dtype))
                    else:
                        buf_chunk = buf_bank[start:stop]
                        buf_chunk.lerp_(grad_chunk, 1.0 - momentum)
                    if (
                        self._workspace_mode == "prealloc"
                        and buf_chunk.dtype == local_grad.dtype
                    ):
                        self._apply_native_style_ns_update_prealloc_(
                            param_chunk=param_bank[start:stop],
                            grad_chunk=grad_chunk,
                            buf_chunk=buf_chunk,
                            momentum=momentum,
                            nesterov=nesterov,
                            ns_steps=ns_steps,
                            eps=eps,
                            compute_dtype=compute_dtype,
                            adjusted_lr=adjusted_lr,
                            buf_already_updated=True,
                        )
                        continue
                    update_chunk = (
                        grad_chunk.lerp(buf_chunk, momentum) if nesterov else buf_chunk
                    )
                    update_chunk = self._native_style_ns(
                        update_chunk,
                        ns_steps=ns_steps,
                        eps=eps,
                        compute_dtype=compute_dtype,
                    )
                    param_bank[start:stop].add_(
                        update_chunk.to(dtype=local_param.dtype),
                        alpha=-adjusted_lr,
                    )

        return loss


@dataclass(kw_only=True, slots=True)
class ParamGroupConfig:
    """Configuration for a parameter group with custom optimizer settings.

    Parameters matching the regex pattern will use lr and weight_decay values
    derived by multiplying the global optimizer values with the specified multipliers.
    """

    pattern: str
    """Regex pattern matched against parameter fully qualified names (FQNs).
    E.g. '.*bias$', '.*norm.*', '.*\\.embed_tokens\\..*'"""

    lr_multiplier: float = 1.0
    """Multiplied with the global optimizer lr to get this group's lr."""

    weight_decay_multiplier: float = 1.0
    """Multiplied with the global optimizer weight_decay to get this group's weight_decay."""

    beta1: float | None = None
    beta2: float | None = None
    """Override betas for this group. None means use the global optimizer betas.
    Each can be overridden independently."""

    optimizer_name: str | None = None
    """Optional optimizer class for this group. If omitted, uses Config.name."""

    rank: int | None = None
    """Optional tensor rank filter. If set, pattern and rank must both match."""


T = TypeVar("T", bound=Optimizer)


class OptimizersContainer(Optimizer, Stateful, Configurable, Generic[T]):
    """A container for multiple optimizers.

    This class is used to wrap multiple optimizers into a single object that can be
    used to reduce the complexity of the training loop. This mimics the behavior of
    ``torch.optim.Optimizer``. This class supports Adam, AdamW, and a slow
    reference Muon implementation for bring-up canaries.

    **Note**
    Users who want to customize the optimizer behavior can inherit from this class and
    extend the functionality as needed. The following methods must follow the same signature
    as ``torch.optim.Optimizer`` class: ``step()``, ``zero_grad()``, ``state_dict()``,
    ``load_state_dict()``.

    **Limitations**
    This class assumes that all the optimizers are the same type and have the same
    configurations. With this assumption, TorchTitan can support lr scheduler resharding
    (e.g., loading a checkpoint with a different number of GPUs and/or different
    parallelization strategy). Note that ``get_optimizer_state_dict`` already enables the
    resharding for the optimizer state but not for the lr scheduler state, hence the limitation.

    Args:
        model_parts (List[nn.Module]): List of model parts to be optimized.
        optimizer_kwargs (Dict[str, Any]): Keyword arguments for the optimizers.
        name (str): Name of the optimizers.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        name: str = "AdamW"
        """Optimizer to use"""

        lr: float = 8e-4
        """Learning rate to use"""

        beta1: float = 0.9
        beta2: float = 0.95
        """Exponential moving average hyperparameters to use"""

        eps: float = 1e-8
        """Epsilon value to use"""

        weight_decay: float = 0.1
        """Weight decay to use"""

        muon_momentum: float = 0.95
        """Muon momentum value used when name or a param group selects Muon."""

        muon_nesterov: bool = True
        """Whether Muon uses the Nesterov-style update."""

        muon_ns_steps: int = 5
        """Newton-Schulz iterations used by the reference Muon implementation."""

        muon_eps: float = 1e-7
        """Numerical epsilon used by the reference Muon orthogonalization."""

        implementation: Literal[
            "for-loop",
            "foreach",
            "fused",
            "fused_opt_states_bf16",
            "swap_opt_states_cpu",
        ] = "fused"
        """
        Specify which optimizer implementation to use:
        - 'fused': Use fused implementation (CUDA only) for best performance.
        - 'foreach': Use some horizontal fusion of tensors for better performance.
        - 'for-loop': Use the default implementation for the optimizer (slowest).
        - 'fused_opt_states_bf16': Like 'fused', but initialize Adam/AdamW
          momentum and variance in bfloat16 via a step pre-hook so the fused
          CUDA kernel uses its mixed-precision path (fp32 params + bf16 states).
          Only supported for Adam/AdamW with OptimizersContainer (not
          OptimizersInBackwardContainer). See docs/bf16_optimizer_states.md.
        - 'swap_opt_states_cpu': Keep Adam/AdamW exp_avg and exp_avg_sq on CPU
          and swap chunked slices to the device only during optimizer.step().
          This is the portable AMD analogue of Ascend SwapOptimizer, not the
          NPU-specific virtual swapped-memory allocator.
          Set TORCHTITAN_OPTIMIZER_SWAP_UPDATE_MODE to select chunked,
          torch_param, fused_param, or foreach_param update style.
        - more info: https://pytorch.org/docs/stable/optim.html
        """

        param_groups: list[ParamGroupConfig] = field(default_factory=list)
        """Optional per-parameter-group overrides. Each entry specifies a regex
        pattern matching parameter FQNs and multipliers for lr and weight_decay.
        Parameters not matching any pattern use the global defaults.
        Patterns are checked in order; first match wins."""

        def __post_init__(self):
            if self.implementation in (
                "fused_opt_states_bf16",
                "swap_opt_states_cpu",
            ):
                optimizer_names = {self.name}
                optimizer_names.update(
                    pg.optimizer_name for pg in self.param_groups if pg.optimizer_name
                )
                unsupported = optimizer_names - {"Adam", "AdamW"}
                if unsupported:
                    raise ValueError(
                        f"implementation='{self.implementation}' is only supported "
                        f"for Adam/AdamW, got {sorted(unsupported)}"
                    )

    optimizers: list[T]
    model_parts: list[nn.Module]

    @staticmethod
    def _resolve_optimizer_cls(name: str) -> type:
        if name == "Muon":
            muon_impl = os.environ.get(
                "TORCHTITAN_OPTIMIZER_MUON_IMPL",
                os.environ.get("CANARY_OPTIMIZER_MUON_IMPL", ""),
            ).strip().lower()
            if muon_impl == "":
                muon_impl = (
                    "batched"
                    if _optimizer_env_flag(
                        "TORCHTITAN_DSV4_EXPERTS_ON_MUON",
                        _optimizer_env_flag("CANARY_DSV4_EXPERTS_ON_MUON", False),
                    )
                    else "native"
                )
            if muon_impl not in {"native", "reference", "batched"}:
                raise ValueError(
                    "TORCHTITAN_OPTIMIZER_MUON_IMPL must be 'native', "
                    f"'reference', or 'batched', got {muon_impl!r}"
                )
            if muon_impl == "batched":
                return BatchedMuonExpertParams
            if muon_impl == "native":
                if hasattr(torch.optim, "Muon"):
                    return NativeMuonWithBatchedParams
                logger.warning(
                    "TORCHTITAN_OPTIMIZER_MUON_IMPL=native requested, but "
                    "torch.optim.Muon is unavailable; falling back to reference Muon."
                )
            return Muon
        optimizer_classes = {
            "Adam": torch.optim.Adam,
            "AdamW": torch.optim.AdamW,
        }
        if name not in optimizer_classes:
            raise NotImplementedError(f"Optimizer {name} not added.")
        return optimizer_classes[name]

    @staticmethod
    def _build_optimizer_kwargs(
        config: Config, optimizer_name: str | None = None
    ) -> dict[str, Any]:
        optimizer_name = optimizer_name or config.name
        if optimizer_name == "Muon":
            return {
                "lr": config.lr,
                "momentum": config.muon_momentum,
                "nesterov": config.muon_nesterov,
                "weight_decay": config.weight_decay,
                "ns_steps": config.muon_ns_steps,
                "eps": config.muon_eps,
            }
        if optimizer_name not in ("Adam", "AdamW"):
            raise NotImplementedError(f"Optimizer {optimizer_name} not added.")
        assert config.implementation in [
            "fused",
            "foreach",
            "for-loop",
            "fused_opt_states_bf16",
            "swap_opt_states_cpu",
        ]
        fused = config.implementation in ("fused", "fused_opt_states_bf16")
        return {
            "lr": config.lr,
            "betas": (config.beta1, config.beta2),
            "eps": config.eps,
            "weight_decay": config.weight_decay,
            "fused": fused,
            "foreach": config.implementation == "foreach",
        }

    @staticmethod
    def _apply_param_group_overrides(
        group_kwargs: dict[str, Any],
        param_group_config: ParamGroupConfig,
    ) -> dict[str, Any]:
        group_kwargs = {**group_kwargs}
        group_kwargs["lr"] = group_kwargs["lr"] * param_group_config.lr_multiplier
        group_kwargs["weight_decay"] = (
            group_kwargs["weight_decay"] * param_group_config.weight_decay_multiplier
        )
        if "betas" in group_kwargs and (
            param_group_config.beta1 is not None
            or param_group_config.beta2 is not None
        ):
            default_beta1, default_beta2 = group_kwargs["betas"]
            group_kwargs["betas"] = (
                param_group_config.beta1
                if param_group_config.beta1 is not None
                else default_beta1,
                param_group_config.beta2
                if param_group_config.beta2 is not None
                else default_beta2,
            )
        elif (
            param_group_config.beta1 is not None
            or param_group_config.beta2 is not None
        ):
            logger.warning(
                "Ignoring beta override for non-Adam optimizer group "
                f"'{param_group_config.optimizer_name}'"
            )
        return group_kwargs

    @staticmethod
    def _build_param_groups(
        model: nn.Module,
        config: Config,
        default_kwargs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Build PyTorch param groups from model parameters and config.

        Each parameter is assigned to the first matching ParamGroupConfig pattern,
        or to the default group if no pattern matches. Returns a list of dicts
        with "params" key and optimizer kwargs, suitable for passing to an optimizer.
        """
        if not config.param_groups:
            params = [p for p in model.parameters() if p.requires_grad]
            return [{"params": params, **default_kwargs}]

        compiled_patterns = [re.compile(pg.pattern) for pg in config.param_groups]

        # group_index -> list of params; None means default group
        grouped_params: dict[int | None, list[nn.Parameter]] = defaultdict(list)

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            matched_index = None
            for i, pat in enumerate(compiled_patterns):
                pg = config.param_groups[i]
                if pat.search(name) and (pg.rank is None or param.ndim == pg.rank):
                    matched_index = i
                    break
            grouped_params[matched_index].append(param)

        # Warn for patterns that matched nothing
        for i, pg in enumerate(config.param_groups):
            if i not in grouped_params:
                logger.warning(
                    f"Optimizer param_groups pattern '{pg.pattern}' "
                    f"matched no parameters"
                )

        result = []
        # Default group first (unmatched params)
        if None in grouped_params:
            result.append(
                {
                    "params": grouped_params[None],
                    "_optimizer_name": config.name,
                    **default_kwargs,
                }
            )

        # Then each matched group in pattern order
        for i, pg in enumerate(config.param_groups):
            if i not in grouped_params:
                continue
            optimizer_name = pg.optimizer_name or config.name
            group_kwargs = OptimizersContainer._build_optimizer_kwargs(
                config, optimizer_name=optimizer_name
            )
            group_kwargs = OptimizersContainer._apply_param_group_overrides(
                group_kwargs, pg
            )
            result.append(
                {
                    "params": grouped_params[i],
                    "_optimizer_name": optimizer_name,
                    **group_kwargs,
                }
            )

        return result

    def __init__(self, config: Config, *, model_parts: list[nn.Module]) -> None:
        optimizer_kwargs = self._build_optimizer_kwargs(config)
        all_params = []
        self.optimizers = []
        self.model_parts = model_parts
        self.optimizer_model_parts = []
        self._use_native_optimizer_state_dict = False
        self._swap_optimizer_states_cpu = config.implementation == "swap_opt_states_cpu"
        self._swap_chunk_mib = self._resolve_swap_chunk_mib()
        self._swap_state_dtype_name = os.environ.get(
            "TORCHTITAN_OPTIMIZER_SWAP_STATE_DTYPE", "float32"
        ).lower()
        self._swap_pin_memory_setting = os.environ.get(
            "TORCHTITAN_OPTIMIZER_SWAP_PIN_MEMORY", "auto"
        ).lower()
        self._swap_update_mode = os.environ.get(
            "TORCHTITAN_OPTIMIZER_SWAP_UPDATE_MODE", "chunked"
        ).lower()
        if self._swap_update_mode not in (
            "chunked",
            "torch_param",
            "fused_param",
            "foreach_param",
        ):
            raise ValueError(
                "TORCHTITAN_OPTIMIZER_SWAP_UPDATE_MODE must be one of "
                "'chunked', 'torch_param', 'fused_param', or 'foreach_param', "
                f"got {self._swap_update_mode!r}"
            )
        self._swap_foreach_max_params = self._resolve_swap_foreach_max_params()
        for model in self.model_parts:
            param_groups = self._build_param_groups(model, config, optimizer_kwargs)
            grouped_param_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for group in param_groups:
                optimizer_name = group.pop("_optimizer_name", config.name)
                grouped_param_groups[optimizer_name].append(group)
                all_params.extend(group["params"])
            for optimizer_name, optimizer_param_groups in grouped_param_groups.items():
                optimizer_cls = self._resolve_optimizer_cls(optimizer_name)
                self.optimizers.append(optimizer_cls(optimizer_param_groups))
                self.optimizer_model_parts.append(model)
        optimizer_types = {type(optimizer) for optimizer in self.optimizers}
        self._use_native_optimizer_state_dict = (
            self._swap_optimizer_states_cpu
            or
            len(self.optimizers) != len(self.model_parts)
            or len(optimizer_types) > 1
            or any(
                not isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW))
                for optimizer in self.optimizers
            )
        )
        if config.implementation == "fused_opt_states_bf16":
            self._register_bf16_optimizer_state_hook()
        if self._swap_optimizer_states_cpu:
            logger.info(
                "Using CPU-swapped Adam optimizer states "
                f"(chunk={self._swap_chunk_mib} MiB, "
                f"state_dtype={self._swap_state_dtype_name}, "
                f"pin_memory={self._swap_pin_memory_setting}, "
                f"update_mode={self._swap_update_mode})"
            )
        self._validate_length(len(self.optimizer_model_parts))
        self._post_init(all_params, optimizer_kwargs)

    def __iter__(self) -> Iterator[T]:
        return iter(self.optimizers)

    def __len__(self) -> int:
        return len(self.optimizers)

    @overload
    def step(self, closure: None = None) -> None:
        ...

    @overload
    def step(self, closure: Callable[[], float]) -> float:
        ...

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        assert closure is None, "OptimizersContainer does not support closures"
        if self._swap_optimizer_states_cpu:
            for optimizer in self.optimizers:
                self._step_cpu_swap_adam(optimizer)
            return None
        for optimizer in self.optimizers:
            _profiled_optimizer_step(
                optimizer,
                context={
                    "profile_context": "container_step",
                    "container_type": type(self).__name__,
                },
            )
        return None

    def zero_grad(self, set_to_none: bool = True) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    @staticmethod
    def _resolve_swap_chunk_mib() -> int:
        raw = os.environ.get("TORCHTITAN_OPTIMIZER_SWAP_CHUNK_MIB", "32")
        try:
            chunk_mib = int(raw)
        except ValueError as exc:
            raise ValueError(
                "TORCHTITAN_OPTIMIZER_SWAP_CHUNK_MIB must be an integer, "
                f"got {raw!r}"
            ) from exc
        if chunk_mib <= 0:
            raise ValueError(
                "TORCHTITAN_OPTIMIZER_SWAP_CHUNK_MIB must be positive, "
                f"got {chunk_mib}"
            )
        return chunk_mib

    @staticmethod
    def _resolve_swap_foreach_max_params() -> int:
        raw = os.environ.get("TORCHTITAN_OPTIMIZER_SWAP_FOREACH_MAX_PARAMS", "4")
        try:
            max_params = int(raw)
        except ValueError as exc:
            raise ValueError(
                "TORCHTITAN_OPTIMIZER_SWAP_FOREACH_MAX_PARAMS must be an integer, "
                f"got {raw!r}"
            ) from exc
        if max_params <= 0:
            raise ValueError(
                "TORCHTITAN_OPTIMIZER_SWAP_FOREACH_MAX_PARAMS must be positive, "
                f"got {max_params}"
            )
        return max_params

    @staticmethod
    def _is_dtensor(tensor: Any) -> bool:
        dtensor_cls = getattr(torch.distributed.tensor, "DTensor", None)
        return dtensor_cls is not None and isinstance(tensor, dtensor_cls)

    @classmethod
    def _local_tensor(cls, tensor: torch.Tensor) -> torch.Tensor:
        if cls._is_dtensor(tensor):
            return tensor.to_local()
        return tensor

    def _swap_state_dtype(self, local_param: torch.Tensor) -> torch.dtype:
        if self._swap_state_dtype_name in ("param", "parameter"):
            return local_param.dtype
        if self._swap_state_dtype_name in ("float32", "fp32"):
            return torch.float32
        if self._swap_state_dtype_name in ("bfloat16", "bf16"):
            return torch.bfloat16
        raise ValueError(
            "TORCHTITAN_OPTIMIZER_SWAP_STATE_DTYPE must be one of "
            "'param', 'float32', or 'bfloat16', "
            f"got {self._swap_state_dtype_name!r}"
        )

    def _swap_pin_memory(self, local_param: torch.Tensor) -> bool:
        if self._swap_pin_memory_setting in ("0", "false", "no", "off"):
            return False
        if self._swap_pin_memory_setting in ("1", "true", "yes", "on"):
            return True
        if self._swap_pin_memory_setting == "auto":
            return local_param.device.type == "cuda" and torch.cuda.is_available()
        raise ValueError(
            "TORCHTITAN_OPTIMIZER_SWAP_PIN_MEMORY must be one of "
            "'auto', 'true', or 'false', "
            f"got {self._swap_pin_memory_setting!r}"
        )

    def _swap_chunk_numel(self, dtype: torch.dtype) -> int:
        element_size = torch.empty((), dtype=dtype).element_size()
        return max(1, self._swap_chunk_mib * 1024 * 1024 // element_size)

    @staticmethod
    def _cpu_zeros_like_local(
        local_param: torch.Tensor,
        dtype: torch.dtype,
        pin_memory: bool,
    ) -> torch.Tensor:
        try:
            tensor = torch.empty_strided(
                tuple(local_param.size()),
                tuple(local_param.stride()),
                dtype=dtype,
                layout=local_param.layout,
                device="cpu",
                pin_memory=pin_memory,
            )
        except RuntimeError:
            if not pin_memory:
                raise
            logger.warning(
                "Pinned CPU optimizer-state allocation failed; falling back to "
                "ordinary CPU memory for this state tensor."
            )
            tensor = torch.empty_strided(
                tuple(local_param.size()),
                tuple(local_param.stride()),
                dtype=dtype,
                layout=local_param.layout,
                device="cpu",
            )
        tensor.zero_()
        return tensor

    def _init_cpu_swap_adam_state(
        self,
        optimizer: Optimizer,
        group: dict[str, Any],
        param: torch.Tensor,
    ) -> dict[str, Any]:
        local_param = self._local_tensor(param)
        state = optimizer.state[param]
        if "step" not in state:
            state["step"] = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        elif torch.is_tensor(state["step"]):
            state["step"] = state["step"].detach().cpu()

        state_dtype = self._swap_state_dtype(local_param)
        pin_memory = self._swap_pin_memory(local_param)
        if "exp_avg" not in state:
            state["exp_avg"] = self._cpu_zeros_like_local(
                local_param, state_dtype, pin_memory
            )
        elif torch.is_tensor(state["exp_avg"]) and state["exp_avg"].device.type != "cpu":
            state["exp_avg"] = self._local_tensor(state["exp_avg"]).detach().cpu()

        if "exp_avg_sq" not in state:
            state["exp_avg_sq"] = self._cpu_zeros_like_local(
                local_param, state_dtype, pin_memory
            )
        elif torch.is_tensor(state["exp_avg_sq"]) and state["exp_avg_sq"].device.type != "cpu":
            state["exp_avg_sq"] = self._local_tensor(state["exp_avg_sq"]).detach().cpu()

        if group.get("amsgrad", False):
            raise RuntimeError(
                "swap_opt_states_cpu does not support Adam amsgrad=True yet"
            )
        return state

    def _materialize_missing_cpu_swap_adam_state(self, optimizer: Optimizer) -> None:
        if not isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW)):
            return
        for group in optimizer.param_groups:
            for param in group["params"]:
                self._init_cpu_swap_adam_state(optimizer, group, param)

    def _move_loaded_cpu_swap_adam_state(self, optimizer: Optimizer) -> None:
        if not isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW)):
            return
        for group in optimizer.param_groups:
            for param in group["params"]:
                state = optimizer.state[param]
                for key in ("step", "exp_avg", "exp_avg_sq"):
                    value = state.get(key)
                    if torch.is_tensor(value):
                        state[key] = self._local_tensor(value).detach().cpu()

    @torch.no_grad()
    def _step_cpu_swap_adam(self, optimizer: Optimizer) -> None:
        if not isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW)):
            raise RuntimeError(
                "swap_opt_states_cpu is only supported for Adam/AdamW optimizers"
            )
        is_adamw = isinstance(optimizer, torch.optim.AdamW)
        if self._swap_update_mode == "foreach_param":
            self._step_cpu_swap_adam_foreach(optimizer, is_adamw=is_adamw)
            if torch.cuda.is_available():
                torch.cuda.current_stream().synchronize()
            return

        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                self._step_cpu_swap_adam_param(
                    optimizer=optimizer,
                    group=group,
                    param=param,
                    is_adamw=is_adamw,
                )
        if torch.cuda.is_available():
            torch.cuda.current_stream().synchronize()

    @torch.no_grad()
    def _step_cpu_swap_adam_foreach(
        self,
        optimizer: Optimizer,
        *,
        is_adamw: bool,
    ) -> None:
        for group in optimizer.param_groups:
            batch: list[tuple[torch.Tensor, torch.Tensor, dict[str, Any]]] = []
            for param in group["params"]:
                if param.grad is None:
                    continue
                local_param = self._local_tensor(param)
                local_grad = self._local_tensor(param.grad)
                if local_grad.is_sparse:
                    raise RuntimeError("swap_opt_states_cpu does not support sparse gradients")
                if local_param.numel() == 0:
                    continue
                if not local_param.is_contiguous() or not local_grad.is_contiguous():
                    raise RuntimeError(
                        "swap_opt_states_cpu currently requires contiguous local param "
                        f"and grad tensors, got param_stride={tuple(local_param.stride())}, "
                        f"grad_stride={tuple(local_grad.stride())}"
                    )
                state = self._init_cpu_swap_adam_state(optimizer, group, param)
                batch.append((local_param, local_grad, state))
                if len(batch) >= self._swap_foreach_max_params:
                    self._step_cpu_swap_adam_foreach_batch(
                        group=group,
                        batch=batch,
                        is_adamw=is_adamw,
                    )
                    batch = []
            self._step_cpu_swap_adam_foreach_batch(
                group=group,
                batch=batch,
                is_adamw=is_adamw,
            )

    @torch.no_grad()
    def _step_cpu_swap_adam_foreach_batch(
        self,
        *,
        group: dict[str, Any],
        batch: list[tuple[torch.Tensor, torch.Tensor, dict[str, Any]]],
        is_adamw: bool,
    ) -> None:
        if not batch:
            return
        if group.get("amsgrad", False):
            raise RuntimeError(
                "swap_opt_states_cpu foreach_param does not support Adam amsgrad=True yet"
            )

        next_steps = [float(item[2]["step"].item()) + 1.0 for item in batch]
        if any(step != next_steps[0] for step in next_steps):
            for local_param, local_grad, state in batch:
                self._step_cpu_swap_adam_param_torch_full(
                    group=group,
                    state=state,
                    local_param=local_param,
                    local_grad=local_grad,
                    is_adamw=is_adamw,
                )
            return
        for _, _, state in batch:
            state["step"].add_(1)
        step = next_steps[0]

        beta1, beta2 = group["betas"]
        lr = group["lr"]
        eps = group["eps"]
        weight_decay = group["weight_decay"]
        maximize = group.get("maximize", False)
        bias_correction1 = 1.0 - beta1**step
        bias_correction2 = 1.0 - beta2**step
        step_size = lr / bias_correction1
        denom_scale = math.sqrt(bias_correction2)

        params = [item[0] for item in batch]
        grads = [item[1].float() for item in batch]
        if maximize:
            grads = torch._foreach_neg(grads)

        exp_avg_cpu = [item[2]["exp_avg"] for item in batch]
        exp_avg_sq_cpu = [item[2]["exp_avg_sq"] for item in batch]
        exp_avgs = [
            state_tensor.to(device=param.device, non_blocking=True).float()
            for state_tensor, param in zip(exp_avg_cpu, params)
        ]
        exp_avg_sqs = [
            state_tensor.to(device=param.device, non_blocking=True).float()
            for state_tensor, param in zip(exp_avg_sq_cpu, params)
        ]

        if is_adamw:
            if weight_decay != 0.0:
                torch._foreach_mul_(params, 1.0 - lr * weight_decay)
        elif weight_decay != 0.0:
            param_fp32 = [param.float() for param in params]
            grads = torch._foreach_add(grads, param_fp32, alpha=weight_decay)

        torch._foreach_mul_(exp_avgs, beta1)
        torch._foreach_add_(exp_avgs, grads, alpha=1.0 - beta1)
        torch._foreach_mul_(exp_avg_sqs, beta2)
        torch._foreach_addcmul_(exp_avg_sqs, grads, grads, value=1.0 - beta2)
        denoms = torch._foreach_sqrt(exp_avg_sqs)
        torch._foreach_div_(denoms, denom_scale)
        torch._foreach_add_(denoms, eps)
        updates = torch._foreach_div(exp_avgs, denoms)
        updates = [
            update.to(dtype=param.dtype)
            for update, param in zip(updates, params)
        ]
        torch._foreach_add_(params, updates, alpha=-step_size)

        for cpu_state, device_state in zip(exp_avg_cpu, exp_avgs):
            cpu_state.copy_(
                device_state.to(dtype=cpu_state.dtype),
                non_blocking=cpu_state.is_pinned(),
            )
        for cpu_state, device_state in zip(exp_avg_sq_cpu, exp_avg_sqs):
            cpu_state.copy_(
                device_state.to(dtype=cpu_state.dtype),
                non_blocking=cpu_state.is_pinned(),
            )

    @torch.no_grad()
    def _step_cpu_swap_adam_param(
        self,
        *,
        optimizer: Optimizer,
        group: dict[str, Any],
        param: torch.Tensor,
        is_adamw: bool,
    ) -> None:
        local_param = self._local_tensor(param)
        local_grad = self._local_tensor(param.grad)
        if local_grad.is_sparse:
            raise RuntimeError("swap_opt_states_cpu does not support sparse gradients")
        if local_param.numel() == 0:
            return
        if not local_param.is_contiguous() or not local_grad.is_contiguous():
            raise RuntimeError(
                "swap_opt_states_cpu currently requires contiguous local param "
                f"and grad tensors, got param_stride={tuple(local_param.stride())}, "
                f"grad_stride={tuple(local_grad.stride())}"
            )

        state = self._init_cpu_swap_adam_state(optimizer, group, param)
        if self._swap_update_mode == "torch_param":
            self._step_cpu_swap_adam_param_torch_full(
                group=group,
                state=state,
                local_param=local_param,
                local_grad=local_grad,
                is_adamw=is_adamw,
            )
            return
        if self._swap_update_mode == "fused_param":
            self._step_cpu_swap_adam_param_fused(
                group=group,
                state=state,
                local_param=local_param,
                local_grad=local_grad,
                is_adamw=is_adamw,
            )
            return

        state["step"].add_(1)
        step = float(state["step"].item())

        beta1, beta2 = group["betas"]
        lr = group["lr"]
        eps = group["eps"]
        weight_decay = group["weight_decay"]
        maximize = group.get("maximize", False)
        bias_correction1 = 1.0 - beta1**step
        bias_correction2 = 1.0 - beta2**step
        step_size = lr / bias_correction1
        denom_scale = math.sqrt(bias_correction2)

        param_flat = local_param.view(-1)
        grad_flat = local_grad.view(-1)
        exp_avg_cpu = state["exp_avg"]
        exp_avg_sq_cpu = state["exp_avg_sq"]
        exp_avg_flat = exp_avg_cpu.view(-1)
        exp_avg_sq_flat = exp_avg_sq_cpu.view(-1)
        chunk_numel = self._swap_chunk_numel(exp_avg_cpu.dtype)

        for start in range(0, local_param.numel(), chunk_numel):
            end = min(start + chunk_numel, local_param.numel())
            param_chunk = param_flat[start:end]
            grad_chunk = grad_flat[start:end]
            if maximize:
                grad_chunk = grad_chunk.neg()

            exp_avg_cpu_chunk = exp_avg_flat[start:end]
            exp_avg_sq_cpu_chunk = exp_avg_sq_flat[start:end]
            exp_avg = exp_avg_cpu_chunk.to(
                device=local_param.device, non_blocking=True
            ).float()
            exp_avg_sq = exp_avg_sq_cpu_chunk.to(
                device=local_param.device, non_blocking=True
            ).float()
            grad_work = grad_chunk.float()

            if is_adamw:
                if weight_decay != 0.0:
                    param_chunk.mul_(1.0 - lr * weight_decay)
            elif weight_decay != 0.0:
                grad_work = grad_work.add(param_chunk.float(), alpha=weight_decay)

            exp_avg.mul_(beta1).add_(grad_work, alpha=1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad_work, grad_work, value=1.0 - beta2)
            denom = exp_avg_sq.sqrt().div_(denom_scale).add_(eps)
            update = exp_avg.div(denom)
            param_chunk.add_(update.to(dtype=param_chunk.dtype), alpha=-step_size)

            exp_avg_cpu_chunk.copy_(
                exp_avg.to(dtype=exp_avg_cpu.dtype),
                non_blocking=exp_avg_cpu.is_pinned(),
            )
            exp_avg_sq_cpu_chunk.copy_(
                exp_avg_sq.to(dtype=exp_avg_sq_cpu.dtype),
                non_blocking=exp_avg_sq_cpu.is_pinned(),
            )

    @torch.no_grad()
    def _step_cpu_swap_adam_param_torch_full(
        self,
        *,
        group: dict[str, Any],
        state: dict[str, Any],
        local_param: torch.Tensor,
        local_grad: torch.Tensor,
        is_adamw: bool,
    ) -> None:
        """Full-parameter PyTorch tensor-op swap update.

        This is the BF16-compatible intermediate rung between the tiny-memory
        chunked path and a true fused/pipelined HIP optimizer. It keeps the same
        math as ``chunked`` but does the whole local parameter at once, avoiding
        the Python loop when transient memory is available.
        """

        state["step"].add_(1)
        step = float(state["step"].item())

        beta1, beta2 = group["betas"]
        lr = group["lr"]
        eps = group["eps"]
        weight_decay = group["weight_decay"]
        maximize = group.get("maximize", False)
        bias_correction1 = 1.0 - beta1**step
        bias_correction2 = 1.0 - beta2**step
        step_size = lr / bias_correction1
        denom_scale = math.sqrt(bias_correction2)

        exp_avg_cpu = state["exp_avg"]
        exp_avg_sq_cpu = state["exp_avg_sq"]
        exp_avg = exp_avg_cpu.to(device=local_param.device, non_blocking=True).float()
        exp_avg_sq = exp_avg_sq_cpu.to(device=local_param.device, non_blocking=True).float()
        grad_work = local_grad.float()
        if maximize:
            grad_work = grad_work.neg()

        if is_adamw:
            if weight_decay != 0.0:
                local_param.mul_(1.0 - lr * weight_decay)
        elif weight_decay != 0.0:
            grad_work = grad_work.add(local_param.float(), alpha=weight_decay)

        exp_avg.mul_(beta1).add_(grad_work, alpha=1.0 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad_work, grad_work, value=1.0 - beta2)
        denom = exp_avg_sq.sqrt().div_(denom_scale).add_(eps)
        update = exp_avg.div(denom)
        local_param.add_(update.to(dtype=local_param.dtype), alpha=-step_size)

        exp_avg_cpu.copy_(
            exp_avg.to(dtype=exp_avg_cpu.dtype),
            non_blocking=exp_avg_cpu.is_pinned(),
        )
        exp_avg_sq_cpu.copy_(
            exp_avg_sq.to(dtype=exp_avg_sq_cpu.dtype),
            non_blocking=exp_avg_sq_cpu.is_pinned(),
        )

    @torch.no_grad()
    def _step_cpu_swap_adam_param_fused(
        self,
        *,
        group: dict[str, Any],
        state: dict[str, Any],
        local_param: torch.Tensor,
        local_grad: torch.Tensor,
        is_adamw: bool,
    ) -> None:
        """Ascend-style full-param swap update using PyTorch fused Adam kernels.

        The default ``chunked`` mode is the conservative fit path: it keeps
        temporary device state tiny but pays Python and tensor-op overhead per
        chunk. This mode instead mirrors the TorchTitan-NPU SwapOptimizer
        contract more closely: make the whole local parameter's Adam moments
        resident on device for the update, run the fused Adam/AdamW kernel, and
        copy the updated moments back to CPU.
        """

        if local_param.device.type != "cuda":
            raise RuntimeError(
                "TORCHTITAN_OPTIMIZER_SWAP_UPDATE_MODE=fused_param requires a "
                "CUDA/HIP parameter device"
            )

        fused_op_name = "_fused_adamw_" if is_adamw else "_fused_adam_"
        fused_op = getattr(torch, fused_op_name, None)
        if fused_op is None:
            raise RuntimeError(
                f"torch.{fused_op_name} is not available in this PyTorch build"
            )
        if group.get("amsgrad", False):
            raise RuntimeError(
                "swap_opt_states_cpu fused_param does not support Adam amsgrad=True yet"
            )
        if local_param.dtype != torch.float32:
            raise RuntimeError(
                "swap_opt_states_cpu fused_param requires float32 parameters in "
                "this ROCm PyTorch build; use chunked or torch_param for BF16 params"
            )

        state["step"].add_(1)
        step_tensor = state["step"].to(device=local_param.device, non_blocking=True)
        exp_avg_cpu = state["exp_avg"]
        exp_avg_sq_cpu = state["exp_avg_sq"]
        exp_avg = exp_avg_cpu.to(device=local_param.device, non_blocking=True)
        exp_avg_sq = exp_avg_sq_cpu.to(device=local_param.device, non_blocking=True)

        fused_op(
            [local_param],
            [local_grad],
            [exp_avg],
            [exp_avg_sq],
            [],
            [step_tensor],
            amsgrad=False,
            beta1=group["betas"][0],
            beta2=group["betas"][1],
            lr=group["lr"],
            weight_decay=group["weight_decay"],
            eps=group["eps"],
            maximize=group.get("maximize", False),
        )

        exp_avg_cpu.copy_(
            exp_avg.to(dtype=exp_avg_cpu.dtype),
            non_blocking=exp_avg_cpu.is_pinned(),
        )
        exp_avg_sq_cpu.copy_(
            exp_avg_sq.to(dtype=exp_avg_sq_cpu.dtype),
            non_blocking=exp_avg_sq_cpu.is_pinned(),
        )

    @staticmethod
    def _materialize_missing_adam_state(optimizer: Optimizer) -> None:
        """Populate lazy Adam state for checkpointable but inactive params."""
        if not isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW)):
            return
        for group in optimizer.param_groups:
            for p in group["params"]:
                state = optimizer.state[p]
                if len(state) != 0:
                    continue
                state["step"] = (
                    torch.zeros((), dtype=torch.float32, device=p.device)
                    if group.get("capturable") or group.get("fused")
                    else torch.tensor(0.0, dtype=torch.float32)
                )
                state["exp_avg"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )
                state["exp_avg_sq"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )
                if group.get("amsgrad"):
                    state["max_exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

    def state_dict(self) -> dict[str, Any]:
        for optimizer in self.optimizers:
            if self._swap_optimizer_states_cpu:
                self._materialize_missing_cpu_swap_adam_state(optimizer)
            else:
                self._materialize_missing_adam_state(optimizer)
        func = functools.partial(
            get_optimizer_state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )
        if self._use_native_optimizer_state_dict:
            return {
                f"optimizer_{idx}": optimizer.state_dict()
                for idx, optimizer in enumerate(self.optimizers)
            }
        return {
            k: v
            for sd in map(func, self.optimizer_model_parts, self.optimizers)
            for k, v in sd.items()
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        options = StateDictOptions(flatten_optimizer_state_dict=True)
        if self._use_native_optimizer_state_dict:
            for idx, optimizer in enumerate(self.optimizers):
                optimizer.load_state_dict(state_dict[f"optimizer_{idx}"])
                if self._swap_optimizer_states_cpu:
                    self._move_loaded_cpu_swap_adam_state(optimizer)
            return

        func = functools.partial(
            set_optimizer_state_dict,
            optim_state_dict=state_dict,
            options=options,
        )
        list(map(func, self.optimizer_model_parts, self.optimizers))

    def _validate_length(self, expected_length: int) -> None:
        assert expected_length == len(self.optimizers), (
            "Must pass one optimizer per model part or per param if "
            "using OptimizersInBackwardContainer."
        )

    def _post_init(
        self, all_params: list[nn.Parameter], optimizer_kwargs: dict[str, Any]
    ) -> None:
        # We need to call Optimizer.__init__() to initialize some necessary optimizer
        # functionality such as hooks.
        Optimizer.__init__(self, all_params, optimizer_kwargs)

    def _register_bf16_optimizer_state_hook(self) -> None:
        """Register a step pre-hook to create Adam optimizer states in bfloat16.

        The hook pre-populates optimizer state before Adam's lazy initialization
        runs, so that ``_init_group`` finds non-empty state and skips its own
        fp32 allocation. The fused CUDA kernel then sees the dtype mismatch
        between fp32 params and bf16 states, dispatching to the mixed-precision
        kernel (``FusedAdamMathFunctorMP``).
        """

        def _bf16_state_init_hook(
            optimizer: Optimizer, args: tuple, kwargs: dict
        ) -> None:
            for group in optimizer.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = optimizer.state[p]
                    if len(state) == 0:
                        state["step"] = (
                            torch.zeros((), dtype=torch.float32, device=p.device)
                            if group.get("capturable") or group.get("fused")
                            else torch.tensor(0.0, dtype=torch.float32)
                        )
                        state["exp_avg"] = torch.zeros_like(
                            p, dtype=torch.bfloat16, memory_format=torch.preserve_format
                        )
                        state["exp_avg_sq"] = torch.zeros_like(
                            p, dtype=torch.bfloat16, memory_format=torch.preserve_format
                        )
                        if group.get("amsgrad"):
                            state["max_exp_avg_sq"] = torch.zeros_like(
                                p,
                                dtype=torch.bfloat16,
                                memory_format=torch.preserve_format,
                            )

        for optim in self.optimizers:
            if isinstance(optim, (torch.optim.Adam, torch.optim.AdamW)):
                optim.register_step_pre_hook(_bf16_state_init_hook)

    def init_cache_state_dict(self) -> None:
        """Initialize cached state dict for TorchFT. No-op for base class."""
        pass


class OptimizersInBackwardContainer(OptimizersContainer):
    """OptimizersContainer for executing ``optim.step()`` in backward pass.

    This class extend ``OptimizersContainer`` to support optimizer step in
    backward pass. ``step()`` and ``zero_grad()`` are no-op in this class.
    Instead, ``register_post_accumulate_grad_hook`` is used to register a hook to
    execute these methods when the gradient is accumulated.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(OptimizersContainer.Config):
        def __post_init__(self) -> None:
            if self.implementation in ("fused_opt_states_bf16", "swap_opt_states_cpu"):
                raise ValueError(
                    f"implementation='{self.implementation}' is not supported with "
                    "OptimizersInBackwardContainer"
                )
            OptimizersContainer.Config.__post_init__(self)

    def __init__(self, config: Config, *, model_parts: list[nn.Module]) -> None:
        optimizer_kwargs = self._build_optimizer_kwargs(config)
        all_params = []
        self.model_parts = model_parts

        # Build a mapping from param -> effective kwargs using param group config
        param_to_kwargs: dict[nn.Parameter, dict[str, Any]] = {}
        param_to_optimizer_name: dict[nn.Parameter, str] = {}
        param_to_name: dict[nn.Parameter, str] = {}
        for model_idx, model in enumerate(self.model_parts):
            for name, param in model.named_parameters():
                param_to_name[param] = f"model{model_idx}.{name}"
        for model in self.model_parts:
            param_groups = self._build_param_groups(model, config, optimizer_kwargs)
            for group in param_groups:
                optimizer_name = group.get("_optimizer_name", config.name)
                group_kwargs = {
                    k: v for k, v in group.items() if k not in ("params", "_optimizer_name")
                }
                for p in group["params"]:
                    param_to_kwargs[p] = group_kwargs
                    param_to_optimizer_name[p] = optimizer_name

        optim_dict = {}
        for model in self.model_parts:
            for p in model.parameters():
                if p.requires_grad:
                    optimizer_cls = self._resolve_optimizer_cls(param_to_optimizer_name[p])
                    optim_dict[p] = optimizer_cls([p], **param_to_kwargs[p])
                all_params.append(p)

        def optim_hook(param) -> None:
            if _optimizer_profile_in_backward_enabled():
                _profiled_optimizer_step(
                    optim_dict[param],
                    context={
                        "profile_context": "in_backward_hook",
                        "container_type": type(self).__name__,
                        "param_name": param_to_name.get(param),
                        "param_shape": [int(dim) for dim in param.shape],
                        "param_ndim": int(param.ndim),
                    },
                )
            else:
                optim_dict[param].step()
            optim_dict[param].zero_grad()

        for model in self.model_parts:
            for param in model.parameters():
                if param.requires_grad:
                    param.register_post_accumulate_grad_hook(optim_hook)

        self.optimizers = list(optim_dict.values())
        self.optimizer_model_parts = []
        self._use_native_optimizer_state_dict = True
        self._swap_optimizer_states_cpu = False

        self._validate_length(
            sum(len(list(model.parameters())) for model in self.model_parts)
        )
        self._post_init(all_params, optimizer_kwargs)

    @overload
    def step(self, closure: None = None) -> None:
        ...

    @overload
    def step(self, closure: Callable[[], float]) -> float:
        ...

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        return None

    def zero_grad(self, set_to_none: bool = True) -> None:
        pass


def register_moe_load_balancing_hook(
    optimizers: OptimizersContainer,
    model_parts: list[nn.Module],
    parallel_dims: ParallelDims,
) -> None:
    """Register an optimizer step pre-hook for MoE auxiliary-loss-free load balancing.

    This function checks if MoE load balancing is enabled and, if so, registers
    a hook that updates expert biases before each optimizer step.

    Args:
        optimizers: The optimizers container to register the hook on.
        model_parts: List of model parts that may contain MoE layers.
        parallel_dims: Parallel dimensions for distributed communication.
    """

    def _should_register_moe_balancing_hook(model_parts: list[nn.Module]) -> bool:
        for model_part in model_parts:
            layers = model_part.get_submodule("layers")
            assert isinstance(layers, nn.ModuleDict)
            for transformer_block in layers.values():
                if transformer_block.moe_enabled:
                    # Assumption: load_balance_coeff is set universally on all moe blocks.
                    # pyrefly: ignore [missing-attribute]
                    return bool(transformer_block.moe.load_balance_coeff)
        return False

    # for MoE auxiliary-loss-free load balancing
    def _is_recomputation_enabled(module):
        return getattr(module, "checkpoint_impl", None) is CheckpointImpl.NO_REENTRANT

    def _update_expert_bias(
        model_parts: list[nn.Module],
        parallel_dims: ParallelDims,
    ):
        loss_mesh = parallel_dims.get_optional_mesh("loss")
        # TODO: Currently this sync is blocking (thus exposed) and happens on the
        # default compute stream. Need to assess if this is OK performance-wise.
        tokens_per_expert_list = []
        for model_part in model_parts:
            layers = model_part.get_submodule("layers")
            assert isinstance(layers, nn.ModuleDict)
            for transformer_block in layers.values():
                if not transformer_block.moe_enabled:
                    continue
                # pyrefly: ignore [missing-attribute]
                if transformer_block.moe.load_balance_coeff is None:
                    return
                # pyrefly: ignore [missing-attribute]
                tokens_per_expert = transformer_block.moe.tokens_per_expert
                if _is_recomputation_enabled(transformer_block):
                    # TODO: This is a hack, we assume with full AC, the tokens_per_expert is counted twice.
                    # This does not affect to expert choice, but affects the experts usage metrics.
                    # We divide by 2 to correct for this double-counting due to recomputation
                    # TODO: new API to help determine if AC is enabled https://github.com/pytorch/pytorch/pull/160888
                    tokens_per_expert = tokens_per_expert // 2
                tokens_per_expert_list.append(tokens_per_expert)

        tokens_per_expert_by_layer = torch.vstack(tokens_per_expert_list)

        if parallel_dims.full_dtensor:
            # full_dtensor: DTensor mesh includes all axes (DP/CP/TP/EP).
            # redistribute Partial→Replicate covers everything.
            assert isinstance(
                tokens_per_expert_by_layer, torch.distributed.tensor.DTensor
            )
            dtensor_mesh = tokens_per_expert_by_layer.device_mesh
            # TODO: This incurs multiple sequential all-reduces, one per
            # SPMD mesh axis. We should provide a utility to do a single all-reduce
            # on the flattened global SPMD mesh.
            tokens_per_expert_by_layer = tokens_per_expert_by_layer.redistribute(
                placements=[Replicate()] * dtensor_mesh.ndim
            )
        else:
            # non-full_dtensor: DTensor mesh only has TP/EP (if enabled).
            # full_tensor() reduces on TP/EP, then all-reduce on loss_mesh
            # covers DP/CP separately.
            is_dtensor = isinstance(
                tokens_per_expert_by_layer, torch.distributed.tensor.DTensor
            )
            if is_dtensor:
                dtensor_mesh = tokens_per_expert_by_layer.device_mesh
                tokens_per_expert_by_layer = tokens_per_expert_by_layer.full_tensor()
            if loss_mesh is not None:
                torch.distributed.all_reduce(
                    tokens_per_expert_by_layer,
                    group=loss_mesh.get_group(),
                    op=torch.distributed.ReduceOp.SUM,
                )
            if is_dtensor:
                tokens_per_expert_by_layer = torch.distributed.tensor.DTensor.from_local(
                    tokens_per_expert_by_layer,
                    # pyrefly: ignore [unbound-name]
                    device_mesh=dtensor_mesh,
                    placements=[Replicate()]
                    * dtensor_mesh.ndim,  # pyrefly: ignore [unbound-name]
                    run_check=False,
                )

        moe_layer_idx = 0
        with torch.no_grad():
            for model_part in model_parts:
                layers = model_part.get_submodule("layers")
                assert isinstance(layers, nn.ModuleDict)
                for transformer_block in layers.values():
                    if not transformer_block.moe_enabled:
                        continue
                    moe = transformer_block.moe

                    tokens_per_expert = tokens_per_expert_by_layer[
                        moe_layer_idx
                    ].float()
                    moe_layer_idx += 1

                    # update the expert bias
                    # this is not exactly the same as https://arxiv.org/pdf/2408.15664 proposed
                    # pyrefly: ignore [missing-attribute]
                    expert_bias_delta = moe.load_balance_coeff * torch.sign(
                        tokens_per_expert.mean() - tokens_per_expert
                    )
                    expert_bias_delta = expert_bias_delta - expert_bias_delta.mean()
                    # pyrefly: ignore [missing-attribute]
                    moe.expert_bias.add_(expert_bias_delta)
                    # pyrefly: ignore [missing-attribute]
                    moe.tokens_per_expert.zero_()

    if _should_register_moe_balancing_hook(model_parts):
        optimizers.register_step_pre_hook(
            lambda *args, **kwargs: _update_expert_bias(
                model_parts, parallel_dims=parallel_dims
            )
        )
