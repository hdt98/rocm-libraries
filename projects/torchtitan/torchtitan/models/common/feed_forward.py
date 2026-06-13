# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
import os

import torch
import torch.nn.functional as F

from torchtitan.models.common.dsv4_profile_timing import (
    dsv4_profile_timing_enabled,
    dsv4_timed_stage,
    flush_dsv4_profile_timing,
)
from torchtitan.models.common.nn_modules import Linear
from torchtitan.protocols.module import Module

__all__ = ["FeedForward", "compute_ffn_hidden_dim"]


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class _ProfiledLinear(torch.autograd.Function):
    """Profiling-only linear wrapper that names FeedForward backward matmuls."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, bias, label: str):
        ctx.has_bias = bias is not None
        ctx.label = str(label)
        if bias is None:
            ctx.save_for_backward(x, weight)
        else:
            ctx.save_for_backward(x, weight, bias)
        return F.linear(x, weight, bias)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        label = ctx.label
        timing_records = [] if dsv4_profile_timing_enabled() else None
        with torch.profiler.record_function(f"{label}.bwd"):
            saved = ctx.saved_tensors
            if ctx.has_bias:
                x, weight, _bias = saved
            else:
                x, weight = saved

            grad_x = grad_weight = grad_bias = None
            grad_output_flat = grad_output.reshape(-1, grad_output.shape[-1])
            x_flat = x.reshape(-1, x.shape[-1])

            if ctx.needs_input_grad[0]:
                with torch.profiler.record_function(f"{label}.bwd.dinput"):
                    with dsv4_timed_stage(timing_records, f"{label}.bwd.dinput"):
                        grad_x = grad_output_flat.to(dtype=weight.dtype).matmul(weight)
                        grad_x = grad_x.reshape_as(x).to(x.dtype)
            if ctx.needs_input_grad[1]:
                with torch.profiler.record_function(f"{label}.bwd.dweight"):
                    with dsv4_timed_stage(timing_records, f"{label}.bwd.dweight"):
                        grad_weight = grad_output_flat.to(dtype=x.dtype).transpose(0, 1).matmul(x_flat)
                        grad_weight = grad_weight.reshape_as(weight).to(weight.dtype)
            if ctx.has_bias and ctx.needs_input_grad[2]:
                with torch.profiler.record_function(f"{label}.bwd.dbias"):
                    with dsv4_timed_stage(timing_records, f"{label}.bwd.dbias"):
                        grad_bias = grad_output_flat.to(dtype=saved[-1].dtype).sum(dim=0)
        flush_dsv4_profile_timing(
            timing_records,
            {
                "kind": "linear_backward",
                "label": label,
                "x_shape": list(x.shape),
                "weight_shape": list(weight.shape),
                "grad_output_shape": list(grad_output.shape),
                "has_bias": bool(ctx.has_bias),
            },
        )
        return grad_x, grad_weight, grad_bias, None


def _profiled_linear(module: Linear, x: torch.Tensor, label: str) -> torch.Tensor:
    if not _env_flag("TORCHTITAN_DSV4_PROFILE_DENSE_LINEAR_BACKWARD", default=False):
        return module(x)
    return _ProfiledLinear.apply(x, module.weight, module.bias, label)


def compute_ffn_hidden_dim(
    dim: int,
    *,
    multiple_of: int = 1,
    ffn_dim_multiplier: float | None = None,
) -> int:
    """Compute the SwiGLU hidden dimension for Llama3/4-style models.

    This applies the 2/3 scaling, optional multiplier, and rounds up to multiple_of.
    """
    hidden_dim = int(2 * 4 * dim / 3)
    if ffn_dim_multiplier is not None:
        hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    return multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)


class FeedForward(Module):
    """SwiGLU feed-forward module shared across models.

    Config takes the **final** hidden_dim (no internal 2/3 scaling).
    Use compute_ffn_hidden_dim() for Llama3/4-style dim computation.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        w1: Linear.Config
        w2: Linear.Config
        w3: Linear.Config
        profile_label_prefix: str = "dsv4.linear.feed_forward"

    def __init__(self, config: Config):
        super().__init__()
        self.w1 = config.w1.build()
        self.w2 = config.w2.build()
        self.w3 = config.w3.build()
        self.profile_label_prefix = config.profile_label_prefix

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        label = self.profile_label_prefix
        w1 = _profiled_linear(self.w1, x, f"{label}.w1")
        w3 = _profiled_linear(self.w3, x, f"{label}.w3")
        hidden = F.silu(w1) * w3
        return _profiled_linear(self.w2, hidden, f"{label}.w2")
