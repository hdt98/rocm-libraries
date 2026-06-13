# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file provides the util functions to apply activation checkpointing to the model.
# Technically, this is not a part of distributed, but distributed module is the best place to put it.

import os

import torch
import torch._functorch.config
import torch.nn as nn
from torch._functorch.partitioners import get_default_op_list
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)
from torch.utils.checkpoint import (
    CheckpointPolicy,
    create_selective_checkpoint_contexts,
)

from torchtitan.config import ActivationCheckpointConfig as ACConfig
from torchtitan.models.common.dsv4_profile_timing import (
    dsv4_current_execution_region,
    dsv4_current_microbatch,
    dsv4_profile_ac_blocks_enabled,
    dsv4_timed_stage,
    flush_dsv4_profile_timing,
)
from torchtitan.tools.logging import logger


class _Dsv4TimedACModule(nn.Module):
    """Opt-in timing wrapper for activation-checkpointed block forwards."""

    def __init__(self, module: nn.Module, *, fqn: str):
        super().__init__()
        self.module = module
        self.fqn = fqn

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            module = self._modules.get("module")
            if module is not None:
                return getattr(module, name)
            raise

    def forward(self, *args, **kwargs):
        if not dsv4_profile_ac_blocks_enabled():
            return self.module(*args, **kwargs)

        region = dsv4_current_execution_region() or "unknown"
        stage = f"dsv4.ac_block.{self.fqn}.{region}"
        timing_records = []
        with torch.profiler.record_function(stage):
            with dsv4_timed_stage(timing_records, stage):
                output = self.module(*args, **kwargs)
        flush_dsv4_profile_timing(
            timing_records,
            {
                "kind": "activation_checkpoint_block_forward",
                "layer": self.fqn,
                "region": region,
                "microbatch": dsv4_current_microbatch(),
                "module_type": type(self.module).__name__,
            },
        )
        return output


def _get_save_ops() -> set:
    """Returns the set of ops whose activations should be saved (compute + comm).

    Each op spec is either an op object (always included) or a tuple
    (root, dotted_path) for conditionally available ops — resolved via
    getattr and silently skipped if not registered.
    """
    # Ops whose outputs are expensive to recompute (matmuls, attention, etc.)
    compute_ops = [
        # SDPA variants
        torch.ops.aten._scaled_dot_product_cudnn_attention.default,
        torch.ops.aten._scaled_dot_product_attention_math.default,
        torch.ops.aten._scaled_dot_product_fused_attention_overrideable.default,
        # For low precision training, always save the absolute maximum used
        # to compute the scaling factor for quantization.
        torch.ops.aten.max.default,
        # FlexAttention (torch.ops.higher_order.flex_attention is the same object)
        torch._higher_order_ops.flex_attention,
        torch.ops.aten.linear.default,
        # topk can be non-deterministic; save to keep MoE expert assignments
        # stable between forward and recompute.
        torch.ops.aten.topk.default,
        # Inductor compiled code (available when torch.compile is used)
        (torch._higher_order_ops, "inductor_compiled_code"),
        # torch_attn custom backend
        (torch.ops, "torch_attn._varlen_attn.default"),
    ]

    # Communication ops whose outputs should be saved to avoid re-communication.
    comm_ops = [
        torch.ops._c10d_functional.reduce_scatter_tensor.default,
        torch.ops._c10d_functional.all_to_all_single.default,
        # DeepEP (available when deepep is installed)
        (torch.ops, "deepep.dispatch.default"),
        (torch.ops, "deepep.combine.default"),
        # HybridEP (available when hybridep is installed)
        (torch.ops, "hybridep.dispatch.default"),
        (torch.ops, "hybridep.combine.default"),
    ]

    def _resolve_ops(op_specs: list) -> dict:
        ops = {}
        for spec in op_specs:
            if isinstance(spec, tuple):
                obj, path = spec
                try:
                    for part in path.split("."):
                        obj = getattr(obj, part)
                    ops[obj] = CheckpointPolicy.MUST_SAVE
                except AttributeError:
                    pass
            else:
                ops[spec] = CheckpointPolicy.MUST_SAVE
        return ops

    aten_op_types = get_default_op_list()
    save_ops = {
        op.default  # pyrefly: ignore [missing-attribute]
        for op in aten_op_types.compute_intensive_ops
    }
    save_ops.update(_resolve_ops(compute_ops))
    save_ops.update(_resolve_ops(comm_ops))
    return save_ops


def _apply_op_sac(
    module: nn.Module,
    ac_config: ACConfig,
    *,
    base_fqn: str | None = None,
) -> nn.Module:
    """Apply per-op selective activation checkpointing to the module."""
    save_ops = _get_save_ops()

    # Collect weight shapes to force-recompute, stored as mm RHS shape
    # (in_f, out_f). For aten.linear we transpose args[1].shape at lookup
    # time to match, since linear's weight is (out_f, in_f).
    mm_recompute_shapes = set()
    mm_recompute_fqns = ac_config.per_op_sac_force_recompute_mm_shapes_by_fqns

    if mm_recompute_fqns:
        for module_fqn, submod in module.named_modules():
            fqn = f"{base_fqn}.{module_fqn}" if base_fqn else module_fqn
            if not any(f in fqn for f in mm_recompute_fqns):
                continue
            if not isinstance(submod, nn.Linear):
                raise ValueError(
                    "per_op_sac_force_recompute_mm_shapes_by_fqns expected to match "
                    f"a nn.Linear, but got: {submod}"
                )
            out_f, in_f = submod.weight.shape
            mm_recompute_shapes.add((in_f, out_f))

    # Some backends (e.g. PrivateUse1) register aten.linear as a leaf op
    # instead of decomposing it into aten.mm, so we must handle both.
    mm_ops = (torch.ops.aten.mm.default, torch.ops.aten.linear.default)

    def _get_custom_policy():
        meta = {"forward_mm_count": 0, "recompute_mm_count": 0}

        def wrapped_policy(ctx, func, *args, **kwargs) -> CheckpointPolicy:
            # Always save CUDA→CPU results to avoid recomputing them
            # (e.g. MoE D2H sync for all-to-all metadata).
            if (
                func == torch.ops.aten._to_copy.default
                and "cuda" in str(args[0].device)
                and "device" in kwargs
                and str(kwargs["device"]) == "cpu"
            ):
                return CheckpointPolicy.MUST_SAVE

            mode = "recompute" if ctx.is_recompute else "forward"
            mm_count_key = f"{mode}_mm_count"

            if func in mm_ops:
                weight_shape = args[1].shape
                # linear weight is (out, in); normalize to (in, out) to match mm
                if func == torch.ops.aten.linear.default:
                    weight_shape = torch.Size((weight_shape[1], weight_shape[0]))
                if weight_shape in mm_recompute_shapes:
                    return CheckpointPolicy.PREFER_RECOMPUTE
                meta[mm_count_key] += 1

            # Save all compute/comm ops, except every second mm/linear.
            if func in save_ops:
                if func in mm_ops and meta[mm_count_key] % 2 == 0:
                    return CheckpointPolicy.PREFER_RECOMPUTE
                return CheckpointPolicy.MUST_SAVE
            return CheckpointPolicy.PREFER_RECOMPUTE

        return wrapped_policy

    return ptd_checkpoint_wrapper(
        module,
        context_fn=lambda: create_selective_checkpoint_contexts(_get_custom_policy()),
        preserve_rng_state=ac_config.preserve_rng_state,
        determinism_check=ac_config.determinism_check,
        early_stop=ac_config.early_stop,
        debug=ac_config.debug,
    )


def _apply_full_ac(module: nn.Module, ac_config: ACConfig) -> nn.Module:
    """Apply full activation checkpointing to the module.

    Args:
        module (nn.Module): The module to apply full activation checkpointing to.
        ac_config (ACConfig): The activation checkpointing config.

    Returns:
        nn.Module: The module with full activation checkpointing applied.
    """
    return ptd_checkpoint_wrapper(
        module,
        preserve_rng_state=ac_config.preserve_rng_state,
        determinism_check=ac_config.determinism_check,
        early_stop=ac_config.early_stop,
        debug=ac_config.debug,
    )


def _apply_ac_to_transformer_block(
    module: nn.Module,
    ac_config: ACConfig,
    *,
    base_fqn: str | None = None,
) -> nn.Module:
    valid_ac_modes = ("full", "selective")
    if ac_config.mode not in valid_ac_modes:
        raise ValueError(
            f"Invalid AC mode: {ac_config.mode}. Valid modes: {valid_ac_modes}"
        )

    if dsv4_profile_ac_blocks_enabled():
        module = _Dsv4TimedACModule(module, fqn=base_fqn or "transformer_block")

    if ac_config.mode == "full":
        return _apply_full_ac(module, ac_config)

    return _apply_op_sac(module, ac_config, base_fqn=base_fqn)


_VALID_SELECTIVE_AC_SCOPES = {
    "transformer_block",
    "attention_only",
    "hyper_connections_only",
    "attention_and_hyper_connections",
    "dense_attention_only",
    "sparse_attention_only",
    "c4_attention_only",
    "c128_attention_only",
    "compress_ratio_attention_only",
}


def _selective_ac_scope() -> str:
    scope = (
        os.environ.get("TORCHTITAN_ACTIVATION_CHECKPOINT_SCOPE")
        or os.environ.get("CANARY_ACTIVATION_CHECKPOINT_SCOPE")
        or "transformer_block"
    )
    scope = scope.strip().lower()
    if scope == "":
        scope = "transformer_block"
    if scope not in _VALID_SELECTIVE_AC_SCOPES:
        raise ValueError(
            "TORCHTITAN_ACTIVATION_CHECKPOINT_SCOPE must be "
            f"one of {sorted(_VALID_SELECTIVE_AC_SCOPES)}, got {scope!r}."
        )
    return scope


def _selective_ac_compress_ratios() -> set[int]:
    raw = (
        os.environ.get("TORCHTITAN_ACTIVATION_CHECKPOINT_COMPRESS_RATIOS")
        or os.environ.get("CANARY_ACTIVATION_CHECKPOINT_COMPRESS_RATIOS")
        or ""
    )
    ratios = {
        int(value.strip())
        for value in raw.split(",")
        if value.strip()
    }
    if not ratios:
        raise ValueError(
            "compress_ratio_attention_only requires "
            "TORCHTITAN_ACTIVATION_CHECKPOINT_COMPRESS_RATIOS or "
            "CANARY_ACTIVATION_CHECKPOINT_COMPRESS_RATIOS, for example '4,128'."
        )
    return ratios


def _attention_matches_selective_scope(
    attention: nn.Module | None,
    scope: str,
    *,
    compress_ratios: set[int] | None = None,
) -> bool:
    if attention is None:
        return False
    if scope == "attention_only":
        return True

    compress_ratio = int(getattr(attention, "compress_ratio", 0) or 0)
    if scope == "dense_attention_only":
        return compress_ratio == 0
    if scope == "sparse_attention_only":
        return compress_ratio > 0
    if scope == "c4_attention_only":
        return compress_ratio == 4
    if scope == "c128_attention_only":
        return compress_ratio == 128
    if scope == "compress_ratio_attention_only":
        assert compress_ratios is not None
        return compress_ratio in compress_ratios
    return False


def _apply_selective_ac_to_attention_only(
    module: nn.Module,
    ac_config: ACConfig,
    *,
    base_fqn: str | None = None,
) -> nn.Module:
    attention = getattr(module, "attention", None)
    if attention is None:
        return module
    wrapped_attention = _apply_op_sac(
        attention,
        ac_config,
        base_fqn=f"{base_fqn}.attention" if base_fqn else "attention",
    )
    setattr(module, "attention", wrapped_attention)
    return module


def apply_ac(
    model: nn.Module,
    ac_config: ACConfig,
    *,
    model_compile_enabled: bool = False,
    base_folder: str = "",
) -> None:
    """Apply activation checkpointing to the model.

    Args:
        model (nn.Module): The model to apply activation checkpointing to.
        ac_config (ACConfig): The activation checkpointing config.
        model_compile_enabled (bool): Whether torch.compile is enabled for the model.

    Returns:
        None
    """
    # Disable dynamo LRU cache to workaround an interaction between SAC, PP, and Flex:
    #
    # When forward runs with a second PP microbatch, it triggers recompilation with dynamic
    # shapes enabled. Now there are two valid compiled graphs. By default, dynamo selects
    # the latest one (the dynamic shapes version), so the runtime wrapper expects an extra
    # symint output. When SAC caches the inductor HOP output from the static graph for
    # batch_idx=0, it would miss that symint and cause an assertion failure. The workaround
    # here is to disable the LRU cache, and select graphs in insertion order instead.
    #
    # Also see: https://github.com/pytorch/pytorch/issues/166926
    # pyrefly: ignore [missing-attribute]
    set_lru_cache = getattr(torch._C._dynamo.eval_frame, "_set_lru_cache", None)
    if set_lru_cache is not None:
        set_lru_cache(False)

    if ac_config.mode == "memory_budget":
        assert model_compile_enabled, "Memory budget mode requires model to be compiled"
        if ac_config.visualize_memory_budget_pareto:
            pareto_dir = os.path.join(base_folder, "memory_budget_pareto")
            if not os.path.exists(pareto_dir):
                os.makedirs(pareto_dir, exist_ok=True)
            torch._functorch.config.memory_budget_pareto_dir = pareto_dir
            torch._functorch.config.visualize_memory_budget_pareto = True

        torch._functorch.config.activation_memory_budget = ac_config.memory_budget
        logger.info(f"Selected {ac_config.memory_budget} budget option")
    else:
        selective_scope = (
            _selective_ac_scope() if ac_config.mode == "selective" else "transformer_block"
        )
        compress_ratios = (
            _selective_ac_compress_ratios()
            if selective_scope == "compress_ratio_attention_only"
            else None
        )
        layers = model.get_submodule("layers")
        for layer_id, transformer_block in layers.named_children():
            layer_fqn = f"layers.{layer_id}"
            attention = getattr(transformer_block, "attention", None)
            if selective_scope == "transformer_block":
                transformer_block = _apply_ac_to_transformer_block(
                    transformer_block,
                    ac_config,
                    base_fqn=layer_fqn,
                )
            elif selective_scope == "hyper_connections_only":
                pass
            elif (
                selective_scope == "attention_and_hyper_connections"
                or _attention_matches_selective_scope(
                    attention,
                    selective_scope,
                    compress_ratios=compress_ratios,
                )
            ):
                transformer_block = _apply_selective_ac_to_attention_only(
                    transformer_block,
                    ac_config,
                    base_fqn=layer_fqn,
                )
            layers.register_module(layer_id, transformer_block)

    if ac_config.mode == "selective":
        logger.info(
            "Applied selective activation checkpointing to the model "
            f"(scope={_selective_ac_scope()})"
        )
    else:
        logger.info(f"Applied {ac_config.mode} activation checkpointing to the model")
