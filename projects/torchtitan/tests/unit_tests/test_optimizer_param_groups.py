# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import unittest

import torch
import torch.nn as nn
from torchtitan.components.optimizer import (
    Muon,
    OptimizersContainer,
    OptimizersInBackwardContainer,
    ParamGroupConfig,
)


class SimpleModel(nn.Module):
    """A small model with diverse parameter names for testing param groups."""

    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(32, 16)
        self.layers = nn.ModuleDict(
            {
                "0": nn.ModuleDict(
                    {
                        "attention": nn.Linear(16, 16),
                        "norm": nn.LayerNorm(16),
                        "ff": nn.Linear(16, 16),
                    }
                ),
            }
        )
        self.output = nn.Linear(16, 32)

    def forward(self, x):
        x = self.embed_tokens(x)
        x = self.layers["0"]["attention"](x)
        x = self.layers["0"]["norm"](x)
        x = self.layers["0"]["ff"](x)
        return self.output(x)


class MixedOptimizerModel(nn.Module):
    """Small model with rank-1, rank-2, and rank-3 trainable parameters."""

    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(32, 16)
        self.norm = nn.LayerNorm(16)
        self.proj = nn.Linear(16, 16, bias=False)
        self.expert_weight = nn.Parameter(torch.randn(2, 8, 16))

    def forward(self, x):
        return self.proj(self.norm(self.embed_tokens(x)))


class UnusedParamModel(nn.Module):
    """Small model with a trainable parameter outside the active forward path."""

    def __init__(self):
        super().__init__()
        self.used = nn.Linear(4, 4, bias=False)
        self.unused = nn.Parameter(torch.zeros(4))

    def forward(self, x):
        return self.used(x)


def _get_param_names_in_group(model, group):
    """Return the set of parameter FQNs in an optimizer param group."""
    param_to_name = {p: n for n, p in model.named_parameters()}
    return {param_to_name[p] for p in group["params"]}


def _is_muon_optimizer(opt):
    return isinstance(opt, Muon) or type(opt).__name__ in {
        "BatchedMuonExpertParams",
        "NativeMuonWithBatchedParams",
    }


class TestParamGroupConfig(unittest.TestCase):
    def test_default_no_param_groups(self):
        """Empty param_groups produces a single group with all params."""
        model = SimpleModel()
        config = OptimizersContainer.Config(lr=1e-3, weight_decay=0.1)
        default_kwargs = OptimizersContainer._build_optimizer_kwargs(config)

        groups = OptimizersContainer._build_param_groups(model, config, default_kwargs)

        self.assertEqual(len(groups), 1)
        all_params = [p for p in model.parameters() if p.requires_grad]
        self.assertEqual(len(groups[0]["params"]), len(all_params))
        self.assertEqual(groups[0]["lr"], 1e-3)
        self.assertEqual(groups[0]["weight_decay"], 0.1)

    def test_single_pattern_weight_decay_zero(self):
        """Pattern matching bias params with weight_decay_multiplier=0."""
        model = SimpleModel()
        config = OptimizersContainer.Config(
            lr=1e-3,
            weight_decay=0.1,
            param_groups=[
                ParamGroupConfig(pattern=r".*\.bias$", weight_decay_multiplier=0.0),
            ],
        )
        default_kwargs = OptimizersContainer._build_optimizer_kwargs(config)
        groups = OptimizersContainer._build_param_groups(model, config, default_kwargs)

        # Should have 2 groups: default + bias group
        self.assertEqual(len(groups), 2)

        # Default group: non-bias params
        default_names = _get_param_names_in_group(model, groups[0])
        for name in default_names:
            self.assertFalse(name.endswith(".bias"), f"{name} should not be in default")
        self.assertEqual(groups[0]["weight_decay"], 0.1)
        self.assertEqual(groups[0]["lr"], 1e-3)

        # Bias group
        bias_names = _get_param_names_in_group(model, groups[1])
        for name in bias_names:
            self.assertTrue(name.endswith(".bias"), f"{name} should end with .bias")
        self.assertEqual(groups[1]["weight_decay"], 0.0)
        self.assertEqual(groups[1]["lr"], 1e-3)

    def test_embed_tokens_pattern(self):
        """Pattern matching embed_tokens with weight_decay_multiplier=0."""
        model = SimpleModel()
        config = OptimizersContainer.Config(
            lr=1e-3,
            weight_decay=0.1,
            param_groups=[
                ParamGroupConfig(
                    pattern=r"embed_tokens\.",
                    weight_decay_multiplier=0.0,
                ),
            ],
        )
        default_kwargs = OptimizersContainer._build_optimizer_kwargs(config)
        groups = OptimizersContainer._build_param_groups(model, config, default_kwargs)

        self.assertEqual(len(groups), 2)

        embed_names = _get_param_names_in_group(model, groups[1])
        self.assertTrue(
            all("embed_tokens" in n for n in embed_names),
            f"Expected embed_tokens params, got {embed_names}",
        )
        self.assertEqual(groups[1]["weight_decay"], 0.0)

    def test_lr_multiplier(self):
        """lr_multiplier correctly scales the base lr."""
        model = SimpleModel()
        config = OptimizersContainer.Config(
            lr=1e-3,
            weight_decay=0.1,
            param_groups=[
                ParamGroupConfig(
                    pattern=r"embed_tokens\.",
                    lr_multiplier=0.1,
                ),
            ],
        )
        default_kwargs = OptimizersContainer._build_optimizer_kwargs(config)
        groups = OptimizersContainer._build_param_groups(model, config, default_kwargs)

        # Default group keeps base lr
        self.assertEqual(groups[0]["lr"], 1e-3)
        # Embed group gets 10% of base lr
        self.assertAlmostEqual(groups[1]["lr"], 1e-4)

    def test_first_match_wins(self):
        """When patterns overlap, the first match wins."""
        model = SimpleModel()
        config = OptimizersContainer.Config(
            lr=1e-3,
            weight_decay=0.1,
            param_groups=[
                # First pattern: all norm params get wd=0
                ParamGroupConfig(pattern=r".*norm.*", weight_decay_multiplier=0.0),
                # Second pattern: broader match that also covers norm
                ParamGroupConfig(pattern=r".*layers.*", lr_multiplier=0.5),
            ],
        )
        default_kwargs = OptimizersContainer._build_optimizer_kwargs(config)
        groups = OptimizersContainer._build_param_groups(model, config, default_kwargs)

        # Norm params should be in group index 0 (the norm pattern), not group 1
        norm_group = groups[1]  # first matched group (after default)
        norm_names = _get_param_names_in_group(model, norm_group)
        self.assertTrue(all("norm" in n for n in norm_names))
        # Norm group should have weight_decay=0 (from first pattern)
        self.assertEqual(norm_group["weight_decay"], 0.0)
        # And default lr (lr_multiplier=1.0 from first pattern)
        self.assertEqual(norm_group["lr"], 1e-3)

    def test_betas_override(self):
        """Per-group betas override works correctly."""
        model = SimpleModel()
        config = OptimizersContainer.Config(
            lr=1e-3,
            beta1=0.9,
            beta2=0.95,
            param_groups=[
                # Override both betas
                ParamGroupConfig(pattern=r"embed_tokens\.", beta1=0.85, beta2=0.99),
                # Override only beta2
                ParamGroupConfig(pattern=r".*\.bias$", beta2=0.999),
            ],
        )
        default_kwargs = OptimizersContainer._build_optimizer_kwargs(config)
        groups = OptimizersContainer._build_param_groups(model, config, default_kwargs)

        # Default group keeps global betas
        self.assertEqual(groups[0]["betas"], (0.9, 0.95))
        # Embed group: both overridden
        self.assertEqual(groups[1]["betas"], (0.85, 0.99))
        # Bias group: only beta2 overridden, beta1 stays global
        self.assertEqual(groups[2]["betas"], (0.9, 0.999))

    def test_warning_on_zero_matches(self):
        """Patterns that match no parameters emit a warning."""
        model = SimpleModel()
        config = OptimizersContainer.Config(
            lr=1e-3,
            param_groups=[
                ParamGroupConfig(pattern=r"nonexistent_layer"),
            ],
        )
        default_kwargs = OptimizersContainer._build_optimizer_kwargs(config)

        with self.assertLogs(level=logging.WARNING) as cm:
            groups = OptimizersContainer._build_param_groups(
                model, config, default_kwargs
            )

        self.assertTrue(
            any("nonexistent_layer" in msg for msg in cm.output),
            f"Expected warning about unmatched pattern, got: {cm.output}",
        )
        # All params should be in the default group
        self.assertEqual(len(groups), 1)

    def test_all_params_covered(self):
        """Every requires_grad param appears in exactly one group."""
        model = SimpleModel()
        config = OptimizersContainer.Config(
            lr=1e-3,
            weight_decay=0.1,
            param_groups=[
                ParamGroupConfig(pattern=r".*\.bias$", weight_decay_multiplier=0.0),
                ParamGroupConfig(pattern=r".*norm.*", weight_decay_multiplier=0.0),
            ],
        )
        default_kwargs = OptimizersContainer._build_optimizer_kwargs(config)
        groups = OptimizersContainer._build_param_groups(model, config, default_kwargs)

        all_grouped_params = []
        for g in groups:
            all_grouped_params.extend(g["params"])
        all_model_params = [p for p in model.parameters() if p.requires_grad]

        self.assertEqual(len(all_grouped_params), len(all_model_params))
        self.assertEqual(
            set(id(p) for p in all_grouped_params), set(id(p) for p in all_model_params)
        )


class TestOptimizersContainerWithParamGroups(unittest.TestCase):
    def test_build_optimizer_with_param_groups(self):
        """End-to-end: build OptimizersContainer with param groups."""
        model = SimpleModel()
        config = OptimizersContainer.Config(
            name="AdamW",
            lr=1e-3,
            weight_decay=0.1,
            implementation="for-loop",
            param_groups=[
                ParamGroupConfig(pattern=r".*\.bias$", weight_decay_multiplier=0.0),
            ],
        )
        container = config.build(model_parts=[model])
        self.assertIsInstance(container, OptimizersContainer)

        # Should have 1 optimizer (one model_part)
        self.assertEqual(len(container.optimizers), 1)
        opt = container.optimizers[0]
        # Should have 2 param groups
        self.assertEqual(len(opt.param_groups), 2)

    def test_build_optimizer_default_groups(self):
        """Empty param_groups produces standard single-group behavior."""
        model = SimpleModel()
        config = OptimizersContainer.Config(
            name="AdamW",
            lr=1e-3,
            weight_decay=0.1,
            implementation="for-loop",
        )
        container = config.build(model_parts=[model])
        opt = container.optimizers[0]
        self.assertEqual(len(opt.param_groups), 1)

    def test_build_mixed_optimizer_classes_with_rank_filters(self):
        """Param groups can select Muon for matrix params and AdamW for the rest."""
        model = MixedOptimizerModel()
        config = OptimizersContainer.Config(
            name="AdamW",
            lr=1e-3,
            weight_decay=0.1,
            implementation="for-loop",
            param_groups=[
                ParamGroupConfig(pattern=r".*\.weight$", rank=1),
                ParamGroupConfig(
                    pattern=r"proj\.weight$", rank=2, optimizer_name="Muon"
                ),
                ParamGroupConfig(
                    pattern=r"expert_weight$", rank=3, optimizer_name="Muon"
                ),
            ],
        )

        container = config.build(model_parts=[model])

        self.assertEqual(len(container.optimizers), 2)
        self.assertTrue(any(isinstance(opt, torch.optim.AdamW) for opt in container))
        self.assertTrue(any(_is_muon_optimizer(opt) for opt in container))

        muon_params = {
            name
            for opt in container
            if _is_muon_optimizer(opt)
            for group in opt.param_groups
            for name in _get_param_names_in_group(model, group)
        }
        self.assertEqual(muon_params, {"proj.weight", "expert_weight"})

        for param in model.parameters():
            param.grad = torch.ones_like(param)
        container.step()

    def test_batched_muon_prealloc_matches_none_one_step(self):
        """Preallocated expert-Muon workspace matches the non-prealloc update."""
        env_keys = [
            "TORCHTITAN_OPTIMIZER_MUON_IMPL",
            "TORCHTITAN_OPTIMIZER_MUON_CHUNK_MATRICES",
            "TORCHTITAN_OPTIMIZER_MUON_NS_COMPUTE_DTYPE",
            "TORCHTITAN_OPTIMIZER_MUON_STATE_DTYPE",
            "TORCHTITAN_OPTIMIZER_MUON_WORKSPACE_MODE",
        ]
        old_env = {key: os.environ.get(key) for key in env_keys}
        try:
            torch.manual_seed(1234)
            reference = MixedOptimizerModel()
            candidate = MixedOptimizerModel()
            candidate.load_state_dict(reference.state_dict())

            os.environ["TORCHTITAN_OPTIMIZER_MUON_IMPL"] = "batched"
            os.environ["TORCHTITAN_OPTIMIZER_MUON_CHUNK_MATRICES"] = "1"
            os.environ["TORCHTITAN_OPTIMIZER_MUON_NS_COMPUTE_DTYPE"] = "float32"
            os.environ["TORCHTITAN_OPTIMIZER_MUON_STATE_DTYPE"] = "float32"

            config = OptimizersContainer.Config(
                name="AdamW",
                lr=1e-3,
                weight_decay=0.01,
                implementation="for-loop",
                muon_ns_steps=2,
                param_groups=[
                    ParamGroupConfig(
                        pattern=r"expert_weight$",
                        rank=3,
                        optimizer_name="Muon",
                    ),
                ],
            )

            os.environ["TORCHTITAN_OPTIMIZER_MUON_WORKSPACE_MODE"] = "none"
            reference_container = config.build(model_parts=[reference])
            os.environ["TORCHTITAN_OPTIMIZER_MUON_WORKSPACE_MODE"] = "prealloc"
            candidate_container = config.build(model_parts=[candidate])

            self.assertTrue(any(_is_muon_optimizer(opt) for opt in candidate_container))

            for idx, ((_, ref_param), (_, cand_param)) in enumerate(
                zip(reference.named_parameters(), candidate.named_parameters())
            ):
                torch.manual_seed(9000 + idx)
                grad = torch.randn_like(ref_param)
                ref_param.grad = grad.clone()
                cand_param.grad = grad.clone()

            reference_container.step()
            candidate_container.step()

            for (name, actual), (_, expected) in zip(
                candidate.named_parameters(),
                reference.named_parameters(),
            ):
                self.assertTrue(
                    torch.allclose(actual, expected, atol=1e-6, rtol=1e-6),
                    f"Parameter mismatch for {name}",
                )
        finally:
            for key, value in old_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_swap_opt_states_cpu_matches_adamw_one_step(self):
        """CPU-swapped AdamW matches plain AdamW on a small float32 model."""
        torch.manual_seed(1234)
        model = SimpleModel()
        reference = SimpleModel()
        reference.load_state_dict(model.state_dict())
        input_ids = torch.randint(0, 32, (2, 4))

        config = OptimizersContainer.Config(
            name="AdamW",
            lr=1e-3,
            beta1=0.9,
            beta2=0.95,
            eps=1e-8,
            weight_decay=0.01,
            implementation="swap_opt_states_cpu",
        )
        container = config.build(model_parts=[model])
        reference_optimizer = torch.optim.AdamW(
            reference.parameters(),
            lr=1e-3,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.01,
            fused=False,
            foreach=False,
        )

        model(input_ids).sum().backward()
        reference(input_ids).sum().backward()
        container.step()
        reference_optimizer.step()

        for actual, expected in zip(model.parameters(), reference.parameters()):
            self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))

        for optimizer in container.optimizers:
            for state in optimizer.state.values():
                self.assertEqual(state["exp_avg"].device.type, "cpu")
                self.assertEqual(state["exp_avg_sq"].device.type, "cpu")

    def test_swap_opt_states_cpu_defaults_to_fp32_states(self):
        """Swapped AdamW defaults to FP32 moments even for BF16 parameters."""
        old_state_dtype = os.environ.pop("TORCHTITAN_OPTIMIZER_SWAP_STATE_DTYPE", None)
        old_pin_memory = os.environ.get("TORCHTITAN_OPTIMIZER_SWAP_PIN_MEMORY")
        os.environ["TORCHTITAN_OPTIMIZER_SWAP_PIN_MEMORY"] = "false"
        try:
            model = nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
            config = OptimizersContainer.Config(
                name="AdamW",
                implementation="swap_opt_states_cpu",
            )
            container = config.build(model_parts=[model])
            for param in model.parameters():
                param.grad = torch.ones_like(param)
            container.step()

            for optimizer in container.optimizers:
                for state in optimizer.state.values():
                    self.assertEqual(state["exp_avg"].dtype, torch.float32)
                    self.assertEqual(state["exp_avg_sq"].dtype, torch.float32)
        finally:
            if old_state_dtype is not None:
                os.environ["TORCHTITAN_OPTIMIZER_SWAP_STATE_DTYPE"] = old_state_dtype
            if old_pin_memory is None:
                os.environ.pop("TORCHTITAN_OPTIMIZER_SWAP_PIN_MEMORY", None)
            else:
                os.environ["TORCHTITAN_OPTIMIZER_SWAP_PIN_MEMORY"] = old_pin_memory

    def test_swap_opt_states_cpu_state_dict_round_trip_keeps_cpu_states(self):
        """Native state_dict load restores swapped AdamW states on CPU."""
        model = SimpleModel()
        config = OptimizersContainer.Config(
            name="AdamW",
            implementation="swap_opt_states_cpu",
        )
        container = config.build(model_parts=[model])
        model(torch.randint(0, 32, (2, 4))).sum().backward()
        container.step()

        state_dict = container.state_dict()
        self.assertEqual(set(state_dict), {"optimizer_0"})

        model2 = SimpleModel()
        container2 = config.build(model_parts=[model2])
        container2.load_state_dict(state_dict)

        for optimizer in container2.optimizers:
            for state in optimizer.state.values():
                self.assertEqual(state["exp_avg"].device.type, "cpu")
                self.assertEqual(state["exp_avg_sq"].device.type, "cpu")

    def test_swap_opt_states_cpu_foreach_param_matches_torch_param(self):
        """Batched full-param swap update matches the scalar full-param path."""
        old_update_mode = os.environ.get("TORCHTITAN_OPTIMIZER_SWAP_UPDATE_MODE")
        old_state_dtype = os.environ.get("TORCHTITAN_OPTIMIZER_SWAP_STATE_DTYPE")
        old_pin_memory = os.environ.get("TORCHTITAN_OPTIMIZER_SWAP_PIN_MEMORY")
        old_foreach_max = os.environ.get("TORCHTITAN_OPTIMIZER_SWAP_FOREACH_MAX_PARAMS")
        try:
            torch.manual_seed(1234)
            reference = SimpleModel()
            model = SimpleModel()
            model.load_state_dict(reference.state_dict())
            input_ids = torch.randint(0, 32, (2, 4))

            os.environ["TORCHTITAN_OPTIMIZER_SWAP_STATE_DTYPE"] = "float32"
            os.environ["TORCHTITAN_OPTIMIZER_SWAP_PIN_MEMORY"] = "false"

            os.environ["TORCHTITAN_OPTIMIZER_SWAP_UPDATE_MODE"] = "torch_param"
            reference_container = OptimizersContainer.Config(
                name="AdamW",
                lr=1e-3,
                beta1=0.9,
                beta2=0.95,
                eps=1e-8,
                weight_decay=0.01,
                implementation="swap_opt_states_cpu",
            ).build(model_parts=[reference])

            os.environ["TORCHTITAN_OPTIMIZER_SWAP_UPDATE_MODE"] = "foreach_param"
            os.environ["TORCHTITAN_OPTIMIZER_SWAP_FOREACH_MAX_PARAMS"] = "2"
            container = OptimizersContainer.Config(
                name="AdamW",
                lr=1e-3,
                beta1=0.9,
                beta2=0.95,
                eps=1e-8,
                weight_decay=0.01,
                implementation="swap_opt_states_cpu",
            ).build(model_parts=[model])

            reference(input_ids).sum().backward()
            model(input_ids).sum().backward()
            reference_container.step()
            container.step()

            for actual, expected in zip(model.parameters(), reference.parameters()):
                self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))
        finally:
            if old_update_mode is None:
                os.environ.pop("TORCHTITAN_OPTIMIZER_SWAP_UPDATE_MODE", None)
            else:
                os.environ["TORCHTITAN_OPTIMIZER_SWAP_UPDATE_MODE"] = old_update_mode
            if old_state_dtype is None:
                os.environ.pop("TORCHTITAN_OPTIMIZER_SWAP_STATE_DTYPE", None)
            else:
                os.environ["TORCHTITAN_OPTIMIZER_SWAP_STATE_DTYPE"] = old_state_dtype
            if old_pin_memory is None:
                os.environ.pop("TORCHTITAN_OPTIMIZER_SWAP_PIN_MEMORY", None)
            else:
                os.environ["TORCHTITAN_OPTIMIZER_SWAP_PIN_MEMORY"] = old_pin_memory
            if old_foreach_max is None:
                os.environ.pop("TORCHTITAN_OPTIMIZER_SWAP_FOREACH_MAX_PARAMS", None)
            else:
                os.environ["TORCHTITAN_OPTIMIZER_SWAP_FOREACH_MAX_PARAMS"] = old_foreach_max

    def test_swap_opt_states_cpu_rejects_non_adam_optimizer(self):
        """The AMD swap port is Adam/AdamW-only for now."""
        with self.assertRaisesRegex(ValueError, "Adam/AdamW"):
            OptimizersContainer.Config(
                name="Muon",
                implementation="swap_opt_states_cpu",
            )


class TestOptimizersInBackwardWithParamGroups(unittest.TestCase):
    def test_build_with_param_groups(self):
        """OptimizersInBackwardContainer respects param groups."""
        model = SimpleModel()
        config = OptimizersInBackwardContainer.Config(
            name="AdamW",
            lr=1e-3,
            weight_decay=0.1,
            implementation="for-loop",
            param_groups=[
                ParamGroupConfig(pattern=r".*\.bias$", weight_decay_multiplier=0.0),
            ],
        )
        container = config.build(model_parts=[model])
        self.assertIsInstance(container, OptimizersInBackwardContainer)

        # Check that bias params have weight_decay=0
        param_to_name = {p: n for n, p in model.named_parameters()}
        for opt in container.optimizers:
            for pg in opt.param_groups:
                for p in pg["params"]:
                    name = param_to_name.get(p, "")
                    if name.endswith(".bias"):
                        self.assertEqual(
                            pg["weight_decay"],
                            0.0,
                            f"Bias param {name} should have weight_decay=0",
                        )


class TestDCPWithParamGroups(unittest.TestCase):
    def test_unused_trainable_param_gets_checkpointable_adam_state(self):
        """Lazy Adam state is materialized for trainable branch-inactive params."""
        model = UnusedParamModel()
        config = OptimizersContainer.Config(
            name="AdamW",
            implementation="for-loop",
        )
        container = config.build(model_parts=[model])

        output = model(torch.randn(2, 4))
        output.sum().backward()
        container.step()

        state_dict = container.state_dict()
        self.assertIn("state.unused.step", state_dict)
        self.assertIn("state.unused.exp_avg", state_dict)
        self.assertIn("state.unused.exp_avg_sq", state_dict)

        model2 = UnusedParamModel()
        container2 = config.build(model_parts=[model2])
        container2.load_state_dict(state_dict)
        self.assertEqual(set(state_dict.keys()), set(container2.state_dict().keys()))

    def test_state_dict_round_trip(self):
        """Optimizer state_dict save/load works with multiple param groups."""
        model = SimpleModel()
        config = OptimizersContainer.Config(
            name="AdamW",
            lr=1e-3,
            weight_decay=0.1,
            implementation="for-loop",
            param_groups=[
                ParamGroupConfig(pattern=r".*\.bias$", weight_decay_multiplier=0.0),
                ParamGroupConfig(pattern=r"embed_tokens\.", lr_multiplier=0.1),
            ],
        )
        container = config.build(model_parts=[model])

        # Run a step to populate optimizer state
        dummy_input = torch.randint(0, 32, (2, 4))
        output = model(dummy_input)
        output.sum().backward()
        container.step()

        # Save state dict
        state_dict = container.state_dict()
        self.assertIsInstance(state_dict, dict)
        self.assertTrue(len(state_dict) > 0)

        # Load into a fresh container
        model2 = SimpleModel()
        container2 = config.build(model_parts=[model2])
        container2.load_state_dict(state_dict)

        # Verify state was restored by checking optimizer states match
        state_dict2 = container2.state_dict()
        self.assertEqual(set(state_dict.keys()), set(state_dict2.keys()))

        for key in state_dict:
            v1 = state_dict[key]
            v2 = state_dict2[key]
            if isinstance(v1, torch.Tensor):
                self.assertTrue(
                    torch.equal(v1, v2),
                    f"State mismatch for key {key}",
                )

    def test_mixed_optimizer_state_dict_round_trip(self):
        """Mixed optimizer-class state_dict save/load keeps states separated."""
        model = MixedOptimizerModel()
        config = OptimizersContainer.Config(
            name="AdamW",
            lr=1e-3,
            weight_decay=0.1,
            implementation="for-loop",
            param_groups=[
                ParamGroupConfig(pattern=r".*\.weight$", rank=1),
                ParamGroupConfig(
                    pattern=r"proj\.weight$", rank=2, optimizer_name="Muon"
                ),
                ParamGroupConfig(
                    pattern=r"expert_weight$", rank=3, optimizer_name="Muon"
                ),
            ],
        )
        container = config.build(model_parts=[model])

        for param in model.parameters():
            param.grad = torch.ones_like(param)
        container.step()

        state_dict = container.state_dict()
        self.assertEqual(set(state_dict), {"optimizer_0", "optimizer_1"})

        model2 = MixedOptimizerModel()
        container2 = config.build(model_parts=[model2])
        container2.load_state_dict(state_dict)

        state_dict2 = container2.state_dict()
        self.assertEqual(set(state_dict.keys()), set(state_dict2.keys()))

    def test_mixed_optimizer_two_model_parts_use_native_state_dict(self):
        """Mixed optimizer state format is stable with several model parts."""
        model1 = MixedOptimizerModel()
        model2 = MixedOptimizerModel()
        config = OptimizersContainer.Config(
            name="AdamW",
            lr=1e-3,
            weight_decay=0.1,
            implementation="for-loop",
            param_groups=[
                ParamGroupConfig(pattern=r".*\.weight$", rank=1),
                ParamGroupConfig(
                    pattern=r"proj\.weight$", rank=2, optimizer_name="Muon"
                ),
                ParamGroupConfig(
                    pattern=r"expert_weight$", rank=3, optimizer_name="Muon"
                ),
            ],
        )
        container = config.build(model_parts=[model1, model2])

        for model in (model1, model2):
            for param in model.parameters():
                param.grad = torch.ones_like(param)
        container.step()

        state_dict = container.state_dict()
        self.assertEqual(
            set(state_dict),
            {"optimizer_0", "optimizer_1", "optimizer_2", "optimizer_3"},
        )

        model3 = MixedOptimizerModel()
        model4 = MixedOptimizerModel()
        container2 = config.build(model_parts=[model3, model4])
        container2.load_state_dict(state_dict)

        state_dict2 = container2.state_dict()
        self.assertEqual(set(state_dict.keys()), set(state_dict2.keys()))


if __name__ == "__main__":
    unittest.main()
