# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import json
import os
import sys
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from typing import Annotated, Any, cast

import torch
import torch.distributed.checkpoint.stateful
import tyro
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.tensor import DTensor

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.dataloader import BaseDataLoader, DataloaderExhaustedError
from torchtitan.components.loss import BaseLoss, ChunkedCELoss, IGNORE_INDEX
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import ensure_pp_loss_visible, MetricsProcessor
from torchtitan.components.optimizer import (
    OptimizersContainer,
    OptimizersInBackwardContainer,
)
from torchtitan.components.quantization.utils import has_quantization
from torchtitan.components.tokenizer import BaseTokenizer, HuggingFaceTokenizer
from torchtitan.components.validate import BaseValidator, Validator
from torchtitan.config import Configurable, TORCH_DTYPE_MAP
from torchtitan.config.configs import (
    ActivationCheckpointConfig,
    CommConfig,
    CompileConfig,
    DebugConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.distributed import full_dtensor, ParallelDims, utils as dist_utils
from torchtitan.distributed.context_parallel import prepare_context_parallel_input

from torchtitan.models.common.attention import FlexAttention, VarlenAttention
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.common.dsv4_profile_timing import (
    Dsv4AdditiveStepTimer,
    clear_dsv4_module_backward_timing,
    dsv4_execution_region,
    dsv4_nvtx_range,
    flush_dsv4_module_backward_timing,
)
from torchtitan.observability import structured_logger as sl
from torchtitan.protocols import BaseModel
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.tools import utils
from torchtitan.tools.logging import logger
from torchtitan.tools.profiler import Profiler


class Trainer(torch.distributed.checkpoint.stateful.Stateful, Configurable):
    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """
        Default container for training configuration.
        """

        # NOTE: model_spec is suppressed from tyro CLI parsing and is always
        # set programmatically by the model registry before Trainer construction.
        model_spec: Annotated[ModelSpec | None, tyro.conf.Suppress] = None

        hf_assets_path: str = "./tests/assets/tokenizer"
        """
        Path to HF assets folder. This folder contains local copies of Hugging Face assets,
        including model weights in .safetensors format, the model.safetensor.index.json file
        (fqn to file mapping), the config.json file, generation_config.json, and tokenizer files.
        """

        dump_folder: str = "./outputs"
        """Folder to dump job outputs"""

        profiler: Profiler.Config = field(default_factory=Profiler.Config)
        metrics: MetricsProcessor.Config = field(
            default_factory=MetricsProcessor.Config
        )
        tokenizer: BaseTokenizer.Config = field(
            default_factory=HuggingFaceTokenizer.Config
        )
        dataloader: BaseDataLoader.Config = field(default_factory=BaseDataLoader.Config)
        optimizer: OptimizersContainer.Config = field(
            default_factory=OptimizersContainer.Config
        )
        lr_scheduler: LRSchedulersContainer.Config = field(
            default_factory=LRSchedulersContainer.Config
        )
        training: TrainingConfig = field(default_factory=TrainingConfig)
        parallelism: ParallelismConfig = field(default_factory=ParallelismConfig)
        checkpoint: CheckpointManager.Config = field(
            default_factory=CheckpointManager.Config
        )
        activation_checkpoint: ActivationCheckpointConfig = field(
            default_factory=ActivationCheckpointConfig
        )
        compile: CompileConfig = field(default_factory=CompileConfig)
        comm: CommConfig = field(default_factory=CommConfig)
        validator: Validator.Config = field(default_factory=Validator.Config)
        debug: DebugConfig = field(default_factory=DebugConfig)
        loss: BaseLoss.Config = field(default_factory=BaseLoss.Config)

        def __post_init__(self):
            if self.debug.batch_invariant:
                raise ValueError(
                    "Batch-invariant mode is not supported in pre-training."
                )
            if isinstance(self.optimizer, OptimizersInBackwardContainer.Config):
                if self.parallelism.expert_parallel_degree > 1:
                    raise NotImplementedError(
                        "Optimizers in backward is not supported with Expert Parallel."
                    )
                if self.parallelism.pipeline_parallel_degree > 1:
                    raise NotImplementedError(
                        "Optimizers in backward is not supported with Pipeline Parallel."
                    )

        def to_dict(self) -> dict[str, Any]:
            d = {}
            for f in dataclasses.fields(self):
                if f.name == "model_spec":
                    assert self.model_spec is not None
                    # ModelSpec contains callables that can't be serialized
                    d["model_spec"] = {
                        "name": self.model_spec.name,
                        "flavor": self.model_spec.flavor,
                    }
                else:
                    val = getattr(self, f.name)
                    if hasattr(val, "to_dict"):
                        d[f.name] = val.to_dict()
                    elif dataclasses.is_dataclass(val):
                        d[f.name] = asdict(val)
                    else:
                        d[f.name] = val
            return d

        def maybe_log(self) -> None:
            if self.debug.print_config:
                logger.info(
                    f"Running with configs: {json.dumps(self.to_dict(), indent=2, ensure_ascii=False)}"
                )

            if self.debug.save_config_file is not None:
                config_file = os.path.join(
                    self.dump_folder, self.debug.save_config_file
                )
                if torch.distributed.is_initialized():
                    if torch.distributed.get_rank() == 0:
                        os.makedirs(os.path.dirname(config_file), exist_ok=True)
                        with open(config_file, "w") as f:
                            json.dump(self.to_dict(), f, indent=2)
                    logger.info(f"Saved job configs to {config_file}")
                else:
                    logger.warning(
                        "Job configs logging is disabled due to torch.distributed not initialized."
                    )

    # core configs
    config: Config
    parallel_dims: ParallelDims

    # swappable training components
    tokenizer: BaseTokenizer
    dataloader: BaseDataLoader
    model_config: BaseModel.Config
    # TODO: we should make this list[BaseModel / Decoder] but this will affect many components.
    # will do this in a separate PR
    model_parts: list[torch.nn.Module]
    loss_fn: BaseLoss
    optimizers: OptimizersContainer
    lr_schedulers: LRSchedulersContainer
    validator: BaseValidator
    metrics_processor: MetricsProcessor
    checkpointer: CheckpointManager

    # runtime utilities
    device: torch.device
    gc_handler: utils.GarbageCollection
    train_context: dist_utils.TrainContext
    gradient_accumulation_steps: int
    pp_has_first_stage: bool
    pp_has_last_stage: bool

    # additional training states
    step: int
    ntokens_seen: int

    @contextmanager
    def _dsv4_additive_phase(self, name: str, **meta: Any):
        timer = getattr(self, "_dsv4_additive_timer", None)
        with dsv4_nvtx_range(f"dsv4.phase.{name}"):
            if timer is None:
                yield
                return
            with timer.phase(name, **meta):
                yield

    @staticmethod
    def _dsv4_env_flag(name: str, default: bool = False) -> bool:
        value = os.environ.get(name)
        if value is None or value == "":
            return default
        return value.strip().lower() in {"1", "true", "yes", "on"}

    def _dsv4_maybe_scan_nonfinite_grads(self) -> None:
        path = os.environ.get("TORCHTITAN_DSV4_GRAD_NONFINITE_SCAN_PATH", "").strip()
        if not path:
            return
        max_steps = int(
            os.environ.get("TORCHTITAN_DSV4_GRAD_NONFINITE_SCAN_MAX_STEPS", "1")
        )
        if max_steps >= 0 and self.step > max_steps:
            return

        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        record: dict[str, Any] = {
            "rank": rank,
            "step": self.step,
            "event": "pre_clip_grad_nonfinite_scan",
            "checked_grads": 0,
            "none_grads": 0,
            "first_nonfinite": None,
        }
        for module_idx, module in enumerate(self.model_parts):
            for name, param in module.named_parameters():
                grad = param.grad
                if grad is None:
                    record["none_grads"] += 1
                    continue
                record["checked_grads"] += 1
                local_grad = grad.to_local() if isinstance(grad, DTensor) else grad
                if bool(torch.isfinite(local_grad).all().item()):
                    continue

                nan_count = int(torch.isnan(local_grad).sum().item())
                posinf_count = int(torch.isposinf(local_grad).sum().item())
                neginf_count = int(torch.isneginf(local_grad).sum().item())
                finite = local_grad[torch.isfinite(local_grad)]
                finite_abs_max = (
                    float(finite.abs().max().item()) if finite.numel() > 0 else None
                )
                record["first_nonfinite"] = {
                    "module_idx": module_idx,
                    "parameter": name,
                    "grad_type": type(grad).__name__,
                    "local_shape": list(local_grad.shape),
                    "dtype": str(local_grad.dtype),
                    "nan_count": nan_count,
                    "posinf_count": posinf_count,
                    "neginf_count": neginf_count,
                    "finite_abs_max": finite_abs_max,
                }
                break
            if record["first_nonfinite"] is not None:
                break

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    # Enable debug tracing on failure: https://pytorch.org/docs/stable/elastic/errors.html
    @record
    def __init__(self, config: Config):
        torch._C._log_api_usage_once("torchtitan.train")

        self.config = config
        assert (
            config.model_spec is not None
        ), "model_spec must be set before creating Trainer"
        model_spec = config.model_spec

        device_module, device_type = utils.device_module, utils.device_type
        # pyrefly: ignore [read-only]
        self.device = torch.device(f"{device_type}:{int(os.environ['LOCAL_RANK'])}")
        # Device has to be set before creating TorchFT manager.
        device_module.set_device(self.device)

        # init distributed and build meshes
        self.parallel_dims = parallel_dims = self.init_distributed()

        # Logging needs to happen after distributed initialized
        config.maybe_log()

        if parallel_dims.dp_enabled:
            batch_mesh = parallel_dims.get_mesh("batch")
            batch_degree, batch_rank = batch_mesh.size(), batch_mesh.get_local_rank()
        else:
            batch_degree, batch_rank = 1, 0

        # take control of garbage collection to avoid stragglers
        self.gc_handler = utils.GarbageCollection(
            gc_freq=config.training.gc_freq, debug=config.training.gc_debug
        )

        # Set random seed, and maybe enable deterministic mode
        # (mainly for debugging, expect perf loss).
        dist_utils.set_determinism(
            parallel_dims,
            self.device,
            config.debug,
            distinct_seed_mesh_dims=["pp"],
        )
        # build tokenizer
        self.tokenizer = config.tokenizer.build(tokenizer_path=config.hf_assets_path)

        # build dataloader
        dataloader_seq_len = config.training.seq_len + max(
            0, int(getattr(config.training, "num_mtp_modules", 0))
        )
        self.dataloader = config.dataloader.build(
            dp_world_size=batch_degree,
            dp_rank=batch_rank,
            tokenizer=self.tokenizer,
            seq_len=dataloader_seq_len,
            local_batch_size=config.training.local_batch_size,
        )

        # build model (using meta init)
        model_config = model_spec.model
        # set the model args from training job configs
        model_config.update_from_config(
            trainer_config=config,
        )
        self.model_config = model_config

        logger.info(f"Building {model_spec.name} {model_spec.flavor}")

        with (
            torch.device("meta"),
            utils.set_default_dtype(TORCH_DTYPE_MAP[config.training.dtype]),
        ):
            model = model_config.build()

        # Verify all submodules satisfy the Module protocol
        # TODO: move this to module validate().
        # This is current put here to verify module build and
        # converter, which should guanrantee Module protocol.
        # On the other hand, some parallelism wrappers don't
        # have this guanrantee, e.g., fully_shard.
        model.verify_module_protocol()

        # metrics logging
        self.metrics_processor = config.metrics.build(
            parallel_dims=parallel_dims,
            dump_folder=config.dump_folder,
            pp_schedule=config.parallelism.pipeline_parallel_schedule,
            config_dict=config.to_dict(),
            has_quantization=has_quantization(model_config),
        )
        color = self.metrics_processor.color

        # calculate model size and flops per token
        (
            model_param_count,
            self.metrics_processor.num_flops_per_token,
        ) = model_config.get_nparams_and_flops(model, config.training.seq_len)

        logger.info(
            f"{color.blue}Model {model_spec.name} {model_spec.flavor} "
            f"{color.red}size: {model_param_count:,} total parameters{color.reset}"
        )

        # move sharded model to CPU/GPU and initialize weights via DTensor
        buffer_device: torch.device | None
        if config.checkpoint.create_seed_checkpoint:
            init_device = "cpu"
            buffer_device = None
        elif config.training.enable_cpu_offload:
            init_device = "cpu"
            buffer_device = torch.device(device_type)
        else:
            init_device = device_type
            buffer_device = None

        self.loss_fn = config.loss.build(
            compile_config=config.compile,
        )
        set_mtp_loss_weight = getattr(self.loss_fn, "set_mtp_loss_weight", None)
        if callable(set_mtp_loss_weight):
            set_mtp_loss_weight(config.training.mtp_loss_weight)

        # verify batch sizes
        global_batch_size = config.training.global_batch_size
        if global_batch_size < 0:
            # This global batch size results in 1 gradient accumulation
            # step.
            global_batch_size = config.training.local_batch_size * batch_degree
        assert global_batch_size > 0
        assert (
            global_batch_size % (config.training.local_batch_size * batch_degree) == 0
        ), (
            f"global batch size must be multiple of local batch size times "
            f"data-parallel degree ({global_batch_size} "
            f"% ({config.training.local_batch_size} * {batch_degree}) != 0)"
        )

        # calculate gradient accumulation steps
        self.gradient_accumulation_steps = global_batch_size // (
            config.training.local_batch_size * batch_degree
        )
        assert self.gradient_accumulation_steps > 0

        # apply parallelisms and initialization
        with sl.log_trace_span("model_parallelism_init"):
            if parallel_dims.pp_enabled:
                if not model_spec.pipelining_fn:
                    raise RuntimeError(
                        f"Pipeline Parallel is enabled but {model_spec.name} "
                        f"does not support pipelining"
                    )

                # apply both Pipeline Parallel and SPMD-style scaling techniques
                (
                    self.pp_schedule,
                    self.model_parts,
                    self.pp_has_first_stage,
                    self.pp_has_last_stage,
                ) = model_spec.pipelining_fn(
                    model,
                    parallel_dims=parallel_dims,
                    training=config.training,
                    parallelism=config.parallelism,
                    compile_config=config.compile,
                    ac_config=config.activation_checkpoint,
                    dump_folder=config.dump_folder,
                    device=self.device,
                    model_config=model_config,
                    parallelize_fn=model_spec.parallelize_fn,
                    loss_fn=self.loss_fn,
                )
                # when PP is enabled, `model` obj is no longer used after this point,
                # model_parts is used instead
                del model

                for m in self.model_parts:
                    m.to_empty(device=init_device)
                    with torch.no_grad():
                        # TODO: Change this back to init_weights once
                        # autoparallel contains the wrap_init_states
                        cast(BaseModel, m).init_weights(buffer_device=buffer_device)
                    m.train()

                # confirm that user will be able to view loss metrics on the console
                ensure_pp_loss_visible(
                    parallel_dims=parallel_dims,
                    pp_schedule=config.parallelism.pipeline_parallel_schedule,
                    color=color,
                )
            else:
                if not config.checkpoint.create_seed_checkpoint:
                    # Skip parallelize_fn for seed checkpoints — nothing from
                    # it is needed (AC, compile, nD parallelism, mixed precision, etc.).
                    model = model_spec.parallelize_fn(
                        model,
                        parallel_dims=parallel_dims,
                        training=config.training,
                        parallelism=config.parallelism,
                        compile_config=config.compile,
                        ac_config=config.activation_checkpoint,
                        dump_folder=config.dump_folder,
                    )

                model.to_empty(device=init_device)
                with torch.no_grad():
                    # TODO: Change this back to init_weights once
                    # autoparallel contains the wrap_init_states
                    cast(BaseModel, model).init_weights(buffer_device=buffer_device)
                model.train()

                self.model_parts = [model]

        # Set lm_head reference for ChunkedCELoss after model construction.
        # Non-PP: single model part always has lm_head.
        # PP: only the last stage has lm_head; non-last stages skip this.
        if isinstance(self.loss_fn, ChunkedCELoss):
            if parallel_dims.pp_enabled:
                if self.pp_has_last_stage:
                    lm_head = self.model_parts[-1].lm_head
                    assert (
                        lm_head is not None
                    ), "Last PP stage must have lm_head for ChunkedCELoss"
                    self.loss_fn.set_lm_head(
                        lm_head  # pyrefly: ignore[bad-argument-type]
                    )
                    self.model_parts[
                        -1
                    ]._skip_lm_head = True  # pyrefly: ignore[bad-argument-type]
            else:
                assert len(self.model_parts) == 1
                lm_head = self.model_parts[0].lm_head
                assert lm_head is not None, "Model must have lm_head for ChunkedCELoss"
                self.loss_fn.set_lm_head(lm_head)  # pyrefly: ignore[bad-argument-type]
                self.model_parts[
                    0
                ]._skip_lm_head = True  # pyrefly: ignore[bad-argument-type]

        # initialize device memory monitor and get peak flops for MFU calculation
        device_memory_monitor = self.metrics_processor.device_memory_monitor
        gpu_peak_flops = utils.get_peak_flops(device_memory_monitor.device_name)
        logger.info(f"Peak FLOPS used for computing MFU: {gpu_peak_flops:.3e}")
        device_mem_stats = device_memory_monitor.get_peak_stats()
        logger.info(
            f"{device_type.upper()} memory usage for model: "
            f"{device_mem_stats.max_reserved_gib:.2f}GiB"
            f"({device_mem_stats.max_reserved_pct:.2f}%)"
        )

        # build optimizer after applying parallelisms to the model
        self.optimizers = config.optimizer.build(model_parts=self.model_parts)
        if model_spec.post_optimizer_build_fn is not None:
            model_spec.post_optimizer_build_fn(
                self.optimizers, self.model_parts, parallel_dims
            )
        self.lr_schedulers = config.lr_scheduler.build(
            optimizers=self.optimizers,
            training_steps=config.training.steps,
        )
        self.metrics_processor.optimizers = self.optimizers
        self.metrics_processor.model_parts = self.model_parts

        # Initialize trainer states that will be saved in checkpoint.
        # These attributes must be initialized before checkpoint loading.
        self.step = 0
        self.ntokens_seen = 0

        self.checkpointer = config.checkpoint.build(
            dataloader=self.dataloader,
            model_parts=self.model_parts,
            optimizers=self.optimizers,
            lr_schedulers=self.lr_schedulers,
            states={"train_state": self},
            sd_adapter=(
                model_spec.state_dict_adapter(model_config, config.hf_assets_path)
                if model_spec.state_dict_adapter
                else None
            ),
            base_folder=config.dump_folder,
        )

        loss_parallel_enabled = (
            parallel_dims.tp_enabled and not config.parallelism.disable_loss_parallel
        )
        self.train_context = dist_utils.get_train_context(loss_parallel_enabled)

        # Build validator if validation is configured
        if config.validator.enable:
            pp_schedule, pp_has_first_stage, pp_has_last_stage = (
                (
                    self.pp_schedule,
                    self.pp_has_first_stage,
                    self.pp_has_last_stage,
                )
                if parallel_dims.pp_enabled
                else (None, None, None)
            )

            self.validator = config.validator.build(
                parallelism=config.parallelism,
                dp_world_size=batch_degree,
                dp_rank=batch_rank,
                tokenizer=self.tokenizer,
                parallel_dims=parallel_dims,
                loss_fn=self.loss_fn,
                validation_context=self.train_context,
                metrics_processor=self.metrics_processor,
                seq_len=dataloader_seq_len,
                local_batch_size=config.training.local_batch_size,
                pp_schedule=pp_schedule,
                pp_has_first_stage=pp_has_first_stage,
                pp_has_last_stage=pp_has_last_stage,
            )

        logger.info(
            "Trainer is initialized with "
            f"local batch size {config.training.local_batch_size}, "
            f"global batch size {global_batch_size}, "
            f"gradient accumulation steps {self.gradient_accumulation_steps}, "
            f"sequence length {config.training.seq_len}, "
            f"total steps {config.training.steps} "
            f"(warmup {config.lr_scheduler.warmup_steps})"
        )

    @sl.log_trace_span("torch_distributed_init")
    def init_distributed(self) -> ParallelDims:
        config = self.config
        world_size = dist_utils.init_distributed(
            config.comm,
            enable_cpu_backend=config.training.enable_cpu_offload,
            base_folder=config.dump_folder,
        )

        return ParallelDims.from_config(config.parallelism, world_size)

    def batch_generator(
        self, data_iterable: Iterable[tuple[dict[str, torch.Tensor], torch.Tensor]]
    ) -> Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]:
        """Returns an iterator that processes batches from the data iterator.

        Note: Tensors are yielded on CPU. The caller is responsible for moving
        them to GPU when needed. This allows for more efficient memory usage
        when doing gradient accumulation.
        """
        data_iterator = iter(data_iterable)

        while True:
            data_load_start = time.perf_counter()
            try:
                batch = next(data_iterator)
            except StopIteration as ex:
                # If data runs out during gradient accumulation, that
                # entire step will not be executed.
                raise DataloaderExhaustedError() from ex
            input_dict, labels = batch
            ntokens_batch = labels.numel()
            self.metrics_processor.ntokens_since_last_log += ntokens_batch
            self.metrics_processor.data_loading_times.append(
                time.perf_counter() - data_load_start
            )

            # Tensors stay on CPU; moved to GPU per-microbatch during training
            yield input_dict, labels

    @sl.log_trace_span("post_dataloading_process")
    def post_dataloading_process(
        self, input_dict: dict[str, torch.Tensor], labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
        """
        Post-processing hook after data loading and before model forward pass.

        This method processes the raw data from the dataloader and prepares it for
        the model's forward pass. It separates the main input tensor from auxiliary
        inputs and constructs additional keyword arguments (e.g., attention masks).

        This method can be overridden in subclasses to customize data processing
        for different training strategies (e.g., converting tensors to DTensors,
        applying custom transformations, etc.).

        Args:
            input_dict: Dictionary containing tensors from the dataloader. Must
                contain an "input" key with the main input tensor. May contain
                additional keys for auxiliary inputs (e.g., position ids).
            labels: Target labels for the batch.

        Returns:
            A tuple of (inputs, labels, extra_inputs, extra_kwargs) where:
                - inputs: Main input tensor extracted from input_dict["input"].
                - labels: Target labels (unchanged from input parameter).
                - extra_inputs: Dict of auxiliary input tensors from input_dict
                    (excluding "input" and "positions"). These are passed to the
                    model forward but are NOT forwarded across pipeline parallel
                    stages.
                - extra_kwargs: Dict of additional keyword arguments for model
                    forward (positions, attention_masks). These ARE forwarded
                    across all pipeline parallel stages.

        Note:
            The distinction between extra_inputs and extra_kwargs is important for
            pipeline parallelism: extra_kwargs are forwarded to all pipeline stages,
            while extra_inputs are only available to the first stage. Positions
            always go into extra_kwargs so every stage can apply RoPE correctly.
        """
        inputs = input_dict["input"]
        extra_inputs = {k: v for k, v in input_dict.items() if k != "input"}
        # extra_kwargs are forwarded to all PP stages; extra_inputs are only
        # available to the first stage.  Positions go into extra_kwargs so
        # every stage can apply RoPE correctly.
        extra_kwargs: dict[str, Any] = {}

        positions = extra_inputs.pop("positions", None)

        if isinstance(self.model_config, Decoder.Config):
            attn_config = self.model_config.layers[0].attention
            inner_attention = attn_config.inner_attention

            if attn_config.mask_type == "block_causal":
                assert (
                    positions is not None
                ), "block_causal mask requires per-document positions from the dataloader"
            else:
                positions = torch.arange(
                    inputs.shape[1], dtype=torch.int32, device=inputs.device
                ).repeat(inputs.shape[0], 1)

            if isinstance(
                inner_attention, (FlexAttention.Config, VarlenAttention.Config)
            ):
                model = cast(Decoder, self.model_parts[0])
                extra_kwargs["attention_masks"] = model.get_attention_masks(
                    positions=positions,
                )

        extra_kwargs["positions"] = positions

        if self.parallel_dims.cp_enabled:
            inputs, labels, extra_kwargs = prepare_context_parallel_input(
                inputs,
                labels,
                extra_kwargs,
                self.parallel_dims.get_mesh("cp"),
                self.device,
                self.config.parallelism.context_parallel_load_balancer,
            )

        # Accumulate after CP sharding so labels.numel() reflects the actual
        # unique tokens this rank processes (not the full pre-split sequence).
        self.ntokens_seen += labels.numel()

        if self.config.parallelism.full_dtensor:
            inputs, labels, extra_kwargs = full_dtensor.parallelize_inputs(
                self.parallel_dims, inputs, labels, extra_kwargs
            )

        return inputs, labels, extra_inputs, extra_kwargs

    @sl.log_trace_span("fwd_bwd")
    def forward_backward_step(
        self,
        *,
        input_dict: dict[str, torch.Tensor],
        labels: torch.Tensor,
        global_valid_tokens: torch.Tensor,
    ) -> torch.Tensor:
        model_parts = self.model_parts
        parallel_dims = self.parallel_dims

        with self._dsv4_additive_phase("forward_backward.post_dataloading_process"):
            inputs, labels, extra_inputs, extra_kwargs = self.post_dataloading_process(
                input_dict, labels
            )

        if parallel_dims.pp_enabled:
            # Pipeline Parallel forward / backward inside step() call
            loss_kwargs = {"global_valid_tokens": global_valid_tokens}
            with self.train_context():
                targets, losses = (
                    (labels, []) if self.pp_has_last_stage else (None, None)
                )
                with self._dsv4_additive_phase("forward_backward.pipeline_step"):
                    if self.pp_has_first_stage:
                        self.pp_schedule.step(
                            inputs,
                            **extra_inputs,
                            **extra_kwargs,
                            target=targets,
                            losses=losses,
                            loss_kwargs=loss_kwargs,
                            return_outputs=False,
                        )
                    else:
                        self.pp_schedule.step(
                            **extra_kwargs,
                            target=targets,
                            losses=losses,
                            loss_kwargs=loss_kwargs,
                            return_outputs=False,
                        )

            # accumulate losses across pipeline microbatches
            # TODO: PP+FSDP unexpectedly puts the loss back to the CPU
            if self.pp_has_last_stage:
                assert losses is not None
                # All loss classes scale by global_valid_tokens internally
                loss = torch.sum(torch.stack(losses)).to(self.device)
            else:
                loss = torch.tensor([-1.0], device=self.device)
        else:
            # Non-PP forward / backward
            assert len(model_parts) == 1
            train_context = self.train_context()
            with self._dsv4_additive_phase("forward_backward.train_context_enter"):
                train_context.__enter__()
            try:
                with dsv4_execution_region("forward"):
                    with self._dsv4_additive_phase("forward_backward.model_forward"):
                        pred = model_parts[0](inputs, **extra_inputs, **extra_kwargs)
                # Under non-full_dtensor, labels stay as plain tensors. See
                # ``cross_entropy_loss`` for why pred must also be plain.
                # Remove once non-full_dtensor is no longer supported.
                if (
                    isinstance(pred, DTensor)
                    and not self.config.parallelism.full_dtensor
                    and self.config.parallelism.disable_loss_parallel
                ):
                    with self._dsv4_additive_phase("forward_backward.pred_to_local"):
                        pred = pred.to_local()
                else:
                    with self._dsv4_additive_phase("forward_backward.pred_to_local"):
                        pass
                with dsv4_execution_region("loss"):
                    with self._dsv4_additive_phase("forward_backward.loss"):
                        loss = self.loss_fn(pred, labels, global_valid_tokens)
                with self._dsv4_additive_phase("forward_backward.drop_pred"):
                    del pred
                with dsv4_execution_region("backward"):
                    with self._dsv4_additive_phase("forward_backward.backward"):
                        clear_dsv4_module_backward_timing()
                        loss.backward()
                        flush_dsv4_module_backward_timing(
                            {
                                "step": self.step,
                                "sequence_length": self.config.training.seq_len,
                                "global_batch_size": (
                                    self.config.training.global_batch_size
                                ),
                                "local_batch_size": (
                                    self.config.training.local_batch_size
                                ),
                                "gradient_accumulation_steps": (
                                    self.gradient_accumulation_steps
                                ),
                            }
                        )
            except BaseException:
                exc_info = sys.exc_info()
                with self._dsv4_additive_phase("forward_backward.train_context_exit"):
                    train_context.__exit__(*exc_info)
                raise
            else:
                with self._dsv4_additive_phase("forward_backward.train_context_exit"):
                    train_context.__exit__(None, None, None)

        with self._dsv4_additive_phase("forward_backward.post_backward_finalize"):
            pass

        with self._dsv4_additive_phase("forward_backward.release_tensors"):
            del inputs, labels, extra_inputs, extra_kwargs

        # The returned loss here is local SUM loss / global_valid_tokens
        with self._dsv4_additive_phase("forward_backward.return_loss"):
            return_detached_loss = self._dsv4_env_flag(
                "TORCHTITAN_DSV4_RETURN_DETACHED_LOSS"
            )
            returned_loss = loss.detach() if return_detached_loss else loss

        if self._dsv4_env_flag("TORCHTITAN_DSV4_RETAIN_ORIGINAL_LOSS_AFTER_BACKWARD"):
            with self._dsv4_additive_phase("forward_backward.retain_original_loss"):
                retained_losses = getattr(self, "_dsv4_retained_original_losses", None)
                if retained_losses is None:
                    retained_losses = []
                    self._dsv4_retained_original_losses = retained_losses
                retained_losses.append(loss)

        if self._dsv4_env_flag("TORCHTITAN_DSV4_DROP_ORIGINAL_LOSS_BEFORE_RETURN"):
            with self._dsv4_additive_phase("forward_backward.drop_original_loss"):
                del loss

        return returned_loss

    def train_step(
        self, data_iterator: Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]
    ):
        with self._dsv4_additive_phase("train_step.zero_grad"):
            self.optimizers.zero_grad()
        # Save the current step learning rate for logging
        with self._dsv4_additive_phase("train_step.lr_read"):
            lr = self.lr_schedulers.schedulers[0].get_last_lr()[0]

        # Keep these variables local to shorten the code as these are
        # the major variables that are used in the training loop.
        parallel_dims = self.parallel_dims

        # Collect all microbatches on CPU and count total valid tokens
        microbatches = []
        local_valid_tokens = torch.tensor(0, dtype=torch.int64)
        with self._dsv4_additive_phase("train_step.fetch_microbatches"):
            for _microbatch in range(self.gradient_accumulation_steps):
                with sl.log_trace_span("fetching_batch"):
                    input_dict, labels = next(data_iterator)
                    num_mtp_modules = max(
                        0, int(getattr(self.config.training, "num_mtp_modules", 0))
                    )
                    main_labels = (
                        labels[:, :-num_mtp_modules]
                        if num_mtp_modules > 0
                        else labels
                    )
                    local_valid_tokens += (main_labels != IGNORE_INDEX).sum()
                    microbatches.append((input_dict, labels))
        sl.log_trace_scalar({"local_valid_tokens": int(local_valid_tokens)})

        # All-reduce to get global token count across DP ranks
        # Move to GPU for distributed communication
        with self._dsv4_additive_phase("train_step.valid_tokens_to_device"):
            local_valid_tokens = local_valid_tokens.to(self.device)
        if parallel_dims.dp_enabled:
            batch_mesh = parallel_dims.get_mesh("batch")
            with self._dsv4_additive_phase("train_step.valid_tokens_all_reduce"):
                global_valid_tokens = dist_utils.dist_sum(
                    local_valid_tokens, batch_mesh
                )
        else:
            global_valid_tokens = local_valid_tokens.float()

        # Process each microbatch: move to GPU, forward/backward, then free
        accumulated_losses = []
        for microbatch_idx, (input_dict, labels) in enumerate(microbatches):
            # Move tensors to GPU
            with self._dsv4_additive_phase(
                "train_step.microbatch_h2d", microbatch=microbatch_idx
            ):
                for k, v in input_dict.items():
                    if isinstance(v, torch.Tensor):
                        input_dict[k] = v.to(self.device)
                labels = labels.to(self.device)

            with dsv4_execution_region("microbatch", microbatch_idx):
                with self._dsv4_additive_phase(
                    "train_step.microbatch_forward_backward",
                    microbatch=microbatch_idx,
                ):
                    with self._dsv4_additive_phase(
                        "train_step.forward_backward_call",
                        microbatch=microbatch_idx,
                    ):
                        loss = self.forward_backward_step(
                            input_dict=input_dict,
                            labels=labels,
                            # pyrefly: ignore [bad-argument-type]
                            global_valid_tokens=global_valid_tokens,
                        )
                    with self._dsv4_additive_phase(
                        "train_step.forward_backward_call_returned",
                        microbatch=microbatch_idx,
                    ):
                        pass
            with self._dsv4_additive_phase(
                "train_step.loss_detach_append", microbatch=microbatch_idx
            ):
                accumulated_losses.append(loss.detach())

        with sl.log_trace_span("optim"):
            with self._dsv4_additive_phase("train_step.grad_nonfinite_scan"):
                self._dsv4_maybe_scan_nonfinite_grads()
            with self._dsv4_additive_phase("train_step.grad_clip"):
                grad_norm = dist_utils.clip_grad_norm_(
                    [p for m in self.model_parts for p in m.parameters()],
                    self.config.training.max_norm,
                    foreach=True,
                    pp_mesh=parallel_dims.get_optional_mesh("pp"),
                    ep_enabled=parallel_dims.ep_enabled,
                )
            with self._dsv4_additive_phase("train_step.checkpoint_wait_for_staging"):
                self.checkpointer.maybe_wait_for_staging()
            with self._dsv4_additive_phase("train_step.optimizer_step"):
                self.optimizers.step()
            with self._dsv4_additive_phase("train_step.lr_scheduler_step"):
                self.lr_schedulers.step()

        # Reduce the data collected over gradient accumulation steps.
        with self._dsv4_additive_phase("train_step.accumulated_loss_reduce"):
            loss = torch.sum(torch.stack(accumulated_losses))

        # log metrics
        if not self.metrics_processor.should_log(self.step):
            return

        with self._dsv4_additive_phase("train_step.collect_dist_metrics"):
            with sl.log_trace_span("collect_dist_metrics"):

                sl.log_trace_scalar({"global_valid_tokens": int(global_valid_tokens)})

                if parallel_dims.dp_cp_enabled:
                    loss = loss.detach()
                    loss_mesh = parallel_dims.get_optional_mesh("loss")

                    # For global_avg_loss, we want the average loss across all ranks:
                    # loss = local_loss_sum / global_valid_tokens
                    # global_avg_loss = sum(local_loss_sum) / global_valid_tokens
                    #                 = sum(loss)
                    #
                    # For global_max_loss, we want the max of local average losses across ranks:
                    # local_avg_loss = local_loss_sum / local_valid_tokens
                    #                = (loss * global_valid_tokens) / local_valid_tokens
                    # global_max_loss = max(local_avg_loss)
                    local_avg_loss = loss * global_valid_tokens / local_valid_tokens
                    global_avg_loss, global_max_loss, global_ntokens_seen = (
                        dist_utils.dist_sum(loss, loss_mesh),
                        dist_utils.dist_max(local_avg_loss, loss_mesh),
                        dist_utils.dist_sum(
                            torch.tensor(
                                self.ntokens_seen,
                                dtype=torch.int64,
                                device=self.device,
                            ),
                            loss_mesh,
                        ),
                    )
                else:
                    global_avg_loss = global_max_loss = float(loss.detach().item())
                    global_ntokens_seen = self.ntokens_seen

        extra_metrics = {
            "n_tokens_seen": global_ntokens_seen,
            "lr": lr,
        }
        with self._dsv4_additive_phase("train_step.metrics_log"):
            self.metrics_processor.log(
                self.step,
                global_avg_loss,
                global_max_loss,
                float(grad_norm.item()),
                extra_metrics=extra_metrics,
            )

    @record
    def train(self):
        config = self.config

        sl.log_trace_instant("training_start")

        self.checkpointer.load(step=config.checkpoint.load_step)

        # Capture loaded step for relative_step calculation.
        # After checkpoint load: self.step = restored step (e.g. 100), or 0 if fresh.
        loaded_step = self.step

        logger.info(f"Training starts at step {self.step + 1}")

        with config.profiler.build(
            global_step=self.step,
            base_folder=config.dump_folder,
        ) as profiler:
            data_iterator = self.batch_generator(self.dataloader)
            while self.should_continue_training():
                self.step += 1
                relative_step = self.step - loaded_step
                sl.set_step(self.step, relative_step=relative_step)
                self._dsv4_additive_timer = Dsv4AdditiveStepTimer(
                    step=self.step,
                    relative_step=relative_step,
                    meta={
                        "local_batch_size": config.training.local_batch_size,
                        "global_batch_size": config.training.global_batch_size,
                        "sequence_length": config.training.seq_len,
                        "gradient_accumulation_steps": self.gradient_accumulation_steps,
                        "parallelism": {
                            "dp_replicate": self.parallel_dims.dp_replicate,
                            "dp_shard": self.parallel_dims.dp_shard,
                            "cp": self.parallel_dims.cp,
                            "tp": self.parallel_dims.tp,
                            "pp": self.parallel_dims.pp,
                            "ep": self.parallel_dims.ep,
                        },
                        "activation_checkpoint_mode": config.activation_checkpoint.mode,
                        "return_detached_loss": self._dsv4_env_flag(
                            "TORCHTITAN_DSV4_RETURN_DETACHED_LOSS"
                        ),
                        "drop_original_loss_before_return": self._dsv4_env_flag(
                            "TORCHTITAN_DSV4_DROP_ORIGINAL_LOSS_BEFORE_RETURN"
                        ),
                        "retain_original_loss_after_backward": self._dsv4_env_flag(
                            "TORCHTITAN_DSV4_RETAIN_ORIGINAL_LOSS_AFTER_BACKWARD"
                        ),
                        "checkpoint_enabled": config.checkpoint.enable,
                    },
                )
                self._dsv4_additive_timer.start()
                step_outcome = "started"

                try:
                    with sl.log_trace_span("step"):
                        with self._dsv4_additive_phase("step.gc"):
                            self.gc_handler.run(self.step)

                        try:
                            with self._dsv4_additive_phase("step.train_step"):
                                self.train_step(data_iterator)
                        except DataloaderExhaustedError:
                            step_outcome = "dataloader_exhausted"
                            logger.warning("Ran out of data; last step was canceled.")
                            break

                        with self._dsv4_additive_phase("step.checkpoint_save"):
                            self.checkpointer.save(
                                self.step,
                                last_step=(self.step == config.training.steps),
                            )

                        # Run validation if validator is available
                        if (
                            self.config.validator.enable
                            and self.validator.should_validate(self.step)
                        ):
                            with self._dsv4_additive_phase("step.validation"):
                                self.validator.validate(self.model_parts, self.step)

                        # signal the profiler that the next profiling step has started
                        with self._dsv4_additive_phase("step.profiler_step"):
                            profiler.step()

                        # Reduce timeout after the first train step of THIS process
                        # (assuming lazy init and compilation are finished). Use the
                        # relative step so this fires on resumed runs too.
                        if relative_step == 1:
                            with self._dsv4_additive_phase("step.set_pg_timeouts"):
                                dist_utils.set_pg_timeouts(
                                    timeout=timedelta(
                                        seconds=config.comm.train_timeout_seconds
                                    ),
                                    parallel_dims=self.parallel_dims,
                                )
                        step_outcome = "completed"
                finally:
                    self._dsv4_additive_timer.finish({"step_outcome": step_outcome})
                    self._dsv4_additive_timer = None

        if torch.distributed.get_rank() == 0:
            logger.info("Sleeping 2 seconds for other ranks to complete")
            time.sleep(2)

        logger.info("Training completed")

    def should_continue_training(self) -> bool:
        return self.step < self.config.training.steps

    def state_dict(self) -> dict[str, Any]:
        return {"step": self.step, "ntokens_seen": self.ntokens_seen}

    def load_state_dict(self, state_dict: dict[str, Any]):
        self.step = state_dict["step"]
        self.ntokens_seen = state_dict["ntokens_seen"]

    def close(self) -> None:
        if hasattr(self, "checkpointer") and self.checkpointer:
            self.checkpointer.close()
        if hasattr(self, "metrics_processor") and self.metrics_processor:
            self.metrics_processor.close()
