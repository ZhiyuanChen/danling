# DanLing
# Copyright (C) 2022-Present  DanLing

# This program is free software: you can redistribute it and/or modify
# it under the terms of the following licenses:
# - The Unlicense
# - GNU Affero General Public License v3.0 or later
# - GNU General Public License v2.0 or later
# - BSD 4-Clause "Original" or "Old" License
# - MIT License
# - Apache License 2.0

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the LICENSE file for more details.

from __future__ import annotations

import os
import random
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from time import time

# pylint: disable=redefined-builtin
import torch
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin
from chanfig import FlatDict, NestedDict
from torch import distributed as dist
from torch import nn, optim, utils
from torch.backends import cudnn
from tqdm import tqdm

try:
    from functools import cached_property
except ImportError:
    from cached_property import cached_property  # type: ignore

try:
    from numpy import random as np_random
except ImportError:
    np_random = None

from danling import defaults
from danling.utils import catch

from .base_runner import BaseRunner
from .config import Config
from .utils import RunnerMode, on_main_process


class AccelerateRunner(BaseRunner, Accelerator):  # pylint: disable=too-many-public-methods
    r"""
    Set up everything for running a job.

    `AccelerateRunner` uses [`accelerate`][accelerate] as distributed backend to
    provide seamless distributed training experience.

    `AccelerateRunner` will automatically [`prepare`][accelerate.Accelerator.prepare] everything,
    including `model`, `criterion`, `optimizer`, `scheduler`, and `dataloaders` for distribute training,
    mixed precision, and deepspeed (optional).

    In fact, you don't even need to create `dataloaders`, just define
    `datasets` and `AccelerateRunner` will create `dataloaders` for you.
    `AccelerateRunner` will inspect the `train` flag in corresponding dataset to
    set `shuffle` and `drop_last` automatically.
    """

    model: nn.Module
    criterion: nn.Module
    optimizer: optim.Optimizer
    scheduler: optim.lr_scheduler._LRScheduler

    _accelerate: FlatDict | None = None

    def __init__(self, config: Config) -> None:
        BaseRunner.__init__(self, config)
        Accelerator.__init__(self, **self.accelerate)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.project_configuration.set_directories(self.dir)
        self.model, self.criterion, self.optimizer = self.prepare(self.model, self.criterion, self.optimizer)
        self.scheduler = self.prepare(self.scheduler)
        if self.datasets:
            datasets = {k: d for k, d in self.datasets.items() if k not in self.dataloaders}
            default_kwargs = self.config.get("dataloader", NestedDict())
            dataloader_kwargs = NestedDict({k: default_kwargs.pop(k) for k in self.datasets if k in default_kwargs})
            for k, d in datasets.items():
                dataloader_kwargs.setdefault(k, NestedDict())
                dataloader_kwargs[k].merge(default_kwargs, overwrite=False)
                dataloader_kwargs[k].setdefault("shuffle", getattr(d, "train", True))
                dataloader_kwargs[k].setdefault("drop_last", not getattr(d, "train", True))
                self.dataloaders[k] = utils.data.DataLoader(d, **dataloader_kwargs[k])
            default_kwargs.update(dataloader_kwargs)
        for k, d in self.dataloaders.items():
            self.dataloaders[k] = self.prepare(d)
        if self.config.get("log_interval") is None:
            self.config.log_interval = max(len(d) for d in self.dataloaders.values()) // 10

    @property
    def accelerate(self) -> FlatDict:
        if self._accelerate is None:
            self._accelerate = self.get_accelerate_config(self.config)
        return self._accelerate

    @accelerate.setter
    def accelerate(self, config: FlatDict) -> None:
        self._accelerate = config

    @property
    def deepspeed(self) -> dict | None:
        if self.state.deepspeed_plugin is not None:
            return self.state.deepspeed_plugin.deepspeed_config
        return None

    def train(self, train_splits: list[str] | None = None, eval_splits: list[str] | None = None) -> NestedDict:
        r"""
        Perform training on `split`.

        Args:
            train_splits (list[str]): list of split to run train.
                Defaults to `["train"]`.
            eval_splits (list[str]): list of split to run evaluate.
                Defaults to `self.dataloaders` except for those in `train_splits`.

        Return:
            NestedDict: train results
        """

        early_stop_counter = 0
        if train_splits is None:
            train_splits = ["train"]
        if eval_splits is None:
            eval_splits = [s for s in self.dataloaders if s not in train_splits]
        self.config.epoch_begin = self.config.epoch
        print(f"Begin training from {self.config.epoch_begin} to {self.config.epoch_end}")
        print(f"Training splits: {train_splits}")
        print(f"Evaluation splits: {eval_splits}")
        patience = self.config.get("patience", float("inf"))
        for epoch in range(self.config.epoch_begin, self.config.epoch_end):  # type: ignore
            self.config.epoch = epoch
            result = NestedDict()
            result.setattr("convert_mapping", True)
            for split in train_splits:
                result[split] = self.train_epoch(split)
            for split in eval_splits:
                result[split] = self.evaluate_epoch(split)
            self.append_result(result)
            print(self.format_epoch_result(result))
            self.save_result()
            if self.config.save_interval is not None:
                self.save_checkpoint(epoch)
            """@nni.report_intermediate_result(self.latest_score)"""
            early_stop_counter = 0 if self.is_best else early_stop_counter + 1
            if early_stop_counter > patience:
                print("early stop")
                break
        """@nni.report_final_result(self.latest_score)"""
        return self.results

    def train_epoch(self, split: str = "train") -> NestedDict:
        r"""
        Train one epoch on `split`.

        Args:
            split (str): split to run train

        Return:
            NestedDict: train result
        """

        self.mode = "train"  # type: ignore
        self.split = split
        loader = self.dataloaders[split]
        length = len(loader) - 1
        last_print_iteration = -1
        log_interval = self.config.get("log_interval", -1)
        self.meters.reset()
        if self.metrics is not None:
            self.metrics.reset()
        batch_time = time()
        if hasattr(loader.batch_sampler, "set_epoch"):
            loader.batch_sampler.set_epoch(self.config.epoch)
        if hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(self.config.epoch)

        for iteration, data in enumerate(loader):
            with self.autocast(), self.accumulate():
                input = data["input"] if isinstance(data, Mapping) else data[0]
                target = data["target"] if isinstance(data, Mapping) else data[1]
                pred = self.model(**input) if isinstance(input, Mapping) else self.model(input)
                loss = self.criterion(pred, target)
                if self.metrics is not None:
                    self.metrics.update(pred.squeeze(-1), target)
                self.advance(loss)

            if log_interval > 0 and (iteration > 0 and iteration % log_interval == 0 or iteration == length):
                interval = iteration - last_print_iteration
                if self.device == torch.device("cuda"):
                    torch.cuda.synchronize()
                if self.scheduler is not None:
                    self.meters.lr.update(self.scheduler.get_last_lr()[0])
                self.meters.time.update((time() - batch_time) / interval)
                batch_time = time()
                reduced_loss = self.reduce(loss).item()
                self.meters.loss.update(reduced_loss)
                self.step_log(split, iteration, length)
                last_print_iteration = iteration

        result = self.meters.average()
        if self.metrics is not None:
            result.merge(self.metrics.average())
        return result

    def advance(self, loss) -> None:
        r"""
        Backward loss and step optimizer & scheduler.

        Args:
            zero_grad: Whether to zero the gradients.
        """

        self.backward(loss)
        if self.sync_gradients:
            if self.config.get("max_grad_value") is not None:
                self.clip_grad_value_(self.model.parameters(), self.config.get("max_grad_value"))
            if self.config.get("max_grad_norm") is not None:
                self.clip_grad_norm_(self.model.parameters(), self.config.get("max_grad_norm"))
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.optimizer.zero_grad()
        self.config.step = self.step

    def evaluate(self, eval_splits: list[str] | None = None) -> NestedDict:
        r"""
        Perform evaluation on `eval_splits`.

        Args:
            eval_splits (list[str]): list of split to run evaluate.
                Defaults to `["eval"]`.

        Return:
            NestedDict: evaluation result
        """

        if eval_splits is None:
            eval_splits = ["eval"]

        print("Begin evaluation")
        print(f"Evaluation splits: {eval_splits}")
        result = NestedDict()
        result.setattr("convert_mapping", True)
        for split in eval_splits:
            result[split] = self.evaluate_epoch(split=split)
        print(self.format_epoch_result(result))
        return result

    @torch.inference_mode()
    def evaluate_epoch(self, split: str = "val") -> NestedDict:
        r"""
        Evaluate one epoch on `split`.

        Args:
            split (str): split to run evaluate

        Return:
            NestedDict: evaluation result
        """

        self.mode = "eval"  # type: ignore
        self.split = split
        loader = self.dataloaders[split]
        length = len(loader) - 1
        last_print_iteration = -1
        log_interval = self.config.get("log_interval", -1)
        self.meters.reset()
        if self.metrics is not None:
            self.metrics.reset()
        batch_time = time()

        for iteration, data in enumerate(loader):
            input = data["input"] if isinstance(data, Mapping) else data[0]
            target = data["target"] if isinstance(data, Mapping) else data[1]
            pred = self.model(**input) if isinstance(input, Mapping) else self.model(input)
            loss = self.criterion(pred, target)
            if self.metrics is not None:
                self.metrics.update(pred.squeeze(-1), target)

            if log_interval > 0 and (iteration > 0 and iteration % log_interval == 0 or iteration == length):
                interval = iteration - last_print_iteration
                if self.device == torch.device("cuda"):
                    torch.cuda.synchronize()
                self.meters.time.update((time() - batch_time) / interval)
                batch_time = time()
                reduced_loss = self.reduce(loss).item()
                self.meters.loss.update(reduced_loss)
                self.step_log(split, iteration, length)
                last_print_iteration = iteration

        result = self.meters.average()
        if self.metrics is not None:
            result.merge(self.metrics.average())
        self.write_result(result, split, self.config.epoch)
        return result

    @torch.inference_mode()
    def inference(self, split: str = "inf") -> list:
        r"""
        Perform inference on `split`.

        Args:
            split (str): split to run inference

        Return:
            Tensor: inference outputs
        """

        self.mode = "inf"  # type: ignore
        loader = self.dataloaders[split]
        self.meters.reset()
        output = []
        for _, data in tqdm(enumerate(loader), total=len(loader)):
            input = data["input"] if isinstance(data, Mapping) else data[0]
            pred = self.model(**input) if isinstance(input, Mapping) else self.model(input)
            output.extend(pred.squeeze(-1).tolist())

        if self.distributed:
            torch.cuda.synchronize()
            output = self.gather_for_metrics(output)
        return output

    def init_distributed(self) -> None:
        r"""
        Set up distributed training.

        Initialise process group and set up DDP variables.
        """

        if self.distributed:
            object_list = [self.id, self.timestamp]
            dist.broadcast_object_list(object_list)
            self.id, self.timestamp = object_list

    @on_main_process
    def init_tensorboard(self, *args, **kwargs) -> None:
        r"""
        Set up Tensoraoard SummaryWriter.
        """
        from torch.utils.tensorboard.writer import SummaryWriter  # pylint: disable=C0415

        if "log_dir" not in kwargs:
            kwargs["log_dir"] = self.dir

        self.writer = SummaryWriter(*args, **kwargs)
        self.writer.add_scalar = catch(OSError, verbose=False)(self.writer.add_scalar)

    def set_seed(self, seed: int = None, bias: int = None) -> int:  # type: ignore[assignment]
        r"""
        Set up random seed.

        Args:
            seed: Random seed to set.
                Defaults to `self.config.seed` (`config.seed`).

            bias: Make the seed different for each processes.
                This is used to ensure the data augmentation are applied differently on every processes.
                Defaults to `self.rank`.
                Set to `False` to disable this feature.
        Returns:
            Random seed set.
        """

        seed = seed or self.config.seed  # type: ignore[assignment]
        if seed is None:
            if self.inited:
                seed = random.randint(0, 2**32 - 1)
                if self.distributed:
                    object_list = [seed]
                    dist.broadcast_object_list(object_list)
                    seed = object_list[0]
                self.config.seed = seed
        else:
            seed = defaults.SEED
        bias = bias or self.rank
        if bias:
            seed += bias
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        if np_random is not None:
            np_random.seed(seed)
        random.seed(seed)
        return seed

    def set_deterministic(self) -> None:
        r"""
        Set up deterministic.
        """

        cudnn.benchmark = False
        cudnn.deterministic = True
        if torch.__version__ >= "1.8.0":
            torch.use_deterministic_algorithms(True)

    def state_dict(self, cls: Callable = dict) -> Mapping:
        r"""
        Return dict of all attributes for checkpoint.
        """

        if self.model is None:
            raise ValueError("Model must be defined when calling state_dict")
        return cls(
            runner=self.config.dict(),
            model=self.unwrap_model(self.model).state_dict(),
            optimizer=self.optimizer.state_dict() if self.optimizer else None,
            scheduler=self.scheduler.state_dict() if self.scheduler else None,
        )

    @contextmanager
    def accumulate(self, *models: nn.Module):
        if not models:
            models = (self.model,)
        yield super().accumulate(*models)

    @property
    def mode(self) -> RunnerMode:
        return self._mode

    @mode.setter
    def mode(self, mode: str | RunnerMode) -> None:
        if isinstance(mode, str):
            mode = RunnerMode(mode)
        self._mode = mode
        if self.model is not None:
            self.model.train(mode == RunnerMode.train)

    @property
    def world_size(self) -> int:
        r"""
        Number of Processes.
        """

        if "state" in self.__dict__:
            return self.state.num_processes
        return 1

    @property
    def rank(self) -> int:
        r"""
        Process index in all processes.
        """

        if "state" in self.__dict__:
            return self.state.process_index
        return 0

    @property
    def local_rank(self) -> int:
        r"""
        Process index in local processes.
        """

        if "state" in self.__dict__:
            return self.state.local_process_index
        return 0

    @cached_property
    def accum_steps(self) -> int:
        r"""
        Accumulated steps.

        Returns:
            (int):
        """

        return self.gradient_accumulation_steps

    @staticmethod
    def get_accelerate_config(config) -> FlatDict:
        accelerate = FlatDict()
        if "accelerate" in config:
            accelerate.update(config.accelerate)
        if "precision" in config:
            accelerate.mixed_precision = config.precision
        if "dynamo" in config:
            accelerate.dynamo_backend = config.dynamo.upper()
        if "accum_steps" in config:
            accelerate.gradient_accumulation_steps = config.accum_steps
        if "kwargs_handlers" not in accelerate:
            accelerate.kwargs_handlers = []
        # Must NOT set project_dir here as timestamp is not synced yet
        # config.project_dir = self.dir
        if os.getenv("ACCELERATE_USE_DEEPSPEED", "false").lower() == "true":
            deepspeed_config = config.get("deepspeed", os.getenv("ACCELERATE_DEEPSPEED_CONFIG_FILE"))
            accelerate.deepspeed_plugin = DeepSpeedPlugin(hf_ds_config=deepspeed_config(deepspeed_config))
        return accelerate
