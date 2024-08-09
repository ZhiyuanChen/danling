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
from collections.abc import Mapping
from contextlib import contextmanager
from math import ceil

import torch
from chanfig import FlatDict, NestedDict
from lazy_imports import try_import
from torch import distributed as dist
from torch import nn, utils

try:
    from functools import cached_property
except ImportError:
    from cached_property import cached_property  # type: ignore

from .config import Config
from .torch_runner import BaseRunner, TorchRunner

with try_import() as ac:
    from accelerate import Accelerator
    from accelerate.utils import DeepSpeedPlugin


class AccelerateRunner(TorchRunner, Accelerator):  # pylint: disable=too-many-public-methods
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

    _accelerate: FlatDict | None = None

    def __init__(self, config: Config) -> None:
        ac.check()
        TorchRunner.__init__(self, config)
        Accelerator.__init__(self, **self.accelerate)
        if self.distributed:
            object_list = [self.id, self.timestamp]
            dist.broadcast_object_list(object_list)
            self.id, self.timestamp = object_list

    def __post_init__(self) -> None:
        BaseRunner.__post_init__(self)
        self.project_configuration.set_directories(self.dir)
        if self.datasets:
            self.build_dataloaders()
        if self.config.get("log_interval") is None:
            self.config.log_interval = max(ceil(max(len(d) for d in self.dataloaders.values()) / 10), 1)
        self.model, self.criterion, self.optimizer, self.scheduler = self.prepare(
            self.model, self.criterion, self.optimizer, self.scheduler
        )

    def train_step(self, data) -> torch.Tensor:
        with self.autocast(), self.accumulate():
            input = data["input"] if isinstance(data, Mapping) else data[0]
            target = data["target"] if isinstance(data, Mapping) else data[1]
            pred = self.model(**input) if isinstance(input, Mapping) else self.model(input)
            loss = self.criterion(pred, target)
            if self.metrics is not None:
                self.metrics.update(pred.squeeze(-1), target)
            self.advance(loss)
        return loss

    def advance(self, loss) -> None:
        r"""
        Backward loss and step optimizer & scheduler.

        Args:
            loss: The loss tensor from which to backpropagate.
        """

        self.backward(loss)
        if self.sync_gradients:
            if self.config.get("max_grad_value") is not None:
                self.clip_grad_value_(self.model.parameters(), self.config["max_grad_value"])
            if self.config.get("max_grad_norm") is not None:
                self.clip_grad_norm_(self.model.parameters(), self.config["max_grad_norm"])
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        if self.ema is not None:
            self.ema.update()
        self.optimizer.zero_grad()
        self.config.steps = self.step
        self.config.iters += 1

    def unwrap(self, model: nn.Module) -> nn.Module:
        return self.unwrap_model(model)

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

    @contextmanager
    def accumulate(self, *models: nn.Module):
        if not models:
            models = (self.model,)
        yield Accelerator.accumulate(self, *models)

    @property
    def device(self) -> torch.device:
        return self.state.device

    @property
    def world_size(self) -> int:
        if "state" in self.__dict__:
            return self.state.num_processes
        return 1

    @property
    def rank(self) -> int:
        if "state" in self.__dict__:
            return self.state.process_index
        return 0

    @property
    def local_rank(self) -> int:
        if "state" in self.__dict__:
            return self.state.local_process_index
        return 0

    @cached_property
    def accum_steps(self) -> int:
        return self.gradient_accumulation_steps

    def get_accelerate_config(self, config) -> FlatDict:
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
            accelerate.deepspeed_plugin = DeepSpeedPlugin(hf_ds_config=self.get_deepspeed_config(deepspeed_config))
        return accelerate

    def build_dataloaders(self):
        datasets = {k: d for k, d in self.datasets.items() if k not in self.dataloaders}
        default_kwargs = self.config.setdefault("dataloader", NestedDict())
        dataloader_kwargs = NestedDict({k: default_kwargs.pop(k) for k in self.datasets if k in default_kwargs})
        for k, d in datasets.items():
            dataloader_kwargs.setdefault(k, NestedDict())
            dataloader_kwargs[k].merge(default_kwargs, overwrite=False)
            dataloader_kwargs[k].setdefault("shuffle", getattr(d, "train", True))
            dataloader_kwargs[k].setdefault("drop_last", not getattr(d, "train", True))
            self.dataloaders[k] = utils.data.DataLoader(d, collate_fn=self.collate_fn, **dataloader_kwargs[k])
        default_kwargs.update(dataloader_kwargs)
        for k, d in self.dataloaders.items():
            self.dataloaders[k] = self.prepare(d)
