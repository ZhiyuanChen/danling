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
import shutil

import torch
from chanfig import NestedDict
from lazy_imports import try_import
from torch import distributed as dist
from torch import nn
from torch.nn.utils import clip_grad_value_

from danling.runner.config import Config
from danling.utils import catch

from .torch_runner import TorchRunner

with try_import() as ds:
    import deepspeed


class DeepSpeedRunner(TorchRunner):

    def __init__(self, config: Config) -> None:
        ds.check()
        super().__init__(config)

    def init_distributed(self) -> None:
        r"""
        Set up distributed training.

        Initialise process group and set up DDP variables.
        """

        backend = self.config.get("backend", os.getenv("BACKEND"))
        init_method = self.config.get("init_method", os.getenv("INIT_METHOD"))
        world_size = int(self.config.get("world_size", os.getenv("WORLD_SIZE", "1")))
        rank = int(self.config.get("rank", os.getenv("RANK", "0")))
        if world_size > 1:
            if torch.cuda.is_available():
                torch.cuda.set_device(self.get_local_rank())
            deepspeed.init_distributed(dist_backend=backend, init_method=init_method, world_size=world_size, rank=rank)
            object_list = [self.id, self.timestamp]
            dist.broadcast_object_list(object_list)
            self.id, self.timestamp = object_list

    def __post_init__(self):
        super().__post_init__()
        self.config.deepspeed = self.get_deepspeed_config()
        self.model, self.optimizer, _, self.scheduler = deepspeed.initialize(
            model=self.model,
            optimizer=self.optimizer,
            lr_scheduler=self.scheduler,
            config=self.config.deepspeed,
        )

    def advance(self, loss) -> None:
        self.backward(loss)
        if self.config.get("max_grad_value") is not None:
            clip_grad_value_(self.model.parameters(), self.config["max_grad_value"])
        self.model.step()
        if self.ema is not None:
            self.ema.update()
        self.config.steps = self.model.global_steps
        self.config.iters += 1

    def backward(self, loss: torch.Tensor) -> None:
        return self.model.backward(loss)

    def get_local_rank(self) -> int:
        local_rank = self.config.get("local_rank", os.getenv("LOCAL_RANK"))
        if local_rank is not None:
            return int(local_rank)
        rank = self.config.get("rank", os.getenv("RANK"))
        world_size = self.config.get("world_size", os.getenv("WORLD_SIZE"))
        if world_size is None or rank is None:
            raise ValueError("Please provide either `local_rank` or `world_size` and `rank`")
        return int(world_size) % int(rank)

    def unwrap(self, model: nn.Module) -> nn.Module:
        while isinstance(model, (deepspeed.DeepSpeedEngine, nn.parallel.DistributedDataParallel)):
            model = model.module
        return model

    @property
    def deepspeed(self) -> NestedDict | None:
        if isinstance(self.model, deepspeed.DeepSpeedEngine):
            return self.model.config
        return None

    @catch
    def save_checkpoint(self, name: str = "latest", epoch: int | None = None, save_best: bool = True) -> None:
        r"""
        Save checkpoint to `self.checkpoint_dir`.

        Args:
            name: Name of the checkpoint. Defaults to `"latest"`.
            epoch: Epoch to save. Defaults to `self.config.epochs`.
            save_best: If `True`, when `self.is_best` is `True`, the checkpoint will also be copied to
                `self.checkpoint_dir/best`.

        If `self.config.save_interval` is positive and `epochs + 1` is a multiple of `save_interval`,
        the checkpoint will also be copied to `self.checkpoint_dir/epoch-{epochs}`.
        """

        epoch = epoch or self.config.epochs
        save_interval = self.config.get("save_interval", -1)
        latest_path = os.path.join(self.checkpoint_dir, name)
        os.makedirs(latest_path, exist_ok=True)
        self.yaml(os.path.join(latest_path, "runner.yaml"))
        self.model.save_checkpoint(
            self.checkpoint_dir, tag=name, client_state={"runner": self.config.dict()}, save_latest=False
        )
        if save_interval > 0 and (epoch + 1) % save_interval == 0:
            save_path = os.path.join(self.checkpoint_dir, f"epoch-{epoch}")
            shutil.copytree(latest_path, save_path, dirs_exist_ok=True)
        if save_best and self.is_best:
            best_path = os.path.join(self.checkpoint_dir, "best")
            shutil.copytree(latest_path, best_path, dirs_exist_ok=True)

    def load_checkpoint(self, checkpoint: bytes | str | os.PathLike, *args, **kwargs) -> None:  # type: ignore[override]
        """
        Load model, optimizer, and scheduler from checkpoint.

        Args:
            checkpoint: Checkpoint (or its path) to load.
            *args: Additional arguments to pass to `self.load`.
            **kwargs: Additional keyword arguments to pass to `self.load`.

        Raises:
            ValueError: If `model` is not defined.
            ValueError: If `model` is not an instance of `deepspeed.DeepSpeedEngine`.

        See Also:
            [`from_checkpoint`][danling.BaseRunner.from_checkpoint]: Build runner from checkpoint.
            [`load_pretrained`][danling.BaseRunner.load_pretrained]: Load model parameters from pretrained checkpoint.
        """

        if self.model is None:
            raise ValueError("model is not defined")
        if not isinstance(self.model, deepspeed.DeepSpeedEngine):
            raise ValueError("model is not an instance of `deepspeed.DeepSpeedEngine`")

        self.model.load_checkpoint(checkpoint)
        self.config.checkpoint = checkpoint

    def load_pretrained(self, checkpoint: bytes | str | os.PathLike, *args, **kwargs) -> None:  # type: ignore[override]
        """
        Load model from pretrained checkpoint.

        This method only loads the model weights.

        Args:
            checkpoint: Pretrained checkpoint directory.
            *args: Additional arguments to pass to `self.load`.
            **kwargs: Additional keyword arguments to pass to `self.load`.

        Raises:
            ValueError: If `model` is not defined.

        See Also:
            [`load_checkpoint`][danling.BaseRunner.load_checkpoint]: Load model, optimizer, and scheduler from
                checkpoint.
        """

        if self.model is None:
            raise ValueError("model is not defined")

        self.model.load_checkpoint(checkpoint, load_module_only=True)
        self.config.pretrained = checkpoint

    def load_config(
        self, checkpoint: bytes | str | os.PathLike, overwrite: bool = False, *args, **kwargs  # type: ignore[override]
    ) -> None:
        r"""
        Load config from checkpoint.

        Args:
            checkpoint: Checkpoint (or its path) to load.
            overwrite: If `True`, overwrite the current config with the loaded config.
                Defaults to `False`.
            *args: Additional arguments to pass to `self.load`.
            **kwargs: Additional keyword arguments to pass to `self.load`.

        Raises:
            FileNotFoundError: If `checkpoint` does not exists.
        """

        if isinstance(checkpoint, bytes):
            checkpoint = checkpoint.decode()

        config = self.load(os.path.join(checkpoint, "runner.yaml"), *args, **kwargs)
        self.config.merge(config, overwrite=overwrite)
        self.config.iter_begin = config["iters"] + 1
        self.config.step_begin = config["steps"] + 1
        self.config.epoch_begin = config["epochs"] + 1
