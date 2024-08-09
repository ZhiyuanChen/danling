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
from collections.abc import Mapping
from contextlib import contextmanager, nullcontext
from math import ceil
from time import time
from typing import Any, Callable, Tuple
from warnings import warn

import torch
import torch.distributed
from chanfig import NestedDict
from torch import distributed as dist
from torch import nn, optim, utils
from torch.backends import cudnn
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
from tqdm import tqdm

try:
    from functools import cached_property
except ImportError:
    from cached_property import cached_property  # type: ignore

try:
    from numpy import random as np_random
except ImportError:
    np_random = None

try:
    import deepspeed as ds
except ImportError:
    ds = None

from danling import defaults
from danling.utils import catch

from .base_runner import BaseRunner
from .utils import RunnerMode, on_main_process, to_device


class TorchRunner(BaseRunner):
    r"""
    Set up everything for running a job.

    `TorchRunner` uses `torch.distributed` as distributed backend to provide
    distributed training experience.
    """

    model: nn.Module
    ema: nn.Module | None = None
    criterion: nn.Module
    optimizer: optim.Optimizer
    scheduler: optim.lr_scheduler._LRScheduler

    def __post_init__(self):
        super().__post_init__()
        if self.datasets:
            self.build_dataloaders()
        if self.config.get("log_interval") is None:
            self.config.log_interval = max(ceil(max(len(d) for d in self.dataloaders.values()) / 10), 1)
        self.model = self.model.to(self.device)
        if self.ema is not None:
            self.ema = self.ema.to(self.device)
        if self.distributed and not isinstance(
            self.model, (nn.parallel.DistributedDataParallel, nn.parallel.DataParallel)
        ):
            self.model = nn.parallel.DistributedDataParallel(self.model)

    def train(self, train_splits: list[str] | None = None, evaluate_splits: list[str] | None = None) -> NestedDict:
        r"""
        Perform training on `split`.

        Args:
            train_splits: list of split to run train.
                Defaults to `["train"]`.
            evaluate_splits: list of split to run evaluate.
                Defaults to `self.dataloaders` except for those in `train_splits`.

        Return:
            NestedDict: train results
        """

        early_stop_counter = 0
        if train_splits is None:
            train_splits = ["train"] if "train" in self.dataloaders else []
        if not train_splits:
            warn("No training split is found. Will only evaluate for one epoch.", stacklevel=2)
            self.config.epoch_end = self.config.epoch_begin + 1
        if evaluate_splits is None:
            evaluate_splits = [s for s in self.dataloaders if s not in train_splits]
        train_splits.sort()
        evaluate_splits.sort()
        print(f"Begin training from {self.config.epoch_begin} to {self.config.epoch_end}")
        print(f"Training splits: {train_splits}")
        print(f"Evaluation splits: {evaluate_splits}")
        patience = self.config.get("patience", float("inf"))
        for epoch in range(self.config.epoch_begin, self.config.epoch_end):  # type: ignore
            self.config.epochs = epoch
            result = NestedDict()
            result.setattr("convert_mapping", True)
            for split in train_splits:
                result[split] = self.train_epoch(split)
            for split in evaluate_splits:
                result[split] = self.evaluate_epoch(split)
            self.append_result(result)
            print(self.format_epoch_result(result))
            self.save_result()
            if self.config.save_interval is not None:
                self.save_checkpoint()
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
        if self.train_metrics is not None:
            self.metrics = self.train_metrics
        if self.metrics is not None:
            self.metrics.reset()
        batch_time = time()
        if hasattr(loader.batch_sampler, "set_epoch"):
            loader.batch_sampler.set_epoch(self.config.epochs)
        if hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(self.config.epochs)

        for iteration, data in enumerate(loader):
            _, loss = self.train_step(data)

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

        result = self.get_epoch_result()
        return result

    def train_step(self, data) -> Tuple[Any, torch.Tensor]:
        with self.autocast(), self.accumulate():
            input = data["input"] if isinstance(data, Mapping) else data[0]
            target = data["target"] if isinstance(data, Mapping) else data[1]
            pred = self.model(**input) if isinstance(input, Mapping) else self.model(input)
            loss = self.criterion(pred, target)
            if self.metrics is not None:
                self.metrics.update(pred.squeeze(-1), target)
            self.advance(loss)
        return pred, loss

    def advance(self, loss) -> None:
        r"""
        Backward loss and step optimizer & scheduler.

        Args:
            loss: The loss tensor from which to backpropagate.
        """

        self.backward(loss / self.accum_steps)
        if self.accum_steps <= 1 or self.config.steps % self.accum_steps == 0:
            if self.config.get("max_grad_value") is not None:
                clip_grad_value_(self.model.parameters(), self.config["max_grad_value"])
            if self.config.get("max_grad_norm") is not None:
                clip_grad_norm_(self.model.parameters(), self.config["max_grad_norm"])
            self.optimizer.step()
            self.optimizer.zero_grad()
            if self.ema is not None:
                self.ema.update()
            if self.scheduler is not None:
                self.scheduler.step()
            self.config.steps += 1
        self.config.iters += 1

    def evaluate(self, evaluate_splits: list[str] | None = None) -> NestedDict:
        r"""
        Perform evaluation on `evaluate_splits`.

        Args:
            evaluate_splits: list of split to run evaluate.
                Defaults to `["eval"]`.

        Return:
            NestedDict: evaluation result
        """

        if evaluate_splits is None:
            evaluate_splits = ["eval"]

        print("Begin evaluation")
        print(f"Evaluation splits: {evaluate_splits}")
        result = NestedDict()
        result.setattr("convert_mapping", True)
        for split in evaluate_splits:
            result[split] = self.evaluate_epoch(split=split)
        print(self.format_epoch_result(result))
        return result

    # torch.inference_mode cause experiments to hang
    # @torch.inference_mode()
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
        if self.evaluate_metrics is not None:
            self.metrics = self.evaluate_metrics
        if self.metrics is not None:
            self.metrics.reset()
        batch_time = time()

        for iteration, data in enumerate(loader):
            _, loss = self.evaluate_step(data)

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

        result = self.get_epoch_result()
        self.write_result(result, split, self.config.epochs)
        return result

    def evaluate_step(self, data) -> Tuple[Any, torch.Tensor]:
        input = data["input"] if isinstance(data, Mapping) else data[0]
        target = data["target"] if isinstance(data, Mapping) else data[1]
        model = self.ema or self.model
        pred = model(**input) if isinstance(input, Mapping) else model(input)
        loss = self.criterion(pred, target)
        if self.metrics is not None:
            self.metrics.update(pred.squeeze(-1), target)
        return pred, loss

    @torch.inference_mode()
    def infer(self, split: str = "inf") -> list[float]:
        r"""
        Perform inference on `split`.

        Args:
            split (str): split to run inference

        Return:
            Tensor: inference outputs
        """

        self.mode = "inf"  # type: ignore
        loader = self.dataloaders[split]
        output: list[float] = []
        model = self.ema or self.model
        for _, data in tqdm(enumerate(loader), total=len(loader)):
            input = data["input"] if isinstance(data, Mapping) else data[0]
            pred = model(**input) if isinstance(input, Mapping) else model(input)
            output.extend(pred.squeeze(-1).tolist())

        if self.distributed:
            torch.cuda.synchronize()
            output = self.gather_for_metrics(output)
        return output

    def backward(self, loss: torch.Tensor) -> None:
        r"""
        Backward loss.

        Args:
            loss: Loss to backward.
        """

        loss.backward()

    def has_nan_inf_grad(self, model: nn.Module | None = None) -> bool:
        r"""
        Check if model has NaN or Inf gradients.

        Args:
            model: Model to check.
                Defaults to `self.model`.

        Return:
            bool: True if NaN or Inf is detected in gradients.
        """
        model = model or self.model
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    print(f"NaN detected in gradients of parameter: {name}")
                    return True
                if torch.isinf(param.grad).any():
                    print(f"Inf detected in gradients of parameter: {name}")
                    return True
        return False

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
                torch.cuda.set_device(self.local_rank)
            dist.init_process_group(backend, init_method, world_size=world_size, rank=rank)
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
        cudnn.benchmark = False
        cudnn.deterministic = True
        if torch.__version__ >= "1.8.0":
            torch.use_deterministic_algorithms(True)

    def state_dict(self, cls: Callable = dict) -> Mapping:
        if self.model is None:
            raise ValueError("Model must be defined when calling state_dict")
        return cls(
            runner=self.config.dict(),
            model=self.unwrap(self.model).state_dict(),
            optimizer=self.optimizer.state_dict() if self.optimizer else None,
            scheduler=self.scheduler.state_dict() if self.scheduler else None,
        )

    def unwrap(self, model: nn.Module) -> nn.Module:
        if isinstance(model, (nn.parallel.DistributedDataParallel, nn.parallel.DataParallel)):
            return model.module
        return model

    def build_dataloaders(self):
        datasets = {k: d for k, d in self.datasets.items() if k not in self.dataloaders}
        default_kwargs = self.config.get("dataloader", NestedDict())
        dataloader_kwargs = NestedDict({k: default_kwargs.pop(k) for k in self.datasets if k in default_kwargs})
        for k, d in datasets.items():
            dataloader_kwargs.setdefault(k, NestedDict())
            dataloader_kwargs[k].merge(default_kwargs, overwrite=False)
            shuffle = dataloader_kwargs[k].pop("shuffle", getattr(d, "train", True))
            if self.distributed:
                sampler = utils.data.distributed.DistributedSampler(d, shuffle=shuffle)
            else:
                sampler = utils.data.RandomSampler(d) if shuffle else utils.data.SequentialSampler(d)
            dataloader_kwargs[k].setdefault("drop_last", not getattr(d, "train", True))
            self.dataloaders[k] = utils.data.DataLoader(
                d, sampler=sampler, collate_fn=self.collate_fn, **dataloader_kwargs[k]
            )

    def collate_fn(self, batch):
        return to_device(utils.data.dataloader.default_collate(batch), self.device)

    @contextmanager
    def autocast(self):
        if self.config.get("precision") is None:
            yield nullcontext()
        else:
            yield torch.autocast(self.device.type, dtype=get_precision(self.config.precision))

    @contextmanager
    def accumulate(self):
        if self.accum_steps <= 1 or self.config.steps % self.accum_steps == 0:
            yield nullcontext()
        else:
            yield self.model.no_sync()

    def get_optimizer(self, name: str):
        if name.lower() == "sgd":
            return optim.SGD
        if name.lower() == "asgd":
            return optim.ASGD
        if name.lower() in {"torch_adam", "torch_adamw"}:
            return optim.Adam
        if ds is not None:
            if name.lower() == "adagrad":
                return ds.ops.adagrad.DeepSpeedCPUAdagrad
            if name.lower() in {"adam", "adamw"}:
                if torch.cuda.device_count() > 0:
                    return ds.ops.adam.FusedAdam
                return ds.ops.adam.DeepSpeedCPUAdam
            if name.lower() in {"cpu", "cpu_adam", "cpuadam", "cpu_adamw", "cpuadamw"}:
                return ds.ops.adam.DeepSpeedCPUAdam
            if name.lower() == "lamb":
                if torch.cuda.device_count() > 0:
                    return ds.ops.lamb.FusedLamb
                return ds.ops.lamb.DeepSpeedCPULamb
            if name.lower() in {"cpulamb", "cpu_lamb"}:
                return ds.ops.lamb.DeepSpeedCPULamb
            if name.lower() == "lion":
                if torch.cuda.device_count() > 0:
                    return ds.ops.lion.FusedLion
                return ds.ops.lion.DeepSpeedCPULion
            if name.lower() in {"cpulion", "cpu_lion"}:
                return ds.ops.lion.DeepSpeedCPULion
        if name.lower() in {"adam", "adamw"}:
            return optim.AdamW
        if name.lower() == "adadelta":
            return optim.Adadelta
        if name.lower() == "adafactor":
            return optim.Adafactor
        if name.lower() == "adagrad":
            return optim.Adagrad
        if name.lower() == "adamax":
            return optim.Adamax
        if name.lower() == "lbfgs":
            return optim.LBFGS
        if name.lower() == "nadam":
            return optim.NAdam
        if name.lower() == "radam":
            return optim.RAdam
        if name.lower() == "rmsprop":
            return optim.RMSprop
        if name.lower() == "rprop":
            return optim.Rprop

    def get_deepspeed_config(self, config: NestedDict | str = None) -> NestedDict:  # pylint: disable=R0912, R0915
        r"""
        Preprocess DeepSpeed config.
        """

        if config is None and "deepspeed" in self.config:
            config = self.config.deepspeed
        if isinstance(config, str):
            config = NestedDict(config)
        if config is None:
            config = NestedDict()
        if config.get("steps_per_print", "auto") == "auto":
            config["steps_per_print"] = self.config.log_interval
        if config.get("train_micro_batch_size_per_gpu", "auto") == "auto":
            config["train_micro_batch_size_per_gpu"] = self.batch_size
        if config.get("gradient_accumulation_steps", "auto") == "auto":
            if self.accum_steps > 1:
                config["gradient_accumulation_steps"] = self.accum_steps
            else:
                config.pop("gradient_accumulation_steps", None)
        if "amp" in config:
            amp = config["amp"]
            if amp.get("enabled", "auto") == "auto":
                amp["enabled"] = "true"
            if amp.get("opt_level", "auto") == "auto":
                amp["opt_level"] = "O1"
        if "zero_optimization" in config:
            zero = config["zero_optimization"]
            if zero.get("allgather_bucket_size") == "auto":
                zero["allgather_bucket_size"] = 1e6
            if zero.get("reduce_bucket_size") == "auto":
                zero["reduce_bucket_size"] = 1e6
            if zero.get("stage3_max_live_parameters") == "auto":
                zero["stage3_max_live_parameters"] = 1e8
            if zero.get("stage3_max_live_gradients") == "auto":
                zero["stage3_max_live_gradients"] = 1e8
            if zero.get("stage3_max_reuse_distance") == "auto":
                zero["stage3_max_reuse_distance"] = 1e8
            if zero.get("stage3_prefetch_bucket_size") == "auto":
                zero["stage3_prefetch_bucket_size"] = 1e6
            if zero.get("stage3_param_persistence_threshold") == "auto":
                zero["stage3_param_persistence_threshold"] = 1e8
            if "amp" in config:
                if "fp16" not in config:
                    config["fp16"] = NestedDict()
                if config["fp16"].get("enabled", "auto"):
                    config["fp16"]["enabled"] = config["amp"]["enabled"]
                warn(
                    f"AMP is not compatible with ZeRO. Automatically set 'fp16' to {config['amp']['enabled']}",
                    stacklevel=2,
                )
                del config["amp"]
        if "optimizer" in config:
            if config["optimizer"].get("type", "auto") == "auto":
                config["optimizer"]["type"] = "Adam"
            if "params" not in config["optimizer"]:
                config["optimizer"]["params"] = NestedDict()
            optimizer = config["optimizer"]["params"]
            if optimizer.get("lr", "auto") == "auto":
                optimizer["lr"] = self.config.get("optim.lr", 1e-3)
            if optimizer.get("weight_decay", "auto") == "auto":
                optimizer["weight_decay"] = self.config.get("optim.weight_decay", 1e-2)
            if optimizer.get("betas") == "auto":
                optimizer["betas"] = (0.9, 0.999)
            if optimizer.get("eps") == "auto":
                optimizer["eps"] = 1e-8
        if "scheduler" in config:
            if config["scheduler"].get("type", "auto") == "auto":
                config["scheduler"]["type"] = "WarmupCosineLR"
            if "params" not in config["scheduler"]:
                config["scheduler"]["params"] = NestedDict()
            scheduler = config["scheduler"]["params"]
            if scheduler.get("total_num_steps", "auto") == "auto":
                scheduler["total_num_steps"] = self.total_steps
            if scheduler.get("warmup_num_steps", "auto") == "auto":
                scheduler["warmup_num_steps"] = scheduler["total_num_steps"] // 20
            if config["scheduler"]["type"] in ("WarmupLR", "WarmupDecayLR"):
                if scheduler.get("warmup_max_lr", "auto") == "auto":
                    if self.optimizer:
                        scheduler["warmup_max_lr"] = self.optimizer.param_groups[0]["lr"]
                    elif "optimizer" in config:
                        scheduler["warmup_max_lr"] = config["optimizer"]["params"]["lr"]
                    else:
                        scheduler["warmup_max_lr"] = self.config.get("optim.lr", 1e-3)
                if scheduler.get("warmup_min_lr", "auto") == "auto":
                    scheduler["warmup_min_lr"] = 1e-9
            else:
                scheduler.pop("warmup_max_lr", None)
                scheduler.pop("warmup_min_lr", None)
        if config.get("gradient_clipping", "auto") == "auto" and self.config.get("max_grad_norm") is not None:
            config["gradient_clipping"] = self.config["max_grad_norm"]
        return config

    @property
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu", self.local_rank)

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
        if self.ema is not None:
            self.ema.train(mode == RunnerMode.train)

    @property
    def rank(self) -> int:
        if self.distributed:
            return dist.get_rank()
        return 0

    @property
    def local_rank(self) -> int:
        if local_rank := os.getenv("LOCAL_RANK"):
            return int(local_rank)
        return 0

    @property
    def world_size(self) -> int:
        r"""
        Number of Processes.
        """

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return dist.get_world_size()
        return 1

    @property
    def distributed(self) -> bool:
        return self.world_size > 1

    @cached_property
    def accum_steps(self) -> int:
        return self.config.get("accum_steps", 1)

    @staticmethod
    def reduce(tensor: torch.Tensor) -> torch.Tensor:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            dist.all_reduce(tensor)
        return tensor


def get_precision(precision: str) -> torch.dtype:
    if precision in ("fp16", "float16", "half"):
        return torch.float16
    if precision in ("bf16", "bfloat16"):
        return torch.bfloat16
    raise ValueError(f"Precision {precision} is not supported")
