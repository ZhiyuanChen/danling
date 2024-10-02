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

# pylint: disable=redefined-builtin
from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import nan
from typing import Callable, Iterable, Optional

import torch
from chanfig import DefaultDict, FlatDict, NestedDict
from torch import Tensor
from torch import distributed as dist
from torcheval.metrics import Metric

from danling.tensors import NestedTensor
from danling.utils import flist, get_world_size

from .preprocesses import preprocess as default_preprocess
from .utils import MultiTaskDict

try:
    from typing import Self  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import Self


class Metrics(Metric):
    r"""
    Metric class wraps around multiple metrics that share the same states.

    Typically, there are many metrics that we want to compute for a single task.
    For example, we usually needs to compute `pearson` and `spearman` for a regression task.
    Unlike `accuracy`, which can uses an average meter to compute the average accuracy,
    `pearson` and `spearman` cannot be computed by averaging the results of multiple batches.
    They need access to all the data to compute the correct results.
    And saving all intermediate results for each tasks is quite inefficient.

    `Metrics` solves this problem by maintaining a shared state for multiple metric functions.

    Attributes:
        metrics: A dictionary of metrics to be computed.A
        ignored_index: Index to be ignored in the computation.
        val: Metric results of current batch on current device.
        avg: Metric results of all results on all devices.
        input: The input tensor of latest batch.
        target: The target tensor of latest batch.
        inputs: All input tensors.
        targets: All target tensors.

    Args:
        *args: A single mapping of metrics.
        **metrics: Metrics.

    Examples:
        >>> from danling.metrics.functional import auroc, auprc
        >>> metrics = Metrics(auroc=auroc, auprc=auprc)
        >>> metrics
        Metrics('auroc', 'auprc')
        >>> metrics.update([0.2, 0.3, 0.5, 0.7], [0, 1, 0, 1])
        >>> metrics.input  # predicted values of current batch
        tensor([0.2000, 0.3000, 0.5000, 0.7000])
        >>> metrics.target  # ground truth of current batch
        tensor([0, 1, 0, 1])
        >>> metrics.inputs  # predicted values of all data
        tensor([0.2000, 0.3000, 0.5000, 0.7000])
        >>> metrics.targets  # ground truth of all data
        tensor([0, 1, 0, 1])
        >>> metrics.val  # Metrics of current batch on current device
        NestedDict(
          ('auroc'): 0.75
          ('auprc'): 0.8333333730697632
        )
        >>> metrics.avg  # Metrics of all data on all devices
        NestedDict(
          ('auroc'): 0.75
          ('auprc'): 0.8333333730697632
        )
        >>> metrics.update([0.1, 0.4, 0.6, 0.8], [0, 0, 1, 0])
        >>> metrics.input  # predicted values of current batch
        tensor([0.1000, 0.4000, 0.6000, 0.8000])
        >>> metrics.target  # ground truth of current batch
        tensor([0, 0, 1, 0])
        >>> metrics.inputs  # predicted values of all data
        tensor([0.2000, 0.3000, 0.5000, 0.7000, 0.1000, 0.4000, 0.6000, 0.8000])
        >>> metrics.targets  # ground truth of all data
        tensor([0, 1, 0, 1, 0, 0, 1, 0])
        >>> metrics.val  # Metrics of current batch on current device
        NestedDict(
          ('auroc'): 0.6666666666666666
          ('auprc'): 0.5
        )
        >>> metrics.avg  # Metrics of all data on all devices
        NestedDict(
          ('auroc'): 0.6666666666666666
          ('auprc'): 0.5555555820465088
        )
        >>> f"{metrics:.4f}"
        'auroc: 0.6667 (0.6667)\tauprc: 0.5000 (0.5556)'
        >>> metrics = Metrics(auroc=auroc, auprc=auprc, ignored_index=-100)
        >>> metrics.update([[0.1, 0.4, 0.6, 0.8], [0.1, 0.4, 0.6]], [[0, -100, 1, 0], [0, -100, 1]])
        >>> metrics.input, metrics.target
        (tensor([0.1000, 0.6000, 0.8000, 0.1000, 0.6000]), tensor([0, 1, 0, 0, 1]))
    """

    metrics: FlatDict[str, Callable]
    preprocess: Callable
    ignored_index: Optional[int] = None
    _input: Tensor
    _target: Tensor
    _inputs: Tensor
    _targets: Tensor
    score_name: str
    best_fn: Callable
    merge_dict: bool = True
    return_nested: bool = False

    def __init__(
        self,
        *args,
        merge_dict: bool | None = None,
        return_nested: bool | None = None,
        device: torch.device | None = None,
        ignored_index: int | None = None,
        preprocess: Callable = default_preprocess,
        **metrics: Callable,
    ):
        super().__init__(device=device)
        self._add_state("_input", torch.empty(0))
        self._add_state("_target", torch.empty(0))
        self._add_state("_inputs", torch.empty(0))
        self._add_state("_targets", torch.empty(0))
        self.world_size = get_world_size()
        self.metrics = FlatDict(*args, **metrics)
        self.preprocess = preprocess
        if merge_dict is not None:
            self.merge_dict = merge_dict
        if return_nested is not None:
            self.return_nested = return_nested
        self.ignored_index = ignored_index

    def update(self, input: Tensor | NestedTensor | Sequence, target: Tensor | NestedTensor | Sequence) -> None:
        # convert input and target to Tensor if they are not
        if not isinstance(input, (Tensor, NestedTensor)):
            try:
                input = torch.tensor(input)
            except ValueError:
                input = NestedTensor(input)
        if not isinstance(target, (Tensor, NestedTensor)):
            try:
                target = torch.tensor(target)
            except ValueError:
                target = NestedTensor(target)
        if input.ndim == target.ndim + 1:
            input = input.squeeze(-1)
        # convert input and target to NestedTensor if one of them is
        if isinstance(input, NestedTensor) or isinstance(target, NestedTensor):
            if isinstance(target, NestedTensor) and isinstance(input, NestedTensor):
                input, target = input.concat, target.concat
            elif isinstance(input, NestedTensor):
                input, mask = input.concat, input.mask
                target = target[mask]
            elif isinstance(target, NestedTensor):
                target, mask = target.concat, target.mask
                input = input[mask]
            else:
                raise ValueError(f"Unknown input and target: {input}, {target}")
        # remove ignored index
        if self.ignored_index is not None:
            if isinstance(input, NestedTensor):
                indices = [i != self.ignored_index for i in target.storage()]
                input = NestedTensor([t[i] for t, i in zip(input.storage(), indices)])
                target = NestedTensor([t[i] for t, i in zip(target.storage(), indices)])
            else:
                input, target = input[target != self.ignored_index], target[target != self.ignored_index]
        if self.world_size > 1:
            input, target = self._sync(input), self._sync(target)
        input, target = input.detach().to(self.device), target.detach().to(self.device)
        self._input = input
        self._target = target
        self._inputs = torch.cat([self._inputs, input]).to(input.dtype)
        self._targets = torch.cat([self._targets, target]).to(target.dtype)

    def value(self) -> NestedDict[str, float | flist]:
        return self.calculate(self.input, self.target)

    def average(self) -> NestedDict[str, float | flist]:
        return self.calculate(self.inputs, self.targets)

    def compute(self) -> NestedDict[str, float | flist]:
        return self.average()

    @property
    def val(self) -> NestedDict[str, float | flist]:
        return self.value()

    @property
    def avg(self) -> NestedDict[str, float | flist]:
        return self.average()

    @torch.inference_mode()
    def calculate(self, input: Tensor, target: Tensor) -> NestedDict[str, flist | float]:
        if (
            isinstance(input, (Tensor, NestedTensor))
            and input.numel() == 0 == target.numel()
            or isinstance(input, (list, dict))
            and len(input) == 0 == len(target)
        ):
            return NestedDict({name: nan for name in self.metrics.keys()})
        ret = NestedDict()
        input, target = self.preprocess(input, target, ignored_index=self.ignored_index)
        for name, metric in self.metrics.items():
            score = self._calculate(metric, input, target, preprocess=False)
            if isinstance(score, Mapping):
                if self.merge_dict:
                    ret.merge(score)
                else:
                    for n, s in score.items():
                        ret[f"{name}.{n}"] = s
            else:
                ret[name] = score
        return ret

    @torch.inference_mode()
    def _calculate(self, metric, input: Tensor, target: Tensor, preprocess: bool = True) -> flist | float:
        if preprocess:
            input, target = self.preprocess(input, target, ignored_index=self.ignored_index)
        score = metric(input, target)
        if isinstance(score, Tensor):
            return score.item() if score.numel() == 1 else flist(score.tolist())
        return score

    @torch.inference_mode()
    def merge_state(self, metrics: Iterable):
        raise NotImplementedError()

    def _sync(self, tensor: Tensor):
        local_size = torch.tensor([tensor.shape[0]], dtype=torch.int64, device=tensor.device)
        size_list = [torch.zeros_like(local_size) for _ in range(self.world_size)]
        dist.all_gather(size_list, local_size)
        sizes = torch.cat(size_list)
        max_size = sizes.max()

        padded_tensor = torch.empty((max_size, *tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device)
        padded_tensor[: tensor.shape[0]] = tensor
        gathered_tensors = [torch.empty_like(padded_tensor) for _ in range(self.world_size)]
        dist.all_gather(gathered_tensors, padded_tensor)
        slices = [gathered_tensors[i][: sizes[i]] for i in range(self.world_size) if sizes[i] > 0]
        return torch.cat(slices, dim=0)

    @property
    def input(self):
        return self._input

    @property
    def target(self):
        return self._target

    @property
    def inputs(self):
        return self._inputs

    @property
    def targets(self):
        return self._targets

    def __repr__(self):
        keys = tuple(i for i in self.metrics.keys())
        return f"{self.__class__.__name__}{keys}"

    def __format__(self, format_spec):
        val, avg = self.value(), self.average()
        return "\t".join(
            [f"{key}: {val[key].__format__(format_spec)} ({avg[key].__format__(format_spec)})" for key in self.metrics]
        )

    def reset(self: Self) -> Self:  # pragma: no cover
        r"""
        Reset the metric state variables to their default value.
        The tensors in the default values are also moved to the device of
        the last ``self.to(device)`` call.
        """
        for state_name, default in self._state_name_to_default.items():
            if isinstance(default, Tensor):
                setattr(self, state_name, default.clone().to(self.device))
            elif isinstance(default, list):
                setattr(
                    self,
                    state_name,
                    flist(tensor.clone().to(self.device) for tensor in default),
                )
            elif isinstance(default, dict):
                setattr(
                    self,
                    state_name,
                    DefaultDict(
                        lambda: torch.tensor(0.0, device=self.device),
                        {key: tensor.clone().to(self.device) for key, tensor in default.items()},
                    ),
                )
            elif isinstance(default, (int, float)):
                setattr(self, state_name, default)
            else:
                raise TypeError(
                    f"Invalid type for default value for {state_name}. Received {type(default)},"
                    "but expected ``Tensor``, a list of ``Tensor``,"
                    "a dictionary with ``Tensor``, int, or float."
                )
        return self


class ScoreMetrics(Metrics):  # pylint: disable=abstract-method
    r"""
    `ScoreMetrics` is a subclass of Metrics that supports scoring.

    Score is a single value that best represents the performance of the model.
    It is the core metrics that we use to compare different models.
    For example, in classification, we usually use auroc as the score.

    `ScoreMetrics` requires two additional arguments: `score_name` and `best_fn`.
    `score_name` is the name of the metric that we use to compute the score.
    `best_fn` is a function that takes a list of values and returns the best value.
    `best_fn` is only not used by `ScoreMetrics`, it is meant to be accessed by other classes.

    Attributes:
        score_name: The name of the metric that we use to compute the score.
        best_fn: A function that takes a list of values and returns the best value.

    Args:
        *args: A single mapping of metrics.
        score_name: The name of the metric that we use to compute the score. Defaults to the first metric.
        best_fn: A function that takes a list of values and returns the best value. Defaults to `max`.
        **metrics: Metrics.
    """

    _score_name: str
    best_fn: Callable

    def __init__(
        self, *args, score_name: str | None = None, best_fn: Callable | None = max, **metrics: NestedDict[str, Callable]
    ):
        super().__init__(*args, **metrics)
        self.score_name = score_name or next(iter(self.metrics.keys()))
        self.metric = self.metrics[self.score_name]
        self.best_fn = best_fn or max

    def get_score(self, scope: str) -> float | flist:
        if scope == "batch":
            return self.batch_score
        if scope == "average":
            return self.average_score
        raise ValueError(f"Unknown scope: {scope}")

    @property
    def batch_score(self) -> float | flist:
        return self._calculate(self.metric, self.input, self.target)

    @property
    def average_score(self) -> float | flist:
        return self._calculate(self.metric, self.inputs, self.targets)

    @property
    def score_name(self) -> str:
        return self._score_name

    @score_name.setter
    def score_name(self, name) -> None:
        if name not in self.metrics:
            raise ValueError(f"score_name must be in {self.metrics.keys()}, but got {name}")
        self._score_name = name


class MultiTaskMetrics(MultiTaskDict):
    r"""
    Examples:
        >>> from danling.metrics.functional import auroc, auprc, pearson, spearman, accuracy, mcc
        >>> metrics = MultiTaskMetrics()
        >>> metrics.dataset1.cls = Metrics(auroc=auroc, auprc=auprc)
        >>> metrics.dataset1.reg = Metrics(pearson=pearson, spearman=spearman)
        >>> metrics.dataset2 = Metrics(auroc=auroc, auprc=auprc)
        >>> metrics
        MultiTaskMetrics(<class 'danling.metrics.metrics.MultiTaskMetrics'>,
          ('dataset1'): MultiTaskMetrics(<class 'danling.metrics.metrics.MultiTaskMetrics'>,
            ('cls'): Metrics('auroc', 'auprc')
            ('reg'): Metrics('pearson', 'spearman')
          )
          ('dataset2'): Metrics('auroc', 'auprc')
        )
        >>> metrics.update({"dataset1.cls": {"input": [0.2, 0.4, 0.5, 0.7], "target": [0, 1, 0, 1]}, "dataset1.reg": {"input": [0.1, 0.4, 0.6, 0.8], "target": [0.2, 0.3, 0.5, 0.7]}, "dataset2": {"input": [0.1, 0.4, 0.6, 0.8], "target": [0, 1, 0, 1]}})
        >>> f"{metrics:.4f}"
        'dataset1.cls: auroc: 0.7500 (0.7500)\tauprc: 0.8333 (0.8333)\ndataset1.reg: pearson: 0.9691 (0.9691)\tspearman: 1.0000 (1.0000)\ndataset2: auroc: 0.7500 (0.7500)\tauprc: 0.8333 (0.8333)'
        >>> metrics.setattr("return_average", True)
        >>> metrics.update({"dataset1.cls": {"input": [0.1, 0.4, 0.6, 0.8], "target": [0, 0, 1, 0]}, "dataset1.reg": {"input": [0.2, 0.3, 0.5, 0.7], "target": [0.2, 0.4, 0.6, 0.8]}, "dataset2": {"input": [0.2, 0.3, 0.5, 0.7], "target": [0, 0, 1, 0]}})
        >>> f"{metrics:.4f}"
        'dataset1.cls: auroc: 0.6667 (0.7000)\tauprc: 0.5000 (0.5556)\ndataset1.reg: pearson: 0.9898 (0.9146)\tspearman: 1.0000 (0.9222)\ndataset2: auroc: 0.6667 (0.7333)\tauprc: 0.5000 (0.7000)'
        >>> metrics.update({"dataset1": {"cls": {"input": [0.1, 0.4, 0.6, 0.8], "target": [1, 0, 1, 0]}}})
        >>> f"{metrics:.4f}"
        'dataset1.cls: auroc: 0.2500 (0.5286)\tauprc: 0.5000 (0.4789)\ndataset1.reg: pearson: 0.9898 (0.9146)\tspearman: 1.0000 (0.9222)\ndataset2: auroc: 0.6667 (0.7333)\tauprc: 0.5000 (0.7000)'
        >>> metrics.update(dict(loss=""))  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ValueError: Metric loss not found in ...
    """  # noqa: E501

    pass
