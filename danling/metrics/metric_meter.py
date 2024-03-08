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

from collections.abc import Mapping, Sequence
from typing import Any, Callable, Optional, Tuple

from torch import Tensor

from ..tensors import NestedTensor
from .average_meter import AverageMeter, AverageMeters, MultiTaskAverageMeters
from .functional import preprocess
from .utils import MultiTaskDict


class MetricMeter(AverageMeter):
    r"""
    Computes metrics and averages them over time.

    Attributes:
        metric: Metric function for computing the value.
        ignored_index: Index to be ignored in the computation.
        val: Results of current batch on current device.
        bat: Results of current batch on all devices.
        avg: Results of all results on all devices.
        sum: Sum of values.
        count: Number of values.

    See Also:
        [`AverageMeter`]: Average meter for computed values.
        [`MetricMeters`]: Manage multiple metric meters in one object.

    Examples:
        >>> from danling.metrics.functional import accuracy
        >>> meter = MetricMeter(accuracy)
        >>> meter.update([0.1, 0.8, 0.6, 0.2], [0, 1, 0, 0])
        >>> meter.val
        0.75
        >>> meter.avg
        0.75
        >>> meter.update([0.1, 0.7, 0.3, 0.2, 0.8, 0.4], [0, 1, 1, 0, 0, 1])
        >>> meter.val
        0.5
        >>> meter.avg
        0.6
        >>> meter.sum
        6.0
        >>> meter.count
        10
        >>> meter.reset()
        >>> meter.val
        0
        >>> meter.avg
        nan
    """

    metric: Callable
    ignored_index: Optional[int] = None

    def __init__(self, metric: Callable, ignored_index: int | None = None) -> None:
        self.metric = metric
        self.ignored_index = ignored_index
        super().__init__()

    def reset(self) -> None:
        r"""
        Resets the meter.
        """

        self.val = 0
        self.n = 1
        self.sum = 0
        self.count = 0

    def update(  # type: ignore[override] # pylint: disable=W0237
        self,
        input: Tensor | NestedTensor | Sequence,  # pylint: disable=W0622
        target: Tensor | NestedTensor | Sequence,
    ) -> None:
        r"""
        Updates the average and current value in the meter.

        Args:
            value: Value to be added to the average.
            n: Number of values to be added.
        """

        input, target = preprocess(input, target, ignored_index=self.ignored_index)
        super().update(self.metric(input, target).item(), n=len(input))


class MetricMeters(AverageMeters):
    r"""
    Manages multiple metric meters in one object.

    Attributes:
        ignored_index: Index to be ignored in the computation.
            Defaults to None.

    See Also:
        [`MetricMeter`]: Computes metrics and averages them over time.
        [`AverageMeters`]: Average meters for computed values.

    >>> from danling.metrics.functional import accuracy, auroc, auprc
    >>> meters = MetricMeters(acc=accuracy, auroc=auroc, auprc=auprc)
    >>> meters.update([0.1, 0.8, 0.6, 0.2], [0, 1, 0, 0])
    >>> meters.sum.dict()
    {'acc': 3.0, 'auroc': 4.0, 'auprc': 4.0}
    >>> meters.count.dict()
    {'acc': 4, 'auroc': 4, 'auprc': 4}
    >>> meters['auroc'].update([0.2, 0.8], [0, 1])
    >>> meters.sum.dict()
    {'acc': 3.0, 'auroc': 6.0, 'auprc': 4.0}
    >>> meters.count.dict()
    {'acc': 4, 'auroc': 6, 'auprc': 4}
    >>> meters.update([[0.1, 0.7, 0.3, 0.2], [0.8, 0.4]], [[0, 0, 1, 0], [0, 0]])
    >>> meters.sum.dict()
    {'acc': 6.0, 'auroc': 8.4, 'auprc': 5.5}
    >>> meters.count.dict()
    {'acc': 10, 'auroc': 12, 'auprc': 10}
    >>> meters['auroc'].update([0.4, 0.8, 0.6, 0.2], [0, 1, 1, 0])
    >>> meters.avg.dict()
    {'acc': 0.6, 'auroc': 0.775, 'auprc': 0.55}
    >>> meters.update(dict(loss=""))  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    TypeError: ...update() missing 1 required positional argument: 'target'
    """

    ignored_index: Optional[int] = None

    def __init__(self, *args, ignored_index: int | None = None, **kwargs) -> None:
        self.setattr("ignored_index", ignored_index)
        for meter in args:
            if callable(meter):
                meter = MetricMeter(meter, ignored_index=self.ignored_index)
            if not isinstance(meter, MetricMeter):
                raise ValueError(f"Expected meter to be an instance of MetricMeter, but got {type(meter)}")
        for name, meter in kwargs.items():
            if callable(meter):
                kwargs[name] = meter = MetricMeter(meter, ignored_index=self.ignored_index)
            if not isinstance(meter, MetricMeter):
                raise ValueError(f"Expected {name} to be an instance of MetricMeter, but got {type(meter)}")
        super().__init__(*args, default_factory=None, **kwargs)  # type: ignore[arg-type]

    def update(  # type: ignore[override] # pylint: disable=W0221
        self,
        input: Tensor | NestedTensor | Sequence,  # pylint: disable=W0622
        target: Tensor | NestedTensor | Sequence,
    ) -> None:
        r"""
        Updates the average and current value in all meters.

        Args:
            input: Input values to compute the metrics.
            target: Target values to compute the metrics.
        """

        input, target = preprocess(input, target, ignored_index=self.ignored_index)
        for meter in self.values():
            meter.update(input, target)

    def set(self, name: str, meter: MetricMeter | Callable) -> None:  # type: ignore[override] # pylint: disable=W0237
        if callable(meter):
            meter = MetricMeter(meter, ignored_index=self.ignored_index)
        if not isinstance(meter, MetricMeter):
            raise ValueError(f"Expected meter to be an instance of MetricMeter, but got {type(meter)}")
        super().set(name, meter)

    def __repr__(self):
        keys = tuple(i for i in self.keys())
        return f"{self.__class__.__name__}{keys}"


class MultiTaskMetricMeters(MultiTaskAverageMeters):
    r"""
    Examples:
        >>> from danling.metrics.functional import accuracy
        >>> metrics = MultiTaskMetricMeters()
        >>> metrics.dataset1.cls = MetricMeters(acc=accuracy)
        >>> metrics.dataset2 = MetricMeters(acc=accuracy)
        >>> metrics
        MultiTaskMetricMeters(<class 'danling.metrics.metric_meter.MultiTaskMetricMeters'>,
          ('dataset1'): MultiTaskMetricMeters(<class 'danling.metrics.metric_meter.MultiTaskMetricMeters'>,
            ('cls'): MetricMeters('acc',)
          )
          ('dataset2'): MetricMeters('acc',)
        )
        >>> metrics.update({"dataset1.cls": {"input": [0.2, 0.4, 0.5, 0.7], "target": [0, 1, 0, 1]}, "dataset2": ([0.1, 0.4, 0.6, 0.8], [1, 0, 0, 0])})
        >>> f"{metrics:.4f}"
        'dataset1.cls: acc: 0.5000 (0.5000)\ndataset2: acc: 0.2500 (0.2500)'
        >>> metrics.setattr("return_average", True)
        >>> metrics.update({"dataset1.cls": [[0.1, 0.4, 0.6, 0.8], [0, 0, 1, 0]], "dataset2": {"input": [0.2, 0.3, 0.5, 0.7], "target": [0, 0, 0, 1]}})
        >>> f"{metrics:.4f}"
        'dataset1.cls: acc: 0.7500 (0.6250)\ndataset2: acc: 0.7500 (0.5000)'
    """  # noqa: E501

    def __init__(self, *args, **kwargs):
        super().__init__(*args, default_factory=MultiTaskMetricMeters, **kwargs)

    def update(  # type: ignore[override] # pylint: disable=W0221
        self,
        values: Mapping[str, Tuple[Tensor | NestedTensor | Sequence, Tensor | NestedTensor | Sequence]],
    ) -> None:
        r"""
        Updates the average and current value in all meters.

        Args:
            input: Input values to compute the metrics.
            target: Target values to compute the metrics.
        """

        for metric, value in values.items():
            if isinstance(value, (Mapping, Sequence)):
                if metric not in self:
                    raise ValueError(f"Metric {metric} not found in {self}")
                if isinstance(self[metric], MultiTaskMetricMeters):
                    for met in self[metric].all_values():
                        if isinstance(value, Mapping):
                            met.update(**value)
                        elif isinstance(value, Sequence):
                            met.update(*value)
                        else:
                            raise ValueError(f"Expected value to be a Mapping or Sequence, but got {type(value)}")
                elif isinstance(self[metric], (MetricMeters, MetricMeter)):
                    if isinstance(value, Mapping):
                        self[metric].update(**value)
                    elif isinstance(value, Sequence):
                        self[metric].update(*value)
                    else:
                        raise ValueError(f"Expected value to be a Mapping or Sequence, but got {type(value)}")
                else:
                    raise ValueError(
                        f"Expected {metric} to be an instance of MultiTaskMetricMeters, MetricMeters, "
                        "or MetricMeter, but got {type(self[metric])}"
                    )
            else:
                raise ValueError(f"Expected values to be a Mapping or Sequence, but got {type(value)}")

    # MultiTaskAverageMeters.get is hacked
    def get(self, name: Any, default=None) -> Any:
        return MultiTaskDict.get(self, name, default)

    def set(  # pylint: disable=W0237
        self,
        name: str,
        meter: MetricMeter | MetricMeters | Callable,  # type: ignore[override]
    ) -> None:
        if callable(meter):
            meter = MetricMeter(meter)
        if not isinstance(meter, (MetricMeter, MetricMeters)):
            raise ValueError(f"Expected meter to be an instance of MetricMeter or MetricMeters, but got {type(meter)}")
        super().set(name, meter)
