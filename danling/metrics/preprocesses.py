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

from collections.abc import Sequence
from warnings import warn

import torch
from torch import Tensor

from danling.tensors import NestedTensor


def infer_task(num_classes: int | None, num_labels: int | None):
    if num_classes is not None and num_labels is not None:
        raise ValueError("Only one of `num_classes` or `num_labels` can be provided.")
    if num_classes is not None:
        return "multiclass"
    if num_labels is not None:
        return "multilabel"
    return "binary"


def preprocess(
    input: Tensor | NestedTensor | Sequence, target: Tensor | NestedTensor | Sequence, ignored_index: int | None = None
):
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
    if isinstance(input, NestedTensor) or isinstance(target, NestedTensor):
        if isinstance(input, NestedTensor) and isinstance(target, Tensor):
            target = input.nested_like(target, strict=False)
        if isinstance(target, NestedTensor) and isinstance(input, Tensor):
            input = target.nested_like(input, strict=False)
        input, target = input.concat, target.concat
    if ignored_index is not None:
        input, target = input[target != ignored_index], target[target != ignored_index]
    if input.numel() == target.numel():
        return input.squeeze(), target.squeeze()
    return input, target


def preprocess_regression(
    input: Tensor | NestedTensor | Sequence,
    target: Tensor | NestedTensor | Sequence,
    num_outputs: int = 1,
    ignored_index: None = None,
):
    if ignored_index is not None:
        warn("`ignored_index` is not used for regression tasks.")
    input, target = preprocess(input, target)
    if num_outputs > 1:
        return input.view(-1, num_outputs), target.view(-1, num_outputs)
    return input.view(-1), target.view(-1)


def preprocess_binary(
    input: Tensor | NestedTensor | Sequence, target: Tensor | NestedTensor | Sequence, ignored_index: int | None = None
):
    input, target = preprocess(input, target, ignored_index=ignored_index)
    input, target = input.view(-1), target.view(-1)
    if input.max() > 1 or input.min() < 0:
        input = input.sigmoid()
    return input, target


def preprocess_multiclass(
    input: Tensor | NestedTensor | Sequence,
    target: Tensor | NestedTensor | Sequence,
    num_classes: int,
    ignored_index: int | None = None,
):
    input, target = preprocess(input, target, ignored_index=ignored_index)
    input, target = input.view(-1, num_classes), target.view(-1)
    if input.max() > 1 or input.min() < 0:
        input = input.softmax(dim=-1)
    return input, target


def preprocess_multilabel(
    input: Tensor | NestedTensor | Sequence,
    target: Tensor | NestedTensor | Sequence,
    num_labels: int,
    ignored_index: int | None = None,
):
    input, target = preprocess(input, target, ignored_index=ignored_index)
    input, target = input.view(-1, num_labels), target.view(-1, num_labels)
    if input.max() > 1 or input.min() < 0:
        input = input.sigmoid()
    return input, target
