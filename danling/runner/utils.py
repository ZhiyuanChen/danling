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
import sys
from collections.abc import Mapping
from contextlib import suppress
from datetime import datetime
from enum import auto
from functools import wraps
from math import isnan
from typing import Any
from warnings import warn

import torch
from chanfig import FlatDict, NestedDict

from danling.utils import base62

try:
    from enum import StrEnum  # type: ignore[attr-defined]
except ImportError:
    from strenum import LowercaseStrEnum as StrEnum  # type: ignore[no-redef]


class RunnerMeta(type):
    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        instance = super().__call__(*args, **kwargs)
        instance.__post_init__()
        return instance


class RunnerMode(StrEnum):  # pylint: disable=too-few-public-methods
    r"""
    `RunnerMode` is an enumeration of running modes.

    Attributes:
        train: Training mode.
        eval: Evaluation mode.
        inf: Inference mode.
    """

    train = auto()
    eval = auto()
    inf = auto()


def get_time_str() -> str:
    time = datetime.now()
    time_tuple = time.isocalendar()[1:] + (
        time.hour,
        time.minute,
        time.second,
        time.microsecond,
    )
    return "".join(base62.encode(i) for i in time_tuple)


def get_git_hash() -> str | None:
    try:
        from git.exc import InvalidGitRepositoryError
        from git.repo import Repo

        try:
            return Repo(search_parent_directories=True).head.object.hexsha
        except ImportError:
            pass  # handle at last
        except (InvalidGitRepositoryError, ValueError):
            warn(
                "Unable to get git hash from CWD, fallback to git has of top-level code environment.",
                category=RuntimeWarning,
                stacklevel=2,
            )
            path = os.path.dirname(os.path.abspath(sys.argv[0]))
            with suppress(InvalidGitRepositoryError, ValueError):
                return Repo(path=path, search_parent_directories=True).head.object.hexsha
    except ImportError:
        warn(
            "GitPython is not installed, unable to fing git hash",
            category=RuntimeWarning,
            stacklevel=2,
        )
    return None


def on_main_process(func):
    """
    Decorator to run func only on main process.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs) -> Any | None:
        if self.is_main_process or not self.distributed:
            return func(self, *args, **kwargs)
        return None

    return wrapper


def on_local_main_process(func):
    """
    Decorator to run func only on local main process.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs) -> Any | None:
        if self.is_local_main_process or not self.distributed:
            return func(self, *args, **kwargs)
        return None

    return wrapper


def format_result(result, format_spec: str = ".4f", depth: int = 0):
    longest_key = max(len(k) for k in result.keys())
    repr_list = [_format_result(result, format_spec)]
    for k, v in result.items():
        if isinstance(v, Mapping):
            initials = " " * (longest_key - len(k)) + "\t" * depth
            repr_list.append(f"{initials}{k}: {format_result(v, format_spec, depth + 1)}")
    return "\n".join(repr_list)


def _format_result(result, format_spec: str = ".4f"):
    repr_list = []
    for k, v in result.items():
        if isinstance(v, (Mapping,)):
            continue
        if isinstance(v, (float,)):
            repr_list.append(f"{k}: {format(v, format_spec)}" if not isnan(v) else f"{k}: NaN\t")
        else:
            repr_list.append(f"{k}: {v}\t")
    return "\t".join(repr_list)


def to_device(data: Any, device: torch.device):
    if isinstance(data, list):
        return [to_device(i, device) for i in data]
    if isinstance(data, tuple):
        return tuple(to_device(i, device) for i in data)
    if isinstance(data, NestedDict):
        return NestedDict({k: to_device(v, device) for k, v in data.all_items()})
    if isinstance(data, dict):
        return FlatDict({k: to_device(v, device) for k, v in data.items()})
    if hasattr(data, "to"):
        return data.to(device)
    return data
