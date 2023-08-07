from __future__ import annotations

from functools import wraps
from typing import Any


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
