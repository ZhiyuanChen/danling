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

# pylint: disable=protected-access
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Callable, Iterable, SupportsFloat, Tuple

import torch
from torch import Tensor
from torch.utils.data._utils.collate import default_collate_fn_map

from ..utils import method_cache
from .functional import mask_tensor, pad_tensor, tensor_mask
from .utils import TorchFuncRegistry


def tensor(data: Any, dtype=None, device=None, requires_grad: bool = False, pin_memory: bool = False) -> PNTensor:
    return PNTensor(torch.tensor(data, dtype=dtype, device=device, requires_grad=requires_grad, pin_memory=pin_memory))


class PNTensor(Tensor):
    r"""
    Wrapper for tensors to be converted to `NestedTensor`.

    `PNTensor` is a subclass of `torch.Tensor`.
    It implements three additional property as `NestedTensor`: `tensor`, `mask`, and `concat`.

    Although it is possible to directly construct `NestedTensor` in dataset,
    the best practice is to do so is in `collate_fn`.
    `PNTensor` is introduced to smoothen the process.

    Convert tensors that will be converted to `NestedTensor` to a `PNTensor`,
    and PyTorch Dataloader will automatically collate `PNTensor` to `NestedTensor`.
    """

    @property
    def tensor(self) -> Tensor:
        r"""
        Identical to `self`.

        Returns:
            (torch.Tensor):

        Examples:
            >>> tensor = torch.tensor([1, 2, 3])
            >>> pn_tensor = PNTensor([1, 2, 3])
            >>> bool((tensor == pn_tensor).all())
            True
            >>> bool((tensor == pn_tensor.tensor).all())
            True
        """

        return self

    @property
    def mask(self) -> Tensor:
        r"""
        Identical to `torch.ones_like(self)`.

        Returns:
            (torch.Tensor):

        Examples:
            >>> tensor = torch.tensor([1, 2, 3])
            >>> pn_tensor = PNTensor([1, 2, 3])
            >>> bool((pn_tensor.mask == torch.ones_like(pn_tensor)).all().item())
            True
        """

        return torch.ones_like(self)

    @property
    def contact(self) -> Tensor:
        r"""
        Identical to `self`.

        Returns:
            (torch.Tensor):

        Examples:
            >>> tensor = torch.tensor([1, 2, 3])
            >>> pn_tensor = PNTensor([1, 2, 3])
            >>> bool((tensor == pn_tensor).all())
            True
            >>> bool((tensor == pn_tensor.contact).all())
            True
        """

        return self

    def new_empty(self, *args, **kwargs):
        return PNTensor(super().new_empty(*args, **kwargs))


class NestedTensor:
    r"""
    Wrap an iterable of tensors into a single tensor with a mask.

    In sequence to sequence tasks, elements of a batch are usually not of the same length.
    This made it tricky to use a single tensor to represent a batch of sequences.

    `NestedTensor` allows to store a sequence of tensors of different lengths in a single object.
    It also provides a mask that can be used to retrieve the original sequence of tensors.

    When calling `__getitem__(arg)` on a `NestedTensor`, it has two return type:
    1. if arg is `int` or `slice`, returns a tuple of two `tensor`s, representing data and padding mask.
    2. if arg is a `tuple`, return a new `NestedTensor` with specified shape.

    Attributes:
        _storage: The sequence of tensors.
        tensor: padded tensor.
        mask: mask tensor.
        concat: concatenated tensor.
        batch_first:  Whether the first dimension of the tensors is the batch dimension.

            If `True`, the first dimension is the batch dimension, i.e., `B, N, *`.

            If `False`, the first dimension is the sequence dimension, i.e., `N, B, *`
        padding_value: The padding value used to in padded tensor.
        mask_value: The mask value used in mask tensor.

    Args:
        tensors:
        batch_first:
        padding_value:
        mask_value:

    Raises:
        ValueError: If `tensors` is not an iterable.
        ValueError: If `tensors` is empty.

    Notes:
        We have rewritten the `__getattr__` function to support as much native tensor operations as possible.
        However, not all operations are tested.

        Please file an issue if you find any bugs.

    Examples:
        >>> nested_tensor = NestedTensor(torch.tensor([1, 2, 3]), torch.tensor([4, 5]))
        >>> nested_tensor.shape
        torch.Size([2, 3])
        >>> nested_tensor.device
        device(type='cpu')
        >>> nested_tensor.dtype
        torch.int64
        >>> nested_tensor.tensor
        tensor([[1, 2, 3],
                [4, 5, 0]])
        >>> nested_tensor.mask
        tensor([[ True,  True,  True],
                [ True,  True, False]])
        >>> nested_tensor.concat
        tensor([1, 2, 3, 4, 5])
        >>> nested_tensor.to(torch.float).tensor
        tensor([[1., 2., 3.],
                [4., 5., 0.]])
        >>> nested_tensor.half().tensor
        tensor([[1., 2., 3.],
                [4., 5., 0.]], dtype=torch.float16)
        >>> nested_tensor[:]
        (tensor([[1, 2, 3],
                [4, 5, 0]]), tensor([[ True,  True,  True],
                [ True,  True, False]]))
        >>> nested_tensor[1]
        tensor([4, 5])
        >>> nested_tensor[:, 1:]
        NestedTensor([[2, 3],
                [5, 0]])
        >>> nested_tensor.tolist()
        [[1, 2, 3], [4, 5]]
        >>> NestedTensor(*[[1, 2, 3], [4, 5]])
        NestedTensor([[1, 2, 3],
                [4, 5, 0]])
    """

    __storage: Sequence[Tensor]
    batch_first: bool = True
    padding_value: SupportsFloat = 0.0
    mask_value: bool = False

    def __init__(
        self,
        *tensors: Iterable[Tensor],
        batch_first: bool = True,
        padding_value: SupportsFloat = 0.0,
        mask_value: bool = False,
    ) -> None:
        if len(tensors) == 1 and isinstance(tensors, Sequence):
            tensors = tensors[0]  # type: ignore
        self._storage = tensors
        self.batch_first = batch_first
        self.padding_value = padding_value
        self.mask_value = mask_value

    @property
    def _storage(self):
        return self.__storage

    @_storage.setter
    def _storage(self, tensors: Sequence):
        if not isinstance(tensors, Iterable):
            raise ValueError(f"tensors must be an Iterable, bug got {type(tensors)}.")
        tensors = list(tensors)
        if len(tensors) == 0:
            raise ValueError("tensors must be a non-empty Iterable.")
        if not isinstance(tensors[0], Tensor):
            tensors = [torch.tensor(t) for t in tensors]
        # if drop_last=False, the last element is likely not a NestedTensor and has an extra batch dimension
        ndims = {t.ndim for t in tensors[:-1]}
        if len(ndims) == 1:
            (ndim,) = ndims
            if tensors[-1].ndim == ndim + 1 and tensors[-1].size(0) == 1:
                tensors[-1] = tensors[-1].squeeze(0)
        self.__storage = tensors

    def storage(self):
        return self._storage

    @property
    def tensor_mask(self) -> Tuple[Tensor, Tensor]:
        r"""
        Return a tuple of padded tensor and mask tensor.

        Returns:
            (torch.Tensor):

        Examples:
            >>> nested_tensor = NestedTensor([torch.tensor([1, 2, 3]), torch.tensor([4, 5])])
            >>> nested_tensor.tensor_mask
            (tensor([[1, 2, 3],
                    [4, 5, 0]]), tensor([[ True,  True,  True],
                    [ True,  True, False]]))
        """

        return self._tensor_mask(tuple(self._storage), self.batch_first, self.padding_value, self.mask_value)

    @property
    def tensor(self) -> Tensor:
        r"""
        Return a single tensor by padding all the tensors.

        Returns:
            (torch.Tensor):

        Examples:
            >>> nested_tensor = NestedTensor([torch.tensor([1, 2, 3]), torch.tensor([4, 5])])
            >>> nested_tensor.tensor
            tensor([[1, 2, 3],
                    [4, 5, 0]])
        """

        return self._tensor(tuple(self._storage), self.batch_first, self.padding_value)

    @property
    def mask(self) -> Tensor:
        r"""
        Padding mask of `tensor`.

        Returns:
            (torch.Tensor):

        Examples:
            >>> nested_tensor = NestedTensor([torch.tensor([1, 2, 3]), torch.tensor([4, 5])])
            >>> nested_tensor.mask
            tensor([[ True,  True,  True],
                    [ True,  True, False]])
        """

        return self._mask(tuple(self._storage), self.batch_first, self.mask_value)

    @property
    def concat(self) -> Tensor:
        r"""
        Concat `tensor` in padding dim.

        This is particularly useful when calculating loss or passing `Linear` to avoid unnecessary computation.

        Returns:
            (torch.Tensor):

        Examples:
            >>> nested_tensor = NestedTensor([torch.randn(9, 8), torch.randn(11, 8)])
            >>> nested_tensor.concat.shape
            torch.Size([20, 8])
            >>> nested_tensor = NestedTensor([torch.randn(9, 9, 8), torch.randn(11, 11, 8)])
            >>> nested_tensor.concat.shape
            torch.Size([202, 8])
            >>> nested_tensor = NestedTensor([torch.randn(9, 9, 8, 6), torch.randn(11, 11, 8, 6)])
            >>> nested_tensor.concat.shape
            torch.Size([202, 8, 6])
            >>> nested_tensor = NestedTensor([torch.randn(9, 9, 8, 7), torch.randn(11, 11, 8, 6)])
            >>> nested_tensor.concat.shape
            torch.Size([1293, 8])
            >>> nested_tensor = NestedTensor([torch.randn(1, 9, 9, 5), torch.randn(1, 11, 11, 5)])
        """
        shape = list(self.size())  # type: ignore[arg-type]
        shape = shape[1:] if self.batch_first else shape[0] + shape[2:]
        elem = self._storage[0]
        if elem.shape == shape:
            return torch.cat(self._storage, dim=1 if self.batch_first else 0)
        static_dims = set(range(len(shape)))
        for i, s in enumerate(shape):
            if not all(t.size(i) == s for t in self._storage):
                shape[i] = -1
                static_dims.remove(i)
        target_shape = [-1] + [s for s in shape if s != -1]
        storage = [i.reshape(target_shape) for i in self._storage]
        return torch.cat(storage, dim=0 if self.batch_first else 1)

    @classmethod
    def from_tensor_mask(cls, tensor: Tensor, mask: Tensor):
        r"""
        Build a `NestedTensor` object from a padded `Tensor` and corresponding mask `Tensor`.

        Args:
            tensor: Padded Tensor.
            mask: Tensor Mask.

        Returns:
            (torch.Tensor):

        Examples:
            >>> padded_tensor = torch.tensor([[1, 2, 3, 0, 0],
            ...                                [4, 5, 0, 0, 0],
            ...                                [6, 7, 8, 9, 0]])
            >>> mask_tensor = torch.tensor([[1, 1, 1, 0, 0],
            ...                             [1, 1, 0, 0, 0],
            ...                             [1, 1, 1, 1, 0]])
            >>> nested_tensor = NestedTensor.from_tensor_mask(padded_tensor, mask_tensor)
            >>> nested_tensor
            NestedTensor([[1, 2, 3, 0],
                    [4, 5, 0, 0],
                    [6, 7, 8, 9]])
        """

        if mask.ndim == 2:
            return cls(t[slice(0, m.sum())] for t, m in zip(tensor, mask))
        return cls(
            t[[slice(0, (m.sum(dim=dim) > 0).sum().item()) for dim in reversed(range(m.dim()))]]
            for t, m in zip(tensor, mask)
        )

    def nested_like(self, tensor: Tensor, strict: bool = True) -> NestedTensor:
        r"""
        Create a new `NestedTensor` from a `Tensor`.
        The newly created `NestedTensor` will have the same shape as current `NestedTensor`.

        Args:
            tensor: The tensor to be converted to `NestedTensor`.
            strict: Check if the shape of `tensor` is the same as the current `NestedTensor`.

        Returns:
            (NestedTensor):

        Examples:
            >>> nested_tensor = NestedTensor([torch.tensor([1, 2, 3]), torch.tensor([4, 5])])
            >>> (nested_tensor == nested_tensor.nested_like(nested_tensor)).all()
            tensor(True)
            >>> tensor = nested_tensor.tensor
            >>> (nested_tensor == nested_tensor.nested_like(tensor)).all()
            tensor(True)
            >>> f = nested_tensor.nested_like(torch.randn(2, 2))
            Traceback (most recent call last):
            ValueError: The shape of NestedTensor and input tensor does not match, torch.Size([2, 3]) != torch.Size([2, 2])
            >>> p = nested_tensor.nested_like(torch.randn(2, 2), False)
            >>> p = nested_tensor.nested_like(torch.randn(3, 3), False)
            Traceback (most recent call last):
            ValueError: The batch size of NestedTensor and input tensor does not match, 2 != 3
        """  # noqa: E501

        if isinstance(tensor, NestedTensor):
            return tensor.clone()

        if strict and self.shape != tensor.shape:
            raise ValueError(
                f"The shape of NestedTensor and input tensor does not match, {self.shape} != {tensor.shape}"
            )
        if self.size(0) != tensor.size(0):
            raise ValueError(
                f"The batch size of NestedTensor and input tensor does not match, {self.size(0)} != {tensor.size(0)}"
            )
        return NestedTensor([o[tuple(slice(0, dim) for dim in t.shape)] for t, o in zip(self._storage, tensor)])

    @property
    def device(self) -> torch.device:
        r"""
        Device of the NestedTensor.

        Returns:
            (torch.Tensor):

        Examples:
            >>> nested_tensor = NestedTensor([torch.tensor([1, 2, 3]), torch.tensor([4, 5])])
            >>> nested_tensor.device
            device(type='cpu')
        """

        return self._device(tuple(self._storage))

    @property
    def shape(self) -> torch.Size | int:
        r"""
        Alias for `size()`.
        """

        return self.size()

    @property
    def ndim(self) -> int:
        r"""
        Alias for `dim()`.
        """

        return self.dim()

    def size(self, dim: int | None = None) -> torch.Size | int:
        r"""
        Returns the size of the self `NestedTensor`.

        Args:
            dim: If not specified, the returned value is a `torch.Size`, a subclass of `tuple`.
                If specified, returns an `int` holding the size of that dimension.
                Defaults to `None`.

        Returns:
            (torch.Size | int):

        Examples:
            >>> nested_tensor = NestedTensor([torch.tensor([1, 2, 3]), torch.tensor([4, 5])])
            >>> nested_tensor.size()
            torch.Size([2, 3])
            >>> nested_tensor.size(0)
            2
            >>> nested_tensor.storage()[1] = torch.tensor([4, 5, 6, 7])
            >>> nested_tensor.shape
            torch.Size([2, 4])
            >>> nested_tensor.size(1)
            4
        """

        return self._size(tuple(self._storage), dim, self.batch_first)

    def dim(self) -> int:
        r"""
        Number of dimension of the NestedTensor.

        Returns:
            (int):

        Examples:
            >>> nested_tensor = NestedTensor([torch.tensor([1, 2, 3]), torch.tensor([4, 5])])
            >>> nested_tensor.dim()
            2
            >>> nested_tensor.storage().append(torch.tensor([6, 7, 8, 9]))
            >>> nested_tensor.ndim
            2
        """

        return self._dim(tuple(self._storage))

    def tolist(self) -> list:
        r"""
        Convert a NestedTensor to a list of lists of values.

        Returns:
            (list):

        Examples:
            >>> nested_tensor = NestedTensor([torch.tensor([1, 2, 3]), torch.tensor([4, 5])])
            >>> nested_tensor.tolist()
            [[1, 2, 3], [4, 5]]
        """

        return [t.tolist() for t in self._storage]

    def all(self, dim: int | None = None, keepdim: bool = False) -> bool | Tensor | NestedTensor:
        r"""
        Tests if all elements in NestedTensor evaluate to True.

        Returns:
            (bool | Tensor):

        Examples:
            >>> nested_tensor = NestedTensor([torch.ones(2, 4, dtype=torch.bool), torch.ones(3, 5, dtype=torch.bool)])
            >>> nested_tensor.all()
            tensor(True)
            >>> nested_tensor.all(dim=0)
            tensor([True, True])
            >>> nested_tensor.all(dim=0, keepdim=True)
            tensor([[True, True]])
            >>> nested_tensor.all(dim=1)
            NestedTensor([[ True,  True,  True,  True, False],
                    [ True,  True,  True,  True,  True]])
            >>> nested_tensor.all(dim=1, keepdim=True)
            NestedTensor([[[ True,  True,  True,  True, False]],
            <BLANKLINE>
                    [[ True,  True,  True,  True,  True]]])
            >>> nested_tensor.batch_first = False
            >>> nested_tensor.all(dim=1)
            tensor([True, True])
            >>> nested_tensor.batch_first = False
            >>> nested_tensor.all(dim=0)
            NestedTensor([[ True,  True,  True,  True, False],
                    [ True,  True,  True,  True,  True]])
            >>> nested_tensor.all(dim=1)
            tensor([True, True])
        """

        if dim is None:
            return torch.tensor(all(i.all() for i in self._storage))
        if (self.batch_first and dim == 0) or (not self.batch_first and dim == 1):
            if keepdim:
                return torch.tensor([i.all() for i in self._storage]).unsqueeze(0 if self.batch_first else 1)
            return torch.tensor([i.all() for i in self._storage])
        if self.batch_first or dim != 0:
            dim -= 1
        return NestedTensor([i.all(dim=dim, keepdim=keepdim) for i in self._storage])

    def where(self, condition: Tensor | NestedTensor, other: Tensor | NestedTensor | SupportsFloat) -> NestedTensor:
        r"""
        Return a NestedTensor of elements selected from either self or other, depending on condition.

        Returns:
            (NestedTensor):

        Examples:
            >>> nested_tensor = NestedTensor([torch.tensor([1, 2, 3]), torch.tensor([4, 5])])
            >>> nested_tensor.where(nested_tensor > 2, torch.tensor([[6, 5, 4], [3, 2, 1]]))
            NestedTensor([[6, 5, 3],
                    [4, 5, 0]])
            >>> nested_tensor.where(nested_tensor > 2, NestedTensor([[6, 5, 4], [3, 2]]))
            NestedTensor([[6, 5, 3],
                    [4, 5, 0]])
            >>> nested_tensor.where(torch.tensor(True), NestedTensor([[6, 5, 4], [3, 2]]))
            NestedTensor([[1, 2, 3],
                    [4, 5, 0]])
        """

        if isinstance(condition, Tensor) and self.shape == condition.shape:
            condition = self.nested_like(condition)
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if isinstance(condition, NestedTensor) and isinstance(other, NestedTensor):
            return NestedTensor(
                [x.where(c, y) for x, c, y in zip(self._storage, condition._storage, other._storage)], **self._state
            )
        if isinstance(condition, NestedTensor):
            return NestedTensor([x.where(c, other) for x, c in zip(self._storage, condition._storage)], **self._state)
        if isinstance(other, NestedTensor):
            return NestedTensor([x.where(condition, y) for x, y in zip(self._storage, other._storage)], **self._state)
        return NestedTensor(x.where(condition, other) for x in self._storage)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func not in NestedTensorFunc or not all(issubclass(t, (torch.Tensor, NestedTensor)) for t in types):
            args = [a.tensor if hasattr(a, "tensor") else a for a in args]
            return func(*args, **kwargs)
        return NestedTensorFunc[func](*args, **kwargs)

    def __getitem__(self, index: int | slice | list | tuple) -> Tensor | tuple[Tensor, Tensor] | NestedTensor:
        if isinstance(index, int):
            return self._storage[index]
        if isinstance(index, (slice, list)):
            storage = tuple(self._storage[index] if isinstance(index, slice) else [self._storage[i] for i in index])
            pad = pad_tensor(
                storage, size=self._size(storage), batch_first=self.batch_first, padding_value=float(self.padding_value)
            )
            mask = mask_tensor(
                storage, size=self._size(storage), batch_first=self.batch_first, mask_value=self.mask_value
            )
            return pad, mask
        if isinstance(index, tuple):
            return NestedTensor([t[index[0]][index[1:]] for t in self._storage])
        raise ValueError(f"Unsupported index type {type(index)}")

    def __getattr__(self, name: str) -> Any:
        if not self._storage:
            raise ValueError(f"Unable to get {name} from an empty {self.__class__.__name__}")
        ret = [getattr(i, name) for i in self._storage]
        elem = ret[0]
        if isinstance(elem, Tensor):
            return NestedTensor(ret, **self._state)
        if callable(elem):
            return NestedTensorFuncWrapper(ret, state=self._state)
        if elem.__hash__ is not None and len(set(ret)) == 1:
            return elem
        return ret

    @property
    def _state(self) -> Mapping:
        return {k: v for k, v in self.__dict__.items() if not (k.startswith("_") or k.endswith("_"))}

    def __state__(self) -> Mapping:
        return self.__dict__

    def __setstate__(self, state: Mapping) -> None:
        self.__dict__.update(state)

    def __len__(self) -> int:
        return len(self._storage)

    def __repr__(self):
        return self.__class__.__name__ + repr(self.tensor)[len(self.tensor.__class__.__name__) :]  # noqa: E203

    def __bool__(self) -> int:
        return all(bool(x) for x in self._storage)

    def __gt__(  # type: ignore[override]
        self, other: Tensor | NestedTensor | SupportsFloat
    ) -> bool | Tensor | NestedTensor:
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if isinstance(other, NestedTensor):
            return NestedTensor(i > j for i, j in zip(self._storage, other._storage))
        if isinstance(other, (int, float, Tensor)):
            return NestedTensor([x > other for x in self._storage], **self._state)
        raise TypeError(f"> not supported between instances of '{type(self)}' and '{type(other)}'")

    def __ge__(  # type: ignore[override]
        self, other: Tensor | NestedTensor | SupportsFloat
    ) -> bool | Tensor | NestedTensor:
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if isinstance(other, NestedTensor):
            return NestedTensor(i >= j for i, j in zip(self._storage, other._storage))
        if isinstance(other, (int, float, Tensor)):
            return NestedTensor([x >= other for x in self._storage], **self._state)
        raise TypeError(f">= not supported between instances of '{type(self)}' and '{type(other)}'")

    def __eq__(  # type: ignore[override]
        self, other: Tensor | NestedTensor | SupportsFloat
    ) -> bool | Tensor | NestedTensor:
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if isinstance(other, NestedTensor):
            return NestedTensor(i == j for i, j in zip(self._storage, other._storage))
        if isinstance(other, (int, float, Tensor)):
            return NestedTensor([x == other for x in self._storage], **self._state)
        return False

    def __le__(  # type: ignore[override]
        self, other: Tensor | NestedTensor | SupportsFloat
    ) -> bool | Tensor | NestedTensor:
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if isinstance(other, NestedTensor):
            return NestedTensor(i <= j for i, j in zip(self._storage, other._storage))
        if isinstance(other, (int, float, Tensor)):
            return NestedTensor([x <= other for x in self._storage], **self._state)
        raise TypeError(f"<= not supported between instances of '{type(self)}' and '{type(other)}'")

    def __lt__(  # type: ignore[override]
        self, other: Tensor | NestedTensor | SupportsFloat
    ) -> bool | Tensor | NestedTensor:
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if isinstance(other, NestedTensor):
            return NestedTensor(i < j for i, j in zip(self._storage, other._storage))
        if isinstance(other, (int, float, Tensor)):
            return NestedTensor([x < other for x in self._storage], **self._state)
        raise TypeError(f"< not supported between instances of '{type(self)}' and '{type(other)}'")

    def __abs__(self):
        return NestedTensor([abs(value) for value in self._storage], **self._state)

    def __add__(self, other: Tensor | NestedTensor | SupportsFloat):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if isinstance(other, NestedTensor):
            return NestedTensor([x + y for x, y in zip(self._storage, other._storage)], **self._state)
        return NestedTensor([value + other for value in self._storage], **self._state)

    def __radd__(self, other: Tensor | NestedTensor | SupportsFloat):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if isinstance(other, NestedTensor):
            return NestedTensor([y + x for x, y in zip(self._storage, other._storage)], **self._state)
        return NestedTensor([other + value for value in self._storage], **self._state)

    def __iadd__(self, other: Tensor | NestedTensor | SupportsFloat):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if hasattr(other, "to"):
            other = other.to(self.dtype)
        if isinstance(other, NestedTensor):
            for x, y in zip(self._storage, other._storage):
                x += y
        else:
            for value in self._storage:
                value += other
        return self

    def __sub__(self, other: Tensor | NestedTensor | SupportsFloat):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if isinstance(other, NestedTensor):
            return NestedTensor([x - y for x, y in zip(self._storage, other._storage)], **self._state)
        return NestedTensor([value - other for value in self._storage], **self._state)

    def __rsub__(self, other: Tensor | NestedTensor | SupportsFloat):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if isinstance(other, NestedTensor):
            return NestedTensor([y - x for x, y in zip(self._storage, other._storage)], **self._state)
        return NestedTensor([other - value for value in self._storage], **self._state)

    def __isub__(self, other: Tensor | NestedTensor | SupportsFloat):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if hasattr(other, "to"):
            other = other.to(self.dtype)
        if isinstance(other, NestedTensor):
            for x, y in zip(self._storage, other._storage):
                x -= y
        else:
            for value in self._storage:
                value -= other
        return self

    def __pos__(self):
        return NestedTensor([+x for x in self._storage])

    def __neg__(self):
        return NestedTensor([-x for x in self._storage])

    def __invert__(self):
        return NestedTensor([~x for x in self._storage])

    def __mul__(self, other: Tensor | NestedTensor | SupportsFloat):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if isinstance(other, NestedTensor):
            return NestedTensor([x * y for x, y in zip(self._storage, other._storage)], **self._state)
        return NestedTensor([value * other for value in self._storage], **self._state)

    def __rmul__(self, other: Tensor | NestedTensor | SupportsFloat):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if isinstance(other, NestedTensor):
            return NestedTensor([y * x for x, y in zip(self._storage, other._storage)], **self._state)
        return NestedTensor([other * value for value in self._storage], **self._state)

    def __imul__(self, other: Tensor | NestedTensor | SupportsFloat):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if hasattr(other, "to"):
            other = other.to(self.dtype)
        if isinstance(other, NestedTensor):
            for x, y in zip(self._storage, other._storage):
                x *= y
        else:
            for value in self._storage:
                value *= other
        return self

    def __pow__(self, other: Tensor | NestedTensor | SupportsFloat):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if isinstance(other, NestedTensor):
            return NestedTensor([x**y for x, y in zip(self._storage, other._storage)], **self._state)
        return NestedTensor([value**other for value in self._storage], **self._state)

    def __rpow__(self, other: Tensor | NestedTensor | SupportsFloat):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if isinstance(other, NestedTensor):
            return NestedTensor([y**x for x, y in zip(self._storage, other._storage)], **self._state)
        return NestedTensor([other**value for value in self._storage], **self._state)

    def __ipow__(self, other: Tensor | NestedTensor | SupportsFloat):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if hasattr(other, "to"):
            other = other.to(self.dtype)
        if isinstance(other, NestedTensor):
            for x, y in zip(self._storage, other._storage):
                x **= y
        else:
            for value in self._storage:
                value **= other
        return self

    def __matmul__(self, other: Tensor | NestedTensor | SupportsFloat):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if isinstance(other, NestedTensor):
            return NestedTensor([x @ y for x, y in zip(self._storage, other._storage)], **self._state)
        return NestedTensor([value @ other for value in self._storage], **self._state)

    def __rmatmul__(self, other: Tensor | NestedTensor | SupportsFloat):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if isinstance(other, NestedTensor):
            return NestedTensor([y @ x for x, y in zip(self._storage, other._storage)], **self._state)
        return NestedTensor([other @ value for value in self._storage], **self._state)

    def __imatmul__(self, other: Tensor | NestedTensor | SupportsFloat):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if hasattr(other, "to"):
            other = other.to(self.dtype)
        if isinstance(other, NestedTensor):
            for x, y in zip(self._storage, other._storage):
                x @= y
        else:
            for value in self._storage:
                value @= other
        return self

    def __truediv__(self, other: Tensor | NestedTensor | SupportsFloat):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if isinstance(other, NestedTensor):
            return NestedTensor([x / y for x, y in zip(self._storage, other._storage)], **self._state)
        return NestedTensor([value / other for value in self._storage], **self._state)

    def __rtruediv__(self, other: Tensor | NestedTensor | SupportsFloat):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if isinstance(other, NestedTensor):
            return NestedTensor([y / x for x, y in zip(self._storage, other._storage)], **self._state)
        return NestedTensor([other / value for value in self._storage], **self._state)

    def __itruediv__(self, other: Tensor | NestedTensor | SupportsFloat):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if hasattr(other, "to"):
            other = other.to(self.dtype)
        if isinstance(other, NestedTensor):
            for x, y in zip(self._storage, other._storage):
                x /= y
        else:
            for value in self._storage:
                value /= other
        return self

    def __floordiv__(self, other: Tensor | NestedTensor | SupportsFloat):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if isinstance(other, NestedTensor):
            return NestedTensor([x // y for x, y in zip(self._storage, other._storage)], **self._state)
        return NestedTensor([value // other for value in self._storage], **self._state)

    def __rfloordiv__(self, other: Tensor | NestedTensor | SupportsFloat):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if isinstance(other, NestedTensor):
            return NestedTensor([y // x for x, y in zip(self._storage, other._storage)], **self._state)
        return NestedTensor([other // value for value in self._storage], **self._state)

    def __ifloordiv__(self, other: Tensor | NestedTensor | SupportsFloat):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if hasattr(other, "to"):
            other = other.to(self.dtype)
        if isinstance(other, NestedTensor):
            for x, y in zip(self._storage, other._storage):
                x //= y
        else:
            for value in self._storage:
                value //= other
        return self

    def __mod__(self, other: Tensor | NestedTensor | SupportsFloat):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if isinstance(other, NestedTensor):
            return NestedTensor([x % y for x, y in zip(self._storage, other._storage)], **self._state)
        return NestedTensor([value % other for value in self._storage], **self._state)

    def __rmod__(self, other: Tensor | NestedTensor | SupportsFloat):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if isinstance(other, NestedTensor):
            return NestedTensor([y % x for x, y in zip(self._storage, other._storage)], **self._state)
        return NestedTensor([other % value for value in self._storage], **self._state)

    def __imod__(self, other: Tensor | NestedTensor | SupportsFloat):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if hasattr(other, "to"):
            other = other.to(self.dtype)
        if isinstance(other, NestedTensor):
            for x, y in zip(self._storage, other._storage):
                x %= y
        else:
            for value in self._storage:
                value %= other
        return self

    def __and__(self, other: Tensor | NestedTensor | SupportsFloat):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if isinstance(other, NestedTensor):
            return NestedTensor([x & y for x, y in zip(self._storage, other._storage)], **self._state)
        return NestedTensor([value & other for value in self._storage], **self._state)

    def __rand__(self, other: Tensor | NestedTensor | SupportsFloat):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if isinstance(other, NestedTensor):
            return NestedTensor([y & x for x, y in zip(self._storage, other._storage)], **self._state)
        return NestedTensor([other & value for value in self._storage], **self._state)

    def __iand__(self, other: Tensor | NestedTensor | SupportsFloat):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if hasattr(other, "to"):
            other = other.to(self.dtype)
        if isinstance(other, NestedTensor):
            for x, y in zip(self._storage, other._storage):
                x &= y
        else:
            for value in self._storage:
                value &= other
        return self

    def __or__(self, other: Tensor | NestedTensor | SupportsFloat):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if isinstance(other, NestedTensor):
            return NestedTensor([x | y for x, y in zip(self._storage, other._storage)], **self._state)
        return NestedTensor([value | other for value in self._storage], **self._state)

    def __ror__(self, other: Tensor | NestedTensor | SupportsFloat):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if isinstance(other, NestedTensor):
            return NestedTensor([y | x for x, y in zip(self._storage, other._storage)], **self._state)
        return NestedTensor([other | value for value in self._storage], **self._state)

    def __ior__(self, other: Tensor | NestedTensor | SupportsFloat):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if hasattr(other, "to"):
            other = other.to(self.dtype)
        if isinstance(other, NestedTensor):
            for x, y in zip(self._storage, other._storage):
                x |= y
        else:
            for value in self._storage:
                value |= other
        return self

    def __xor__(self, other: Tensor | NestedTensor | SupportsFloat):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if isinstance(other, NestedTensor):
            return NestedTensor([x ^ y for x, y in zip(self._storage, other._storage)], **self._state)
        return NestedTensor([value ^ other for value in self._storage], **self._state)

    def __rxor__(self, other: Tensor | NestedTensor | SupportsFloat):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if isinstance(other, NestedTensor):
            return NestedTensor([y ^ x for x, y in zip(self._storage, other._storage)], **self._state)
        return NestedTensor([other ^ value for value in self._storage], **self._state)

    def __ixor__(self, other: Tensor | NestedTensor | SupportsFloat):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if hasattr(other, "to"):
            other = other.to(self.dtype)
        if isinstance(other, NestedTensor):
            for x, y in zip(self._storage, other._storage):
                x ^= y
        else:
            for value in self._storage:
                value ^= other
        return self

    def __lshift__(self, other: Tensor | NestedTensor | SupportsFloat):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if isinstance(other, NestedTensor):
            return NestedTensor([x << y for x, y in zip(self._storage, other._storage)], **self._state)
        return NestedTensor([value << other for value in self._storage], **self._state)

    def __rlshift__(self, other: Tensor | NestedTensor | SupportsFloat):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if isinstance(other, NestedTensor):
            return NestedTensor([y << x for x, y in zip(self._storage, other._storage)], **self._state)
        return NestedTensor([other << value for value in self._storage], **self._state)

    def __ilshift__(self, other: Tensor | NestedTensor | SupportsFloat):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if hasattr(other, "to"):
            other = other.to(self.dtype)
        if isinstance(other, NestedTensor):
            for x, y in zip(self._storage, other._storage):
                x <<= y
        else:
            for value in self._storage:
                value <<= other
        return self

    def __rshift__(self, other: Tensor | NestedTensor | SupportsFloat):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if isinstance(other, NestedTensor):
            return NestedTensor([x >> y for x, y in zip(self._storage, other._storage)], **self._state)
        return NestedTensor([value >> other for value in self._storage], **self._state)

    def __rrshift__(self, other: Tensor | NestedTensor | SupportsFloat):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if isinstance(other, NestedTensor):
            return NestedTensor([y >> x for x, y in zip(self._storage, other._storage)], **self._state)
        return NestedTensor([other >> value for value in self._storage], **self._state)

    def __irshift__(self, other: Tensor | NestedTensor | SupportsFloat):
        if isinstance(other, Tensor) and self.shape == other.shape:
            other = self.nested_like(other)
        if hasattr(other, "to"):
            other = other.to(self.dtype)
        if isinstance(other, NestedTensor):
            for x, y in zip(self._storage, other._storage):
                x >>= y
        else:
            for value in self._storage:
                value >>= other
        return self

    @method_cache(maxsize=1)
    def _tensor_mask(self, storage: tuple, batch_first: bool, padding_value: SupportsFloat, mask_value: bool) -> Tensor:
        if storage[0].dim() == 0:
            return torch.stack(storage, dim=0), torch.full(
                (len(storage),), not mask_value, dtype=torch.bool, device=self.device
            )
        return tensor_mask(
            storage,
            size=self.size(),
            batch_first=batch_first,
            padding_value=float(padding_value),
            mask_value=mask_value,
        )

    @method_cache(maxsize=1)
    def _tensor(self, storage: tuple, batch_first: bool, padding_value: SupportsFloat) -> Tensor:
        if storage[0].dim() == 0:
            return torch.stack(storage, dim=0)
        return pad_tensor(storage, size=self.size(), batch_first=batch_first, padding_value=float(padding_value))

    @method_cache(maxsize=1)
    def _mask(self, storage: tuple, batch_first: bool, mask_value: bool) -> Tensor:
        if storage[0].dim() == 0:
            return torch.full((len(storage),), not mask_value, dtype=torch.bool, device=self.device)
        return mask_tensor(storage, size=self.size(), batch_first=batch_first, mask_value=mask_value)

    @method_cache(maxsize=1)
    def _device(self, storage) -> torch.device:
        return storage[0].device

    @method_cache(maxsize=1)
    def _size(self, storage, dim: int | None = None, batch_first: bool = True) -> torch.Size | int:
        if dim is not None:
            if dim == 0:
                return len(storage)
            return max(t.size(dim - 1) for t in storage)
        if max(t.dim() for t in storage) == 0:
            return torch.Size((len(storage),))
        ndim = max(t.dim() for t in storage)
        size = [max(t.shape[i] if i < len(t.shape) else 0 for t in storage) for i in range(ndim)]
        size.insert(0 if batch_first else 1, len(storage))
        return torch.Size(size)

    @method_cache(maxsize=1)
    def _dim(self, storage) -> torch.Size:
        return max(t.dim() for t in storage) + 1

    def view(self, *shape) -> Tensor:
        r"""
        Returns a torch tensor with a different shape.

        Note:
            since NestedTensor is a collection of tensors, the view operation is ambiguous.

            Therefore, it is converted to a tensor and then reshaped.

        Args:
            shape: The desired size of each dimension.

        Returns:
            (Tensor):

        Examples:
            >>> nested_tensor = NestedTensor([torch.tensor([1, 2, 3]), torch.tensor([4, 5])])
            >>> nested_tensor.view(3, 2)
            tensor([[1, 2],
                    [3, 4],
                    [5, 0]])
            >>> nested_tensor.view(2, 3)
            tensor([[1, 2, 3],
                    [4, 5, 0]])
        """

        return self.tensor.view(*shape)

    def reshape(self, *shape) -> Tensor:
        r"""
        Returns a torch tensor with a different shape.

        Note:
            since NestedTensor is a collection of tensors, the reshape operation is ambiguous.

            Therefore, it is converted to a tensor and then reshaped.

        Args:
            shape: The desired size of each dimension.

        Returns:
            (Tensor):

        Examples:
            >>> nested_tensor = NestedTensor([torch.tensor([1, 2, 3]), torch.tensor([4, 5])])
            >>> nested_tensor.reshape(3, 2)
            tensor([[1, 2],
                    [3, 4],
                    [5, 0]])
            >>> nested_tensor.reshape(2, 3)
            tensor([[1, 2, 3],
                    [4, 5, 0]])
        """

        return self.tensor.reshape(*shape)

    def __iter__(self):
        return iter(self._storage)


NestedTensorFunc = TorchFuncRegistry()


@NestedTensorFunc.implement(torch.cat)
def cat(tensors, dim: int = 0):
    if dim != 0:
        raise NotImplementedError(f"NestedTensor only supports cat when dim=0, but got {dim}")
    return NestedTensor([t for tensor in tensors for t in tensor._storage], tensors[0]._state)


@NestedTensorFunc.implement(torch.isin)
def isin(elements, test_elements, *, assume_unique: bool = False, invert: bool = False):
    if isinstance(elements, NestedTensor):
        elements = elements.tensor
    if isinstance(test_elements, NestedTensor):
        test_elements = test_elements.tensor
    return torch.isin(elements, test_elements, assume_unique=assume_unique, invert=invert)


@NestedTensorFunc.implement(torch.log)
def log(tensor):
    return NestedTensor([torch.log(t) for t in tensor._storage])


@NestedTensorFunc.implement(torch.mean)
def mean(
    input,
    dim: int | None = None,
    keepdim: bool = False,
    *,
    dtype: torch.dtype | None = None,
):
    return input.mean(dim=dim, keepdim=keepdim, dtype=dtype)


@NestedTensorFunc.implement(torch.sqrt)
def sqrt(tensor):
    return NestedTensor([torch.sqrt(t) for t in tensor._storage])


@NestedTensorFunc.implement(torch.stack)
def stack(*args, **kwargs):
    raise NotImplementedError("NestedTensor does not support stack as of now")


class NestedTensorFuncWrapper:  # pylint: disable=R0903
    r"""
    Function Wrapper to handle NestedTensor as input.
    """

    __storage: Sequence[Callable] = []
    state: Mapping = {}

    def __init__(self, *callables: Iterable[Callable], state: Mapping | None = None) -> None:
        if len(callables) == 1 and isinstance(callables, Sequence):
            callables = callables[0]  # type: ignore
        self._storage = callables  # type: ignore
        if state is None:
            state = {}
        self.state = state

    @property
    def _storage(self):
        return self.__storage

    @_storage.setter
    def _storage(self, callables: Sequence):
        if not isinstance(callables, Sequence):
            raise ValueError(f"callables must be a Sequence, bug got {type(callables)}")
        if len(callables) == 0:
            raise ValueError("callables must be a non-empty Sequence.")
        if not callable(callables[0]):
            raise ValueError(f"callables must be a Sequence of Callable, bug got {type(callables[0])}")
        self.__storage = callables

    def __call__(self, *args, **kwargs) -> NestedTensor | Sequence[Tensor]:
        ret = [call(*args, **kwargs) for call in self._storage]
        elem = ret[0]
        if isinstance(elem, Tensor):
            return NestedTensor(ret, **self.state)
        if elem.__hash__ is not None and len(set(ret)) == 1:
            return elem
        return ret


def collate_pn_tensor_fn(batch, *, collate_fn_map: dict[type | tuple[type, ...], Callable] | None = None):
    return NestedTensor(batch)


default_collate_fn_map[PNTensor] = collate_pn_tensor_fn
