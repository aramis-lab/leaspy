"""This module contains utility functions related to the weighted_tensor module."""

from typing import Tuple, TypeVar, Union

import torch

from ._weighted_tensor import TensorOrWeightedTensor, WeightedTensor

S = TypeVar("S")


def expand_left(
    t: TensorOrWeightedTensor[S], *, shape: Tuple[int, ...]
) -> TensorOrWeightedTensor[S]:
    """Expand shape of tensor at left with provided shape."""
    return t.expand(shape + t.shape)


def expand_right(
    t: TensorOrWeightedTensor[S], *, shape: Tuple[int, ...]
) -> TensorOrWeightedTensor[S]:
    """Expand shape of tensor at right with provided shape."""
    return t.expand(t.shape + shape)


def unsqueeze_right(
    t: TensorOrWeightedTensor[S], *, ndim: int
) -> TensorOrWeightedTensor[S]:
    """
    Adds `ndim` dimensions to tensor, from right-side, without
    copy (useful for right broadcasting which is non-standard).
    """
    # Nota: `unsqueeze_left` is useless since it is automatically done with standard broadcasting
    assert isinstance(ndim, int) and ndim >= 0, f"Can not unsqueeze {ndim} dimensions"
    if ndim == 0:
        return t
    return t.view(t.shape + (1,) * ndim)


def sum_dim(
    x: TensorOrWeightedTensor,
    *,
    fill_value=0,
    dim: Union[None, int, Tuple[int, ...]] = None,
    but_dim: Union[None, int, Tuple[int, ...]] = None,
    **kws,
) -> torch.Tensor:
    """
    Sum dimension(s) of provided tensor (regular or weighted -
    filling with `fill_value` aggregates without any summed weighting if any).
    """
    dim = _get_dim(x, dim=dim, but_dim=but_dim)
    if isinstance(x, WeightedTensor):
        return x.sum(fill_value=fill_value, dim=dim, **kws)
    return x.sum(dim=dim, **kws)


def wsum_dim(
    x: WeightedTensor,
    *,
    fill_value=0,
    dim: Union[None, int, Tuple[int, ...]] = None,
    but_dim: Union[None, int, Tuple[int, ...]] = None,
    **kws,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sum dimension(s) of provided weighted tensor.

    The weighted tensor is filled with `fill_value` if provided.
    The function returns the sum of weights as well.

    Parameters
    ----------
    x : WeightedTensor
        The weighted tensor on which to compute the sum.
    fill_value : float
        The value to use for filling the weighted tensor.
    dim : int or tuple of int, optional
        The dimension(s) on which to sum.
    but_dim : int or tuple of int, optional
        The dimension(s) to omit when summing.

    Returns
    -------
    weighted_sum : torch.Tensor
        Weighted sum, with totally un-weighted aggregates filled with `fill_value`.
    sum_weights : torch.Tensor
        The sum of weights.
    """
    dim = _get_dim(x, dim=dim, but_dim=but_dim)
    return x.wsum(fill_value=fill_value, dim=dim, **kws)


def wsum_dim_return_weighted_sum_only(
    x: WeightedTensor,
    *,
    fill_value=0,
    dim: Union[None, int, Tuple[int, ...]] = None,
    but_dim: Union[None, int, Tuple[int, ...]] = None,
    **kws,
) -> torch.Tensor:
    """
    Sum dimension(s) of provided weighted tensor like `wsum_dim` but
    only return the weighted sum and not the sum of weights.

    Parameters
    ----------
    x : WeightedTensor
        The weighted tensor on which to compute the sum.
    fill_value : float
        The value to use for filling the weighted tensor.
    dim : int or tuple of int, optional
        The dimension(s) on which to sum.
    but_dim : int or tuple of int, optional
        The dimension(s) to omit when summing.

    Returns
    -------
    torch.Tensor :
        Weighted sum, with totally un-weighted aggregates filled with `fill_value`.
    """
    return wsum_dim(x, fill_value=fill_value, dim=dim, but_dim=but_dim, **kws)[0]


def wsum_dim_return_sum_of_weights_only(
    x: WeightedTensor,
    *,
    fill_value=0,
    dim: Union[None, int, Tuple[int, ...]] = None,
    but_dim: Union[None, int, Tuple[int, ...]] = None,
    **kws,
) -> torch.Tensor:
    """
    Sum dimension(s) of provided weighted tensor like `wsum_dim` but
    only return the sum of weights and not the weighted sum.

    Parameters
    ----------
    x : WeightedTensor
        The weighted tensor on which to compute the sum.
    fill_value : float
        The value to use for filling the weighted tensor.
    dim : int or tuple of int, optional
        The dimension(s) on which to sum.
    but_dim : int or tuple of int, optional
        The dimension(s) to omit when summing.

    Returns
    -------
    torch.Tensor :
        The sum of weights.
    """
    return wsum_dim(x, fill_value=fill_value, dim=dim, but_dim=but_dim, **kws)[1]


def _get_dim(
    x: TensorOrWeightedTensor,
    *,
    dim: Union[None, int, Tuple[int, ...]] = None,
    but_dim: Union[None, int, Tuple[int, ...]] = None,
) -> Union[int, Tuple[int, ...]]:
    if (dim is not None) and (but_dim is not None):
        raise ValueError("`dim` and `but_dim` should not be both defined.")
    if but_dim is not None:
        ndim = x.ndim
        if isinstance(but_dim, int):
            but_dim = {but_dim}
        but_dim = {i if i >= 0 else ndim + i for i in but_dim}
        assert all(i >= 0 for i in but_dim), but_dim
        dim = tuple(i for i in range(ndim) if i not in but_dim)
    elif dim is None:
        # full summation by default
        dim = ()
    return dim
