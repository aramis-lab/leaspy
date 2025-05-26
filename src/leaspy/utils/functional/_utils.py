"""This module contains utility functions related to the functional module."""

import operator
from functools import reduce
from typing import Callable, Iterable, Optional, Set, Tuple, TypeVar

import torch

from leaspy.utils.typing import KwargsType

from ..weighted_tensor import TensorOrWeightedTensor, WeightedTensor
from ._named_input_function import NamedInputFunction

__all__ = [
    "get_named_parameters",
]

S = TypeVar("S")


def _prod(iterable: Iterable[S], start: int = 1) -> S:
    """Product of all elements of the provided iterable, starting from `start`."""
    return reduce(operator.mul, iterable, start)


def _prod_args(
    *args: TensorOrWeightedTensor[S], **start_kw
) -> TensorOrWeightedTensor[S]:
    """Product of tensors with variadic input instead of standard iterable input."""
    return _prod(args, **start_kw)


def _identity(x: S) -> S:
    """Unary identity function."""
    return x


def _affine_from_vector(
    coeffs: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """
    Affine transformation from (slope, intercept) and input x:
    y = slope * x + intercept

    Supports broadcasting.
    """
    slope, intercept = coeffs.unbind(-1)
    return slope * x + intercept


def get_named_parameters(f: Callable) -> Tuple[str, ...]:
    """
    Get the names of parameters of the input function `f`, which should be
    a `NamedInputFunction` or a function with keyword-only parameters.
    """
    from inspect import signature

    if isinstance(f, NamedInputFunction):
        return f.parameters
    params = signature(f).parameters
    non_kw_only_params = [
        p_name for p_name, p in params.items() if p.kind is not p.KEYWORD_ONLY
    ]
    # nota: we do not check annotation of returned type for now (to remain lighter)
    if len(non_kw_only_params):
        raise ValueError(non_kw_only_params)
    return tuple(params)


def _arguments_checker(
    *,
    nb_arguments: Optional[int] = None,
    mandatory_kws: Optional[Set[str]] = None,
    possible_kws: Optional[Set[str]] = None,
) -> Callable:
    """
    Factory to check basic properties of parameters names and keyword-only arguments.

    Parameters
    ----------
    nb_arguments : int, optional
        Fixed number of positional arguments required by the function.
    mandatory_kws : set[str], optional
        Mandatory keyword-arguments for the function.
    possible_kws : set[str], optional
        Set of ALL possible keyword-arguments for the function.

    Returns
    -------
    function (args: tuple[str, ...], kws: dict[str, Any]) -> None
    """
    if nb_arguments is not None and (
        not isinstance(nb_arguments, int) or nb_arguments < 0
    ):
        raise ValueError(
            "Number of arguments should be a positive or null integer or None. "
            f"You provided a {type(nb_arguments)}."
        )
    nb_arguments_error_msg = None
    if nb_arguments == 1:
        nb_arguments_error_msg = f"Single name expected for positional parameters"
    elif nb_arguments is not None:
        nb_arguments_error_msg = (
            f"{nb_arguments} names expected for positional parameters"
        )
    if mandatory_kws is not None and possible_kws is not None:
        unknown_mandatory_kws = mandatory_kws.difference(possible_kws)
        if len(unknown_mandatory_kws) != 0:
            raise ValueError(
                f"Some mandatory kws are not allowed: {sorted(list(unknown_mandatory_kws))}."
            )

    def check_arguments(args: tuple, kws: KwargsType) -> None:
        """Positional and keyword arguments checker."""
        if nb_arguments_error_msg is not None:
            if len(args) != nb_arguments:
                raise ValueError(nb_arguments_error_msg)
        if mandatory_kws is not None:
            missing_kws = mandatory_kws.difference(kws)
            if len(missing_kws) != 0:
                raise ValueError(
                    f"Missing mandatory keyword-arguments: {sorted(list(missing_kws))}."
                )
        if possible_kws is not None:
            unknown_kws = set(kws).difference(possible_kws)
            if len(unknown_kws) != 0:
                raise ValueError(
                    f"Unknown keyword-arguments: {sorted(list(unknown_kws))}."
                )

    return check_arguments


def _sum_args(*args: TensorOrWeightedTensor, **start_kw) -> TensorOrWeightedTensor:
    """Summation of regular tensors with variadic input instead of standard iterable input."""
    summation = sum(args, **start_kw)
    if not isinstance(summation, (torch.Tensor, WeightedTensor)):
        # If args is empty, sum returns a float 0 that needs to be converted to a tensor
        return torch.tensor(summation)
    return summation
