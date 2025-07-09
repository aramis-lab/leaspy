"""This module defines commonly used named input functions."""

from __future__ import annotations

import torch

from ..linalg import compute_orthonormal_basis, compute_orthonormal_basis_batch
from ..weighted_tensor import factory_weighted_tensor_unary_operator, sum_dim
from ._named_input_function import NamedInputFunction
from ._utils import _affine_from_vector, _arguments_checker, _batch_matmul_by_index, _identity, _index_of, _prod_args, _sum_args, _unique_wrapper

__all__ = [
    "Prod",
    "Identity",
    "MatMul",
    "OrthoBasis",
    "Exp",
    "Sqr",
    "Mean",
    "Std",
    "SumDim",
    "Sum",
    "AffineFromVector",
    "Unique",
    "IndexOf",
    "batch_matmul_by_index",
]


Prod = NamedInputFunction.bound_to(
    _prod_args,
    _arguments_checker(
        possible_kws={"start"},
    ),
)


Identity = NamedInputFunction.bound_to(
    _identity,
    _arguments_checker(
        nb_arguments=1,
        possible_kws=set(),
    ),
)


MatMul = NamedInputFunction.bound_to(
    torch.matmul,
    _arguments_checker(
        nb_arguments=2,
        possible_kws=set(),
    ),
)


BatchMatMulByIndex = NamedInputFunction.bound_to(
    _batch_matmul_by_index,
    _arguments_checker(
        nb_arguments=3,
        possible_kws=set()
    ),
)


OrthoBasis = NamedInputFunction.bound_to(
    compute_orthonormal_basis,
    _arguments_checker(
        nb_arguments=2,
        possible_kws=set("strip_col"),
    ),
)


OrthoBasisBatch = NamedInputFunction.bound_to(
    compute_orthonormal_basis_batch,
    _arguments_checker(
        nb_arguments=2,
        possible_kws=set("strip_col"),
    ),
)


Exp = NamedInputFunction.bound_to(
    factory_weighted_tensor_unary_operator(torch.exp),
    _arguments_checker(
        nb_arguments=1,
        possible_kws=set(),
    ),
)


Sqr = NamedInputFunction.bound_to(
    factory_weighted_tensor_unary_operator(torch.square),
    _arguments_checker(
        nb_arguments=1,
        possible_kws=set(),
    ),
)


Mean = NamedInputFunction.bound_to(
    # <!> never compute mean directly on WeightedTensor (use `.wsum()` instead)
    torch.mean,
    _arguments_checker(
        nb_arguments=1,
        possible_kws={"dim"},
    ),
)


Std = NamedInputFunction.bound_to(
    # <!> never compute std directly on WeightedTensor
    torch.std,
    _arguments_checker(
        nb_arguments=1,
        possible_kws={"dim", "unbiased"},
    ),
)


SumDim = NamedInputFunction.bound_to(
    sum_dim,
    _arguments_checker(
        nb_arguments=1,  # with `dim` XOR `but_dim`
    ),
)


Sum = NamedInputFunction.bound_to(
    _sum_args,
    _arguments_checker(
        possible_kws={"start"},
    ),
)


AffineFromVector = NamedInputFunction.bound_to(
    _affine_from_vector,
    _arguments_checker(
        nb_arguments=2,
        possible_kws=set(),
    ),
)

Unique = NamedInputFunction.bound_to(
    _unique_wrapper,
    _arguments_checker(
        nb_arguments=1,
        possible_kws=set(),
    ),
)

IndexOf = NamedInputFunction.bound_to(
    _index_of,
    _arguments_checker(
        nb_arguments=2,
        possible_kws=set()
    )
)


# Filled = NamedInputFunction.bound_to(
#     WeightedTensor.filled, arguments_checker(nb_arguments=1, mandatory_kws={"fill_value"})
# )
# Negate = NamedInputFunction.bound_to(
#     operator.neg, arguments_checker(nb_arguments=1, possible_kws=set())
# )
# ItemGetter = NamedInputFunction.bound_to(
#     operator.itemgetter, arguments_checker(nb_arguments=1, possible_kws=set())
# )
