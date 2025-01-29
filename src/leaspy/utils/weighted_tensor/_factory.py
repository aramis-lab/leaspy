from functools import wraps
from typing import Callable, Optional, TypeVar

import torch

from ._weighted_tensor import TensorOrWeightedTensor, WeightedTensor

__all__ = [
    "factory_weighted_tensor_unary_operator",
]


VT = TypeVar("VT")


def factory_weighted_tensor_unary_operator(
    f: Callable[[torch.Tensor], torch.Tensor],
    *,
    fill_value: Optional[VT] = None,
) -> Callable[[TensorOrWeightedTensor[VT]], TensorOrWeightedTensor[VT]]:
    """Factory/decorator to create a weighted-tensor compatible function from the provided unary-tensor function."""

    @wraps(f)
    def f_compatible(
        x: TensorOrWeightedTensor[VT], *args, **kws
    ) -> TensorOrWeightedTensor[VT]:
        if not isinstance(x, WeightedTensor):
            return f(x, *args, **kws)
        r = f(x.filled(fill_value), *args, **kws)
        conv = x.valued
        if isinstance(r, (tuple, list, set, frozenset)):
            return type(r)(map(conv, r))
        return conv(r)

    return f_compatible
