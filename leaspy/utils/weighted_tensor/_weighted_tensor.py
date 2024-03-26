from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union, Callable, TypeVar, Generic
import operator

import torch

VT = TypeVar("VT")


@dataclass(frozen=True)
class WeightedTensor(Generic[VT]):
    """
    A torch.tensor, with optional (non-negative) weights (0 <-> masked).

    Parameters
    ----------
    value : torch.Tensor (of type VT)
        Raw values, without any mask.
    weight : None (default) or torch.Tensor (of booleans, integers or floats, with same shape as value)
        If None, weighted tensor boils down to a regular tensor.

    Attributes
    ----------
    value : torch.Tensor (of type VT)
        Raw values, without any mask.
    weight : None or torch.Tensor (of booleans, integers or floats, with same shape as value)
        Relative weights for values.
        If None, weighted tensor boils down to a regular tensor (as if all weights equal 1).
        If weight is a tensor[bool], it can be seen as a mask (valid value <-> weight is True).
        More generally, meaningless values <-> indices where weights equal 0.
    """

    value: torch.Tensor
    weight: Optional[torch.Tensor] = None

    def __post_init__(self):
        if not isinstance(self.value, torch.Tensor):
            assert not isinstance(
                self.value, WeightedTensor
            ), "You should NOT init a `WeightedTensor` with another"
            object.__setattr__(self, "value", torch.tensor(self.value))
        if self.weight is not None:
            if not isinstance(self.weight, torch.Tensor):
                assert not isinstance(
                    self.weight, WeightedTensor
                ), "You should NOT use a `WeightedTensor` for weights"
                object.__setattr__(self, "weight", torch.tensor(self.weight))
            assert (self.weight >= 0).all(), "Weights must be non-negative"
            # we forbid implicit broadcasting of weights for safety
            assert (
                self.weight.shape == self.value.shape
            ), f"Bad shapes: {self.weight.shape} != {self.value.shape}"
            assert (
                self.weight.device == self.value.device
            ), f"Bad devices: {self.weight.device} != {self.value.device}"

    @property
    def weighted_value(self) -> torch.Tensor:
        if self.weight is None:
            return self.value
        return self.weight * self.filled(0)

    def __getitem__(self, indices):
        if self.weight is None:
            return WeightedTensor(self.value.__getitem__(indices), None)
        return WeightedTensor(self.value.__getitem__(indices), self.weight.__getitem__(indices))

    def filled(self, fill_value: Optional[VT] = None) -> torch.Tensor:
        """Return the values tensor filled with `fill_value` where the `weight` is exactly zero.

        If `fill_value` is None or `weight` is None, return the value as is.
        """
        if fill_value is None or self.weight is None:
            return self.value
        return self.value.masked_fill(self.weight == 0, fill_value)

    def valued(self, value: torch.Tensor) -> WeightedTensor:
        """Return a new WeightedTensor with same weight as self but with new value provided."""
        return type(self)(value, self.weight)

    def map(
        self,
        func: Callable[[torch.Tensor], torch.Tensor],
        *args,
        fill_value: Optional[VT] = None,
        **kws,
    ) -> WeightedTensor:
        """Apply a function that only operates on values.

        This has no impact on weights (e.g. log-likelihood(value)).
        """
        return self.valued(func(self.filled(fill_value), *args, **kws))

    def map_both(
        self,
        func: Callable[[torch.Tensor], torch.Tensor],
        *args,
        fill_value: Optional[VT] = None,
        **kws,
    ) -> WeightedTensor:
        """Apply a function that operates both on values and weight, the same way (e.g. `expand` sizes)."""
        return type(self)(
            func(self.filled(fill_value), *args, **kws),
            func(self.weight, *args, **kws) if self.weight is not None else None,
        )

    def index_put(
        self,
        indices: Tuple[torch.Tensor, ...],  # of ints
        values: torch.Tensor,  # of VT
        *,
        accumulate: bool = False,
    ) -> WeightedTensor[VT]:
        """Out-of-place `torch.index_put` on values (no modification of weights)."""
        return self.map(
            torch.index_put, indices=indices, values=values, accumulate=accumulate
        )

    def wsum(self, *, fill_value: VT = 0, **kws) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the weighted sum of tensor together with sum of weights.

        <!> The result is NOT a `WeightedTensor` any more since weights are already taken into account.
        <!> We always fill values with 0 prior to weighting to prevent 0 * nan = nan that would propagate nans in sums.

        Parameters
        ----------
        fill_value : VT (default = 0)
            The value to fill the sum with for aggregates where weights were all zero.
        **kws
            Optional keyword-arguments for torch.sum (such as `dim=...` or `keepdim=...`)

        Returns
        -------
        weighted_sum : torch.Tensor[VT]
            Weighted sum, with totally un-weighted aggregates filled with `fill_value`.
        sum_weights : torch.Tensor (may be of other type than `cls.weight_dtype`)
            The sum of weights (useful if some average are needed).
            <!> We do NOT recast `sum_weights` to `cls.weight_dtype` to prevent lossy case: bool -> int -> bool.
        """
        weight = self.weight
        if weight is None:
            weight = torch.ones_like(self.value, dtype=torch.bool)
        weighted_values = weight * self.filled(0)
        weighted_sum = weighted_values.sum(**kws)
        sum_weights = weight.sum(**kws)
        return weighted_sum.masked_fill(sum_weights == 0, fill_value), sum_weights

    def sum(self, *, fill_value: VT = 0, **kws) -> torch.Tensor:
        """Get the weighted sum of tensor (discarding the sum of weights) - refer to `.wsum()`."""
        if self.weight is None:
            # more efficient in this case
            return self.value.sum(**kws)
        return self.wsum(fill_value=fill_value, **kws)[0]

    def view(self, *shape) -> WeightedTensor[VT]:
        """View of the tensor with another shape."""
        return self.map_both(torch.Tensor.view, *shape)

    def expand(self, *shape) -> WeightedTensor[VT]:
        """Expand the tensor with another shape."""
        return self.map_both(torch.Tensor.expand, *shape)

    def to(self, *, device: torch.device) -> WeightedTensor[VT]:
        """Apply the `torch.to` out-of-place function to both values and weights (only to move to device for now)."""
        return self.map_both(torch.Tensor.to, device=device)

    def cpu(self) -> WeightedTensor[VT]:
        return self.map_both(torch.Tensor.cpu)

    def __pow__(self, exponent: Union[int, float]) -> WeightedTensor[VT]:
        return self.valued(self.value**exponent)

    @property
    def shape(self) -> torch.Size:
        return self.value.shape

    @property
    def ndim(self) -> int:
        return self.value.ndim

    @property
    def dtype(self) -> torch.dtype:
        """Type of values."""
        return self.value.dtype

    @property
    def device(self) -> torch.device:
        return self.value.device

    @property
    def requires_grad(self) -> bool:
        return self.value.requires_grad

    def abs(self) -> WeightedTensor:
        return self.__abs__()

    def all(self) -> bool:
        return self.value.all()

    def clamp(self, min: float, max: float) -> WeightedTensor:
        return WeightedTensor(torch.clamp(self.value, min, max), self.weight)

    def __neg__(self) -> WeightedTensor:
        return _factory_for_unary_operators("neg")(self)

    def __abs__(self) -> WeightedTensor:
        return _factory_for_unary_operators("abs")(self)

    def __add__(self, other: WeightedTensor) -> WeightedTensor:
        return _factory_for_binary_operator("add")(self, other)

    def __radd__(self, other: WeightedTensor) -> WeightedTensor:
        return _factory_for_binary_operator("add", rev=True)(self, other)

    def __sub__(self, other: WeightedTensor) -> WeightedTensor:
        return _factory_for_binary_operator("sub")(self, other)

    def __rsub__(self, other: WeightedTensor) -> WeightedTensor:
        return _factory_for_binary_operator("sub", rev=True)(self, other)

    def __mul__(self, other:  WeightedTensor) -> WeightedTensor:
        return _factory_for_binary_operator("mul")(self, other)

    def __rmul__(self, other:  WeightedTensor) -> WeightedTensor:
        return _factory_for_binary_operator("mul", rev=True)(self, other)

    def __truediv__(self, other: WeightedTensor) -> WeightedTensor:
        return _factory_for_binary_operator("truediv")(self, other)

    def __rtruediv__(self, other: WeightedTensor) -> WeightedTensor:
        return _factory_for_binary_operator("truediv", rev=True)(self, other)

    def __lt__(self, other: WeightedTensor):
        return _factory_for_binary_operator("lt", fill_value=None)(self, other)

    def __le__(self, other: WeightedTensor):
        return _factory_for_binary_operator("le", fill_value=None)(self, other)

    def __eq__(self, other: WeightedTensor):
        return _factory_for_binary_operator("eq", fill_value=None)(self, other)

    def __ne__(self, other: WeightedTensor):
        return _factory_for_binary_operator("ne", fill_value=None)(self, other)

    def __gt__(self, other: WeightedTensor):
        return _factory_for_binary_operator("gt", fill_value=None)(self, other)

    def __ge__(self, other: WeightedTensor):
        return _factory_for_binary_operator("ge", fill_value=None)(self, other)

    @staticmethod
    def get_filled_value_and_weight(
        t: TensorOrWeightedTensor[VT], *, fill_value: Optional[VT] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Method to get tuple (value, weight) for both regular and weighted tensors."""
        if isinstance(t, WeightedTensor):
            return t.filled(fill_value), t.weight
        else:
            if not isinstance(t, torch.Tensor):
                t = torch.tensor(t)
            return t, None

    @classmethod
    def _binary_op(
        cls,
        op: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        a: TensorOrWeightedTensor[VT],
        b: TensorOrWeightedTensor[VT],
        *,
        same_weights_only: bool,
        fill_value: Optional[VT],
        **kws,
    ) -> WeightedTensor[VT]:
        """Binary operation between regular or weighted tensors (without shapes modifications)."""
        a_value, a_weight = cls.get_filled_value_and_weight(a, fill_value=fill_value)
        b_value, b_weight = cls.get_filled_value_and_weight(b, fill_value=fill_value)
        if a_weight is None and b_weight is None:
            result_weight = None
        elif a_weight is not None and b_weight is not None:
            if same_weights_only:
                # <!> Addition (and comparisons) of weighted tensors with unequal weights would be strange
                # We could allow it by weighting values at this stage but it seems a bit too implicit
                # and we would lost the sum of weights that could be needed for weighted means...

                # check strict identity of weights for efficiency first
                if a_weight is not b_weight and (
                    a_weight.shape != b_weight.shape
                    or a_weight.dtype != b_weight.dtype
                    or a_weight.device != b_weight.device
                    or not torch.equal(a_weight, b_weight)
                ):
                    # return NotImplemented (not much debug info in stack trace...)
                    raise NotImplementedError(
                        f"Binary operation '{op.__name__}' on two weighted tensors is not implemented "
                        "when their weights differ"
                    )
                result_weight = a_weight  # == b_weight
            else:
                # weights are multiplied (including broadcasting if needed)
                result_weight = a_weight * b_weight
        else:
            result_shape = torch.broadcast_shapes(a_value.shape, b_value.shape)
            if a_weight is not None:
                result_weight = a_weight.expand(result_shape)
            else:
                result_weight = b_weight.expand(result_shape)
        return WeightedTensor(op(a_value, b_value, **kws), result_weight)


TensorOrWeightedTensor = Union[torch.Tensor, WeightedTensor[VT]]


def _factory_for_unary_operators(name_of_operator: str, *, fill_value: Optional[VT] = 0):
    op = getattr(operator, name_of_operator)

    def _unary_op(self: WeightedTensor[VT]) -> WeightedTensor[VT]:
        return self.map(op, fill_value=fill_value)

    return _unary_op


def _factory_for_binary_operator(
    name_of_operator: str,
    *,
    fill_value: Optional[VT] = 0,
    rev: bool = False,
):
    op = getattr(operator, name_of_operator)
    kws = dict(
        fill_value=fill_value,
        same_weights_only=name_of_operator not in {"mul", "truediv"},
    )
    if rev:
        def _binary_rop(
            self: WeightedTensor[VT], other: TensorOrWeightedTensor[VT]
        ) -> WeightedTensor[VT]:
            return self._binary_op(op, other, self, **kws)

        return _binary_rop
    else:
        def _binary_op(
            self: WeightedTensor[VT], other: TensorOrWeightedTensor[VT]
        ) -> WeightedTensor[VT]:
            return self._binary_op(op, self, other, **kws)

        return _binary_op
