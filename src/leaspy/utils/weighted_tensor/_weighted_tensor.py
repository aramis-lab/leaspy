from __future__ import annotations

import operator
from dataobjes import dataclass
from typing import Callable, Generic, Optional, Tuple, TypeVar, Union

import torch

__all__ = [
    "WeightedTensor",
    "TensorOrWeightedTensor",
]


VT = TypeVar("VT")


@dataclass(frozen=True)
class WeightedTensor(Generic[VT]):
    """
    A torch.tensor, with optional (non-negative) weights (0 <-> masked).

    Parameters
    ----------
    value : :obj:`torch.Tensor` (of type VT)
        Raw values, without any mask.
    weight : :obj:`torch.Tensor', optional
        Relative weights for values.
        Default: None

    Attributes
    ----------
    value : :obj:`torch.Tensor` (of type VT)
        Raw values, without any mask.
    weight : :obj:`torch.Tensor` 
        Relative weights for values.
        If weight is a tensor[bool], it can be seen as a mask (valid value <-> weight is True).
        More generally, meaningless values <-> indices where weights equal 0.
        Default: None
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
        return WeightedTensor(
            self.value.__getitem__(indices), self.weight.__getitem__(indices)
        )

    def filled(self, fill_value: Optional[VT] = None) -> torch.Tensor:
        """
        Return the values tensor filled with `fill_value` where the `weight` is exactly zero.

        Parameters
        ----------
        fill_value : :obj:`torch.Tensor` (of type VT), optional
            The value to fill the tensor with for aggregates where weights were all zero.
            Default: None
        
        Returns
        -------
        :obj:`torch.Tensor` (of type VT):
            The filled tensor.
            If `weight` is None, the original tensor is returned.

        """
        if fill_value is None or self.weight is None:
            return self.value
        return self.value.masked_fill(self.weight == 0, fill_value)

    def valued(self, value: torch.Tensor) -> WeightedTensor:
        """
        Return a new WeightedTensor with same weight as self but with new value provided.
        
        Parameters
        ----------
        value : :obj:`torch.Tensor` 
            The new value to be set.
        
        Returns
        -------
        :obj:`WeightedTensor`:
            A new WeightedTensor with the same weight as self but with the new value provided.
        """
        return type(self)(value, self.weight)

    def map(
        self,
        func: Callable[[torch.Tensor], torch.Tensor],
        *args,
        fill_value: Optional[VT] = None,
        **kws,
    ) -> WeightedTensor:
        """
        Apply a function that only operates on values.

        Parameters
        ----------
        func : :obj:`Callable[[torch.Tensor], torch.Tensor]`
            The function to be applied to the values.
        *args : :obj:`Any`
            Positional arguments to be passed to the function.
        fill_value : :obj:`VT`, optional
            The value to fill the tensor with for aggregates where weights were all zero.
            Default: None
        **kws : :obj:`Any`
            Keyword arguments to be passed to the function.
        
        Returns
        -------
        :obj:`WeightedTensor`:
            A new `WeightedTensor` with the result of the operation and the same weights.

        """
        return self.valued(func(self.filled(fill_value), *args, **kws))

    def map_both(
        self,
        func: Callable[[torch.Tensor], torch.Tensor],
        *args,
        fill_value: Optional[VT] = None,
        **kws,
    ) -> WeightedTensor:
        """
        Apply a function that operates both on values and weight.
        
        Parameters
        ----------
        func : :obj:`Callable[[torch.Tensor], torch.Tensor]`
            The function to be applied to both values and weights.
        *args : :obj:`Any`
            Positional arguments to be passed to the function.
        fill_value : :obj:`VT`, optional
            The value to fill the tensor with for aggregates where weights were all zero.
            Default: None
        **kws : :obj:`Any`      
            Keyword arguments to be passed to the function.
        
        Returns
        -------
        :obj:`WeightedTensor`:
            A new `WeightedTensor` with the result of the operation and the appropriate weights.

        
        """
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
        """
        Out-of-place `torch.index_put` on values (no modification of weights).
        
        Parameters
        ----------
        
        indices : :obj:`Tuple[torch.Tensor, ...]`
            The indices to put the values at.
        values : :obj:`torch.Tensor` (of type VT)
            The values to put at the specified indices.
        accumulate : :obj:`bool`, optional
            Whether to accumulate the values at the specified indices.
            Default: False
        
        Returns
        -------
        :obj:`WeightedTensor`[VT]:
            A new `WeightedTensor` with the updated values and the same weights.
        """
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
        fill_value : :obj:`VT`, optional
            The value to fill the sum with for aggregates where weights were all zero.
            Default: 0
        **kws
            Optional keyword-arguments for torch.sum (such as `dim=...` or `keepdim=...`)

        Returns
        -------
        
        :obj:`Tuple[torch.Tensor, torch.Tensor]`:
        Tuple containing: 
            - weighted_sum : :obj:`torch.Tensor`[VT]
                Weighted sum, with totally un-weighted aggregates filled with `fill_value`.
            - sum_weights : :obj:`torch.Tensor` (may be of other type than `cls.weight_dtype`)
                The sum of weights (useful if some average are needed).
        """
        weight = self.weight
        if weight is None:
            weight = torch.ones_like(self.value, dtype=torch.bool)
        weighted_values = weight * self.filled(0)
        weighted_sum = weighted_values.sum(**kws)
        sum_weights = weight.sum(**kws)
        return weighted_sum.masked_fill(sum_weights == 0, fill_value), sum_weights

    def sum(self, *, fill_value: VT = 0, **kws) -> torch.Tensor:
        """
        Get the weighted sum of tensor.
        
        Parameters
        ----------
        fill_value : :obj:`VT`, optional
            The value to fill the sum with for aggregates where weights were all zero.
            Default: 0
        **kws
            Optional keyword-arguments 
        
        Returns
        -------
        :obj:`torch.Tensor`[VT]:
            The weighted sum, with totally un-weighted aggregates filled with `fill_value`.

        """
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

    def __neg__(self) -> WeightedTensor:
        return WeightedTensor(-1 * self.value, self.weight)

    def __abs__(self) -> WeightedTensor:
        return WeightedTensor(abs(self.value), self.weight)

    def __add__(self, other: TensorOrWeightedTensor) -> WeightedTensor:
        return _apply_operation(self, other, "add")

    def __radd__(self, other: TensorOrWeightedTensor) -> WeightedTensor:
        return _apply_operation(self, other, "add")

    def __sub__(self, other: TensorOrWeightedTensor) -> WeightedTensor:
        return _apply_operation(self, other, "sub")

    def __rsub__(self, other: TensorOrWeightedTensor) -> WeightedTensor:
        return _apply_operation(self, other, "sub", reverse=True)

    def __mul__(self, other: TensorOrWeightedTensor) -> WeightedTensor:
        return _apply_operation(self, other, "mul")

    def __rmul__(self, other: TensorOrWeightedTensor) -> WeightedTensor:
        return _apply_operation(self, other, "mul")

    def __truediv__(self, other: TensorOrWeightedTensor) -> WeightedTensor:
        return _apply_operation(self, other, "truediv")

    def __rtruediv__(self, other: TensorOrWeightedTensor) -> WeightedTensor:
        return _apply_operation(self, other, "truediv", reverse=True)

    def __lt__(self, other: TensorOrWeightedTensor) -> WeightedTensor:
        return _apply_operation(self, other, "lt")

    def __le__(self, other: TensorOrWeightedTensor) -> WeightedTensor:
        return _apply_operation(self, other, "le")

    def __eq__(self, other: TensorOrWeightedTensor) -> WeightedTensor:
        return _apply_operation(self, other, "eq")

    def __ne__(self, other: TensorOrWeightedTensor) -> WeightedTensor:
        return _apply_operation(self, other, "ne")

    def __gt__(self, other: TensorOrWeightedTensor) -> WeightedTensor:
        return _apply_operation(self, other, "gt")

    def __ge__(self, other: TensorOrWeightedTensor) -> WeightedTensor:
        return _apply_operation(self, other, "ge")

    @staticmethod
    def get_filled_value_and_weight(
        t: TensorOrWeightedTensor[VT], *, fill_value: Optional[VT] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Method to get tuple (value, weight) for both regular and weighted tensors.
        
        Parameters
        ----------
        t : :obj:`TensorOrWeightedTensor`
            The tensor to be converted.
        fill_value : :obj:`VT`, optional
            The value to fill the tensor with for aggregates where weights were all zero.
            Default: None
        
        Returns
        -------
        :obj:`Tuple[torch.Tensor, Optional[torch.Tensor]]`:
            Tuple containing:
            - value : :obj:`torch.Tensor`
                The filled tensor.
                If `weight` is None, the original tensor is returned.
            - weight : :obj:`torch.Tensor`, optional
                The weight tensor.
        """
        if isinstance(t, WeightedTensor):
            return t.filled(fill_value), t.weight
        else:
            if not isinstance(t, torch.Tensor):
                t = torch.tensor(t)
            return t, None


TensorOrWeightedTensor = Union[torch.Tensor, WeightedTensor[VT]]


def _apply_operation(
    a: WeightedTensor,
    b: TensorOrWeightedTensor,
    operator_name: str,
    reverse: bool = False,
) -> WeightedTensor:
    
    """
    Apply a binary operation on two tensors, with the first one being a `WeightedTensor`.
    The second one can be a `WeightedTensor` or a regular tensor.
    The operation is applied to the values of the tensors, and the weights are handled accordingly.

    Parameters
    ----------
    a : :obj:`WeightedTensor`
        The first tensor, which is a `WeightedTensor`.
    b : :obj:`TensorOrWeightedTensor`
        The second tensor, which can be a `WeightedTensor` or a regular tensor.
    operator_name : :obj:`str`
        The name of the binary operation to be applied.
    reverse : :obj:`bool`, optional
        If True, the operation is applied in reverse order (b operator a).
        Default: False
    Returns
    -------
    :obj:`WeightedTensor`:
        A new `WeightedTensor` with the result of the operation and the appropriate weights.
    """

    operation = getattr(operator, operator_name)
    if isinstance(b, WeightedTensor):
        result_value = (
            operation(b.value, a.value) if reverse else operation(a.value, b.value)
        )
        if a.weight is None:
            if b.weight is None:
                return WeightedTensor(result_value)
            else:
                return WeightedTensor(
                    result_value,
                    b.weight.expand(result_value.shape).clone()
                    if b.weight.shape != result_value.shape
                    else b.weight.clone(),
                )
        else:
            if b.weight is None:
                return WeightedTensor(
                    result_value,
                    a.weight.expand(result_value.shape).clone()
                    if a.weight.shape != result_value.shape
                    else a.weight.clone(),
                )
            else:
                if not torch.equal(a.weight, b.weight):
                    raise NotImplementedError(
                        f"Binary operation '{operator_name}' on two weighted tensors is "
                        "not implemented when their weights differ."
                    )
                return WeightedTensor(
                    result_value,
                    a.weight.expand(result_value.shape).clone()
                    if a.weight.shape != result_value.shape
                    else a.weight.clone(),
                )
    result_value = operation(b, a.value) if reverse else operation(a.value, b)
    result_weight = None
    if a.weight is not None:
        return WeightedTensor(
            result_value,
            (
                a.weight.expand(result_value.shape).clone()
                if a.weight.shape != result_value.shape
                else a.weight.clone()
            ),
        )

    return WeightedTensor(result_value)
