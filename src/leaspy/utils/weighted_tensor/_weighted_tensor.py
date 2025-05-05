from __future__ import annotations

import operator
from dataclasses import dataclass
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
    value : :obj:`torch.Tensor`
        Raw values, without any mask.
    weight : :obj:`torch.Tensor`, optional
        Relative weights for values.
        Default: None

    Attributes
    ----------
    value : :obj:`torch.Tensor`
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
        """
        Post-initialization method to ensure that the value and weight tensors are properly initialized.

        Raises:
        ------
        AssertionError:
            - If `value` is a `WeightedTensor` (disallowed for initialization).
            - If `weight` is a `WeightedTensor` (disallowed for weights).
            - If `weight` contains negative values.
            - If `weight` and `value` have mismatched shapes (no implicit broadcasting allowed).
            - If `weight` and `value` are on different devices.
        """
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
        """
        Get the weighted value tensor.
        This is the value tensor multiplied by the weight tensor.

        Returns
        -------
        :obj:`torch.Tensor`:
            The weighted value tensor.
            If weight is None, the value tensor is returned.
        """
        if self.weight is None:
            return self.value
        return self.weight * self.filled(0)

    def __getitem__(self, indices) -> WeightedTensor:
        """
        Get the weighted tensor at the specified indices.

        Parameters
        ----------
        indices : :obj:`torch.Tensor`
            The indices to get the weighted tensor at.

        Returns
        -------
        :class:`WeightedTensor`:
            A new `WeightedTensor` with the values and weights at the specified indices.
        """

        if self.weight is None:
            return WeightedTensor(self.value.__getitem__(indices), None)
        return WeightedTensor(
            self.value.__getitem__(indices), self.weight.__getitem__(indices)
        )

    def filled(self, fill_value: Optional[VT] = None) -> torch.Tensor:
        """Return the values tensor with masked zeros filled with the specified value.

        Return the values tensor filled with `fill_value` where the `weight` is exactly zero.

        Parameters
        ----------
        fill_value : :obj:`VT`, optional
            The value to fill the tensor with for aggregates where weights were all zero.
            Default: None

        Returns
        -------
        :obj:`torch.Tensor`:
            The filled tensor.
            If `weight` or fill_value is None, the original tensor is returned.

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
        """Apply a function to the values tensor while preserving weights.

        The function is applied only to the values tensor after optionally filling
        zero-weight positions. The weights remain unchanged in the returned tensor.

        Parameters
        ----------
        func : Callable[[ :obj:`torch.Tensor` ], :obj:`torch.Tensor` ]
            The function to be applied to the values.
        *args :
            Positional arguments to be passed to the function.
        fill_value : :obj:`VT`, optional
            The value to fill the tensor with for aggregates where weights were all zero.
            Default: None
        **kws :
            Keyword arguments to be passed to the function.

        Returns
        -------
        :class:`WeightedTensor`:
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
        """Apply a function to both values and weights tensors.

        The same function is applied to both components of the weighted tensor.
        Zero-weight positions in the values tensor are filled before applying the function.

        Parameters
        ----------
        func : Callable[[ :obj`torch.Tensor` ], :obj:`torch.Tensor` ]
            The function to be applied to both values and weights.
        *args :
            Positional arguments to be passed to the function.
        fill_value : :obj:`VT`, optional
            The value to fill the tensor with for aggregates where weights were all zero.
            Default: None
        **kws :
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
        Out-of-place :func:`torch.index_put` on values (no modification of weights).

        Parameters
        ----------
        indices : :obj:`tuple` [ :obj:`torch.Tensor`, ...]
            The indices to put the values at.
        values : :obj:`torch.Tensor`
            The values to put at the specified indices.
        accumulate : :obj:`bool`, optional
            Whether to accumulate the values at the specified indices.
            Default: False

        Returns
        -------
        :class:`~leaspy.utils.weighted_tensor.WeightedTensor` [ :obj:`VT` ]
            A new :class:`~leaspy.utils.weighted_tensor.WeightedTensor` with the updated values and the same weights.
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

        :obj:`tuple` [ :obj:`torch.Tensor`, :obj:`torch.Tensor` ]:
        Tuple containing:
            - weighted_sum : :obj:`torch.Tensor`
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
        """Compute weighted sum of values.

        For unweighted tensors, this is equivalent to regular :func:`torch.sum`.
        For weighted tensors, returns the same as the first element of wsum().

        Parameters
        ----------
        fill_value : :obj:`VT`, optional
            The value to fill the sum with for aggregates where weights were all zero.
            Default: 0
        **kws
            Optional keyword-arguments

        Returns
        -------
        :obj:`torch.Tensor`:
            The weighted sum, with totally un-weighted aggregates filled with `fill_value`.
        """
        if self.weight is None:
            # more efficient in this case
            return self.value.sum(**kws)
        return self.wsum(fill_value=fill_value, **kws)[0]

    def view(self, *shape) -> WeightedTensor[VT]:
        """Return a view of the weighted tensor with a different shape.

        Parameters
        ----------
        shape : :obj:`tuple` [ :obj:`int`, ...]
            The new shape to be set.

        Returns
        -------
        :class:`~leaspy.utils.weighted_tensor.WeightedTensor` [ :obj:`VT` ]:
            A new :class:`~leaspy.utils.weighted_tensor.WeightedTensor` with the same weights but with the new shape provided.
        """
        return self.map_both(torch.Tensor.view, *shape)

    def expand(self, *shape) -> WeightedTensor[VT]:
        """Expand the weighted tensor to a new shape.

        Parameters
        ----------
        shape : :obj:`tuple` [ :obj:`int`, ...]
            The new shape to be set.

        Returns
        -------
        :class:`~leaspy.utils.weighted_tensor.WeightedTensor` [ :obj:`VT` ]:
            A new :class:`~leaspy.utils.weighted_tensor.WeightedTensor` with the same weights but with the new shape provided.
        """
        return self.map_both(torch.Tensor.expand, *shape)

    def to(self, *, device: torch.device) -> WeightedTensor[VT]:
        """Move the weighted tensor to a different device.

        Parameters
        ----------
        device : :obj:`torch.device`
            The device to be set.

        Returns
        -------
        :obj:`WeightedTensor`[:obj:`VT]:
            A new `WeightedTensor` with the same weights but with the new device provided.
        """
        return self.map_both(torch.Tensor.to, device=device)

    def cpu(self) -> WeightedTensor[VT]:
        """Move the weighted tensor to CPU memory.

        Applies the `torch.Tensor.cpu()` operation to both the value tensor and
        weight tensor (if present), returning a new weighted tensor with all
        components on the CPU.

        Returns
        -------
        :class:`~leaspy.utils.weighted_tensor.WeightedTensor` [ :obj:`VT` ]:
            A new :class:`~leaspy.utils.weighted_tensor.WeightedTensor` with the same weights but with the new device provided.
        """
        return self.map_both(torch.Tensor.cpu)

    def __pow__(self, exponent: Union[int, float]) -> WeightedTensor[VT]:
        """
        Apply the power of the tensor to the specified exponent.

        Parameters
        ----------
        exponent : :obj:`int` or :obj:`float`
            The exponent to be applied.

        Returns
        -------
        :obj:`WeightedTensor`[:obj:`VT]:
            A new `WeightedTensor` with the same weights but with the new exponent applied.
        """
        return self.valued(self.value**exponent)

    @property
    def shape(self) -> torch.Size:
        """Shape of the values tensor.

        Returns
        -------
        :obj:`torch.Size`:
            The shape of the values tensor.
        """
        return self.value.shape

    @property
    def ndim(self) -> int:
        """Number of dimensions of the values tensor.

        Returns
        -------
        :obj:`int`:
            The number of dimensions of the values tensor.
        """
        return self.value.ndim

    @property
    def dtype(self) -> torch.dtype:
        """Type of values.

        Returns
        -------
        :obj:`torch.dtype`:
            The type of values.
        """

        return self.value.dtype

    @property
    def device(self) -> torch.device:
        """Device of values.

        Returns
        -------
        :obj:`torch.device`:
            The device of values.
        """
        return self.value.device

    @property
    def requires_grad(self) -> bool:
        """Whether the values tensor requires gradients.

        Returns
        -------
        :obj:`bool`:
            Whether the values tensor requires gradients.
        """
        return self.value.requires_grad

    def abs(self) -> WeightedTensor:
        """Compute the absolute value of the weighted tensor.

        Returns
        -------
        :class:`~leaspy.utils.weighted_tensor.WeightedTensor`
            A new `WeightedTensor` with the absolute value of the values tensor.
        """
        return self.__abs__()

    def all(self) -> bool:
        """Check if all values are non-zero.

        Returns
        -------
        :obj:`bool`:
            Whether all values are non-zero.
        """
        return self.value.all()

    def __neg__(self) -> WeightedTensor:
        """Compute the negative of the weighted tensor.

        Returns
        -------
        :obj:`WeightedTensor`:
            A new `WeightedTensor` with the negative of the values tensor.
        """
        return WeightedTensor(-1 * self.value, self.weight)

    def __abs__(self) -> WeightedTensor:
        """Compute the absolute value of the weighted tensor.

        Returns
        -------
        :obj:`WeightedTensor`:
            A new `WeightedTensor` with the absolute value of the values tensor, the weight stay the same.
        """
        return WeightedTensor(abs(self.value), self.weight)

    def __add__(self, other: TensorOrWeightedTensor) -> WeightedTensor:
        """Compute the sum of the weighted tensor and another tensor.

        Returns a new weighted tensor containing:
        - The sum of the value tensors
        - The weight according to the following rules:
            - If both tensors have weights: weights must be identical
            - If only one tensor has weights: those weights are retained
            - If neither tensor has weights: result has no weights

        Parameters
        ----------
        other : class:`TensorOrWeightedTensor`
            The tensor to be added to the weighted tensor.
        Returns
        -------
        :obj:`WeightedTensor`:
            A new `WeightedTensor` with the summed values

        """
        return _apply_operation(self, other, "add")

    def __radd__(self, other: TensorOrWeightedTensor) -> WeightedTensor:
        """Compute the sum of another tensor and the weighted tensor.

        Equivalent to `__add__` but with operands reversed. See `__add__` for details.

        Parameters
        ----------
        other : class:`TensorOrWeightedTensor`
            The tensor to be added to the weighted tensor.

        Returns
        -------
        :obj:`WeightedTensor`:
            A new `WeightedTensor` with the sum of the other tensor and the values tensor.
        """
        return _apply_operation(self, other, "add", reverse=True)

    def __sub__(self, other: TensorOrWeightedTensor) -> WeightedTensor:
        """Compute the difference between the weighted tensor and another tensor.

        Returns a new weighted tensor containing:
            - The difference between this tensor's values and the other tensor's values
            - The weight according to the same rules as __add__

        Parameters
        ----------
        other : class:`TensorOrWeightedTensor`
            The tensor to be subtracted from the weighted tensor.

        Returns
        -------
        :obj:`WeightedTensor`:
            A new `WeightedTensor` with the differences and appropriate weights.
        """
        return _apply_operation(self, other, "sub")

    def __rsub__(self, other: TensorOrWeightedTensor) -> WeightedTensor:
        """Compute the difference between another tensor and the weighted tensor.

        Equivalent to `__sub__` but with operands reversed. See `__sub__` for details.

        Parameters
        ----------
        other : class:`TensorOrWeightedTensor`
            The tensor to be subtracted from the weighted tensor.

        Returns
        -------
        :obj:`WeightedTensor`:
            A new `WeightedTensor` containing the differences and appropriate weights.
        """
        return _apply_operation(self, other, "sub", reverse=True)

    def __mul__(self, other: TensorOrWeightedTensor) -> WeightedTensor:
        """Compute the product of the weighted tensor and another tensor.

        Returns a new weighted tensor containing:
            - The product of the value tensors
            - The weight according to the same rules as __add__

        Parameters
        ----------
        other : class:`TensorOrWeightedTensor`
            The tensor to be multiplied with the weighted tensor.

        Returns
        -------
        :obj:`WeightedTensor`:
            A new `WeightedTensor` containing the products and appropriate weights
        """
        return _apply_operation(self, other, "mul")

    def __rmul__(self, other: TensorOrWeightedTensor) -> WeightedTensor:
        """Compute the product of another tensor and the weighted tensor.

        Equivalent to `__mul__` but with operands reversed. See `__mul__` for details.

        Parameters
        ----------
        other : class:`TensorOrWeightedTensor`
            The tensor to be multiplied with the weighted tensor.
        Returns
        -------
        :obj:`WeightedTensor`:
            A new `WeightedTensor` containing the products and appropriate weights
        """
        return _apply_operation(self, other, "mul", reverse=True)

    def __truediv__(self, other: TensorOrWeightedTensor) -> WeightedTensor:
        """Compute the division of the weighted tensor by another tensor.

        Returns a new weighted tensor containing:
            - The quotient of the value tensors
            - The weight according to the same rules as __add__

        Parameters
        ----------
        other : class:`TensorOrWeightedTensor`
            The tensor to divide the weighted tensor by.

        Returns
        -------
        :obj:`WeightedTensor`:
            A new `WeightedTensor` containing the quotients and appropriate weights.

        """
        return _apply_operation(self, other, "truediv")

    def __rtruediv__(self, other: TensorOrWeightedTensor) -> WeightedTensor:
        """Compute the division of another tensor by the weighted tensor.

        Equivalent to `__truediv__` but with operands reversed. See `__truediv__` for details.

        Parameters
        ----------
        other : class:`TensorOrWeightedTensor`
            The tensor to be divided by the weighted tensor.

        Returns
        -------
        :obj:`WeightedTensor`:
            A new `WeightedTensor` containing the quotients and appropriate weights.
        """
        return _apply_operation(self, other, "truediv", reverse=True)

    def __lt__(self, other: TensorOrWeightedTensor) -> WeightedTensor:
        """Compute the less-than comparison between the weighted tensor and another tensor.

        Returns a new weighted tensor containing boolean values indicating where:
            - This tensor's values are less than the other tensor's values
            - The weight according to the same rules as __add__

        Parameters
        ----------
        other : class:`TensorOrWeightedTensor`
            The tensor to compare against

        Returns
        -------
        :obj:`WeightedTensor`:
            A new `WeightedTensor` with the result of the less-than comparison and appropriate weights.
        """
        return _apply_operation(self, other, "lt")

    def __le__(self, other: TensorOrWeightedTensor) -> WeightedTensor:
        """Compute the less-than-or-equal-to comparison between the weighted tensor and another tensor.

        Returns a new weighted tensor containing boolean values indicating where:
            - This tensor's values are less than or equal to the other tensor's values
            - The weight according to the same rules as __add__

        Parameters
        ----------
        other : class:`TensorOrWeightedTensor`
            The tensor to compare against.

        Returns
        -------
        :obj:`WeightedTensor`:
            A new `WeightedTensor` with the result of the less-than-or-equal-to comparison and appropriate weights.
        """
        return _apply_operation(self, other, "le")

    def __eq__(self, other: TensorOrWeightedTensor) -> WeightedTensor:
        """Compute the equality comparison between the weighted tensor and another tensor.

        Returns a new weighted tensor containing boolean values indicating where:
            - This tensor's values equal the other tensor's values
            - The weight according to the same rules as __add__

        Parameters
        ----------
        other : class:`TensorOrWeightedTensor`
            The tensor to compare against.

        Returns
        -------
        :obj:`WeightedTensor`:
            A new `WeightedTensor` with the result of the equality comparison and appropriate weights..
        """
        return _apply_operation(self, other, "eq")

    def __ne__(self, other: TensorOrWeightedTensor) -> WeightedTensor:
        """Compute the not-equal-to comparison between the weighted tensor and another tensor.

        Returns a new weighted tensor containing boolean values indicating where:
            - This tensor's values differ from the other tensor's values
            - The weight according to the same rules as __add__

        Parameters
        ----------
        other : class:`TensorOrWeightedTensor`
            The tensor to compare against.

        Returns
        -------
        :obj:`WeightedTensor`:
            A new `WeightedTensor` with the result of the not-equal-to comparison and appropriate weights..
        """
        return _apply_operation(self, other, "ne")

    def __gt__(self, other: TensorOrWeightedTensor) -> WeightedTensor:
        """Compute the greater-than comparison between the weighted tensor and another tensor.
        Returns a new weighted tensor containing boolean values indicating where:
            - This tensor's values exceed the other tensor's values
            - The weight according to the same rules as __add__

        Parameters
        ----------
        other : class:`TensorOrWeightedTensor`
            The tensor to compare against.

        Returns
        -------
        :obj:`WeightedTensor`:
            A new `WeightedTensor` with the result of the greater-than comparison and appropriate weights.
        """
        return _apply_operation(self, other, "gt")

    def __ge__(self, other: TensorOrWeightedTensor) -> WeightedTensor:
        """Compute the greater-than-or-equal-to comparison between the weighted tensor and another tensor.

        Returns a new weighted tensor containing boolean values indicating where:
            - This tensor's values are greater than or equal to the other tensor's values
            - The weight according to the same rules as __add__

        Parameters
        ----------
        other : class:`TensorOrWeightedTensor`
            The tensor to compare against.

        Returns
        -------
        :obj:`WeightedTensor`:
            A new `WeightedTensor` with the result of the greater-than-or-equal-to compariso and appropriate weights.
        """
        return _apply_operation(self, other, "ge")

    @staticmethod
    def get_filled_value_and_weight(
        t: TensorOrWeightedTensor[VT], *, fill_value: Optional[VT] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Method to get tuple (value, weight) for both regular and weighted tensors.

        Parameters
        ----------
        t : class:`TensorOrWeightedTensor`
            The tensor to be converted.
        fill_value : :obj:`VT`, optional
            The value to fill the tensor with for aggregates where weights were all zero.
            Default: None

        Returns
        -------
        :obj:`Tuple`[:obj:`torch.Tensor`, Optional[:obj:`torch.Tensor`]]:
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
    a : :class:`WeightedTensor`
        The first tensor, which is a `WeightedTensor`.
    b : class:`TensorOrWeightedTensor`
        The second tensor, which can be a `WeightedTensor` or a regular tensor.
    operator_name : :obj:`str`
        The name of the binary operation to be applied.
    reverse : :obj:`bool`, optional
        If True, the operation is applied in reverse order (b operator a).
        Default: False
    Returns
    -------
    :class:`WeightedTensor`:
        A new `WeightedTensor` with the result of the operation and the appropriate weights.

    Raises
    ------
    :exc:`NotImplementedError`
        If the operation is not implemented for the given combination of tensors.
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
