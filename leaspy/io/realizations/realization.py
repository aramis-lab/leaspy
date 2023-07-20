from __future__ import annotations

import abc
from typing import Tuple, Optional, TYPE_CHECKING

import torch

from leaspy.exceptions import LeaspyInputError
from leaspy.utils.typing import ParamType, KwargsType

if TYPE_CHECKING:
    from leaspy.models.abstract_model import AbstractModel


class AbstractRealization:
    """
    Abstract class for Realization.

    Parameters
    ----------
    name : ParamType
        The name of the variable associated with the realization.
    shape : Tuple[int, ...]
        The shape of the tensor realization.
    tensor : torch.Tensor, optional
        If not None, the tensor realization to be stored.
    tensor_copy : bool (default True)
        Whether the `tensor` provided is copied or not.
    **kwargs : dict
        Additional parameters.

    Attributes
    ----------
    name : ParamType
        The name of the variable associated with the realization.
    shape : Tuple[int, ...]
        The shape of the tensor realization.
    tensor : torch.Tensor
        The tensor realization.
    """
    def __init__(
        self,
        name: ParamType,
        shape: Tuple[int, ...],
        *,
        tensor: Optional[torch.Tensor] = None,
        tensor_copy: bool = True,
        **kwargs
    ):
        self.name = name
        self.shape = shape
        self._tensor_realizations: Optional[torch.Tensor] = None

        if tensor is not None:
            if tensor_copy:
                tensor = tensor.clone().detach()
            self.tensor = tensor

    def to_dict(self) -> KwargsType:
        """Return a serialized dictionary of realization attributes."""
        return dict(
            name=self.name,
            shape=self.shape,
            tensor=self.tensor,
        )

    def __deepcopy__(self, memo):
        """
        Deep-copy the Realization object (magic method invoked with using copy.deepcopy)

        It clones the underlying tensor and detach it from the computational graph

        Returns
        -------
        A new instance from self class.
        """
        return self.__class__(**self.to_dict(), tensor_copy=True)

    @property
    def tensor(self) -> torch.Tensor:
        if self._tensor_realizations is None:
            raise LeaspyInputError(
                f"You can not access tensor realization for {self.name} until properly initialized."
            )
        return self._tensor_realizations

    @tensor.setter
    def tensor(self, tensor: torch.Tensor):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"Expected a torch tensor object but received a {type(tensor)} instead."
            )
        # <!> no copy of provided tensor
        self._tensor_realizations = tensor

    def set_tensor_realizations_element(self, element: torch.Tensor, dim: tuple[int, ...]) -> None:
        """
        Manually change the value (in-place) of `tensor_realizations` at dimension `dim`.

        Parameters
        ----------
        element : torch.Tensor
            The element to put in the tensor realization.
        dim : Tuple[int, ...]
            The dimension where to put the element.
        """
        if not isinstance(element, torch.Tensor):
            raise TypeError(
                f"Expected a torch tensor object but received a {type(element)} instead."
            )
        self.tensor[dim] = element

    @abc.abstractmethod
    def initialize(
        self,
        model: AbstractModel,
        **kwargs: KwargsType,
    ):
        """
        Initialize realization from a given model.

        Parameters
        ----------
        model : :class:`.AbstractModel`
            The model you want realizations for.
        **kwargs : KwargsType
            Additional parameters for initialization.

        Raises
        ------
        :exc:`.LeaspyModelInputError`
            if unknown variable type
        """

    def set_autograd(self) -> None:
        """
        Set autograd for tensor of realizations.

        TODO remove? only in legacy code

        Raises
        ------
        :class:`ValueError`
            if inconsistent internal request

        See Also
        --------
        torch.Tensor.requires_grad_
        """
        if self.tensor.requires_grad:
            raise ValueError("Realizations are already using autograd")
        self.tensor.requires_grad_(True)

    def unset_autograd(self) -> None:
        """
        Unset autograd for tensor of realizations

        TODO remove? only in legacy code

        Raises
        ------
        :class:`ValueError`
            if inconsistent internal request

        See Also
        --------
        torch.Tensor.requires_grad_
        """
        if not self.tensor.requires_grad:
            raise ValueError("Realizations are already detached")
        self.tensor.requires_grad_(False)

    def __str__(self):
        s = f"Realization of {self.name}\n"
        s += f"Shape : {self.shape}\n"
        return s


class IndividualRealization(AbstractRealization):
    """
    Class for realizations of individual variables.

    Parameters
    ----------
    name : ParamType
        The name of the variable associated with the realization.
    shape : Tuple[int, ...]
        The shape of the tensor realization.
    n_individuals : int
        The number of individuals related to this realization.
    **kwargs : dict
        Additional parameters (including `tensor` and `tensor_copy`).
    """
    def __init__(
        self,
        name: ParamType,
        shape: Tuple[int, ...],
        *,
        n_individuals: int,
        **kwargs
    ):
        super().__init__(name, shape, **kwargs)
        self.n_individuals = n_individuals
        self.distribution_factory = torch.distributions.Normal

    def to_dict(self):
        """Return a serialized dictionary of realization attributes."""
        return dict(super().to_dict(), n_individuals=self.n_individuals)

    def initialize(
        self,
        model: AbstractModel,
        *,
        init_at_mean: bool = False,
        **kwargs: KwargsType,
    ):
        """
        Initialize the realization from a model instance.

        Parameters
        ----------
        model : AbstractModel
            The model from which to initialize the realization.
        init_at_mean : bool, optional
            If True, the realization is initialized at the corresponding
            variable mean value, otherwise it the initial value is sampled
            around its mean value with a normal distribution.
        **kwargs : KwargsType
            Additional parameters for initialization.
        """
        if init_at_mean:
            self.initialize_at_mean(model.parameters[f"{self.name}_mean"])
        else:
            self.initialize_around_mean(
                model.parameters[f"{self.name}_mean"],
                model.parameters[f"{self.name}_std"],
            )

    def initialize_at_mean(self, mean: torch.Tensor) -> None:
        """
        Initialize the realization at provided mean value.

        Parameters
        ----------
        mean : torch.Tensor
            The mean at which to initialize the realization.
        """
        self.tensor = mean * torch.ones((self.n_individuals, *self.shape))

    def initialize_around_mean(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """
        Initialize the realization around the provided mean value.

        The initial value is sampled according to a normal distribution with
        provided mean and std parameters.

        Parameters
        ----------
        mean : torch.Tensor
            Mean value around which to sample the initial value.
        std : torch.Tensor
            Standard deviation for the normal distribution used to
            sample the initial value.
        """
        distribution = self.distribution_factory(loc=mean, scale=std)
        self.tensor = distribution.sample(
            sample_shape=(self.n_individuals, *self.shape)
        )

    def __str__(self):
        s = super().__str__()
        s += f"Variable type : individual"
        return s



class PopulationRealization(AbstractRealization):
    """
    Class for realizations of population variables.

    Parameters
    ----------
    name : ParamType
        The name of the variable associated with the realization.
    shape : Tuple[int, ...]
        The shape of the tensor realization.
    tensor : torch.Tensor, optional
        If not None, the tensor realization to be stored.
    tensor_copy : bool (default True)
        Whether the `tensor` provided is copied or not.
    **kwargs : dict
        Additional parameters.
    """

    def initialize(
        self,
        model: AbstractModel,
        **kwargs: KwargsType,
    ) -> None:
        """
        Initialize the realization from a model instance.

        Parameters
        ----------
        model : AbstractModel
            The model from which to initialize the realization.
        **kwargs : KwargsType
            Additional parameters for initialization.
        """
        self.tensor = model.parameters[self.name].reshape(self.shape).clone().detach()

    def __str__(self):
        s = super().__str__()
        s += f"Variable type : population"
        return s

class DeterministicRealization(AbstractRealization):
    """
        Class for realizations of deterministic variables.

        Parameters
        ----------
        name : ParamType
            The name of the variable associated with the realization.
        shape : Tuple[int, ...]
            The shape of the tensor realization.
        n_individuals : int
            The number of individuals related to this realization.
        **kwargs : dict
            Additional parameters (including `tensor` and `tensor_copy`).
        """

    def __init__(
            self,
            name: ParamType,
            shape: Tuple[int, ...],
            *,
            n_individuals: int,
            **kwargs
    ):
        super().__init__(name, shape, **kwargs)
        self.n_individuals = n_individuals
        self.init_function = kwargs["init_function"]
        self.update_function = kwargs["update_function"]

    def to_dict(self):
        """Return a serialized dictionary of realization attributes."""
        return dict(super().to_dict(), n_individuals=self.n_individuals)

    def initialize(
            self,
            model: AbstractModel,
            **kwargs: KwargsType,
    ):
        """
        Initialize the realization from a model instance.

        Parameters
        ----------
        model : AbstractModel
            The model from which to initialize the realization.
        init_at_mean : bool, optional
            If True, the realization is initialized at the corresponding
            variable mean value, otherwise it the initial value is sampled
            around its mean value with a normal distribution.
        **kwargs : KwargsType
            Additional parameters for initialization.
        """

        self.tensor = self.init_function(self.n_individuals, model, **kwargs)

    def __str__(self):
        s = super().__str__()
        s += f"Variable type : individual deterministic"
        return s

