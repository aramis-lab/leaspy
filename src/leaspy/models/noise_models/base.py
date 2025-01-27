"""BaseNoiseModel defines the common interface for noise models in Leaspy."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    ClassVar,
    FrozenSet,
    Iterable,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import torch

from leaspy.exceptions import LeaspyInputError
from leaspy.io.data.dataset import Dataset
from leaspy.models.utilities import tensor_to_list
from leaspy.utils.typing import DictParamsTorch, KwargsType

T = TypeVar("T")


def constant_return_factory(x: T) -> Callable[[], T]:
    """
    Return a function returning the input value.

    Parameters
    ----------
    x : T
        The input value.

    Returns
    -------
    Callable :
        The constant return factory.
    """

    def constant_return():
        """
        Return constant value.
        """
        return x

    return constant_return


def value_to_tensor(x: Any) -> torch.Tensor:
    """
    Transform input values to tensors.

    Not intended to be used on values not castable to tensors, e.g. None.

    Parameters
    ----------
    x : Any
        The value that should be converted to a tensor.

    Returns
    -------
    :class:`torch.Tensor`
        The value converted to a tensor.
    """
    if isinstance(x, torch.Tensor):
        return x
    return torch.tensor(x)


@dataclass
class DistributionFamily:
    """
    Base class for a distribution family being able to sample "around" user-provided values.

    Parameters
    ----------
    parameters : :obj:`dict` [ :obj:`str`, :class:`torch.Tensor` ] or None
        Values for all the free parameters of the distribution family.
        All of them must have values before using the sampling methods.

    Attributes
    ----------
    free_parameters : :obj:`frozenset` of :obj:`str`
        Name of all the free parameters (but `loc`) needed to characterize the distribution.
        Nota: for each parameter, if a method named "validate_xxx" exists (torch.Tensor -> torch.Tensor),
        then it will be used for user-input validation of parameter "xxx".
    factory : None or function(free parameters values) -> :class:`torch.distributions.distribution.Distribution`
        The factory for the distribution family.
    parameters : :obj:`dict` [ :obj:`str`, :class:`torch.Tensor` ] or None
        Values for all the free parameters of the distribution family.
        All of them must have values before using the sampling methods.
    """

    parameters: Optional[DictParamsTorch] = None

    free_parameters: ClassVar[FrozenSet[str]]
    factory: ClassVar[Optional[Callable[..., torch.distributions.Distribution]]]

    def __post_init__(self):
        # we allow partially defined families until the actual use of sampling methods
        if self.parameters is None:
            return
        self.parameters = self.validate(**self.parameters)

    def validate(self, **params: Any) -> DictParamsTorch:
        """
        Validation function for parameters (based on 'validate_xxx' methods).

        Parameters
        ----------
        **params : Any
            The parameters to validate.

        Returns
        -------
        :class:`.DictParamsTorch` :
            The validated parameters.
        """
        self.raise_if_unknown_parameters(params)
        return {
            k: getattr(self, f"validate_{k}", lambda x: x)(value_to_tensor(v))
            for k, v in params.items()
            if v is not None
        }

    @classmethod
    def raise_if_unknown_parameters(cls, params: Optional[Iterable]) -> None:
        """
        Raise an error if the provided parameters are not part of the free parameters.

        Parameters
        ----------
        params : Iterable, optional
            The list of parameters to analyze.
        """
        unknown_params = set(params or ()).difference(cls.free_parameters)
        if len(unknown_params):
            raise LeaspyInputError(
                f"Cannot set unknown parameters {unknown_params} for distribution family {cls}."
            )

    def raise_if_partially_defined(self) -> None:
        """
        Raise an error if some of the free parameters are not defined.
        """
        missing_params = self.free_parameters.difference(self.parameters or ())
        if len(missing_params):
            raise LeaspyInputError(
                f"You must provide values for these free parameters: {set(missing_params)}"
            )

    def to_dict(self) -> KwargsType:
        """
        Serialize instance as dictionary.

        Returns
        -------
        :class:`.KwargsType` :
            The instance serialized as a dictionary.
        """
        return {k: tensor_to_list(v) for k, v in (self.parameters or {}).items()}

    def move_to_device(self, device: torch.device) -> None:
        """
        Move all torch tensors stored in this instance to
        the provided device (parameters & hyperparameters).

        Parameters
        ----------
        device : :class:`torch.device`
            Torch device on which to move the tensors.
        """
        for k, v in vars(self).items():
            if isinstance(v, torch.Tensor):
                setattr(self, k, v.to(device))
        if self.parameters is not None:
            self.parameters = {k: v.to(device) for k, v in self.parameters.items()}

    def update_parameters(
        self, *, validate: bool = False, **parameters: torch.Tensor
    ) -> None:
        """
        (Partial) update of the free parameters of the distribution family.

        Parameters
        ----------
        validate : :obj:`bool`, optional
            If True, the provided parameters are validated before being updated.
            Default=False.

        **parameters : :class:`torch.Tensor`
            The new parameters.
        """
        if validate:
            parameters = self.validate(**parameters)
        if self.parameters is None:
            self.parameters = parameters
        elif len(parameters):
            self.parameters.update(parameters)

    def sample_around(self, loc: torch.Tensor) -> torch.Tensor:
        """
        Realization around `loc` with respect to partially defined distribution.

        Parameters
        ----------
        loc : :class:`torch.Tensor`
            The loc around which to sample.

        Returns
        -------
        :class:`torch.Tensor` :
            The requested sample.
        """
        return self.sampler_around(loc)()

    def sampler_around(self, loc: torch.Tensor) -> Callable[[], torch.Tensor]:
        """
        Return the sampling function around input values.

        Parameters
        ----------
        loc : :class:`torch.Tensor`
            The loc around which to sample.

        Returns
        -------
        Callable :
            The sampler.
        """
        if self.factory is None:
            return constant_return_factory(loc)
        return self.rv_around(loc).sample

    def rv_around(self, loc: torch.Tensor) -> torch.distributions.Distribution:
        """
        Return the torch distribution centred around values (only if noise is not None).

        Parameters
        ----------
        loc : :class:`torch.Tensor`
            The loc around which to sample.

        Returns
        -------
        :class:`torch.distributions.distribution.Distribution` :
            The torch distribution centered around the loc.
        """
        if self.factory is None:
            raise LeaspyInputError(
                "Random variable around values is undefined with null distribution family."
            )
        self.raise_if_partially_defined()
        params = self.parameters or {}
        return self.factory(loc, validate_args=True, **params)


class NoNoise(DistributionFamily):
    """
    A dummy noise model that only returns the provided values.

    This model may be useful for simulation.

    Parameters
    ----------
    parameters : :obj:`dict` [ :obj:`str`, :class:`torch.Tensor` ] or None
        Values for all the free parameters of the distribution family.
        All of them must have values before using the sampling methods.

    Attributes
    ----------
    free_parameters : :obj:`frozenset` of :obj:`str`
        Name of all the free parameters (but `loc`) needed to characterize the distribution.
        Nota: for each parameter, if a method named "validate_xxx" exists (torch.Tensor -> torch.Tensor),
        then it will be used for user-input validation of parameter "xxx".
    factory : None or function(free parameters values) -> :class:`torch.distributions.distribution.Distribution`
        The factory for the distribution family.
    """

    factory = None
    free_parameters = frozenset()


NO_NOISE = NoNoise()


class BaseNoiseModel(ABC, DistributionFamily):
    """
    Base class for valid noise models that may be used in probabilistic models.

    The negative log-likelihood (nll, to be minimized) always corresponds to attachment term in models.

    Parameters
    ----------
    parameters : :obj:`dict` [ :obj:`str`, :class:`torch.Tensor` ] or None
        Values for all the free parameters of the distribution family.
        All of them must have values before using the sampling methods.

    Attributes
    ----------
    parameters : :obj:`dict` [ :obj:`str`, :class:`torch.Tensor` ] or  None
        All values for the free parameters of the distribution family.
        None is to be used if and only if there are no `free_parameters` at all.
    canonical_loss_properties : :obj:`tuple` [ :obj:`str`, :obj:`str` ]
        Tuple of strings characterizing the canonical loss of the noise model.
    """

    canonical_loss_properties: ClassVar = ("(neg) log-likelihood for attachment", ".3f")

    @abstractmethod
    def compute_nll(
        self,
        data: Dataset,
        predictions: torch.Tensor,
        *,
        with_gradient: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute negative log-likelihood of data given model predictions
        (no summation), and its gradient w.r.t. predictions if requested.

        Parameters
        ----------
        data : :class:`.Dataset`
            The dataset related to the computation of the log likelihood.
        predictions : :class:`torch.Tensor`
            The model's predictions from which to compute the log likelihood.
        with_gradient : :obj:`bool`, optional
            If True, returns also the gradient of the negative log likelihood
            wrt the predictions.
            If False, only returns the negative log likelihood.
            Default=False.

        Returns
        -------
        :class:`torch.Tensor` or :obj:`tuple` of :class:`torch.Tensor`
            The negative log likelihood (and its jacobian if requested).
        """

    def compute_canonical_loss(
        self,
        data: Dataset,
        predictions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute a human-friendly overall loss (independent from instance parameters),
        useful as a measure of goodness-of-fit after personalization (nll by default -
        assuming no free parameters).

        Parameters
        ----------
        data : :class:`.Dataset`
            The dataset related to the computation of the log likelihood.
        predictions : :class:`torch.Tensor`
            The model's predictions from which to compute the canonical loss.

        Returns
        -------
        :class:`torch.Tensor`
            The computed loss.
        """
        return self.compute_nll(data, predictions).sum()

    def compute_sufficient_statistics(
        self,
        data: Dataset,
        predictions: torch.Tensor,
    ) -> DictParamsTorch:
        """
        Computes the set of noise-related sufficient statistics
        and metrics (to be extended in child class).

        Parameters
        ----------
        data : :class:`.Dataset`
            The dataset related to the computation of the sufficient statistics.
        predictions : :class:`torch.Tensor`
            The model's predictions from which to compute the sufficient statistics.

        Returns
        -------
        :class:`.DictParamsTorch` :
            The sufficient statistics.
        """
        return {}

    def update_parameters_from_sufficient_statistics(
        self,
        data: Dataset,
        sufficient_statistics: DictParamsTorch,
    ) -> None:
        """
        Updates noise-model parameters in-place (nothing done by default).

        Parameters
        ----------
        data : :class:`.Dataset`
            The dataset related to the computation of the log likelihood.
        sufficient_statistics : :class:`DictParamsTorch`
            The sufficient statistics to use for parameter update.
        """
        pass

    def update_parameters_from_predictions(
        self,
        data: Dataset,
        predictions: torch.Tensor,
    ) -> None:
        """
        Updates noise-model parameters in-place (nothing done by default).

        Parameters
        ----------
        data : :class:`.Dataset`
            The dataset related to the computation of the log likelihood.
        predictions : :class:`torch.Tensor`
            The model's predictions from which to update the parameters.
        """
        pass
