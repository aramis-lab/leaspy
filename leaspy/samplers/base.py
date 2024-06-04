from __future__ import annotations
from typing import TYPE_CHECKING, Tuple, Optional
from abc import ABC, abstractmethod

import torch

from leaspy.exceptions import LeaspyModelInputError

if TYPE_CHECKING:
    from leaspy.io.data.dataset import Dataset
    from leaspy.models.abstract_model import AbstractModel
from leaspy.io.realizations import CollectionRealization


class AbstractSampler(ABC):
    """
    Abstract sampler class.

    Parameters
    ----------
    name : :obj:`str`
        The name of the random variable to sample.
    shape : :obj:`tuple` of :obj:`int`
        The shape of the random variable to sample.
    acceptation_history_length : :obj:`int` > 0 (default 25)
        Deepness (= number of iterations) of the history kept for computing the mean acceptation rate.
        (It is the same for population or individual variables.)

    Attributes
    ----------
    name : :obj:`str`
        Name of variable
    shape : :obj:`tuple` of :obj:`int`
        Shape of variable
    acceptation_history_length : :obj:`int`
        Deepness (= number of iterations) of the history kept for computing the mean acceptation rate.
        (Same for population or individual variables by default.)
    acceptation_history : :class:`torch.Tensor`
        History of binary acceptations to compute mean acceptation rate for the sampler in MCMC-SAEM algorithm.
        It keeps the history of the last `acceptation_history_length` steps.

    Raises
    ------
    :exc:`.LeaspyModelInputError`
    """
    def __init__(
        self,
        name: str,
        shape: Tuple[int, ...],
        *,
        acceptation_history_length: int = 25,
    ):
        self.name = name
        self.shape = shape
        self.acceptation_history_length = acceptation_history_length
        self.acceptation_history = torch.zeros((self.acceptation_history_length, *self.shape_acceptation))

    @property
    @abstractmethod
    def shape_acceptation(self) -> Tuple[int, ...]:
        """
        Return the shape of acceptation tensor for a single iteration.

        Returns
        -------
        :obj:`tuple` of :obj:`int` :
            The shape of the acceptation history.
        """

    @abstractmethod
    def sample(
        self,
        dataset: Dataset,
        model: AbstractModel,
        realizations: CollectionRealization,
        temperature_inv: float,
        **attachment_computation_kws,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample new realization (either population or individual) for a given
        :class:`~.io.realizations.collection_realization.CollectionRealization`
        state, :class:`.Dataset`, :class:`.AbstractModel`, and temperature.

        <!> Modifies in-place the realizations object,
        <!> as well as the model through its `update_MCMC_toolbox` for population variables.

        Parameters
        ----------
        dataset : :class:`.Dataset`
            Dataset class object build with leaspy class object Data, model & algo
        model : :class:`.AbstractModel`
            Model for loss computations and updates
        realizations : :class:`~.io.realizations.collection_realization.CollectionRealization`
            Contain the current state & information of all the variables of interest
        temperature_inv : :obj:`float` > 0
            Inverse of the temperature used in tempered MCMC-SAEM
        **attachment_computation_kws
            Optional keyword arguments for attachment computations.
            As of now, we only use it for individual variables, and only `attribute_type`.
            It is used to know whether to compute attachments from the MCMC toolbox (esp. during fit)
            or to compute it from regular model parameters (esp. during personalization in mean/mode realization)

        Returns
        -------
        attachment, regularity_var : :class:`torch.Tensor`
            The attachment and regularity tensors (only for the current variable)
            at the end of this sampling step (globally or per individual, depending on variable type).
            The tensors are 0D for population variables, or 1D for individual variables (with length `n_individuals`).
        """

    def _group_metropolis_step(self, alpha: torch.Tensor) -> torch.Tensor:
        """
        Compute the acceptance decision (0. for False & 1. for True).

        Parameters
        ----------
        alpha : :class:`torch.FloatTensor` > 0

        Returns
        -------
        accepted : :class:`torch.FloatTensor`, same shape as `alpha`
            Acceptance decision (0. or 1.).
        """
        accepted = torch.rand(alpha.size()) < alpha
        return accepted.float()

    def _metropolis_step(self, alpha: float) -> bool:
        """
        Compute the Metropolis acceptance decision.

        If better (alpha>=1): accept
        If worse (alpha<1): accept with probability alpha

        <!> This function is critical for the reproducibility between machines.
        Different architectures might lead to different rounding errors on alpha
        (e.g: 1. - 1e-6 vs 1. + 1e-6). If we were to draw only for alpha < 1 (and not when alpha >= 1),
        then it would cause the internal seed of pytorch to change or not depending on the case
        which would lead to very different results afterwards (all the random numbers would be affected).

        Parameters
        ----------
        alpha : :obj:`float` > 0

        Returns
        -------
        :obj:`bool`
            Acceptance decision (False or True).
        """
        # Sample a realization from uniform law
        # Choose to keep iff realization is < alpha (probability alpha)
        # <!> Always draw a number even if it seems "useless" (cf. docstring warning)
        return torch.rand(1).item() < alpha

    def _update_acceptation_rate(self, accepted: torch.Tensor):
        """
        Update history of acceptation rates with latest accepted rates

        Parameters
        ----------
        accepted : :class:`torch.FloatTensor` (0. or 1.)

        Raises
        ------
        :exc:`.LeaspyModelInputError`
        """
        # Concatenate the new acceptation result at end of new one (forgetting the oldest acceptation rate)
        old_acceptation_history = self.acceptation_history[1:]
        self.acceptation_history = torch.cat([old_acceptation_history, accepted.unsqueeze(0)])


class AbstractIndividualSampler(AbstractSampler):
    """
    Abstract class for samplers of individual random variables.

    Parameters
    ----------
    name : :obj:`str`
        The name of the random variable to sample.
    shape : :obj:`tuple` of :obj:`int`
        The shape of the random variable to sample.
    n_patients : :obj:`int`
        Number of patients.
    acceptation_history_length : :obj:`int` > 0 (default 25)
        Deepness (= number of iterations) of the history kept for computing the mean acceptation rate.
        (It is the same for population or individual variables.)

    Attributes
    ----------
    name : :obj:`str`
        Name of variable
    shape : :obj:`tuple` of :obj:`int`
        Shape of variable
    n_patients : :obj:`int`
        Number of patients.
    acceptation_history_length : :obj:`int`
        Deepness (= number of iterations) of the history kept for computing the mean acceptation rate.
        (It is the same for population or individual variables.)
    acceptation_history : :class:`torch.Tensor`
        History of binary acceptations to compute mean acceptation rate for the sampler in MCMC-SAEM algorithm.
        It keeps the history of the last `acceptation_history_length` steps.
    """

    def __init__(
        self,
        name: str,
        shape: Tuple[int, ...],
        *,
        n_patients: int,
        acceptation_history_length: int = 25,
    ):
        self.n_patients = n_patients
        super().__init__(name, shape, acceptation_history_length=acceptation_history_length)

        # Initialize the acceptation history
        if len(self.shape) != 1:
            raise LeaspyModelInputError("Dimension of individual variable should be 1")

        # The dimension(s) to sum when computing regularity of individual parameters
        # For now there's only one extra dimension whether it's tau, xi or sources
        # but in the future it could be extended. We never sum dimension 0 which
        # will always be the individual dimension.
        self.ind_param_dims_but_individual = tuple(range(1, 1 + len(self.shape)))  # for now it boils down to (1,)


class AbstractPopulationSampler(AbstractSampler):
    """
    Abstract class for samplers of population random variables.

    Parameters
    ----------
    name : :obj:`str`
        The name of the random variable to sample.
    shape : :obj:`tuple` of :obj:`int`
        The shape of the random variable to sample.
    acceptation_history_length : :obj:`int` > 0 (default 25)
        Deepness (= number of iterations) of the history kept for computing the mean acceptation rate.
        (It is the same for population or individual variables.)
    mask : :class:`torch.Tensor`, optional
        If not None, mask should be 0/1 tensor indicating the sampling variable to adapt variance from
        1 indices are kept for sampling while 0 are excluded.

    Attributes
    ----------
    name : :obj:`str`
        Name of variable
    shape : :obj:`tuple` of :obj:`int`
        Shape of variable
    acceptation_history_length : :obj:`int`
        Deepness (= number of iterations) of the history kept for computing the mean acceptation rate.
        (It is the same for population or individual variables.)
    acceptation_history : :class:`torch.Tensor`
        History of binary acceptations to compute mean acceptation rate for the sampler in MCMC-SAEM algorithm.
        It keeps the history of the last `acceptation_history_length` steps.
    mask : :class:`torch.Tensor` of `obj`:bool, optional
        If not None, mask should be 0/1 tensor indicating the sampling variable to adapt variance from
        1 (True) indices are kept for sampling while 0 (False) are excluded.
    """

    def __init__(
        self,
        name: str,
        shape: Tuple[int, ...],
        *,
        acceptation_history_length: int = 25,
        mask: Optional[torch.Tensor] = None,
    ):
        super().__init__(name, shape, acceptation_history_length=acceptation_history_length)
        if len(self.shape) not in {1, 2}:
            # convention: shape of pop variable is 1D or 2D
            raise LeaspyModelInputError("Dimension of population variable should be 1 or 2")
        self.mask = mask
        if self.mask is not None:
            if not isinstance(self.mask, torch.Tensor) or self.mask.shape != self.shape:
                raise LeaspyModelInputError(
                    f"Mask for sampler should be of size {self.shape} but is of shape {self.mask.shape}"
                )
            self.mask = self.mask.to(bool)
