from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch

from leaspy.exceptions import LeaspyModelInputError
from leaspy.variables.state import State

__all__ = [
    "AbstractSampler",
    "AbstractIndividualSampler",
    "AbstractPopulationSampler",
]


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
        self.acceptation_history = torch.zeros(
            (self.acceptation_history_length, *self.shape_acceptation)
        )

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return len(self.shape)

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
        state: State,
        *,
        temperature_inv: float,
    ) -> None:
        """
        Apply a sampling step

        <!> It will modify in-place the internal state, caching all intermediate values needed to efficient.

        Parameters
        ----------
        state : :class:`.State`
            Object containing values for all model variables, including latent variables
        temperature_inv : :obj:`float` > 0
            Inverse of the temperature used in tempered MCMC-SAEM
        """

    def _group_metropolis_step(self, alpha: torch.Tensor) -> torch.Tensor:
        """
        Compute the boolean acceptance decisions.

        Parameters
        ----------
        alpha : :class:`torch.FloatTensor` > 0

        Returns
        -------
        accepted : :class:`torch.Tensor[bool]`, same shape as `alpha`
            Acceptance decision (False or True).
        """
        accepted = torch.rand(alpha.shape) < alpha
        return accepted

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
        return torch.rand(()) < alpha

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
        self.acceptation_history = torch.cat(
            [old_acceptation_history, accepted.unsqueeze(0)]
        )


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
        super().__init__(
            name, shape, acceptation_history_length=acceptation_history_length
        )

        # Initialize the acceptation history
        #if self.ndim != 1:
        #    raise LeaspyModelInputError("Dimension of individual variable should be 1")


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
        A binary (0/1) tensor indicating which elements to sample.
        Elements with value 1 (True) are included in the sampling; elements with 0 (False) are excluded.

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
    mask : :class:`torch.Tensor` of :obj:`bool`, optional
        A binary (0/1) tensor indicating which elements to sample.
        Elements with value 1 (True) are included in the sampling; elements with 0 (False) are excluded.
    """

    def __init__(
        self,
        name: str,
        shape: Tuple[int, ...],
        *,
        acceptation_history_length: int = 25,
        mask: Optional[torch.Tensor] = None,
    ):
        super().__init__(
            name, shape, acceptation_history_length=acceptation_history_length
        )
        if self.ndim not in {1, 2}:
            # convention: shape of pop variable is 1D or 2D
            raise LeaspyModelInputError(
                "Dimension of population variable should be 1 or 2"
            )

        # TODO / WIP
        self.mask = mask
        if self.mask is not None:
            raise NotImplementedError(
                "WIP: Masked samplers are not supported yet with new variable interface / new weighted tensors "
                "--> cf. comments in LatentVariable & StatelessDistributionFamily"
            )
            if not isinstance(self.mask, torch.Tensor) or self.mask.shape != self.shape:
                raise LeaspyModelInputError(
                    f"Mask for sampler should be of shape {self.shape} but is of shape {self.mask.shape}"
                )
            self.mask = self.mask.to(torch.bool)
