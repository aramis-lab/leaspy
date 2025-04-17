"""This module defines the `TensorMCMCSAEM` class."""

from ..base import AlgorithmName
from .abstract_mcmc import AbstractFitMCMC

__all__ = ["TensorMCMCSAEM"]


class TensorMCMCSAEM(AbstractFitMCMC):
    """
    Main algorithm for MCMC-SAEM.

    Parameters
    ----------
    settings : :class:`~leaspy.algo.AlgorithmSettings`
        MCMC fit algorithm settings

    See Also
    --------
    :class:`~leaspy.algo.fit.AbstractFitMCMC`
    """

    name: AlgorithmName = AlgorithmName.FIT_MCMC_SAEM
