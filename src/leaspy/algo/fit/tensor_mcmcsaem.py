from ..factory import AlgorithmName
from .abstract_mcmc import AbstractFitMCMC

__all__ = ["TensorMCMCSAEM"]


class TensorMCMCSAEM(AbstractFitMCMC):
    """
    Main algorithm for MCMC-SAEM.

    Parameters
    ----------
    settings : :class:`.AlgorithmSettings`
        MCMC fit algorithm settings

    See Also
    --------
    :class:`.AbstractFitMCMC`
    """

    name: AlgorithmName = AlgorithmName.FIT_MCMC_SAEM
