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

    name = "mcmc_saem"  # OLD: "MCMC_SAEM (tensor)"
