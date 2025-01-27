from leaspy.algo.fit.abstract_mcmc import AbstractFitMCMC


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
