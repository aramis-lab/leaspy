import torch

from leaspy.utils.typing import DictParamsTorch

from .abstract_mcmc_personalize import AbstractMCMCPersonalizeAlgo

__all__ = ["MeanReal"]


class MeanReal(AbstractMCMCPersonalizeAlgo):
    """
    Sampler based algorithm, individual parameters are derived as the mean realization for `n_iter` samplings.

    TODO: harmonize naming in paths realiSation vs. realiZation...

    Parameters
    ----------
    settings : :class:`.AlgorithmSettings`
        Settings of the algorithm.
    """

    name = "mean_real"

    def _compute_individual_parameters_from_samples_torch(
        self,
        realizations: DictParamsTorch,
        attachments: torch.Tensor,
        regularities: torch.Tensor,
    ) -> DictParamsTorch:
        """
        Compute dictionary of individual parameters from stacked realizations, attachments and regularities.

        Parameters
        ----------
        realizations : dict[ind_var_name: str, `torch.Tensor[float]` of shape (n_iter, n_individuals, *ind_var.shape)]
            The stacked history of realizations for individual latent variables.
        attachments : `torch.Tensor[float]` of shape (n_iter, n_individuals)
            The stacked history of attachments (per individual).
        regularities : `torch.Tensor[float]` of shape (n_iter, n_individuals)
            The stacked history of regularities (per individual; but summed on all individual variables and all of their dimensions).

        Returns
        -------
        dict[ind_var_name: str, `torch.Tensor[float]` of shape (n_individuals, *ind_var.shape)]
        """
        # Only compute the mean of realizations (attachments & regularities not taken into account)
        return {
            ind_var_name: reals_var.mean(dim=0)
            for ind_var_name, reals_var in realizations.items()
        }
