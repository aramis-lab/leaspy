"""This module defines the `MeanPosterior` sampler based personalize algorithm."""

import torch

from leaspy.utils.typing import DictParamsTorch

from ..base import AlgorithmName
from .abstract_mcmc_personalize import AbstractMCMCPersonalizeAlgo

__all__ = ["MeanPosterior"]


class MeanPosterior(AbstractMCMCPersonalizeAlgo):
    """
    Sampler based algorithm, individual parameters are derived as the mean posterior for `n_iter` samplings.

    Parameters
    ----------
    settings : :class:`.AlgorithmSettings`
        Settings of the algorithm.
    """

    name: AlgorithmName = AlgorithmName.PERSONALIZE_MEAN_POSTERIOR

    def _compute_individual_parameters_from_samples_torch(
        self,
        values: DictParamsTorch,
        attachments: torch.Tensor,
        regularities: torch.Tensor,
    ) -> DictParamsTorch:
        """
        Compute dictionary of individual parameters from stacked values, attachments and regularities.

        Parameters
        ----------
        values : dict[ind_var_name: str, `torch.Tensor[float]` of shape (n_iter, n_individuals, *ind_var.shape)]
            The stacked history of values for individual latent variables.
        attachments : `torch.Tensor[float]` of shape (n_iter, n_individuals)
            The stacked history of attachments (per individual).
        regularities : `torch.Tensor[float]` of shape (n_iter, n_individuals)
            The stacked history of regularities (per individual; but summed on all individual variables and all of their dimensions).

        Returns
        -------
        dict[ind_var_name: str, `torch.Tensor[float]` of shape (n_individuals, *ind_var.shape)]
        """
        # Only compute the mean of posterior values (attachments & regularities not taken into account)
        return {
            ind_var_name: value_var.mean(dim=0)
            for ind_var_name, value_var in values.items()
        }
