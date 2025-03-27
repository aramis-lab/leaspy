"""This module defines the `ModePost` sampler based personalize algorithm."""

import torch

from leaspy.utils.typing import DictParamsTorch

from ..base import AlgorithmName
from .abstract_mcmc_personalize import AbstractMCMCPersonalizeAlgo

__all__ = ["ModePost"]


class ModePost(AbstractMCMCPersonalizeAlgo):
    """
    Sampler based algorithm, individual parameters are derived as the most frequent posterior value for `n_iter` samplings.

    TODO? we could derive some confidence intervals on individual parameters thanks to this personalization algorithm...

    Parameters
    ----------
    settings : :class:`.AlgorithmSettings`
        Settings of the algorithm.
    """

    name: AlgorithmName = AlgorithmName.PERSONALIZE_MODE_POST
    regularity_factor: float = 1.0
    """Weighting of regularity term in the final loss to be minimized."""

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
        values : :obj:`dict`[ind_var_name: str, `torch.Tensor[float]` of shape (n_iter, n_individuals, *ind_var.shape)]
            The stacked history of values for individual latent variables.
        attachments : `torch.Tensor[float]` of shape (n_iter, n_individuals)
            The stacked history of attachments (per individual).
        regularities : `torch.Tensor[float]` of shape (n_iter, n_individuals)
            The stacked history of regularities (per individual; but summed on all individual variables and all of their dimensions).

        Returns
        -------
        :obj:`dict`[ind_var_name: :obj:`str`, `torch.Tensor[float]` of shape (n_individuals, *ind_var.shape)]
        """
        # Indices of iterations where loss (= negative log-likelihood) was minimal
        # (per individual, but tradeoff on ALL individual parameters)
        indices_iter_best = torch.argmin(
            attachments + self.regularity_factor * regularities, dim=0
        )  # shape (n_individuals,)
        indices_individuals = torch.arange(len(indices_iter_best))
        return {
            ind_var_name: value_var[indices_iter_best, indices_individuals]
            for ind_var_name, value_var in values.items()
        }
