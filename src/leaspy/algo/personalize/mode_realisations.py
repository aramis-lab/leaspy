"""This module defines the `ModeReal` sampler based personalize algorithm."""

import torch

from leaspy.utils.typing import DictParamsTorch

from ..base import AlgorithmName
from .abstract_mcmc_personalize import AbstractMCMCPersonalizeAlgo

__all__ = ["ModeReal"]


class ModeReal(AbstractMCMCPersonalizeAlgo):
    """
    Sampler based algorithm, individual parameters are derived as the most frequent realization for `n_iter` samplings.

    TODO? we could derive some confidence intervals on individual parameters thanks to this personalization algorithm...

    TODO: harmonize naming in paths realiSation vs. realiZation...

    Parameters
    ----------
    settings : :class:`.AlgorithmSettings`
        Settings of the algorithm.
    """

    name: AlgorithmName = AlgorithmName.PERSONALIZE_MODE_REAL
    regularity_factor: float = 1.0
    """Weighting of regularity term in the final loss to be minimized."""

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
        realizations : :obj:`dict`[ind_var_name: str, `torch.Tensor[float]` of shape (n_iter, n_individuals, *ind_var.shape)]
            The stacked history of realizations for individual latent variables.
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
            ind_var_name: reals_var[indices_iter_best, indices_individuals]
            for ind_var_name, reals_var in realizations.items()
        }
