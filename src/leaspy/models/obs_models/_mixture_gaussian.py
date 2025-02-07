from __future__ import annotations
from typing import (
    Dict,
    Callable,
)

import torch

from leaspy.models.utilities import compute_std_from_variance
from leaspy.variables.distributions import Categorical, Normal, MixtureNormal
from leaspy.utils.weighted_tensor import (
    WeightedTensor,
    sum_dim,
    wsum_dim_return_weighted_sum_only,
    wsum_dim_return_sum_of_weights_only,
    wsum_dim,
)
from leaspy.utils.functional import Sqr, Prod
from leaspy.variables.specs import (
    VarName,
    VariableInterface,
    LinkedVariable,
    ModelParameter,
    Collect,
    LVL_FT, Hyperparameter,
)
from leaspy.io.data.dataset import Dataset
from leaspy.variables.state import State

from ._gaussian import FullGaussianObservationModel

__all__ = [
    "MixtureGaussianObservationModel",
]

class MixtureGaussianObservationModel(FullGaussianObservationModel):
    """
    Specialized observational model when the data come from a mixture normal distribution.
    """

    def __init__(self, probs: VariableInterface, **extra_vars: VariableInterface):
        super().__init__(
            probs=probs,
            **extra_vars,
        )

    @classmethod
    def update_probs_ind(cls, *, state: State) -> torch.Tensor:
        """
        Update rule for the individual probabilities of each individual i to belong to the cluster c.
        It uses the parts pf the likelihood corresponding to the data attachment and the attachment to the
        random effects, as calculated in the previous iteration.
        ----------
        Parameters
        ----------
        state

        Returns
        -------
        probs_ind : a 2D tensor (n_individuals x n_clusters)
        with the corresponding probabilities of each individual i to belong to the cluster i
        """
        probs_ind = state['probs_ind'] #from the previous iteration
        n_inds = probs_ind.size()[0]
        n_clusters = probs_ind.size()[1]
        probs = probs_ind.sum(dim=0) / n_inds #from the previous iteration
        nll_ind = state['nll_attach_y']
        nll_random = probs * (
                state['nll_regul_xi_ind'] + state['nll_regul_tau_ind'] + state['nll_regul_sources_ind'])

        denominator = (probs * nll_ind * nll_random).sum(dim=1)  # sum for all the clusters
        nominator = probs * nll_ind * nll_random
        for c in range(n_clusters):
            probs_ind[:, c] = nominator[:, c] / denominator

        return probs_ind

    @classmethod
    def update_probs(cls) -> torch.Tensor:
        """
        Update rule for the probabilities of occurrence of each cluster.
        -------
        Returns
        -------
        probs : an 1D tensor (n_cluster)
        with the probabilities of occurrence of each cluster
        """
        probs_ind = cls.update_probs_ind
        n_inds = probs_ind.size()[0]
        probs = probs_ind.sum(dim=0) / n_inds

        return probs

    @classmethod
    def probs_specs(cls, n_clusters:int):
        """
        Default specifications for probs parameter.
        """
        update_rule = cls.update_probs

        return ModelParameter(
            shape = n_clusters,
            update_rule = update_rule
        )

    @classmethod
    def with_probs_as_model_parameter(cls, n_clusters:int):
        """
        Default instance of MixtureGaussianObservationModel
        """
        if not isinstance(n_clusters, int) or n_clusters < 2:
            raise ValueError(
                f"Number of clusters should be an integer >= 2. You provided {n_clusters}."
            )

        return cls(noise_std=cls.probs_specs(n_clusters))

    def to_string(self) -> str:
        """method for parameter saving"""
        return "mixture-gaussian"