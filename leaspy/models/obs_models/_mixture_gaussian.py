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
    def with_probs(cls, n_clusters: int):
        """
        Default instance
        """

        if not isinstance(n_clusters, int) or n_clusters >= 2:
            raise ValueError(f"Number of clusters should be an integer >=2. You provided {n_clusters}.")

        for c in range(n_clusters):
            extra_vars = {
                "y_L2_per_cluster": LinkedVariable(Sqr("y").then(wsum_dim_return_weighted_sum_only, but_dim=LVL_FT)),
                "n_obs_per_cluster": LinkedVariable(Sqr("y").then(wsum_dim_return_sum_of_weights_only, but_dim=LVL_FT)),
                # fix dim to lvl_clusters
            }

        return cls(**extra_vars)

    #to fix none of this really exists in state

    def to_string(self) -> str:
        """method for parameter saving"""
        return "mixture-gaussian"
"""
    @classmethod
    def compute_probs_update(
            cls,
            *,
            state: State,
            n_clusters: int,
            n_inds: int,
            probs_ind=None) -> torch.Tensor:  # tuple[torch.Tensor, torch.Tensor]:
        

        nll_ind_per_cluster = state["nll_ind_per_cluster"]
        nll_random_per_cluster = state["nll_random_per_cluster"]

        for i in range(n_inds):
            denominator = 0
            for c in range(n_clusters):
                denominator = denominator + state["probs"][c] * nll_ind_per_cluster[i, c] * nll_random_per_cluster[i, c]

            for c in range(n_clusters):
                probs_ind[i, c] = state["probs"][c] * nll_ind_per_cluster[i, c] * nll_random_per_cluster[
                    i, c] / denominator

        # probs = probs_ind.sum(dim=0)

        return probs_ind  # , probs

    @classmethod
    def probs_specs(cls, n_clusters: int) -> LinkedVariable:
    return LinkedVariable(cls.compute_probs_update(n_clusters))
    """



