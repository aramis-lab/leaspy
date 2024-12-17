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

from ._base import ObservationModel

class GaussianObservationModel(ObservationModel):
    """Specialized `ObservationModel` for noisy observations with Gaussian residuals assumption."""

    def __init__(
        self,
        name: VarName,
        getter: Callable[[Dataset], WeightedTensor],
        loc: VarName,
        scale: VarName,
        **extra_vars: VariableInterface,
    ):
        super().__init__(name, getter, Normal(loc, scale), extra_vars=extra_vars)

class MixtureGaussianObservationModel(GaussianObservationModel):
    """
    Specialized observational model when the data come from a mixture normal distribution.
    """

    tol_noise_variance = 1e-5

    def __init__(
            self,
            loc: VarName,
            scale: VarName,
            n_clusters: Hyperparameter,
            probs : VarName,
            **extra_vars: VariableInterface,
    ):

        super().__init__(
            name="cluster",
            getter=self.y_getter,
            n_clusters = n_clusters,
            dist = MixtureNormal(Categorical(probs),Normal(loc,scale)),
            **extra_vars,
        )

    @staticmethod
    def y_getter(dataset: Dataset) -> WeightedTensor:
        assert dataset.values is not None
        assert dataset.mask is not None
        return WeightedTensor(dataset.values, weight=dataset.mask.to(torch.bool))

    @classmethod
    def compute_probs_update(
            cls,
            *,
            state: State,
            n_clusters: int,
            n_inds : int,
            probs_ind=None) -> torch.Tensor : #tuple[torch.Tensor, torch.Tensor]:
        """
        Update rule for 'probs' from state
        probs_ind refers to the probability of each individual i to belong to each cluster c
        probs refers to the probability of each cluster c
        """

        nll_ind_per_cluster = state["nll_ind_per_cluster"]
        nll_random_per_cluster = state["nll_random_per_cluster"]

        for i in range(n_inds):
            denominator = 0
            for c in range(n_clusters):
                denominator = denominator + state["probs"][c] * nll_ind_per_cluster[i,c] * nll_random_per_cluster[i,c]

            for c in range(n_clusters):
                probs_ind[i,c] = state["probs"][c] * nll_ind_per_cluster[i,c] * nll_random_per_cluster[i,c] / denominator

        #probs = probs_ind.sum(dim=0)

        return probs_ind#, probs

    @classmethod
    def probs_specs(cls, n_clusters: int) -> LinkedVariable:
        """
        Default specifications of 'probs'.
        """
        return LinkedVariable(cls.compute_probs_update(n_clusters))

    @classmethod
    def with_probs_as_model_parameter(cls, n_clusters: int):
        """
        Default instance
        """
        if not isinstance(n_clusters, int) or n_clusters < 2:
            raise ValueError(f"Number of clusters should be an integer >=2. You provided {n_clusters}.")

        if n_clusters == 1 :
            extra_vars = {
                "y_L2_per_ft": LinkedVariable(Sqr("y").then(wsum_dim_return_weighted_sum_only, but_dim=LVL_FT)),
                "n_obs_per_ft": LinkedVariable(Sqr("y").then(wsum_dim_return_sum_of_weights_only, but_dim=LVL_FT)),
            }
        elif n_clusters >= 2 :
            extra_vars = {
                "y_L2_per_cluster_per_ft": LinkedVariable(
                    Sqr("y").then(wsum_dim_return_weighted_sum_only, but_dim=[n_clusters, LVL_FT])),
                "n_obs_per_cluster_per_ft": LinkedVariable(
                    Sqr("y").then(wsum_dim_return_sum_of_weights_only, but_dim=[n_clusters, LVL_FT])),
                # fix dim to lvl_clusters
            }

        return cls(probs=cls.probs_specs(n_clusters), **extra_vars)

    #to fix none of this really exists in state

    @classmethod
    def noise_std_suff_stats(cls) -> Dict[VarName, LinkedVariable]:
        """Dictionary of sufficient statistics needed for `noise_std` (when directly a model parameter)."""
        return dict(
            y_x_model=LinkedVariable(Prod("y", "model")),
            model_x_model=LinkedVariable(Sqr("model")),
        )

    # @classmethod #not used for now - use the mixture model only with the diagonal noise update
    # def scalar_noise_std_update(
    #        cls,
    #        *,
    #        state: State,
    #        y_x_model: WeightedTensor[float],
    #        model_x_model: WeightedTensor[float],
    # ) -> torch.Tensor:
    #    """Update rule for scalar `noise_std` (when directly a model parameter), from state & sufficient statistics."""
    #    y_l2 = state["y_L2"]
    #    n_obs = state["n_obs"]
    #    # TODO? by linearity couldn't we only require `-2*y_x_model + model_x_model` as summary stat?
    #    # and couldn't we even collect the already summed version of it?
    #    s1 = sum_dim(y_x_model)
    #    s2 = sum_dim(model_x_model)
    #    noise_var = (y_l2 - 2 * s1 + s2) / n_obs.float()
    #    return compute_std_from_variance(
    #        noise_var,
    #        varname="noise_std",
    #        tol=cls.tol_noise_variance,
    #    )

    @classmethod
    def diagonal_noise_std_update(
            cls,
            *,
            state: State,
            y_x_model: WeightedTensor[float],
            model_x_model: WeightedTensor[float],
    ) -> torch.Tensor:
        """
        Update rule for feature-wise `noise_std` (when directly a model parameter),
        from state & sufficient statistics.
        """
        y_l2_per_ft = state["y_L2_per_ft"]
        n_obs_per_ft = state["n_obs_per_ft"]
        # TODO: same remark as in `.scalar_noise_std_update()`
        # sum must be done after computation to use weights of y in model to mask missing data
        summed = sum_dim(- 2 * y_x_model + model_x_model, but_dim=LVL_FT)
        noise_var = (y_l2_per_ft + summed) / n_obs_per_ft.float()

        return compute_std_from_variance(
            noise_var,
            varname="noise_std",
            tol=cls.tol_noise_variance,
        )

    @classmethod
    def noise_std_specs(cls, dimension: int) -> ModelParameter:
        """
        Default specifications of `noise_std` variable when directly
        modelled as a parameter (no latent population variable).
        """
        update_rule = cls.diagonal_noise_std_update
        return ModelParameter(
            shape=(dimension,),
            suff_stats=Collect(**cls.noise_std_suff_stats()),
            update_rule=update_rule,
        )

    #@classmethod   #not used for now in the mixture model only in the full gaussian - leave here for reference
    #def with_noise_std_as_model_parameter(cls, dimension: int):
    #    """
    #    Default instance of `FullGaussianObservationModel` with `noise_std`
    #    (scalar or diagonal depending on `dimension`) being a `ModelParameter`.
    #    """
    #    if not isinstance(dimension, int) or dimension < 1:
    #        raise ValueError(f"Dimension should be an integer >= 1. You provided {dimension}.")
    #    if dimension == 1:
    #        extra_vars = {
    #            "y_L2": LinkedVariable(Sqr("y").then(wsum_dim_return_weighted_sum_only)),
    #            "n_obs": LinkedVariable(Sqr("y").then(wsum_dim_return_sum_of_weights_only)),
    #        }
    #    else:
    #        extra_vars = {
    #            "y_L2_per_ft": LinkedVariable(Sqr("y").then(wsum_dim_return_weighted_sum_only, but_dim=LVL_FT)),
    #            "n_obs_per_ft": LinkedVariable(Sqr("y").then(wsum_dim_return_sum_of_weights_only, but_dim=LVL_FT)),
    #        }
    #    return cls(noise_std=cls.noise_std_specs(dimension), **extra_vars)

    # Util functions not directly used in code

    @classmethod
    def compute_rmse(
        cls,
        *,
        y: WeightedTensor[float],
        model: WeightedTensor[float],
    ) -> torch.Tensor:
        """Compute root mean square error."""
        l2: WeightedTensor[float] = (model - y) ** 2
        l2_sum, n_obs = wsum_dim(l2)
        return (l2_sum / n_obs.float()) ** 0.5

    @classmethod
    def compute_rmse_per_ft(
        cls,
        *,
        y: WeightedTensor[float],
        model: WeightedTensor[float],
    ) -> torch.Tensor:
        """Compute root mean square error, per feature."""
        l2: WeightedTensor[float] = (model - y) ** 2
        l2_sum_per_ft, n_obs_per_ft = wsum_dim(l2, but_dim=LVL_FT)
        return (l2_sum_per_ft / n_obs_per_ft.float()) ** 0.5

    def to_string(self) -> str:
        """method for parameter saving"""
        return "mixture-gaussian"



