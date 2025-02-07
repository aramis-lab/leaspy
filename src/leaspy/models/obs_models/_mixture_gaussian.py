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
            name="y",
            getter=self.y_getter,
            loc="model",
            scale="noise_std",
            noise_std=noise_std,
            probs=probs,
            **extra_vars,
        )

    @classmethod
    def compute_probs_ind(cls, *, state: State) -> torch.Tensor:
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
    def compute_probs(cls) -> torch.Tensor:
        """
        Update rule for the probabilities of occurrence of each cluster.
        -------
        Returns
        -------
        probs : an 1D tensor (n_cluster)
        with the probabilities of occurrence of each cluster
        """
        probs_ind = cls.compute_probs_ind
        n_inds = probs_ind.size()[0]
        probs = probs_ind.sum(dim=0) / n_inds

        return probs

    @classmethod
    def probs_specs(cls, n_clusters:int):
        """
        Default specifications for probs parameter.
        """

        return LinkedVariable(cls.compute_probs)

    @staticmethod
    def y_getter(dataset: Dataset) -> WeightedTensor:
        assert dataset.values is not None
        assert dataset.mask is not None
        return WeightedTensor(dataset.values, weight=dataset.mask.to(torch.bool))

    @classmethod
    def noise_std_suff_stats(cls) -> Dict[VarName, LinkedVariable]:
        """Dictionary of sufficient statistics needed for `noise_std` (when directly a model parameter)."""
        return dict(
            y_x_model=LinkedVariable(Prod("y", "model")),
            model_x_model=LinkedVariable(Sqr("model")),
        )

    @classmethod
    def scalar_noise_std_update(
            cls,
            *,
            state: State,
            y_x_model: WeightedTensor[float],
            model_x_model: WeightedTensor[float],
    ) -> torch.Tensor:
        """Update rule for scalar `noise_std` (when directly a model parameter), from state & sufficient statistics."""
        y_l2 = state["y_L2"]
        n_obs = state["n_obs"]
        # TODO? by linearity couldn't we only require `-2*y_x_model + model_x_model` as summary stat?
        # and couldn't we even collect the already summed version of it?
        s1 = sum_dim(y_x_model)
        s2 = sum_dim(model_x_model)
        noise_var = (y_l2 - 2 * s1 + s2) / n_obs.float()
        return compute_std_from_variance(
            noise_var,
            varname="noise_std",
            tol=cls.tol_noise_variance,
        )

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
        summed = sum_dim(-2 * y_x_model + model_x_model, but_dim=LVL_FT)
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
        update_rule = (
            cls.scalar_noise_std_update
            if dimension == 1
            else cls.diagonal_noise_std_update
        )
        return ModelParameter(
            shape=(dimension,),
            suff_stats=Collect(**cls.noise_std_suff_stats()),
            update_rule=update_rule,
        )

    # Util functions not directly used in code

    @classmethod
    def with_probs(cls, dimension: int, n_clusters:int):
        """
        Default instance of MixtureGaussianObservationModel
        """
        if not isinstance(n_clusters, int) or n_clusters < 2:
            raise ValueError(
                f"Number of clusters should be an integer >= 2. You provided {n_clusters}."
            )

        if not isinstance(dimension, int) or dimension < 1:
            raise ValueError(
                f"Dimension should be an integer >= 1. You provided {dimension}."
            )
        if dimension == 1:
            extra_vars = {
                "y_L2": LinkedVariable(
                    Sqr("y").then(wsum_dim_return_weighted_sum_only)
                ),
                "n_obs": LinkedVariable(
                    Sqr("y").then(wsum_dim_return_sum_of_weights_only)
                ),
            }
        else:
            extra_vars = {
                "y_L2_per_ft": LinkedVariable(
                    Sqr("y").then(wsum_dim_return_weighted_sum_only, but_dim=LVL_FT)
                ),
                "n_obs_per_ft": LinkedVariable(
                    Sqr("y").then(wsum_dim_return_sum_of_weights_only, but_dim=LVL_FT)
                ),
            }

        return cls(probs=cls.probs_specs(n_clusters), noise_std=cls.noise_std_specs(dimension), **extra_vars)

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