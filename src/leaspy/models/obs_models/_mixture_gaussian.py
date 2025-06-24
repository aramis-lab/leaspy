from __future__ import annotations

from typing import (
    Callable,
    Dict,
)

import torch

from leaspy.io.data.dataset import Dataset
from leaspy.models.utilities import compute_std_from_variance
from leaspy.utils.functional import Prod, Sqr, Identity, Sum
from leaspy.utils.weighted_tensor import (
    WeightedTensor,
    sum_dim,
    wsum_dim,
    wsum_dim_return_sum_of_weights_only,
    wsum_dim_return_weighted_sum_only,
)
from leaspy.variables.distributions import MixtureNormal, Normal
from leaspy.variables.specs import (
    LVL_FT,
    Collect,
    Hyperparameter,
    LinkedVariable,
    NamedVariables,
    ModelParameter,
    VariableInterface,
    VariableName,
)
from leaspy.variables.state import State

from ._base import ObservationModel

__all__ = [
    "MixtureGaussianObservationModel",
]

from ...utils.functional import Identity


class MixtureGaussianObservationModel(ObservationModel):
    """Specialized observational model when the data come from a mixture normal distribution."""

    tol_noise_variance = 1e-5

    def __init__(
        self,
        probs: VariableInterface,
        noise_std: VariableInterface,
        **extra_vars: VariableInterface,
    ):
        extra_vars = {
            **extra_vars,
            **{
                "noise_std": noise_std,
                "probs": probs,
            },
        }
        super().__init__(
            name="y",
            getter=self.y_getter,

            dist = Normal("model", "noise_std"),
            extra_vars=extra_vars,
        )

    ## dist=MixtureNormal("model", "noise_std", "probs"),
    @classmethod
    def individual_probabilities_update(
            cls,
            *,
            state: State, #probs_ind: torch.Tensor
            nll_cluster_ind: WeightedTensor[float],

    ) -> torch.Tensor:
        """Update rule for the individual probabilities of each individual i to belong to the cluster c.

        It uses the parts pf the likelihood corresponding to the data attachment and the attachment to the
        random effects, as calculated in the previous iteration.

        Parameters
        ----------
        state

        Returns
        -------
        probs_ind : a 2D tensor (n_individuals x n_clusters)
        with the corresponding probabilities of each individual i to belong to the cluster i
        """
        probs_ind = state['probs_ind'] #from the previous iteration
        #n_inds = probs_ind.size()[0]
        n_clusters = probs_ind.size()[1]
        #probs = probs_ind.sum(dim=0) / n_inds  # from the previous iteration
        #nll_ind = state["nll_attach_ind"]
        #nll_random = probs * (state["nll_regul_xi_ind"] + state["nll_regul_tau_ind"]+ state["nll_regul_sources_ind"])
        #nll_random = state['nll_regul_ind_sum_ind']

        denominator = nll_cluster_ind.sum(dim=1)  # sum for all the clusters
        nominator = nll_cluster_ind
        for c in range(n_clusters):
            probs_ind[:, c] = nominator[:, c] / denominator

        return probs_ind

    @classmethod
    def compute_cluster_probabilities(cls,
                                     *,
                                     nll_regul_ind_sum_ind: WeightedTensor[float], ) -> torch.Tensor:
        n_individuals = nll_regul_ind_sum_ind.shape[0]
        denominator = nll_regul_ind_sum_ind.sum(dim=1)  # sum for all the clusters
        nominator = nll_regul_ind_sum_ind
        probs_list = []

        for id_cluster in range(nominator.shape[1]):
            probs_ind_cluster = nominator[:, id_cluster] / denominator
            probs_list.append(probs_ind_cluster.value)

        probs_ind = torch.stack(probs_list, dim=1)

        return probs_ind.sum(dim=0) / n_individuals
    """
    @classmethod
    def probs_specs(cls):
        return LinkedVariable(cls.compute_cluster_probabilities)
    

    @classmethod
    def cluster_probabilities_update(cls,
                                     *,
                                     nll_cluster: torch.Tensor,) -> torch.Tensor:
        n_individuals = nll_cluster.shape[0]
        denominator = nll_cluster.sum(dim=1)  # sum for all the clusters
        nominator = nll_cluster
        probs_list = []

        for id_cluster in range(nominator.shape[1]):
            probs_ind_cluster = nominator[:, id_cluster] / denominator
            probs_list.append(probs_ind_cluster.value)

        probs_ind = torch.stack(probs_list, dim=1)

        return probs_ind.sum(dim=0) / n_individuals
    """

    @classmethod
    def probs_suff_stats(cls)-> Dict[VarName, LinkedVariable]:
        return dict(
            nll_cluster=LinkedVariable(Prod("probs","nll_attach_ind","nll_regul_ind_sum_ind")),
        )

    @classmethod
    def probs_specs(cls, n_clusters: int) -> ModelParameter:
        return ModelParameter(
            shape=(n_clusters,),
            #suff_stats=Collect(**cls.probs_suff_stats()),
            suff_stats = Collect("probs", "nll_attach_ind", "nll_attach_ind"),
            update_rule=cls.cluster_probabilities_update,
        )

    """
        
    @classmethod
    def probs_ind_suff_stats(cls) -> Dict[VarName, LinkedVariable]:
        return dict(
            nll_cluster_ind=LinkedVariable(Prod("probs", "nll_attach_ind", "nll_regul_ind_sum_ind")),
        )
    
    @classmethod
    def probs_ind_specs(cls,
                        #state: State, #probs_ind: torch.Tensor
                        #n_individuals : int,
                        #n_clusters : int
                        ) -> ModelParameter:

        #n_individuals = state['y'].shape[0]
        #return ModelParameter(
        #    shape = (n_individuals, n_clusters,),
        #    suff_stats= Collect(**cls.probs_ind_suff_stats()),
        #    update_rule=cls.individual_probabilities_update
        #)
        return IndividualLatentVariable(Categorical("probs"))


    def get_variables_specs(
            self,
            named_attach_vars: bool = True,
    ) -> Dict[VarName, VariableInterface]:
        
        d = super().get_variables_specs(named_attach_vars)
        d.update(
            probs_ind= self.probs_ind_specs()
        )

        return d
    
    @classmethod
    def probs_ind_specs(cls):
        return LinkedVariable(cls.compute_individual_probabilities)
    """

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
        """Update rule for feature-wise `noise_std` (when directly a model parameter), from state & sufficient statistics."""
        y_l2_per_ft = state["y_L2_per_ft"]
        n_obs_per_ft = state["n_obs_per_ft"]
        summed = sum_dim(-2 * y_x_model + model_x_model, but_dim=LVL_FT)
        noise_var = (y_l2_per_ft + summed) / n_obs_per_ft.float()
        return compute_std_from_variance(
            noise_var,
            varname="noise_std",
            tol=cls.tol_noise_variance,
        )

    @classmethod
    def noise_std_specs(cls, dimension: int) -> ModelParameter:
        """Default specifications of `noise_std` variable when directly modelled as a parameter (no latent population variable)."""
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
    def with_probs(cls, dimension: int, n_clusters: int):
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

        return cls(
            probs=cls.probs_specs(n_clusters),
            noise_std=cls.noise_std_specs(dimension),
            **extra_vars,
        )

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
