import pandas as pd
import torch
from abc import abstractmethod
from operator import itemgetter

from typing import Iterable, Optional
from leaspy.utils.weighted_tensor import WeightedTensor, TensorOrWeightedTensor
from leaspy.models.base import InitializationMethod
from leaspy.models.multivariate import LogisticMultivariateModel
from leaspy.models.utils.ordinal import OrdinalModelMixin
from leaspy.io.data.dataset import Dataset

from leaspy.utils.docs import doc_with_super

from leaspy.variables.state import State
from leaspy.variables.specs import (
    NamedVariables,
    ModelParameter,
    PopulationLatentVariable,
    LinkedVariable,
    Hyperparameter,
    SuffStatsRW,
    VariablesValuesRO,
)
from leaspy.variables.distributions import Normal
from leaspy.utils.functional import Exp, Sqr, OrthoBasis
from leaspy.utils.weighted_tensor import unsqueeze_right

from leaspy.models.obs_models import FullGaussianObservationModel


# TODO refact? implement a single function
# compute_individual_tensorized(..., with_jacobian: bool) -> returning either
# model values or model values + jacobians wrt individual parameters

# TODO refact? subclass or other proper code technique to extract model's concrete
#  formulation depending on if linear, logistic, mixed log-lin, ...


@doc_with_super()
class OrdinalMultivariateModel(LogisticMultivariateModel, OrdinalModelMixin):
    """
    Manifold model for multiple variables of interest (logistic or linear formulation).

    Parameters
    ----------
    name : :obj:`str`
        The name of the model.
    **kwargs
        Hyperparameters of the model (including `noise_model`)

    Raises
    ------
    :exc:`.LeaspyModelInputError`
        * If hyperparameters are inconsistent
    """

    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)

        deltas = ["deltas"]

        # Remove the batch deltas option : should be default
        #if self.batch_deltas:
        #    deltas = ["deltas"]
        #else:
        #    deltas = [f"deltas_{ft}" for ft, ft_max_level in self.noise_model.max_levels.items()]

        self.tracked_variables = self.tracked_variables.union(set(deltas))

    def ordinal_reparametrized_time(self,
                                    *,
                                    rt: TensorOrWeightedTensor[float],
                                    deltas: torch.Tensor,
                                    ):
        reparametrized_time = unsqueeze_right(rt, ndim=2) # add dim of ordinal level
        t0 = torch.zeros((self.dimension, 1))
        deltas = torch.cat((t0, deltas), dim=-1) # Add zero for P(X >= 1), parametrized by standard leaspy
        deltas = deltas[None, None, ...] #add (ind, tpts) dimensions
        reparametrized_time = reparametrized_time - deltas.cumsum(dim=-1)
        return reparametrized_time

    def get_variables_specs(self) -> NamedVariables:
        """
        Return the specifications of the variables (latent variables, derived variables,
        model 'parameters') that are part of the model.

        Returns
        -------
        NamedVariables :
            The specifications of the model's variables.
        """
        d = super().get_variables_specs()
        d.update(
            # PRIORS
            log_deltas_mean=ModelParameter.for_pop_mean(
                "log_deltas",
                shape=(self.dimension,self.obs_models[0].max_level - 1),
            ),
            log_deltas_std=Hyperparameter(0.1),
            # LATENT VARS
            log_deltas=PopulationLatentVariable(
                Normal("log_deltas_mean", "log_deltas_std"),
                #sampling_kws={"mask": self.obs_models[0]._mask}, # does not work yet, WIP in samplers
            ),
            # DERIVED VARS
            deltas=LinkedVariable(
                Exp("log_deltas"),
            ),
            ordinal_rt=LinkedVariable(self.ordinal_reparametrized_time),
        )

        # For not batched option, need to concatenate all the deltas
        #if not self.batch_deltas:
        #    d.update(
        #        deltas=LinkedVariable( ??? )
        #    )

        # TODO: WIP
        #variables_info.update(self.get_additional_ordinal_population_random_variable_information())
        #self.update_ordinal_population_random_variable_information(variables_info)

        return d

    @classmethod
    def model_no_sources(cls, *, ordinal_rt: torch.Tensor, metric, v0, g) -> torch.Tensor:
        """Returns a model without source. A bit dirty?"""
        return cls.model_with_sources(
            ordinal_rt=ordinal_rt,
            metric=metric,
            v0=v0,
            g=g,
            space_shifts=torch.zeros((1, 1)),
        )

    @staticmethod
    def metric(*, g: torch.Tensor) -> torch.Tensor:
        """Used to define the corresponding variable."""
        return (g + 1) ** 2 / g

    @classmethod
    def model_with_sources(
        cls,
        *,
        ordinal_rt: TensorOrWeightedTensor[float],
        space_shifts: TensorOrWeightedTensor[float],
        metric: TensorOrWeightedTensor[float],
        v0: TensorOrWeightedTensor[float],
        g: TensorOrWeightedTensor[float],
    ) -> torch.Tensor:
        """Returns a model with sources."""
        # Shape: (Ni, Nt, Nfts)
        pop_s = (None, None, ..., None)
        w_model_logit = metric[pop_s] * (v0[pop_s] * ordinal_rt + space_shifts[:, None, ..., None]) - torch.log(g[pop_s])
        model_logit, weights = WeightedTensor.get_filled_value_and_weight(w_model_logit, fill_value=0.)
        model = torch.sigmoid(model_logit).nan_to_num(0.0) # Fill nan with 0.
        return WeightedTensor(model, weights).weighted_value

    def initialize(self, dataset: Optional[Dataset] = None, method: Optional[InitializationMethod] = None) -> None:

        # Need to initialize ordinal_infos to create proper DAG
        self.compute_ordinal_infos(dataset)
        super().initialize(dataset, method)

    def _compute_initial_values_for_model_parameters(
            self,
            dataset: Dataset,
            method: InitializationMethod,
    ) -> VariablesValuesRO:
        """Compute initial values for model parameters and for the ordinal deltas parameters
        and initializes ordinal noise_model attributes.
        """
        parameters = super()._compute_initial_values_for_model_parameters(dataset, method)
        df = self._get_dataframe_from_dataset(dataset)

        deltas = {}
        for ft, s in df.items():  # preserve feature order
            max_lvl = int(s.max())  # possible levels not observed in calibration data do not exist for us
            # we do not model P >= 0 (since constant = 1)
            # we compute stats on P(Y >= k) in our data
            first_age_gte = {}
            for k in range(1, max_lvl + 1):
                s_gte_k = (s >= k).groupby('ID')
                first_age_gte[k] = s_gte_k.idxmax().map(itemgetter(1)).where(
                    s_gte_k.any())  # (ID, TIME) tuple -> TIME or nan
            # we do not need a delta for our anchor curve P >= 1
            # so we start at k == 2
            delays = [(first_age_gte[k] - first_age_gte[k - 1]).mean(skipna=True).item()
                      for k in range(2, max_lvl + 1)]
            deltas[ft] = torch.log(torch.clamp(torch.tensor(delays), min=0.1))

        # Should not be used yet
        #if self.batch_deltas:
        # we set the undefined deltas to be infinity to extend validity of formulas for them as well (and to avoid computations)
        deltas_ = float('inf') * torch.ones((len(deltas), self.obs_models[0].max_level - 1))
        for i, name in enumerate(deltas):
            deltas_[i, :len(deltas[name])] = deltas[name]
        parameters["log_deltas_mean"] = deltas_
        #else:
         #   for ft in self.features:
          #      parameters["deltas_" + ft] = deltas[ft]
        return parameters
