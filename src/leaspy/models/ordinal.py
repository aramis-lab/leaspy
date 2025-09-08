from operator import itemgetter
from typing import Optional

import numpy as np
import torch

from leaspy.exceptions import LeaspyModelInputError
from leaspy.io.data.dataset import Dataset
from leaspy.models.base import InitializationMethod
from leaspy.models.obs_models import (
    FullGaussianObservationModel,
    observation_model_factory,
)
from leaspy.utils.distributions import compute_ordinal_pdf_from_ordinal_sf
from leaspy.utils.docs import doc_with_super
from leaspy.utils.functional import Exp
from leaspy.utils.typing import DictParams, DictParamsTorch, KwargsType
from leaspy.utils.weighted_tensor import (
    TensorOrWeightedTensor,
    WeightedTensor,
    unsqueeze_right,
)
from leaspy.variables.distributions import Normal
from leaspy.variables.specs import (
    LVL_FT,
    Hyperparameter,
    LinkedVariable,
    ModelParameter,
    NamedVariables,
    PopulationLatentVariable,
    VariableNameToValueMapping,
)
from leaspy.variables.state import State

from .logistic import LogisticModel


@doc_with_super(if_other_signature="force")
class OrdinalModel(LogisticModel):
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

        self.tracked_variables = self.tracked_variables.union(set(deltas))

    def initialize(self, dataset: Optional[Dataset] = None) -> None:
        """Overloads base model initialization (in particular to handle internal model State).

        <!> We do not put data variables in internal model state at this stage (done in algorithm)

        Parameters
        ----------
        dataset : :class:`~leaspy.io.data.dataset.Dataset`, optional
            Input dataset from which to initialize the model.
        """
        self.max_levels = dataset.get_max_levels()
        self.max_level = max(self.max_levels.values())
        super().initialize(dataset=dataset)

    def ordinal_time_reparametrization(
        self,
        *,
        t: TensorOrWeightedTensor[float],
        alpha: torch.Tensor,
        tau: torch.Tensor,
        deltas: torch.Tensor,
    ) -> TensorOrWeightedTensor[float]:
        """
        Tensorized time reparametrization formula.

        .. warning::
            Shapes of tensors must be compatible between them.

        Parameters
        ----------
        t : :class:`torch.Tensor`
            Timepoints to reparametrize
        alpha : :class:`torch.Tensor`
            Acceleration factors of individual(s)
        tau : :class:`torch.Tensor`
            Time-shift(s) of individual(s)

        Returns
        -------
        :class:`torch.Tensor`
            Reparametrized time of same shape as `timepoints`
        """
        reparametrized_time = alpha * (t - tau)
        reparametrized_time = unsqueeze_right(
            reparametrized_time, ndim=2
        )  # add dim of ordinal level
        t0 = torch.zeros((self.dimension, 1))
        deltas = torch.cat(
            (t0, deltas), dim=-1
        )  # Add zero for P(X >= 1), parametrized by standard leaspy
        deltas = deltas[None, None, ...]  # add (ind, tpts) dimensions
        reparametrized_time = reparametrized_time - deltas.cumsum(dim=-1)
        return reparametrized_time

    def _compute_initial_values_for_model_parameters(
        self,
        dataset: Dataset,
    ) -> VariableNameToValueMapping:
        """Compute initial values for model parameters and for the ordinal deltas parameters
        and initializes ordinal noise_model attributes.
        """
        parameters = super()._compute_initial_values_for_model_parameters(dataset)

        df = dataset.to_pandas(apply_headers=True)

        deltas = {}
        for feature, s in df.items():  # preserve feature order
            max_level = int(
                s.max()
            )  # possible levels not observed in calibration data do not exist for us
            # we do not model P >= 0 (since constant = 1)
            # we compute stats on P(Y >= k) in our data
            first_age_gte = {}
            for k in range(1, max_level + 1):
                s_gte_k = (s >= k).groupby("ID")
                first_age_gte[k] = (
                    s_gte_k.idxmax().map(itemgetter(1)).where(s_gte_k.any())
                )  # (ID, TIME) tuple -> TIME or nan
            # we do not need a delta for our anchor curve P >= 1
            # so we start at k == 2
            delays = [
                (first_age_gte[k] - first_age_gte[k - 1]).mean(skipna=True).item()
                for k in range(2, max_level + 1)
            ]
            deltas[feature] = torch.log(torch.clamp(torch.tensor(delays), min=0.1))

        # Should not be used yet
        # if self.batch_deltas:
        # we set the undefined deltas to be infinity to extend validity of formulas for them as well (and to avoid computations)
        deltas_ = float("inf") * torch.ones((len(deltas), self.max_level - 1))
        for i, name in enumerate(deltas):
            deltas_[i, : len(deltas[name])] = deltas[name]
        # parameters["log_deltas_mean"] = deltas_[:, 0]
        parameters["log_deltas_mean"] = deltas_
        return parameters

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
        del d["rt"]
        d.update(
            # PRIORS
            log_deltas_mean=ModelParameter.for_pop_mean(
                "log_deltas",
                shape=(self.dimension, self.max_level - 1),
                # shape=(self.dimension),
            ),
            log_deltas_std=Hyperparameter(0.1),
            # LATENT VARS
            log_deltas=PopulationLatentVariable(
                Normal("log_deltas_mean", "log_deltas_std"),
            ),
            # DERIVED VARS
            deltas=LinkedVariable(Exp("log_deltas")),
            # ordinal_rt=LinkedVariable(self.ordinal_reparametrized_time(rt=self.time_reparametrization, deltas="deltas")),
            rt=LinkedVariable(self.ordinal_time_reparametrization),
        )
        return d

    def model_with_sources(
        cls,
        *,
        rt: TensorOrWeightedTensor[float],
        space_shifts: TensorOrWeightedTensor[float],
        metric: TensorOrWeightedTensor[float],
        v0: TensorOrWeightedTensor[float],
        g: TensorOrWeightedTensor[float],
    ) -> torch.Tensor:
        """
        Return the model output when sources(spatial components) are present.

        Parameters
        ----------
        rt : TensorOrWeightedTensor[float]
            Tensor containing the reparametrized time.
        space_shifts : TensorOrWeightedTensor[float]
            Tensor containing the values of the space-shifts
        metric : TensorOrWeightedTensor[float]
            Tensor containing the metric tensor used for computing the spatial/temporal influence.
        v0 : TensorOrWeightedTensor[float]
            Tensor containing the values of the population parameter `v0` for each feature.
        g : TensorOrWeightedTensor[float]
            Tensor containing the values of the population parameter `g` for each feature.

        Returns
        -------
         :class:`torch.Tensor`
            Weighted value tensor after applying sigmoid transformation,
            representing the model output with sources.
        """
        # Shape: (Ni, Nt, Nfts)
        pass
