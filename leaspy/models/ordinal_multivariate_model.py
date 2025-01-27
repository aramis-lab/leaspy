import torch
import numpy as np
from operator import itemgetter

from typing import Optional, Union, Hashable
from leaspy.utils.weighted_tensor import WeightedTensor, TensorOrWeightedTensor
from leaspy.models.base import InitializationMethod
from leaspy.models.multivariate import LogisticMultivariateModel
from leaspy.io.data.dataset import Dataset
from leaspy.utils.distributions import compute_ordinal_pdf_from_ordinal_sf
from leaspy.utils.docs import doc_with_super

from leaspy.variables.specs import (
    NamedVariables,
    ModelParameter,
    PopulationLatentVariable,
    LinkedVariable,
    Hyperparameter,
    VariablesValuesRO,
)
from leaspy.variables.distributions import Normal
from leaspy.utils.functional import Exp
from leaspy.utils.weighted_tensor import unsqueeze_right
from enum import Enum, auto
from leaspy.utils.typing import DictParamsTorch, DictParams, KwargsType
from leaspy.exceptions import LeaspyModelInputError


class OrdinalMethod(Enum):
    """Possible ordinal methods.

    - maximum_likelihood: maximum likelihood estimator for each point (int)
    - expectation
    - probabilities returns probabilities of all-possible levels for a given feature: {feature_name: array[float]<0..max_level_ft>}
    """

    MAXIMUM_LIKELIHOOD = auto()
    EXPECTATION = auto()
    PROBABILITIES = auto()


@doc_with_super()
class OrdinalMultivariateModel(LogisticMultivariateModel):
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

    def __init__(self, name: str, max_levels: Optional[dict[str, int]] = None, **kwargs):
        super().__init__(name, **kwargs)
        #deltas = ["deltas"]
        self.tracked_variables.add("deltas")
        self.max_levels: dict[str, int] = max_levels or {}
        #self.mask: Optional[torch.Tensor] = None

    @property
    def max_level(self) -> int:
        if self.max_levels:
            return max(self.max_levels.values())
        return 0

    def to_dict(self, *, with_mixing_matrix: bool = True) -> KwargsType:
        model_settings = super().to_dict(with_mixing_matrix=with_mixing_matrix)
        model_settings["max_levels"] = self.max_levels
        return model_settings

    def ordinal_reparametrized_time(
        self,
        *,
        rt: TensorOrWeightedTensor[float],
        deltas: torch.Tensor,
    ) -> WeightedTensor[float]:
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
                shape=(self.dimension, self.max_level - 1),
            ),
            log_deltas_std=Hyperparameter(0.1),
            # LATENT VARS
            log_deltas=PopulationLatentVariable(
                Normal("log_deltas_mean", "log_deltas_std"),
                #sampling_kws={"mask": self.obs_models[0]._mask}, # does not work yet, WIP in samplers
            ),
            # DERIVED VARS
            deltas=LinkedVariable(Exp("log_deltas")),
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
        self.max_levels = dataset.get_max_levels()
        #self.mask = dataset.get_mask()
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
        for feature, s in df.items():  # preserve feature order
            max_level = int(s.max())  # possible levels not observed in calibration data do not exist for us
            # we do not model P >= 0 (since constant = 1)
            # we compute stats on P(Y >= k) in our data
            first_age_gte = {}
            for k in range(1, max_level + 1):
                s_gte_k = (s >= k).groupby('ID')
                first_age_gte[k] = s_gte_k.idxmax().map(itemgetter(1)).where(
                    s_gte_k.any())  # (ID, TIME) tuple -> TIME or nan
            # we do not need a delta for our anchor curve P >= 1
            # so we start at k == 2
            delays = [
                (first_age_gte[k] - first_age_gte[k - 1]).mean(skipna=True).item()
                for k in range(2, max_level + 1)
            ]
            deltas[feature] = torch.log(torch.clamp(torch.tensor(delays), min=0.1))

        # Should not be used yet
        #if self.batch_deltas:
        # we set the undefined deltas to be infinity to extend validity of formulas for them as well (and to avoid computations)
        deltas_ = float('inf') * torch.ones((len(deltas), self.max_level - 1))
        for i, name in enumerate(deltas):
            deltas_[i, :len(deltas[name])] = deltas[name]
        parameters["log_deltas_mean"] = deltas_
        return parameters

    def postprocess_model_estimation(
        self,
        estimation: np.ndarray,
        *,
        ordinal_method: Optional[OrdinalMethod] = None,
        **kws,
    ) -> Union[np.ndarray, dict[Hashable, np.ndarray]]:
        """
        Extra layer of processing used to output nice estimated values in main API `Leaspy.estimate`.

        Parameters
        ----------
        estimation : numpy.ndarray[float]
            The raw estimated values by model (from `compute_individual_trajectory`)
        ordinal_method : OrdinalMethod, optional
            Default = Maximum Likelihood
        **kws
            Some extra keywords arguments that may be handled in the future.

        Returns
        -------
        numpy.ndarray[float] or dict[str, numpy.ndarray[float]]
            Post-processed values.
            In case using 'probabilities' mode, the values are a dictionary with keys being:
            `(feature_name: str, feature_level: int<0..max_level_for_feature>)`
            Otherwise it is a standard numpy.ndarray corresponding to different model features (in order)
        """
        ordinal_method = ordinal_method or OrdinalMethod.MAXIMUM_LIKELIHOOD
        if ordinal_method == OrdinalMethod.MAXIMUM_LIKELIHOOD:
            return estimation.argmax(axis=-1)
        if ordinal_method == OrdinalMethod.EXPECTATION:
            return np.flip(estimation, axis=-1).cumsum(axis=-1).sum(axis=-1) - 1.0
        if ordinal_method == OrdinalMethod.PROBABILITIES:
            d_ests = {}
            for ft_i, (ft, ft_max_level) in enumerate(self.max_levels.items()):
                for ft_lvl in range(ft_max_level + 1):
                    d_ests[(ft, ft_lvl)] = estimation[..., ft_i, ft_lvl]
            return d_ests

    def compute_appropriate_ordinal_model(
        self, model_or_model_grad: torch.Tensor
    ) -> torch.Tensor:
        """Post-process model values (or their gradient) if needed."""
        return compute_ordinal_pdf_from_ordinal_sf(model_or_model_grad)

    # Not working anymore
    def _ordinal_grid_search_value(
        self,
        grid_timepoints: torch.Tensor,
        values: torch.Tensor,
        *,
        individual_parameters: DictParamsTorch,
        feat_index: int,
    ) -> torch.Tensor:
        """Search first timepoint where ordinal MLE is >= provided values."""
        grid_model = self.compute_individual_tensorized_logistic(
            grid_timepoints.unsqueeze(0), individual_parameters, attribute_type=None
        )[:, :, [feat_index], :]

        if self.is_ordinal_ranking:
            grid_model = compute_ordinal_pdf_from_ordinal_sf(grid_model)

        # we search for the very first timepoint of grid where ordinal MLE was >= provided value
        # TODO? shouldn't we return the timepoint where P(X = value) is highest instead?
        MLE = grid_model.squeeze(dim=2).argmax(
            dim=-1
        )  # squeeze feat_index (after computing pdf when needed)
        index_cross = (MLE.unsqueeze(1) >= values.unsqueeze(-1)).int().argmax(dim=-1)

        return grid_timepoints[index_cross]

    # Not good anymore
    def _check_ordinal_parameters_consistency(self) -> None:
        """Check consistency of ordinal model parameters."""
        deltas_p = {k: v for k, v in self.parameters.items() if k.startswith("deltas")}
        if deltas_p.keys() != {"deltas"}:
            raise LeaspyModelInputError(
                f"Unexpected delta parameters. Expected 'deltas' but got {deltas_p.keys()}"
            )
        if self.max_levels is None:
            raise LeaspyModelInputError("Your ordinal noise model is incomplete (missing `max_levels`).")
        deltas = self.parameters["deltas"]
        if deltas.shape != (self.dimension, self.max_level - 1):
            raise LeaspyModelInputError("Shape of deltas is inconsistent with noise model.")
        if not torch.equal(
            torch.isinf(self.parameters["deltas"]), ~self.mask[:, 1:].bool()
        ):
            raise LeaspyModelInputError("Mask on deltas is inconsistent with noise model.")


    def get_additional_ordinal_population_random_variable_information(self) -> DictParams:
        """Return the information of additional population random variables for the ordinal model."""
        # Nota for shapes: the >= level-0 is not included (always = 1)
        return {
            "deltas": {
                "name": "deltas",
                "shape": torch.Size([self.dimension, self.max_level - 1]),
                "rv_type": "multigaussian",
                "mask": self.mask[:, 1:],
                "scale": 0.5,
            }
        }
