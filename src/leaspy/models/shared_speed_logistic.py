import torch

from leaspy.io.data.dataset import Dataset
from leaspy.utils.functional import Exp, OrthoBasis
from leaspy.utils.weighted_tensor import (
    TensorOrWeightedTensor,
    WeightedTensor,
    unsqueeze_right,
)
from leaspy.variables.distributions import Normal
from leaspy.variables.specs import (
    Hyperparameter,
    LinkedVariable,
    ModelParameter,
    NamedVariables,
    PopulationLatentVariable,
    VariableNameToValueMapping,
)

from .logistic import LogisticInitializationMixin
from .time_reparametrized import TimeReparametrizedModel

__all__ = ["SharedSpeedLogisticModel"]


class SharedSpeedLogisticModel(LogisticInitializationMixin, TimeReparametrizedModel):
    """
    Logistic model for multiple variables of interest, imposing same average
    evolution pace for all variables (logistic curves are only time-shifted).

    Parameters
    ----------
    name : :obj:`str`
        The name of the model.
    **kwargs
        Hyperparameters of the model.
    """

    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)

    def _compute_initial_values_for_model_parameters(
        self,
        dataset: Dataset,
    ) -> VariableNameToValueMapping:
        parameters = super()._compute_initial_values_for_model_parameters(dataset)
        parameters["log_g_mean"] = parameters["log_g_mean"].mean()
        parameters["xi_mean"] = parameters["log_v0_mean"].mean()
        del parameters["log_v0_mean"]
        parameters["deltas_mean"] = torch.zeros((self.dimension - 1,))
        return parameters

    @staticmethod
    def metric(*, g_deltas_exp: torch.Tensor) -> torch.Tensor:
        """Used to define the corresponding variable."""
        return (g_deltas_exp + 1) ** 2 / g_deltas_exp

    @staticmethod
    def deltas_exp(*, deltas_padded: torch.Tensor) -> torch.Tensor:
        return torch.exp(-1 * deltas_padded)

    @staticmethod
    def g_deltas_exp(*, g: torch.Tensor, deltas_exp: torch.Tensor) -> torch.Tensor:
        return g * deltas_exp

    @staticmethod
    def pad_deltas(*, deltas: torch.Tensor) -> torch.Tensor:
        """Prepend deltas with a zero as delta_1 is set to zero in the equations."""
        return torch.cat((torch.tensor([0.0]), deltas))

    @staticmethod
    def denom(*, g_deltas_exp: torch.Tensor) -> torch.Tensor:
        return 1 + g_deltas_exp

    @staticmethod
    def gamma_t0(*, denom: torch.Tensor) -> torch.Tensor:
        return 1 / denom

    @staticmethod
    def g_metric(*, gamma_t0: torch.Tensor) -> torch.Tensor:
        return 1 / (gamma_t0 * (1 - gamma_t0)) ** 2

    @staticmethod
    def collin_to_d_gamma_t0(
        *, deltas_exp: torch.Tensor, denom: torch.Tensor
    ) -> torch.Tensor:
        return deltas_exp / denom**2

    @classmethod
    def model_with_sources(
        cls,
        *,
        rt: TensorOrWeightedTensor[float],
        space_shifts: TensorOrWeightedTensor[float],
        metric: TensorOrWeightedTensor[float],
        deltas_padded: TensorOrWeightedTensor[float],
        log_g: TensorOrWeightedTensor[float],
    ) -> torch.Tensor:
        """Returns a model with sources."""
        # Shape: (Ni, Nt, Nfts)
        pop_s = (None, None, ...)
        rt = unsqueeze_right(rt, ndim=1)  # .filled(float('nan'))
        w_model_logit = (
            metric[pop_s] * space_shifts[:, None, ...]
            + rt
            + deltas_padded
            - log_g[pop_s]
        )
        model_logit, weights = WeightedTensor.get_filled_value_and_weight(
            w_model_logit, fill_value=0.0
        )
        return WeightedTensor(torch.sigmoid(model_logit), weights).weighted_value

    @classmethod
    def model_no_sources(
        cls,
        *,
        rt: TensorOrWeightedTensor[float],
        metric: TensorOrWeightedTensor[float],
        deltas_padded: TensorOrWeightedTensor[float],
        log_g: TensorOrWeightedTensor[float],
    ) -> torch.Tensor:
        """Returns a model without source. A bit dirty?"""
        return cls.model_with_sources(
            rt=rt,
            metric=metric,
            deltas_padded=deltas_padded,
            log_g=log_g,
            space_shifts=torch.zeros((1, 1)),
        )

    def get_variables_specs(self) -> NamedVariables:
        d = super().get_variables_specs()
        d.update(
            log_g_mean=ModelParameter.for_pop_mean("log_g", shape=(1,)),
            log_g_std=Hyperparameter(0.01),
            log_g=PopulationLatentVariable(Normal("log_g_mean", "log_g_std")),
            g=LinkedVariable(Exp("log_g")),
            xi_mean=ModelParameter.for_ind_mean("xi", shape=(1,)),
            deltas_mean=ModelParameter.for_pop_mean(
                "deltas",
                shape=(self.dimension - 1,),
            ),
            deltas_std=Hyperparameter(0.01),
            deltas=PopulationLatentVariable(
                Normal("deltas_mean", "deltas_std"),
                sampling_kws={"scale": 1.0},
            ),
            deltas_padded=LinkedVariable(self.pad_deltas),
            deltas_exp=LinkedVariable(self.deltas_exp),
            g_deltas_exp=LinkedVariable(self.g_deltas_exp),
            metric=LinkedVariable(self.metric),
        )
        if self.source_dimension >= 1:
            d.update(
                denom=LinkedVariable(self.denom),
                gamma_t0=LinkedVariable(self.gamma_t0),
                collin_to_d_gamma_t0=LinkedVariable(self.collin_to_d_gamma_t0),
                g_metric=LinkedVariable(self.g_metric),
                orthonormal_basis=LinkedVariable(
                    OrthoBasis("collin_to_d_gamma_t0", "g_metric"),
                ),
                model=LinkedVariable(self.model_with_sources),
            )
        else:
            d["model"] = LinkedVariable(self.model_no_sources)

        return d
