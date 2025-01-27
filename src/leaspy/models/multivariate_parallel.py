import torch

from leaspy.io.data.dataset import Dataset
from leaspy.models.abstract_multivariate_model import AbstractMultivariateModel
from leaspy.models.base import InitializationMethod
from leaspy.models.multivariate import LogisticMultivariateInitializationMixin
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
    VariablesValuesRO,
)


class MultivariateParallelModel(
    LogisticMultivariateInitializationMixin, AbstractMultivariateModel
):
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
        method: InitializationMethod,
    ) -> VariablesValuesRO:
        parameters = super()._compute_initial_values_for_model_parameters(
            dataset, method=method
        )
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

    """
    def compute_jacobian_tensorized(
        self,
        timepoints: torch.Tensor,
        individual_parameters: dict,
        *,
        attribute_type=None,
    ) -> DictParamsTorch:
        # TODO: refact highly inefficient (many duplicated code from `compute_individual_tensorized`)

        # Population parameters
        g, deltas, mixing_matrix = self._get_attributes(attribute_type)

        # Individual parameters
        xi, tau = individual_parameters['xi'], individual_parameters['tau']
        reparametrized_time = self.time_reparametrization(timepoints, xi, tau)

        # Reshaping
        reparametrized_time = reparametrized_time.unsqueeze(-1)  # (n_individuals, n_timepoints, -> n_features)
        alpha = torch.exp(xi).reshape(-1, 1, 1)

        # Model expected value
        t = reparametrized_time + deltas
        if self.source_dimension != 0:
            sources = individual_parameters['sources']
            wi = sources.matmul(mixing_matrix.t())  # (n_individuals, n_features)
            g_deltas_exp = g * torch.exp(-deltas)
            b = (1. + g_deltas_exp) ** 2 / g_deltas_exp  # (n_features, )
            t += (b * wi).unsqueeze(-2)  # to get n_timepoints dimension
        model = 1. / (1. + g * torch.exp(-t))

        # Jacobian of model expected value w.r.t. individual parameters
        c = model * (1. - model)

        derivatives = {
            'xi': (c * reparametrized_time).unsqueeze(-1),
            'tau': (c * -alpha).unsqueeze(-1),
        }
        if self.source_dimension > 0:
            b = b.reshape((1, 1, -1, 1))  # n_features is third
            derivatives['sources'] = c.unsqueeze(-1) * b * mixing_matrix.expand((1, 1, -1, -1))

        return derivatives

    def compute_individual_ages_from_biomarker_values_tensorized(
        self,
        value: torch.Tensor,
        individual_parameters: dict,
        feature: str,
    ) -> torch.Tensor:
        raise NotImplementedError("Open an issue on Gitlab if needed.")  # pragma: no cover

    ##############################
    ### MCMC-related functions ###
    ##############################

    #def compute_model_sufficient_statistics(
    #    self,
    #    state #: State,
    #) -> DictParamsTorch:
    #    # unlink all sufficient statistics from updates in realizations!
    #    realizations = realizations.clone()
    #
    #    sufficient_statistics = realizations[["g", "deltas", "tau", "xi"]].tensors_dict
    #    if self.source_dimension != 0:
    #        sufficient_statistics['betas'] = realizations["betas"].tensor
    #    for param in ("tau", "xi"):
    #        sufficient_statistics[f"{param}_sqrd"] = torch.pow(realizations[param].tensor, 2)
    #
    #    return sufficient_statistics

    # def update_model_parameters_burn_in(self, data: Dataset, sufficient_statistics: DictParamsTorch) -> None:
    #     for param in ("g", "deltas"):
    #         self.parameters[param] = sufficient_statistics[param]
    #     if self.source_dimension != 0:
    #         self.parameters['betas'] = sufficient_statistics['betas']
    #     for param in ("xi", "tau"):
    #         param_realizations = sufficient_statistics[param]
    #         self.parameters[f"{param}_mean"] = torch.mean(param_realizations)
    #         self.parameters[f"{param}_std"] = torch.std(param_realizations)

    #def update_model_parameters_normal(self, data: Dataset, sufficient_statistics: DictParamsTorch) -> None:
    #    # TODO? factorize `update_model_parameters_***` methods?
    #    from .utilities import compute_std_from_variance
    #
    #    for param in ("g", "deltas"):
    #        self.parameters[param] = sufficient_statistics[param]
    #    if self.source_dimension != 0:
    #        self.parameters['betas'] = sufficient_statistics['betas']
    #
    #    for param in ("tau", "xi"):
    #        param_old_mean = self.parameters[f"{param}_mean"]
    #        param_cur_mean = torch.mean(sufficient_statistics[param])
    #        param_variance_update = (
    #            torch.mean(sufficient_statistics[f"{param}_sqrd"]) -
    #            2. * param_old_mean * param_cur_mean
    #        )
    #        param_variance = param_variance_update + param_old_mean ** 2
    #        self.parameters[f"{param}_std"] = compute_std_from_variance(
    #            param_variance, varname=f"{param}_std"
    #        )
    #        self.parameters[f"{param}_mean"] = param_cur_mean
    """
