from abc import abstractmethod
from typing import Iterable, Optional

import pandas as pd
import torch

from leaspy.io.data.dataset import Dataset
from leaspy.utils.docs import doc_with_super
from leaspy.utils.functional import Affine, Exp, OrthoBasis, Sqr
from leaspy.utils.weighted_tensor import (
    TensorOrWeightedTensor,
    WeightedTensor,
    unsqueeze_right,
)
from leaspy.variables.distributions import Normal, NormalCovariateLinear
from leaspy.variables.specs import (
    DataVariable,
    Hyperparameter,
    LinkedVariable,
    ModelParameter,
    NamedVariables,
    PopulationLatentVariable,
    SuffStatsRW,
    VariableName,
    VariableNameToValueMapping,
)
from leaspy.variables.state import State

from .base import InitializationMethod
from .covariate_abstract_multivariate_model import CovariateAbstractMultivariateModel
from .obs_models import FullGaussianObservationModel

# TODO refact? implement a single function
# compute_individual_tensorized(..., with_jacobian: bool) -> returning either
# model values or model values + jacobians wrt individual parameters

# TODO refact? subclass or other proper code technique to extract model's concrete
#  formulation depending on if linear, logistic, mixed log-lin, ...


__all__ = [
    "CovariateMultivariateModel",
    # "CovariateLinearMultivariateInitializationMixin",
    # "CovariateLinearMultivariateModel",
    "CovariateLogisticMultivariateInitializationMixin",
    "CovariateLogisticMultivariateModel",
]


@doc_with_super()
class CovariateMultivariateModel(CovariateAbstractMultivariateModel):
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

    def __init__(
        self,
        name: str,
        variables_to_track: Optional[Iterable[VariableName]] = None,
        **kwargs,
    ):
        super().__init__(name, **kwargs)

        default_variables_to_track = [
            "g",
            "v0",
            "noise_std",
            "tau_mean",
            "tau_std",
            "xi_mean",
            "xi_std",
            "nll_attach",
            "nll_regul_log_g",
            "nll_regul_log_v0",
            "xi",
            "tau",
            "nll_regul_pop_sum",
            "nll_regul_all_sum",
            "nll_tot",
        ]

        if self.source_dimension:
            default_variables_to_track += [
                "sources",
                "betas",
                "mixing_matrix",
                "space_shifts",
            ]

        variables_to_track = variables_to_track or default_variables_to_track
        self.tracked_variables = self.tracked_variables.union(set(variables_to_track))

    @classmethod
    def _center_xi_realizations(cls, state: State) -> None:
        """
        Center the ``xi`` realizations in place.

        .. note::
            This operation does not change the orthonormal basis
            (since the resulting ``v0`` is collinear to the previous one)
            Nor all model computations (only ``v0 * exp(xi_i)`` matters),
            it is only intended for model identifiability / ``xi_i`` regularization
            <!> all operations are performed in "log" space (``v0`` is log'ed)

        Parameters
        ----------
        realizations : :class:`.CollectionRealization`
            The realizations to use for updating the :term:`MCMC` toolbox.
        """
        mean_xi = torch.mean(state["xi"])
        state["xi"] = state["xi"] - mean_xi
        state["log_v0"] = state["log_v0"] + mean_xi

        # TODO: find a way to prevent re-computation of orthonormal basis since it should
        #  not have changed (v0_collinear update)
        # self.update_MCMC_toolbox({'v0_collinear'}, realizations)

    @classmethod
    def compute_sufficient_statistics(cls, state: State) -> SuffStatsRW:
        """
        Compute the model's :term:`sufficient statistics`.

        Parameters
        ----------
        state : :class:`.State`
            The state to pick values from.

        Returns
        -------
        SuffStatsRW :
            The computed sufficient statistics.
        """
        # <!> modify 'xi' and 'log_v0' realizations in-place
        # TODO: what theoretical guarantees for this custom operation?
        cls._center_xi_realizations(state)

        return super().compute_sufficient_statistics(state)

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
            phi_v0_mean=ModelParameter.for_pop_mean(
                ("phi_v0"), shape=(self.dimension, 2)
            ),
            phi_v0_std=Hyperparameter((0.001, 0.01)),
            rho_v0=ModelParameter.for_correlation_covariate_linear(
                ("phi_v0"), shape=(self.dimension,)
            ),
            xi_mean=Hyperparameter(0.0),
            # LATENT VARS
            phi_v0=PopulationLatentVariable(
                NormalCovariateLinear(
                    "phi_v0_mean", "phi_v0_std", "rho_v0", "covariates"
                )
            ),  # phi_v0 = (phi_mod_v0, phi_ref_v0)
            # LINKED VARS
            # log_v0=LinkedVariable(
            #     Affine("phi_v0", "covariates")
            # ),
            # v0=LinkedVariable(Exp("log_v0")),
            metric=LinkedVariable(
                self.metric
            ),  # for linear model: metric & metric_sqr are fixed = 1.
        )

        if self.source_dimension >= 1:
            d.update(
                model=LinkedVariable(self.model_with_sources),
                metric_sqr=LinkedVariable(Sqr("metric")),
                orthonormal_basis=LinkedVariable(OrthoBasis("v0", "metric_sqr")),
            )
        else:
            d["model"] = LinkedVariable(self.model_no_sources)

        # TODO: WIP
        # variables_info.update(self.get_additional_ordinal_population_random_variable_information())
        # self.update_ordinal_population_random_variable_information(variables_info)

        return d

    @staticmethod
    @abstractmethod
    def metric(*, g: torch.Tensor) -> torch.Tensor:
        pass

    @classmethod
    def model_no_sources(cls, *, rt: torch.Tensor, metric, v0, g) -> torch.Tensor:
        """Returns a model without source. A bit dirty?"""
        return cls.model_with_sources(
            rt=rt,
            metric=metric,
            v0=v0,
            g=g,
            space_shifts=torch.zeros((1, 1)),
        )

    @classmethod
    @abstractmethod
    def model_with_sources(
        cls,
        *,
        rt: torch.Tensor,
        space_shifts: torch.Tensor,
        metric,
        v0,
        g,
    ) -> torch.Tensor:
        pass


# class CovariateLinearMultivariateInitializationMixin:
#     """Compute initial values for model parameters."""

#     def _compute_initial_values_for_model_parameters(
#         self,
#         dataset: Dataset,
#         method: InitializationMethod,
#     ) -> VariableNameToValueMapping:
#         from leaspy.models.utilities import (
#             compute_linear_regression_subjects,
#             get_log_velocities,
#             torch_round,
#         )

#         df = dataset.to_pandas(apply_headers=True)
#         times = df.index.get_level_values("TIME").values
#         t0 = times.mean()

#         d_regress_params = compute_linear_regression_subjects(df, max_inds=None)
#         df_all_regress_params = pd.concat(d_regress_params, names=["feature"])
#         df_all_regress_params["position"] = (
#             df_all_regress_params["intercept"] + t0 * df_all_regress_params["slope"]
#         )
#         df_grp = df_all_regress_params.groupby("feature", sort=False)
#         positions = torch.tensor(df_grp["position"].mean().values)
#         velocities = torch.tensor(df_grp["slope"].mean().values)

#         parameters = {
#             "g_mean": positions,
#             "log_v0_mean": get_log_velocities(velocities, self.features),
#             # "betas": torch.zeros((self.dimension - 1, self.source_dimension)),
#             "tau_mean": torch.tensor(t0),
#             "tau_std": self.tau_std,
#             # "xi_mean": torch.tensor(0.),
#             "xi_std": self.xi_std,
#             # "sources_mean": torch.tensor(0.),
#             # "sources_std": torch.tensor(SOURCES_STD),
#         }
#         if self.source_dimension >= 1:
#             parameters["betas_mean"] = torch.zeros(
#                 (self.dimension - 1, self.source_dimension)
#             )
#         rounded_parameters = {
#             str(p): torch_round(v.to(torch.float32)) for p, v in parameters.items()
#         }
#         obs_model = next(iter(self.obs_models))  # WIP: multiple obs models...
#         if isinstance(obs_model, FullGaussianObservationModel):
#             rounded_parameters["noise_std"] = self.noise_std.expand(
#                 obs_model.extra_vars["noise_std"].shape
#             )
#         return rounded_parameters


# class CovariateLinearMultivariateModel(
#     CovariateLinearMultivariateInitializationMixin, CovariateMultivariateModel
# ):
#     """Manifold model for multiple variables of interest (linear formulation)."""

#     def __init__(self, name: str, **kwargs):
#         super().__init__(name, **kwargs)

#     def get_variables_specs(self) -> NamedVariables:
#         """
#         Return the specifications of the variables (latent variables, derived variables,
#         model 'parameters') that are part of the model.

#         Returns
#         -------
#         NamedVariables :
#             The specifications of the model's variables.
#         """
#         d = super().get_variables_specs()
#         d.update(
#             g_mean=ModelParameter.for_pop_mean("g", shape=(self.dimension,)),
#             g_std=Hyperparameter(0.01),
#             g=PopulationLatentVariable(Normal("g_mean", "g_std")),
#         )

#         return d

#     @staticmethod
#     def metric(*, g: torch.Tensor) -> torch.Tensor:
#         """Used to define the corresponding variable."""
#         return torch.ones_like(g)

#     @classmethod
#     def model_with_sources(
#         cls,
#         *,
#         rt: torch.Tensor,
#         space_shifts: torch.Tensor,
#         metric,
#         v0,
#         g,
#     ) -> torch.Tensor:
#         """Returns a model with sources."""
#         pop_s = (None, None, ...)
#         rt = unsqueeze_right(rt, ndim=1)  # .filled(float('nan'))
#         return (g[pop_s] + v0[pop_s] * rt + space_shifts[:, None, ...]).weighted_value


class CovariateLogisticMultivariateInitializationMixin:
    def _compute_initial_values_for_model_parameters(
        self,
        dataset: Dataset,
        method: InitializationMethod,
    ) -> VariableNameToValueMapping:
        """Compute initial values for model parameters."""
        from leaspy.models.utilities import (
            compute_patient_slopes_distribution,
            compute_patient_time_distribution,
            compute_patient_values_distribution,
            get_log_velocities,
            torch_round,
        )

        df = dataset.to_pandas(apply_headers=True)
        slopes_mu, slopes_sigma = compute_patient_slopes_distribution(df)
        values_mu, values_sigma = compute_patient_values_distribution(df)
        time_mu, time_sigma = compute_patient_time_distribution(df)

        if method == InitializationMethod.DEFAULT:
            slopes = slopes_mu
            values = values_mu
            t0 = time_mu
            betas = torch.zeros((self.dimension - 1, self.source_dimension))

        if method == InitializationMethod.RANDOM:
            slopes = torch.normal(slopes_mu, slopes_sigma)
            values = torch.normal(values_mu, values_sigma)
            t0 = torch.normal(time_mu, time_sigma)
            betas = torch.distributions.normal.Normal(loc=0.0, scale=1.0).sample(
                sample_shape=(self.dimension - 1, self.source_dimension)
            )

        # Enforce values are between 0 and 1
        values = values.clamp(
            min=1e-2, max=1 - 1e-2
        )  # always "works" for ordinal (values >= 1)

        parameters = {
            "phi_tau_mean": torch.Tensor([0.0, t0]),
            "phi_g_mean": torch.stack(
                [
                    torch.zeros(self.dimension),  # slopes
                    torch.log(1.0 / values - 1.0),  # intercepts (logit)
                ],
                dim=0,
            ).T,
            "phi_v0_mean": torch.stack(
                [
                    torch.zeros(self.dimension),  # slopes
                    get_log_velocities(slopes, self.features),  # intercepts
                ],
                dim=0,
            ).T,
            "rho_tau": torch.zeros(1),
            "rho_g": torch.zeros(self.dimension),
            "rho_v0": torch.zeros(self.dimension),
            "xi_std": self.xi_std,
        }
        if self.source_dimension >= 1:
            parameters["betas_mean"] = betas
        rounded_parameters = {
            str(p): torch_round(v.to(torch.float32)) for p, v in parameters.items()
        }
        obs_model = next(iter(self.obs_models))  # WIP: multiple obs models...
        if isinstance(obs_model, FullGaussianObservationModel):
            rounded_parameters["noise_std"] = self.noise_std.expand(
                obs_model.extra_vars["noise_std"].shape
            )
        return rounded_parameters


class CovariateLogisticMultivariateModel(
    CovariateLogisticMultivariateInitializationMixin, CovariateMultivariateModel
):
    """Manifold model for multiple variables of interest (logistic formulation)."""

    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)

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
            phi_g_mean=ModelParameter.for_pop_mean(
                ("phi_g"), shape=(self.dimension, 2)
            ),
            phi_g_std=Hyperparameter((0.001, 0.01)),
            rho_g=ModelParameter.for_correlation_covariate_linear(
                ("phi_g"), shape=(self.dimension,)
            ),
            # LATENT VARS
            phi_g=PopulationLatentVariable(
                NormalCovariateLinear("phi_g_mean", "phi_g_std", "rho_g", "covariates")
            ),  # phi_g = (phi_mod_g, phi_ref_g)
            # # LINKED VARS
            # log_g=LinkedVariable(
            #     Affine("phi_g", "covariates")
            # ),  # log_g=phi_mod_g*covariate+phi_ref_g
            # g=LinkedVariable(Exp("log_g")),
        )

        return d

    @staticmethod
    def metric(*, g: torch.Tensor) -> torch.Tensor:
        """Used to define the corresponding variable."""
        return (g + 1) ** 2 / g

    @classmethod
    def model_with_sources(
        cls,
        *,
        rt: TensorOrWeightedTensor[float],
        space_shifts: TensorOrWeightedTensor[float],
        metric: TensorOrWeightedTensor[float],
        v0: TensorOrWeightedTensor[float],
        g: TensorOrWeightedTensor[float],
    ) -> torch.Tensor:
        """Returns a model with sources."""
        # Shape: (Ni, Nt, Nfts)
        pop_s = (None, None, ...)
        rt = unsqueeze_right(rt, ndim=1)  # .filled(float('nan'))
        w_model_logit = metric[pop_s] * (
            v0[pop_s] * rt + space_shifts[:, None, ...]
        ) - torch.log(g[pop_s])
        model_logit, weights = WeightedTensor.get_filled_value_and_weight(
            w_model_logit, fill_value=0.0
        )
        return WeightedTensor(torch.sigmoid(model_logit), weights).weighted_value
