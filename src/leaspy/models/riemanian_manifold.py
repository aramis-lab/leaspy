from abc import abstractmethod
from typing import Iterable, Optional

import pandas as pd
import torch

from leaspy.io.data.dataset import Dataset
from leaspy.utils.functional import Exp, OrthoBasis, Sqr
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
    SuffStatsRW,
    VariableName,
    VariableNameToValueMapping,
)
from leaspy.variables.state import State

from .base import InitializationMethod
from .obs_models import FullGaussianObservationModel
from .time_reparametrized import TimeReparametrizedModel

# TODO refact? implement a single function
# compute_individual_tensorized(..., with_jacobian: bool) -> returning either
# model values or model values + jacobians wrt individual parameters

# TODO refact? subclass or other proper code technique to extract model's concrete
#  formulation depending on if linear, logistic, mixed log-lin, ...


__all__ = [
    "RiemanianManifoldModel",
    "LinearInitializationMixin",
    "LinearModel",
    "LogisticInitializationMixin",
    "LogisticModel",
]


class RiemanianManifoldModel(TimeReparametrizedModel):
    """Manifold model for multiple variables of interest (logistic or linear formulation).

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
        self.track_variables(variables_to_track or default_variables_to_track)

    @classmethod
    def _center_xi_realizations(cls, state: State) -> None:
        """
        Center the ``xi`` realizations in place.
        
        Parameters
        ----------
        state : :class:`.State`
            The dictionary-like object representing current model state, which
            contains keys such as``"xi"`` and ``"log_v0"``.

        Notes
        -----
        This transformation preserves the orthonormal basis since the new ``v0`` remains
        collinear to the previous one. It is a purely internal operation meant to reduce
        redundancy in the parameter space (i.e., improve identifiability and stabilize
        inference).
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
            A dictionary-like object mapping variable names to their specifications.
            These include `ModelParameter`, `Hyperparameter`, `PopulationLatentVariable`,
            and `LinkedVariable` instances.
        """
        d = super().get_variables_specs()
        d.update(
            # PRIORS
            log_v0_mean=ModelParameter.for_pop_mean(
                "log_v0",
                shape=(self.dimension,),
            ),
            log_v0_std=Hyperparameter(0.01),
            xi_mean=Hyperparameter(0.0),
            # LATENT VARS
            log_v0=PopulationLatentVariable(
                Normal("log_v0_mean", "log_v0_std"),
            ),
            # DERIVED VARS
            v0=LinkedVariable(
                Exp("log_v0"),
            ),
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
        """
        Return the model output when no sources(spatial components) are present.

        Parameters
        ----------
        rt : torch.Tensor
            The reparametrized time.
        metric : Any
            The metric tensor used for computing the spatial/temporal influence.
        v0 : Any
            The values of the popoulation parameter `v0` for each feature.
        g : Any
            The values of the population parameter `g` for each feature.

        Returns
        -------
         :class:`torch.Tensor`
            The model output without contribution from source shifts.

        Notes
        -----
        This implementation delegates to `model_with_sources` with `space_shifts`
        set to a zero tensor of shape (1, 1), effectively removing source effects.
        """
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


class LinearInitializationMixin:
    """Compute initial values for model parameters."""

    def _compute_initial_values_for_model_parameters(
        self,
        dataset: Dataset,
    ) -> VariableNameToValueMapping:
        """
        Compute and return initial values for the model parameters using linear regression.
        
        Parameters
        ----------
        dataset : :class:`Dataset`
            The dataset from which to extract observations and masks.

        Returns
        -------
        VariableNameToValueMapping
            A dictionary mapping parameter names (as strings) to their initialized
            torch.Tensor values.

        Notes
        -----
        - The initial positions are computed using `intercept + t0 * slope` where `t0`
        is the mean of all observation times.
        - Velocities are averaged per feature, and log-transformed using
        `get_log_velocities`.
        - Time parameters (`tau_mean`, `tau_std`) and variability parameters
        (`xi_std`) are also initialized.
        - If `self.source_dimension >= 1`, zero initialization is used for `betas_mean`.
        - If the observation model is a `FullGaussianObservationModel`, the
        `noise_std` parameter is also added, expanded to the correct shape.
        """
        from leaspy.models.utilities import (
            compute_linear_regression_subjects,
            get_log_velocities,
            torch_round,
        )

        df = dataset.to_pandas(apply_headers=True)
        times = df.index.get_level_values("TIME").values
        t0 = times.mean()

        d_regress_params = compute_linear_regression_subjects(df, max_inds=None)
        df_all_regress_params = pd.concat(d_regress_params, names=["feature"])
        df_all_regress_params["position"] = (
            df_all_regress_params["intercept"] + t0 * df_all_regress_params["slope"]
        )
        df_grp = df_all_regress_params.groupby("feature", sort=False)
        positions = torch.tensor(df_grp["position"].mean().values)
        velocities = torch.tensor(df_grp["slope"].mean().values)

        parameters = {
            "g_mean": positions,
            "log_v0_mean": get_log_velocities(velocities, self.features),
            "tau_mean": torch.tensor(t0),
            "tau_std": self.tau_std,
            "xi_std": self.xi_std,
        }
        if self.source_dimension >= 1:
            parameters["betas_mean"] = torch.zeros(
                (self.dimension - 1, self.source_dimension)
            )
        rounded_parameters = {
            str(p): torch_round(v.to(torch.float32)) for p, v in parameters.items()
        }
        obs_model = next(iter(self.obs_models))  # WIP: multiple obs models...
        if isinstance(obs_model, FullGaussianObservationModel):
            rounded_parameters["noise_std"] = self.noise_std.expand(
                obs_model.extra_vars["noise_std"].shape
            )
        return rounded_parameters


class LinearModel(LinearInitializationMixin, RiemanianManifoldModel):
    """Manifold model for multiple variables of interest (linear formulation)."""

    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)

    def get_variables_specs(self) -> NamedVariables:
        """
        Return the specifications of the variables (latent variables, derived variables,
        model 'parameters') that are part of the model.

        Returns
        -------
        NamedVariables :
            A dictionary-like object mapping variable names to their specifications.
        """
        d = super().get_variables_specs()
        d.update(
            g_mean=ModelParameter.for_pop_mean("g", shape=(self.dimension,)),
            g_std=Hyperparameter(0.01),
            g=PopulationLatentVariable(Normal("g_mean", "g_std")),
        )

        return d

    @staticmethod
    def metric(*, g: torch.Tensor) -> torch.Tensor:
        """
        Compute the metric tensor for the model.
    
        Parameters
        ----------
        g :  :class:`torch.Tensor`
            Input tensor with values of the population parameter `g` for each feature.

        Returns
        -------
         :class:`torch.Tensor`
            A tensor of ones with the same shape as `g`.
        """
        return torch.ones_like(g)

    @classmethod
    def model_with_sources(
        cls,
        *,
        rt: torch.Tensor,
        space_shifts: torch.Tensor,
        metric,
        v0,
        g,
    ) -> torch.Tensor:
        """
        Return the model output when sources(spatial components) are present.

        Parameters
        ----------
        rt :  :class:`torch.Tensor`
            The reparametrized time.
        space_shifts :  :class:`torch.Tensor`
            The values of the space-shifts
        metric : Any
            The metric tensor used for computing the spatial/temporal influence.
        v0 : Any
            The values of the popoulation parameter `v0` for each feature.
        g : Any
            The values of the population parameter `g` for each feature.

        Returns
        -------
         :class:`torch.Tensor`
            The model output with contribution from sources.
        """
        pop_s = (None, None, ...)
        rt = unsqueeze_right(rt, ndim=1)  # .filled(float('nan'))
        return (g[pop_s] + v0[pop_s] * rt + space_shifts[:, None, ...]).weighted_value


class LogisticInitializationMixin:
    def _compute_initial_values_for_model_parameters(
        self,
        dataset: Dataset,
    ) -> VariableNameToValueMapping:
        """
        Compute initial values for model parameters.

        Parameters
        ----------
        dataset : :class:`Dataset`
            The dataset from which to extract observations and masks.

        Returns
        -------
        VariableNameToValueMapping
            A dictionary mapping parameter names (as strings) to their initialized
            torch.Tensor values.

        Notes
        -----
        - If the initialization method is `DEFAULT`, patient means are used.
        - If `RANDOM`, parameters are sampled from normal distributions
        centered at patient means with estimated standard deviations.
        - `values` are clamped between 0.01 and 0.99 to avoid boundary issues.
        - If the model includes sources (source_dimension >= 1),
        regression coefficients `betas_mean` are initialized accordingly.
        - If the observation model is a `FullGaussianObservationModel`,
        the noise standard deviation parameter is expanded to the correct shape.
        """
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

        if self.initialization_method == InitializationMethod.DEFAULT:
            slopes = slopes_mu
            values = values_mu
            t0 = time_mu
            betas = torch.zeros((self.dimension - 1, self.source_dimension))

        if self.initialization_method == InitializationMethod.RANDOM:
            slopes = torch.normal(slopes_mu, slopes_sigma)
            values = torch.normal(values_mu, values_sigma)
            t0 = torch.normal(time_mu, time_sigma)
            betas = torch.distributions.normal.Normal(loc=0.0, scale=1.0).sample(
                sample_shape=(self.dimension - 1, self.source_dimension)
            )

        # Enforce values are between 0 and 1
        values = values.clamp(min=1e-2, max=1 - 1e-2)

        parameters = {
            "log_g_mean": torch.log(1.0 / values - 1.0),
            "log_v0_mean": get_log_velocities(slopes, self.features),
            "tau_mean": t0,
            "tau_std": self.tau_std,
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


class LogisticModel(LogisticInitializationMixin, RiemanianManifoldModel):
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
            A dictionary-like object mapping variable names to their specifications.
        """
        d = super().get_variables_specs()
        d.update(
            log_g_mean=ModelParameter.for_pop_mean("log_g", shape=(self.dimension,)),
            log_g_std=Hyperparameter(0.01),
            log_g=PopulationLatentVariable(Normal("log_g_mean", "log_g_std")),
            g=LinkedVariable(Exp("log_g")),
        )

        return d

    @staticmethod
    def metric(*, g: torch.Tensor) -> torch.Tensor:
        """
        Compute the metric tensor from input tensor `g`.
        This function calculates the metric as \((g + 1)^2 / g\) element-wise.

        Parameters
        ----------
        g : t :class:`torch.Tensor`
            Input tensor with values of the population parameter `g` for each feature.

        Returns
        -------
         :class:`torch.Tensor`
            The computed metric tensor, same shape as g(number of features)
        """
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
            Tensor containing the values of the popoulation parameter `v0` for each feature.
        g : TensorOrWeightedTensor[float]
            Tensor containing the values of the population parameter `g` for each feature.

        Returns
        -------
         :class:`torch.Tensor`
            Weighted value tensor after applying sigmoid transformation,
            representing the model output with sources.
        """
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
