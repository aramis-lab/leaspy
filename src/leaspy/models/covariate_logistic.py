import torch

from leaspy.io.data.dataset import Dataset
from leaspy.utils.functional import AffineMatrix, Exp
from leaspy.utils.weighted_tensor import (
    TensorOrWeightedTensor,
    WeightedTensor,
    unsqueeze_right,
)
from leaspy.variables.distributions import MultivariateNormal, Normal
from leaspy.variables.specs import (
    Hyperparameter,
    LinkedVariable,
    ModelParameter,
    NamedVariables,
    PopulationLatentVariable,
    VariableNameToValueMapping,
)

from .base import InitializationMethod
from .covariate_riemannian_manifold import CovariateRiemannianManifoldModel
from .obs_models import FullGaussianObservationModel

__all__ = [
    "CovariateLogisticInitializationMixin",
    "CovariateLogisticModel",
]


class CovariateLogisticInitializationMixin:
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
        :class:`~leaspy.variables.specs.VariableNameToValueMapping
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
            "t0_mean": t0,
            "tau_std": self.tau_std,
            "xi_std": self.xi_std,
            "delta_t0_mean": torch.zeros((self.nb_cov,)),
            "delta_v0_mean": torch.zeros((self.dimension, self.nb_cov)),
            "delta_g_mean": torch.zeros((self.dimension, self.nb_cov)),
            "delta_t0_sigma": torch.eye(self.nb_cov),  # matrice identité
            "delta_v0_sigma": torch.eye(self.nb_cov),
            "delta_g_sigma": torch.eye(self.nb_cov),
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


class CovariateLogisticModel(
    CovariateLogisticInitializationMixin, CovariateRiemannianManifoldModel
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
            A dictionary-like object mapping variable names to their specifications.
        """
        d = super().get_variables_specs()
        d.update(
            log_g_mean=ModelParameter.for_pop_mean("log_g", shape=(self.dimension,)),
            log_g_std=Hyperparameter(0.01),
            log_g=PopulationLatentVariable(Normal("log_g_mean", "log_g_std")),
            g=LinkedVariable(Exp("log_g")),
            delta_g_mean=ModelParameter.for_pop_mean(
                "delta_g", shape=(self.dimension, self.nb_cov)
            ),
            delta_g_sigma=ModelParameter.for_pop_cov_matrix(
                "delta_g", shape=(self.nb_cov, self.nb_cov)
            ),
            delta_g=PopulationLatentVariable(
                MultivariateNormal("delta_g_mean", "delta_g_sigma"),
                sampling_kws={"scale": 0.1},
            ),
            g_patient=LinkedVariable(
                AffineMatrix("g", "delta_g", "covariates"),
            ),
        )

        return d

    @staticmethod
    def metric(*, g_patient: torch.Tensor) -> torch.Tensor:
        r"""
        Compute the metric tensor from input tensor `g`.
        This function calculates the metric as \((g + 1)^2 / g\) element-wise.

        Parameters
        ----------
        g : :class:`torch.Tensor`
            Input tensor with values of the population parameter `g` for each feature.

        Returns
        -------
        :class:`torch.Tensor`
            The computed metric tensor, same shape as g(number of features)
        """
        return (g_patient + 1) ** 2 / g_patient

    @classmethod
    def model_with_sources(
        cls,
        *,
        rt: TensorOrWeightedTensor[float],
        space_shifts: TensorOrWeightedTensor[float],
        metric: TensorOrWeightedTensor[float],
        v0_patient: TensorOrWeightedTensor[float],
        g_patient: TensorOrWeightedTensor[float],
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
        rt = unsqueeze_right(rt, ndim=1)  # (N, N_i, 1)

        # v0_patient et g_patient : (N, K) -> unsqueeze pour broadcaster sur N_i
        w_model_logit = metric[:, None, :] * (
            v0_patient[:, None, :] * rt + space_shifts[:, None, :]
        ) - torch.log(g_patient[:, None, :])

        model_logit, weights = WeightedTensor.get_filled_value_and_weight(
            w_model_logit, fill_value=0.0
        )
        return WeightedTensor(torch.sigmoid(model_logit), weights).weighted_value
