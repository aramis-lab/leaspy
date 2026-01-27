import pandas as pd
import torch

from leaspy.io.data.dataset import Dataset
from leaspy.utils.functional import Exp, OrthoBasis, Sqr
from leaspy.utils.weighted_tensor import (
    unsqueeze_right,
)
from leaspy.variables.distributions import Normal
from leaspy.variables.specs import (
    Hyperparameter,
    ModelParameter,
    NamedVariables,
    PopulationLatentVariable,
    VariableNameToValueMapping,
)

from .obs_models import FullGaussianObservationModel
from .riemanian_manifold import RiemanianManifoldModel

__all__ = [
    "LinearInitializationMixin",
    "LinearModel",
]


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
        :class:`~leaspy.variables.specs.VariableNameToValueMapping`
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
        :class:`~leaspy.variables.specs.NamedVariables :
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
            The values of the population parameter `v0` for each feature.
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
