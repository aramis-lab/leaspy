from __future__ import annotations
from typing import (
    Dict,
    Callable,
)

import torch
import torch.optim as optim
from leaspy.models.utilities import compute_std_from_variance
from leaspy.variables.distributions import Beta
from leaspy.utils.weighted_tensor import (
    WeightedTensor,
    sum_dim,
    wsum_dim_return_weighted_sum_only,
    wsum_dim_return_sum_of_weights_only,
    wsum_dim,
)
from leaspy.utils.functional import Sqr, Prod, Sum
from leaspy.variables.specs import (
    VarName,
    VariableInterface,
    LinkedVariable,
    ModelParameter,
    Collect,
    LVL_FT,
)
from leaspy.io.data.dataset import Dataset
from leaspy.variables.state import State

from ._base import ObservationModel


class BetaObservationModel(ObservationModel):
    """Specialized `ObservationModel` for noisy observations with Gaussian residuals assumption."""

    def __init__(
        self,
        name: VarName,
        getter: Callable[[Dataset], WeightedTensor],
        model: VarName,
        scale: VarName,
        **extra_vars: VariableInterface,
    ):

        super().__init__(name, getter, Beta(model, scale),
                                            extra_vars=extra_vars)


class FullBetaObservationModel(BetaObservationModel):
    """
    Specialized `GaussianObservationModel` when all data share the same observation model, with default naming.

    The default naming is:
        - 'y' for observations
        - 'model' for model predictions
        - 'noise_std' for scale of residuals

    We also provide a convenient factory `default` for most common case, which corresponds
    to `noise_std` directly being a `ModelParameter` (it could also be a `PopulationLatentVariable`
    with positive support). Whether scale of residuals is scalar or diagonal depends on the
    `dimension` argument of this method.
    """

    tol_noise_variance = 1e-5

    def __init__(self, noise_std: VariableInterface,
                 model:VarName, **extra_vars: VariableInterface):
        super().__init__(
            name="y",
            getter=self.y_getter,
            model = model,
            scale = "noise_std",
            noise_std= noise_std,
            **extra_vars,
        )

    @staticmethod
    def y_getter(dataset: Dataset) -> WeightedTensor:
        assert dataset.values is not None
        assert dataset.mask is not None
        return WeightedTensor(dataset.values, weight=dataset.mask.to(torch.bool))

    @classmethod
    def noise_std_suff_stats(cls) -> Dict[VarName, LinkedVariable]:
        """Dictionary of sufficient statistics needed for `noise_std` (when directly a model parameter)."""
        return dict(
        )

    @classmethod
    def scalar_noise_std_update(
        cls,
        *,
        state: State,
    ) -> torch.Tensor:
        """Update rule for scalar `noise_std` (when directly a model parameter), from state & sufficient statistics."""
        # Parameters we want to optimize: alpha and beta
        # We initialize them randomly, and make sure they're positive using softplus

        # We initialize beta randomly, and make sure they're positive using softplus
        variance = torch.tensor([state["noise_std"]], requires_grad=True) # TODO: use the past noise_std

        # Optimizer (you can use any optimizer, Adam is often a good choice)
        optimizer = optim.Adam([variance], lr=0.1)

        # Number of iterations for optimization
        num_iterations = 10

        for i in range(num_iterations):
            optimizer.zero_grad()

            # We use the softplus to ensure alpha and beta are positive
            variance_pos = torch.nn.functional.softplus(variance)

            # Define the beta distribution with current alpha and beta
            variance_dist = torch.distributions.Beta(state["model"].clip(min=0.001, max=0.99) * variance_pos,
                                                     (1 - state["model"].clip(min=0.001, max=0.99)) * variance_pos)

            # Compute the negative log-likelihood of the data under this beta distribution
            nll = WeightedTensor(-variance_dist.log_prob(state["y"].weighted_value),state["y"].weight).weighted_value.mean()

            # Backpropagate to compute gradients
            nll.backward()

            # Take an optimization step
            optimizer.step()

        # Final optimized parameters
        variance_optimized = torch.nn.functional.softplus(variance)
        return variance_optimized.detach()

    @classmethod
    def diagonal_noise_std_update(
        cls,
        *,
        state: State,
        y_x_model: WeightedTensor[float],
        model_x_model: WeightedTensor[float],
    ) -> torch.Tensor:
        """
        Update rule for feature-wise `noise_std` (when directly a model parameter),
        from state & sufficient statistics.
        """
        raise(NotImplementedError)

    @classmethod
    def noise_std_specs(cls, dimension: int) -> ModelParameter:
        """
        Default specifications of `noise_std` variable when directly
        modelled as a parameter (no latent population variable).
        """
        update_rule = cls.scalar_noise_std_update if dimension == 1 else cls.diagonal_noise_std_update
        return ModelParameter(
            shape=(dimension,),
            suff_stats=Collect(**cls.noise_std_suff_stats()),
            update_rule=update_rule,
        )

    @classmethod
    def with_noise_std_as_model_parameter(cls, dimension: int, **kwargs):
        """
        Default instance of `FullGaussianObservationModel` with `noise_std`
        (scalar or diagonal depending on `dimension`) being a `ModelParameter`.
        """
        if not isinstance(dimension, int) or dimension < 1:
            raise ValueError(f"Dimension should be an integer >= 1. You provided {dimension}.")

        return cls(noise_std=cls.noise_std_specs(dimension),
                   model = kwargs.pop("model", "model"))

    def to_string(self) -> str:
        """method for parameter saving"""
        if self.extra_vars['noise_std'].shape == (1,):
            return "beta-scalar"
        return "beta-diagonal"