from __future__ import annotations

from typing import Callable

import torch

from leaspy.io.data.dataset import Dataset
from leaspy.models.utilities import compute_std_from_variance
from leaspy.utils.functional import Prod, Sqr
from leaspy.utils.weighted_tensor import (
    WeightedTensor,
    sum_dim,
    wsum_dim,
    wsum_dim_return_sum_of_weights_only,
    wsum_dim_return_weighted_sum_only,
)
from leaspy.variables.distributions import Normal
from leaspy.variables.specs import (
    LVL_FT,
    Collect,
    LinkedVariable,
    ModelParameter,
    VariableInterface,
    VariableName,
)
from leaspy.variables.state import State

from ._base import ObservationModel

__all__ = [
    "GaussianObservationModel",
    "FullGaussianObservationModel",
]


class GaussianObservationModel(ObservationModel):
    """
    Specialized `ObservationModel` for noisy observations with Gaussian residuals assumption.
    
    Parameters
    ----------
    name : :obj:`str`
        The name of observed variable (to name the data variable & attachment term related to this observation).
    getter : function :class:`~leaspy.io.data.dataset.Dataset` -> :class:`.WeightedTensor`
        The way to retrieve the observed values from the :class:`~leaspy.io.data.dataset.Dataset` (as a :class:`.WeightedTensor`):
        e.g. all values, subset of values - only x, y, z features, one-hot encoded features, ...
    loc : :obj:`str`
        The name of the variable representing the mean (location) of the Gaussian
    scale : :obj:`str`
        The name of the variable representing the standard deviation (scale) of the Gaussian (`noise_std`)
    **extra_vars : VariableInterface
        Additional variables required by the model

    Notes
    -----
    - The model uses `leaspy.variables.distributions.Normal` internally for computing
      the log-likelihood and related operations.
    """
    def __init__(
        self,
        name: VariableName,
        getter: Callable[[Dataset], WeightedTensor],
        loc: VariableName,
        scale: VariableName,
        **extra_vars: VariableInterface,
    ):
        super().__init__(name, getter, Normal(loc, scale), extra_vars=extra_vars)


class FullGaussianObservationModel(GaussianObservationModel):
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

    def __init__(self, noise_std: VariableInterface, **extra_vars: VariableInterface):
        super().__init__(
            name="y",
            getter=self.y_getter,
            loc="model",
            scale="noise_std",
            noise_std=noise_std,
            **extra_vars,
        )

    @staticmethod
    def y_getter(dataset: Dataset) -> WeightedTensor:
        """
        Extracts the observation values and mask from a dataset.

        Parameters
        ----------
        dataset : :class:`~leaspy.io.data.dataset.Dataset`
            A dataset object containing 'values' and 'mask' attributes

        Returns
        -------
        :class:`.WeightedTensor`
            A tensor containing the observed values and a boolean mask used as weights
            for likekelihood and loss computations

        Raises
        ------
        AssertionError
            If either `dataset.values`or `dataset.mask`is `None`.
        """
        assert dataset.values is not None
        assert dataset.mask is not None
        return WeightedTensor(dataset.values, weight=dataset.mask.to(torch.bool))

    @classmethod
    def noise_std_suff_stats(cls) -> dict[VariableName, LinkedVariable]:
        """
        Dictionary of sufficient statistics needed for `noise_std` (when directly a model parameter).
        
        Returns
        -------
        :obj:`dict` [ :class:`~leaspy.variables.specs.VariableName`, :class:`~leaspy.variables.specs.LinkedVariable`]
            A dictionary containing the sufficient statistics:

            * `"y_x_model"`: Product of the observed values (`"y"`) and the model predictions (`"model"`).
            * `"model_x_model"`: Squared values of the model predictions (`"model"`).
        """
        return dict(
            y_x_model=LinkedVariable(Prod("y", "model")),
            model_x_model=LinkedVariable(Sqr("model")),
        )

    @classmethod
    def scalar_noise_std_update(
        cls,
        *,
        state: State,
        y_x_model: WeightedTensor[float],
        model_x_model: WeightedTensor[float],
    ) -> torch.Tensor:
        """
        Update rule for scalar `noise_std` (when directly a model parameter), 
        from state & sufficient statistics.
        
        Computes a common `noise_std` for all the features

        Parameters
        ----------
        state : :class:`State`
            A state dictionary containing precomputed values.
        y_x_model : WeightedTensor[float]
            The weighted inner product between the observations and the model predictions.
        model_x_model : WeightedTensor[float]
            The weighted inner product of the model predictions with themselves.

        Returns
        -------
        :class:`torch.Tensor`
            The updated scalar value of the `noise_std`.
        """
        y_l2 = state["y_L2"]
        n_obs = state["n_obs"]
        # TODO? by linearity couldn't we only require `-2*y_x_model + model_x_model` as summary stat?
        # and couldn't we even collect the already summed version of it?
        s1 = sum_dim(y_x_model)
        s2 = sum_dim(model_x_model)
        noise_var = (y_l2 - 2 * s1 + s2) / n_obs.float()
        return compute_std_from_variance(
            noise_var,
            varname="noise_std",
            tol=cls.tol_noise_variance,
        )

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

        Computes one `noise_std` per feature.

        Parameters
        ----------
        state: :class:`State`
            A state dictionary containing precomputed values
        y_x_model : :class:`.WeightedTensor`[:obj:`float`]
            The weighted inner product between the observations and the model predictions.
        model_x_model : :class:`.WeightedTensor`[:obj:`float`]
            The weighted inner product of the model predictions with themselves.

        Returns
        -------
        :class:`torch.Tensor`
            The updated value of the `noise_std` for each feature.
        """
        y_l2_per_ft = state["y_L2_per_ft"]
        n_obs_per_ft = state["n_obs_per_ft"]
        # TODO: same remark as in `.scalar_noise_std_update()`
        # sum must be done after computation to use weights of y in model to mask missing data
        summed = sum_dim(-2 * y_x_model + model_x_model, but_dim=LVL_FT)
        noise_var = (y_l2_per_ft + summed) / n_obs_per_ft.float()

        return compute_std_from_variance(
            noise_var,
            varname="noise_std",
            tol=cls.tol_noise_variance,
        )

    @classmethod
    def noise_std_specs(cls, dimension: int) -> ModelParameter:
        """
        Default specifications of `noise_std` variable when directly
        modelled as a parameter (no latent population variable).

        Parameters
        ----------
        dimension : :obj:`int`
            The dimension of the `noise_std`.
            - If `dimension == 1`, a scalar `noise_std` deviation is assumed.
            - If `dimension > 1`, feature-wise independent `noise_std` deviations
            are assumed (diagonal noise).

        Returns
        -------
        ModelParameter
            The specification of the `noise_std`, including:
            - `shape`: tuple defining the parameter shape.
            - `suff_stats`: collected sufficient statistics needed for updates.
            - `update_rule`: method to update the parameter based on statistics.
        """
        update_rule = (
            cls.scalar_noise_std_update
            if dimension == 1
            else cls.diagonal_noise_std_update
        )
        return ModelParameter(
            shape=(dimension,),
            suff_stats=Collect(**cls.noise_std_suff_stats()),
            update_rule=update_rule,
        )

    @classmethod
    def with_noise_std_as_model_parameter(cls, dimension: int):
        """
        Default instance of `FullGaussianObservationModel` with `noise_std`
        (scalar or diagonal depending on `dimension`) being a `ModelParameter`.

        Parameters
        ----------
        dimension : :obj:`int`
            The dimension of the `noise_std`.
            - If `dimension == 1`, a scalar `noise_std` is assumed.
            - If `dimension > 1`, feature-wise independent `noise_std` deviations
            are assumed (diagonal noise).

        Returns
        -------
        FullGaussianObservationModel
            A configured instance with `noise_std` as a `ModelParameter`, along with the
            necessary sufficient statistics for inference.

        Raises
        ------
        ValueError
            If `dimension` is not a positive integer.
        """
        if not isinstance(dimension, int) or dimension < 1:
            raise ValueError(
                f"Dimension should be an integer >= 1. You provided {dimension}."
            )
        if dimension == 1:
            extra_vars = {
                "y_L2": LinkedVariable(
                    Sqr("y").then(wsum_dim_return_weighted_sum_only)
                ),
                "n_obs": LinkedVariable(
                    Sqr("y").then(wsum_dim_return_sum_of_weights_only)
                ),
            }
        else:
            extra_vars = {
                "y_L2_per_ft": LinkedVariable(
                    Sqr("y").then(wsum_dim_return_weighted_sum_only, but_dim=LVL_FT)
                ),
                "n_obs_per_ft": LinkedVariable(
                    Sqr("y").then(wsum_dim_return_sum_of_weights_only, but_dim=LVL_FT)
                ),
            }
        return cls(noise_std=cls.noise_std_specs(dimension), **extra_vars)

    # Util functions not directly used in code

    @classmethod
    def compute_rmse(
        cls,
        *,
        y: WeightedTensor[float],
        model: WeightedTensor[float],
    ) -> torch.Tensor:
        """
        Computes the Root Mean Square Error (RMSE) between predictions and observations.

        Parameters
        ----------
        y : :class:`.WeightedTensor`[:obj:`float`]
            The observed target values with associated weights.
        model : :class:`.WeightedTensor`[:obj:`float`]
            The model predictions with the same shape and weighting scheme as `y`.

        Returns
        -------
        :class:`torch.Tensor`
            A scalar tensor representing the RMSE between `model` and `y`.

        """
        l2: WeightedTensor[float] = (model - y) ** 2
        l2_sum, n_obs = wsum_dim(l2)
        return (l2_sum / n_obs.float()) ** 0.5

    @classmethod
    def compute_rmse_per_ft(
        cls,
        *,
        y: WeightedTensor[float],
        model: WeightedTensor[float],
    ) -> torch.Tensor:
        """
        Computes the Root Mean Square Error (RMSE) between predictions and observations
        separately for each feature.

        Parameters
        ----------
        y : :class:`.WeightedTensor`[:obj:`float`]
            The observed target values with associated weights.
        model : :class:`.WeightedTensor`[:obj:`float`]
            The model predictions with the same shape and weighting scheme as `y`.

        Returns
        -------
        :class:`torch.Tensor`
            A scalar tensor representing the RMSE between `model` and `y`.
        """
        l2: WeightedTensor[float] = (model - y) ** 2
        l2_sum_per_ft, n_obs_per_ft = wsum_dim(l2, but_dim=LVL_FT)
        return (l2_sum_per_ft / n_obs_per_ft.float()) ** 0.5

    def to_string(self) -> str:
        """method for parameter saving"""
        if self.extra_vars["noise_std"].shape == (1,):
            return "gaussian-scalar"
        return "gaussian-diagonal"
