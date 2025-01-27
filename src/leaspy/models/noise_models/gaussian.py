"""Module defining gaussian noise models."""

from __future__ import annotations

import abc
import math
from dataclasses import dataclass
from typing import ClassVar, Optional, Tuple, Union

import torch

from leaspy.exceptions import LeaspyInputError
from leaspy.io.data.dataset import Dataset
from leaspy.models.utilities import compute_std_from_variance
from leaspy.utils.typing import DictParamsTorch

from .base import BaseNoiseModel, DistributionFamily

TWO_PI = torch.tensor(2 * math.pi)


class GaussianFamily(DistributionFamily):
    """
    Gaussian distribution family for Gaussian noise models.

    Parameters
    ----------
    parameters : :obj:`dict` [ :obj:`str`, :class:`torch.Tensor` ] or None
        Values for all the free parameters of the distribution family.
        All of them must have values before using the sampling methods.

    Attributes
    ----------
    free_parameters : :obj:`frozenset`
        The set of free parameters. For GaussianFamily this set
        is composed of a unique element "scale".
    factory : :class:`torch.distributions.distribution.Distribution`
        The underlying distribution.
    """

    free_parameters = frozenset({"scale"})
    factory = torch.distributions.Normal

    def validate_scale(self, scale: torch.Tensor) -> torch.Tensor:
        """
        Scale parameter validation (may be extended in children classes).

        Parameters
        ----------
        scale : :class:`torch.Tensor`
            The scale to validate.

        Returns
        -------
        :class:`torch.Tensor`
            The validated scale.
        """
        scale = scale.float()
        if (scale <= 0).any():
            raise LeaspyInputError(
                "The noise `scale` parameter should be > 0, "
                f"which is not the case for {scale}."
            )
        return scale


@dataclass
class AbstractGaussianNoiseModel(GaussianFamily, BaseNoiseModel):
    """
    Base class for Gaussian noise models.

    Parameters
    ----------
    parameters : :obj:`dict` [ :obj:`str`, :class:`torch.Tensor` ] or None
        Values for all the free parameters of the distribution family.
        All of them must have values before using the sampling methods.
    scale_dimension : :obj:`int`, optional
        The scale dimension.

    Attributes
    ----------
    scale_dimension : :obj:`int`, optional
        The scale dimension.
    parameters : :obj:`dict`
        Contains the parameters relative to the noise model.
    canonical_loss_properties : :obj:`tuple`
        The properties for the canonical loss.
    """

    scale_dimension: Optional[int] = None
    canonical_loss_properties: ClassVar = ("standard-deviation of the noise", ".2%")

    def validate_scale(self, scale: torch.Tensor) -> torch.Tensor:
        """
        Add a size-validation for scale parameter.

        Parameters
        ----------
        scale : :class:`torch.Tensor`
            The scale to validate.

        Returns
        -------
        :class:`torch.Tensor`
            The validated scale.
        """
        scale = super().validate_scale(scale)
        if self.scale_dimension is not None and scale.numel() != self.scale_dimension:
            raise LeaspyInputError(
                f"You have provided a noise `scale` ({scale}) of size {scale.numel()} "
                f"whereas a size = {self.scale_dimension} was expected."
            )
        return scale

    @staticmethod
    def compute_residuals(data: Dataset, predictions: torch.Tensor) -> torch.Tensor:
        """
        Compute the residuals of the given predictions.

        Parameters
        ----------
        data : :class:`.Dataset`
            The dataset related to the computation of the residuals.
        predictions : :class:`torch.Tensor`
            The model's predictions from which to compute the residuals.

        Returns
        -------
        :class:`torch.Tensor` :
            The residuals.
        """
        return data.mask.float() * (predictions - data.values)

    @classmethod
    def compute_l2_residuals(
        cls, data: Dataset, predictions: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the squared residuals of the given predictions.

        Parameters
        ----------
        data : :class:`.Dataset`
            The dataset related to the computation of the squared residuals.
        predictions : :class:`torch.Tensor`
            The model's predictions from which to compute the squared residuals.

        Returns
        -------
        :class:`torch.Tensor` :
            The squared residuals.
        """
        res = cls.compute_residuals(data, predictions)
        return res * res

    def _compute_noise_var_in_dimension(self, dimension: int) -> torch.Tensor:
        """
        Compute the noise variance and expand it to the provided dimension.

        Parameters
        ----------
        dimension : :obj:`int`
            The dimension in which to expand the noise variance.

        Returns
        -------
        :class:`torch.Tensor` :
            The noise variance expanded in requested dimension.
        """
        self.raise_if_partially_defined()
        noise_var = self.parameters["scale"] ** 2
        # shape: (n_individuals, n_visits, n_features)
        return noise_var.expand((1, 1, dimension))

    def _compute_nll_from_residuals(
        self,
        data: Dataset,
        residuals: torch.Tensor,
        *,
        incl_const: bool = True,
        with_gradient: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Return negative log-likelihood (without summation),
        and optionally its jacobian w.r.t prediction.

        Parameters
        ----------
        data : :class:`.Dataset`
            The dataset related to the computation of the log likelihood.
        residuals : :class:`torch.Tensor`
            The residuals from which to compute the negative log likelihood.
        incl_const : :obj:`bool`, optional
            If True, adds a constant term to the negative log likelihood.
            Default=True.
        with_gradient : :obj:`bool`, optional
            If True, returns also the jacobian of the negative log likelihood
            wrt the predictions. If False, only returns the negative log likelihood.
            Default=False.

        Returns
        -------
        :class:`torch.Tensor` or :obj:`tuple` of :class:`torch.Tensor`
            The negative log likelihood (and its jacobian if requested).
        """
        noise_var = self._compute_noise_var_in_dimension(data.dimension)
        nll = 0.5 / noise_var * residuals * residuals
        if incl_const:
            nll += 0.5 * torch.log(TWO_PI * noise_var) * data.mask.float()
        if not with_gradient:
            return nll
        return nll, residuals / noise_var

    def compute_nll(
        self,
        data: Dataset,
        predictions: torch.Tensor,
        *,
        with_gradient: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Negative log-likelihood without summation (and its
        gradient w.r.t. predictions if requested).

        Parameters
        ----------
        data : :class:`.Dataset`
            The dataset related to the computation of the log likelihood.
        predictions : :class:`torch.Tensor`
            The model's predictions from which to compute the log likelihood.
        with_gradient : :obj:`bool`, optional
            If True, returns also the gradient of the negative log likelihood
            wrt the predictions.
            If False, only returns the negative log likelihood.
            Default=False.

        Returns
        -------
        :class:`torch.Tensor` or :obj:`tuple` of :class:`torch.Tensor`
            The negative log likelihood (and its jacobian if requested).
        """
        residuals = self.compute_residuals(data, predictions)
        return self._compute_nll_from_residuals(
            data, residuals, incl_const=True, with_gradient=with_gradient
        )

    def compute_sufficient_statistics(
        self, data: Dataset, predictions: torch.Tensor
    ) -> DictParamsTorch:
        """
        Compute the specific sufficient statistics and metrics for this noise-model.

        Parameters
        ----------
        data : :class:`.Dataset`
            The dataset related to the computation of the sufficient statistics.
        predictions : :class:`torch.Tensor`
            The model's predictions from which to compute the sufficient statistics.

        Returns
        -------
        DictParamsTorch :
            The sufficient statistics.
        """
        predictions = data.mask.float() * predictions
        return {
            "obs_x_reconstruction": data.values * predictions,
            "reconstruction_x_reconstruction": predictions * predictions,
        }

    def update_parameters_from_sufficient_statistics(
        self,
        data: Dataset,
        sufficient_statistics: DictParamsTorch,
    ) -> None:
        """
        In-place update of free parameters from provided sufficient statistics.

        Parameters
        ----------
        data : :class:`.Dataset`
            The dataset related to the computation of the log likelihood.
        sufficient_statistics : DictParamsTorch
            The sufficient statistics to use for parameter update.
        """
        noise_var = self._compute_noise_variance_from_sufficient_statistics(
            data, sufficient_statistics
        )
        self.update_parameters(
            scale=compute_std_from_variance(noise_var, varname="noise_std")
        )

    @classmethod
    @abc.abstractmethod
    def _compute_noise_variance_from_sufficient_statistics(
        cls,
        data: Dataset,
        sufficient_statistics: dict,
    ) -> torch.FloatTensor:
        """
        Compute the noise variance from the provided sufficient statistics.

        Parameters
        ----------
        data : :class:`.Dataset`
            The dataset related to the computation of the log likelihood.
        sufficient_statistics : :obj:`dict`
            The sufficient statistics to use for variance computation.
        """
        ...

    def update_parameters_from_predictions(
        self, data: Dataset, predictions: torch.Tensor
    ) -> None:
        """
        In-place update of free parameters from provided predictions.

        Parameters
        ----------
        data : :class:`.Dataset`
            The dataset related to the computation of the log likelihood.
        predictions : :class:`torch.Tensor`
            The model's predictions from which to update the parameters.
        """
        self.update_parameters(scale=self.compute_rmse(data, predictions))

    @classmethod
    def compute_rmse(cls, data: Dataset, predictions: torch.Tensor) -> torch.Tensor:
        """
        Computes root mean squared error of provided data vs. predictions.

        Parameters
        ----------
        data : :class:`.Dataset`
            The dataset related to the computation of the root mean squared error.
        predictions : :class:`torch.Tensor`
            The model's predictions from which to compute the root mean squared error.

        Returns
        -------
        :class:`torch.Tensor`
            The root mean squared error.
        """
        l2_res = cls.compute_l2_residuals(data, predictions)
        mse = cls._compute_mse_from_l2_residuals(data, l2_res)
        return torch.sqrt(mse)

    @classmethod
    @abc.abstractmethod
    def _compute_mse_from_l2_residuals(
        cls,
        data: Dataset,
        l2_residuals: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the mean squared error from the squared residuals.

        Parameters
        ----------
        data : :class:`.Dataset`
            The dataset related to the computation of the log likelihood.
        l2_residuals : :class:`torch.Tensor`
            The squared residuals from which to compute the mean squared error.

        Returns
        -------
        :class:`torch.Tensor`
            The mean squared error.
        """
        ...

    @classmethod
    def compute_canonical_loss(
        cls,
        data: Dataset,
        predictions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute a human-friendly overall loss (RMSE).

        Parameters
        ----------
        data : :class:`.Dataset`
            The dataset related to the computation of the log likelihood.
        predictions : :class:`torch.Tensor`
            The model's predictions from which to compute the canonical loss.

        Returns
        -------
        :class:`torch.Tensor`
            The computed loss.
        """
        return cls.compute_rmse(data, predictions)


class GaussianScalarNoiseModel(AbstractGaussianNoiseModel):
    """
    Class implementing scalar Gaussian noise models.

    Parameters
    ----------
    parameters : :obj:`dict` [ :obj`str`, :class:`torch.Tensor` ] or None
        Values for all the free parameters of the distribution family.
        All of them must have values before using the sampling methods.
    scale_dimension : :obj:`int`, optional
        The scale dimension.

    Attributes
    ----------
    scale_dimension : :obj:`int`, optional
        The scale dimension.
    parameters : :obj:`dict`
        Contains the parameters relative to the noise model.
    """

    scale_dimension: int = 1

    def validate_scale(self, scale: torch.Tensor) -> torch.Tensor:
        """
        Ensure the scale is valid.

        Parameters
        ----------
        scale : :class:`torch.Tensor`
            The scale to validate.

        Returns
        -------
        :class:`torch.Tensor`
            The validated scale.
        """
        return super().validate_scale(scale).view(())

    @classmethod
    def _compute_mse_from_l2_residuals(
        cls, data: Dataset, l2_res: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the mean squared error from squared residuals.
        Also sum on features.

        Parameters
        ----------
        data : :class:`.Dataset`
            The dataset related to the computation of the log likelihood.
        l2_res : :class:`torch.Tensor`
            The squared residuals.

        Returns
        -------
        :class:`torch.Tensor`
            The computed mean squared error.
        """
        return l2_res.sum() / data.n_observations

    @classmethod
    def _compute_noise_variance_from_sufficient_statistics(
        cls,
        data: Dataset,
        sufficient_statistics: dict,
    ) -> torch.Tensor:
        """
        Compute the noise variance from provided sufficient statistics.
        Sum on features.

        Parameters
        ----------
        data : :class:`.Dataset`
            The dataset related to the computation of the log likelihood.
        sufficient_statistics : :obj:`dict`
            The sufficient statistics from which to compute the noise variance.

        Returns
        -------
        :class:`torch.Tensor` :
            The computed noise variance.
        """
        s1 = sufficient_statistics["obs_x_reconstruction"].sum()
        s2 = sufficient_statistics["reconstruction_x_reconstruction"].sum()
        return (data.L2_norm - 2.0 * s1 + s2) / data.n_observations


class GaussianDiagonalNoiseModel(AbstractGaussianNoiseModel):
    """
    Class implementing diagonal Gaussian noise models.

    Parameters
    ----------
    parameters : :obj:`dict` [ :obj:`str`, :class:`torch.Tensor` ] or None
        Values for all the free parameters of the distribution family.
        All of them must have values before using the sampling methods.
    scale_dimension : :obj:`int`, optional
        The scale dimension.

    Attributes
    ----------
    scale_dimension : :obj:`int`, optional
        The scale dimension.
    parameters : :obj:`dict`
        Contains the parameters relative to the noise model.
    """

    def validate_scale(self, scale: torch.Tensor) -> torch.Tensor:
        """
        Ensure the scale is valid.

        Parameters
        ----------
        scale : :class:`torch.Tensor`
            The scale to validate.

        Returns
        -------
        :class:`torch.Tensor`
            The validated scale.
        """
        return super().validate_scale(scale).view(-1)

    @classmethod
    def _compute_mse_from_l2_residuals(
        cls, data: Dataset, l2_res: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the mean squared error from squared residuals.
        Do not sum on features.

        Parameters
        ----------
        data : :class:`.Dataset`
            The dataset related to the computation of the log likelihood.
        l2_res : :class:`torch.Tensor`
            The squared residuals.

        Returns
        -------
        :class:`torch.Tensor`
            The computed mean squared error.
        """
        return l2_res.sum(dim=(0, 1)) / data.n_observations_per_ft.float()

    @classmethod
    def _compute_noise_variance_from_sufficient_statistics(
        cls,
        data: Dataset,
        sufficient_statistics: dict,
    ) -> torch.Tensor:
        """
        Compute the noise variance from provided sufficient statistics.
        Do not sum on features.

        Parameters
        ----------
        data : :class:`.Dataset`
            The dataset related to the computation of the log likelihood.
        sufficient_statistics : :obj:`dict`
            The sufficient statistics from which to compute the noise variance.

        Returns
        -------
        :class:`torch.Tensor` :
            The computed noise variance.
        """
        s1 = sufficient_statistics["obs_x_reconstruction"].sum(dim=(0, 1))
        s2 = sufficient_statistics["reconstruction_x_reconstruction"].sum(dim=(0, 1))
        return (
            data.L2_norm_per_ft - 2.0 * s1 + s2
        ) / data.n_observations_per_ft.float()
