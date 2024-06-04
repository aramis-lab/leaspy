"""Module defining the Bernoulli noise model."""

from __future__ import annotations

import torch
from typing import Union, Tuple

from .base import DistributionFamily, BaseNoiseModel
from leaspy.io.data.dataset import Dataset


class BernoulliFamily(DistributionFamily):
    """
    Distribution family for Bernoulli noise model.

    Parameters
    ----------
    parameters : :obj:`dict` [ :obj:`str`, :class:`torch.Tensor` ] or None
        Values for all the free parameters of the distribution family.
        All of them must have values before using the sampling methods.

    Attributes
    ----------
    free_parameters : :obj:`frozenset` of :obj:`str`
        Name of all the free parameters (but `loc`) needed to characterize the distribution.
        .. note::
            For each parameter, if a method named
            `validate_xxx(torch.Tensor -> torch.Tensor)` exists,
            then it will be used for user-input validation of parameter "xxx".
    factory : None or function(free parameters values) -> :class:`torch.distributions.distribution.Distribution`
        The factory for the distribution family.
    """
    factory = torch.distributions.Bernoulli
    free_parameters = frozenset()


class BernoulliNoiseModel(BernoulliFamily, BaseNoiseModel):
    """
    Class implementing Bernoulli noise models.

    Parameters
    ----------
    parameters : :obj:`dict` [ :obj:`str`, :class:`torch.Tensor` ] or None
        Values for all the free parameters of the distribution family.
        All of them must have values before using the sampling methods.
    """

    def compute_nll(
        self,
        data: Dataset,
        predictions: torch.Tensor,
        *,
        with_gradient: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute the negative log-likelihood and its gradient wrt predictions.

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
        predictions = torch.clamp(predictions, 1e-7, 1.0 - 1e-7)
        ll = data.values * torch.log(predictions) + (1.0 - data.values) * torch.log(
            1.0 - predictions
        )
        nll = -data.mask.float() * ll
        if not with_gradient:
            return nll
        nll_grad = (
            data.mask.float()
            * (predictions - data.values)
            / (predictions * (1.0 - predictions))
        )
        return nll, nll_grad
