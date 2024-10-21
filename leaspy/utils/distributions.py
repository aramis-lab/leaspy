"""Module defining useful distribution for ordinal noise model."""

from typing import Union, Optional, ClassVar

import numpy as np
import torch
from torch.distributions.constraints import unit_interval


def discrete_sf_from_pdf(pdf: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    """
    Compute the discrete survival function values [P(X > l), i.e.
    P(X >= l+1), l=0..L-1] (l=0..L-1) from the discrete probability density
    [P(X = l), l=0..L] (assuming discrete levels are in last dimension).

    Parameters
    ----------
    pdf : torch.Tensor or np.ndarray
        The discrete probability density.

    Returns
    -------
    np.ndarray :
        The discrete survival function values.
    """
    return (1 - pdf.cumsum(-1))[..., :-1]

def compute_ordinal_pdf_from_ordinal_sf(
    ordinal_sf: torch.Tensor,
    dim_ordinal_levels: int = 3,
) -> torch.Tensor:
    """
    Computes the probability density (or its jacobian) of an ordinal
    model [P(X = l), l=0..L] from `ordinal_sf` which are the survival
    function probabilities [P(X > l), i.e. P(X >= l+1), l=0..L-1] (or its jacobian).

    Parameters
    ----------
    ordinal_sf : `torch.FloatTensor`
        Survival function values : ordinal_sf[..., l] is the proba to be superior or equal to l+1
        Dimensions are:
        * 0=individual
        * 1=visit
        * 2=feature
        * 3=ordinal_level [l=0..L-1]
        * [4=individual_parameter_dim_when_gradient]
    dim_ordinal_levels : int, default = 3
        The dimension of the tensor where the ordinal levels are.

    Returns
    -------
    ordinal_pdf : `torch.FloatTensor` (same shape as input, except for dimension 3 which has one more element)
        ordinal_pdf[..., l] is the proba to be equal to l (l=0..L)
    """
    # nota: torch.diff was introduced in v1.8 but would not highly improve performance of this routine anyway
    s = list(ordinal_sf.shape)
    s[dim_ordinal_levels] = 1
    last_row = torch.zeros(size=tuple(s))
    if len(s) == 5:  # in the case of gradient we added a dimension
        first_row = last_row  # gradient(P>=0) = 0
    else:
        first_row = torch.ones(size=tuple(s))  # (P>=0) = 1
    sf_sup = torch.cat([first_row, ordinal_sf], dim=dim_ordinal_levels)
    sf_inf = torch.cat([ordinal_sf, last_row], dim=dim_ordinal_levels)
    pdf = sf_sup - sf_inf
    return pdf

class MultinomialDistribution(torch.distributions.Multinomial):
    """
    Class for a multinomial distribution with only one sample based on the Multinomial torch distrib.

    Parameters
    ----------
    probs : torch.Tensor
        The pdf of the multinomial distribution.

    Attributes
    ----------
    """

    arg_constraints: ClassVar = {}

    def __init__(self, probs, **kwargs):
        super().__init__(total_count=1, probs=probs, **kwargs)

    @classmethod
    def from_sf(cls, sf: torch.Tensor, **kws):
        """
        Generate a new MultinomialDistribution from its survival
        function instead of its probability density function.

        Parameters
        ----------
        pdf : :class:`torch.Tensor`
            The input probability density function.
        **kws
            Additional keyword arguments to be passed for instance initialization.
        """
        return cls(compute_ordinal_pdf_from_ordinal_sf(sf, dim_ordinal_levels=-1), **kws)


class MultinomialCdfDistribution(torch.distributions.Distribution):
    """
    Class for a multinomial distribution with only sample method.

    Parameters
    ----------
    sf : torch.Tensor
        Values of the survival function [P(X > l) for l=0..L-1 where L is max_level]
        from which the distribution samples.
        Ordinal levels are assumed to be in the last dimension.
        Those values must be in [0, 1], and decreasing when ordinal level increases (not checked).
    validate_args : bool, optional (default True)
        Whether to validate the arguments or not (None for default behavior, which changed after torch >= 1.8 to True).

    Attributes
    ----------
    cdf : torch.Tensor
        The cumulative distribution function [P(X <= l) for l=0..L] from which the distribution samples.
        The shape of latest dimension is L+1 where L is max_level.
        We always have P(X <= L) = 1
    arg_constraints : dict
        Contains the constraints on the arguments.
    """

    arg_constraints: ClassVar = {}

    def __init__(self, sf: torch.Tensor, *, validate_args: Optional[bool] = True):
        super().__init__(validate_args=validate_args)
        if self._validate_args:
            assert unit_interval.check(
                sf
            ).all(), "Bad probabilities in MultinomialDistribution"
        # shape of the sample (we discard the last dimension, used to store the different ordinal levels)
        self._sample_shape = sf.shape[:-1]
        # store the cumulative distribution function with trailing P(X <= L) = 1
        self.cdf = torch.cat((1.0 - sf, torch.ones((*self._sample_shape, 1))), dim=-1)

    @classmethod
    def from_pdf(cls, pdf: torch.Tensor, **kws):
        """
        Generate a new MultinomialDistribution from its probability density
        function instead of its survival function.

        Parameters
        ----------
        pdf : :class:`torch.Tensor`
            The input probability density function.
        **kws
            Additional keyword arguments to be passed for instance initialization.
        """
        return cls(discrete_sf_from_pdf(pdf), **kws)

    def sample(self) -> torch.Tensor:
        """
        Multinomial sampling.

        We sample uniformly on [0, 1( but for the latest dimension corresponding
        to ordinal levels this latest dimension will be broadcast when comparing
        with `cdf`.

        Returns
        -------
        torch.Tensor :
            Vector of integer values corresponding to the multinomial sampling.
            Result is in [[0, L]]
        """
        r = torch.rand(self._sample_shape).unsqueeze(-1)
        return (r < self.cdf).int().argmax(dim=-1)
