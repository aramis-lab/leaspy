from typing import Optional

import numpy as np
import statsmodels.api as sm
import torch

from leaspy.io.outputs import IndividualParameters
from leaspy.utils.typing import DictParamsTorch

from .stateless import StatelessModel

__all__ = ["LMEModel"]


class LMEModel(StatelessModel):
    r"""LMEModel is a benchmark model that fits and personalize a linear mixed-effects model.

    The model specification is the following:

    .. math:: y_{ij} = fixed_{intercept} + random_{intercept_i} + (fixed_{slopeAge} + random_{slopeAge_i}) * age_{ij} + \epsilon_{ij}

    with:
        * :math:`y_{ij}`: value of the feature of the i-th subject at his j-th visit,
        * :math:`age_{ij}`: age of the i-th subject at his j-th visit.
        * :math:`\epsilon_{ij}`: residual Gaussian noise (independent between visits)

    .. warning::
        This model must be fitted on one feature only (univariate model).

    TODO? add some covariates in this very simple model.

    Parameters
    ----------
    name : :obj:`str`
        The model's name.
    **kwargs
        Model hyperparameters:
            * with_random_slope_age : :obj:`bool` (default ``True``).

    Attributes
    ----------
    name : :obj:`str`
        The model's name.
    is_initialized : :obj:`bool`
        ``True`` if the model is initialized, ``False`` otherwise.
    with_random_slope_age : :obj:`bool` (default ``True``)
        Has the LME a random slope for subject's age?
        Otherwise it only has a random intercept per subject.
    features : :obj:`list` of :obj:`str`
        List of the model features.

        .. warning::
            LME has only one feature.

    dimension : :obj:`int`
        Will always be 1 (univariate).
    parameters : :obj:`dict`
        Contains the model parameters. In particular:
            * ``ages_mean`` : :obj:`float`
                Mean of ages (for normalization).
            * ``ages_std`` : :obj:`float`
                Std-dev of ages (for normalization).
            * ``fe_params`` : :class:`np.ndarray` of :obj:`float`
                Fixed effects.
            * ``cov_re`` : :class:`np.ndarray`
                Variance-covariance matrix of random-effects.
            * ``cov_re_unscaled_inv`` : :class:`np.ndarray`
                Inverse of unscaled (= divided by variance of noise) variance-covariance matrix of random-effects.
                This matrix is used for personalization to new subjects.
            * ``noise_std`` : :obj:`float`
                Std-dev of Gaussian noise.
            * ``bse_fe``, ``bse_re`` : :class:`np.ndarray` of :obj:`float`
                Standard errors on fixed-effects and random-effects respectively (not used in ``Leaspy``).

    See Also
    --------
    :class:`~leaspy.algo.others.lme_fit.LMEFitAlgorithm`
    :class:`~leaspy.algo.others.lme_personalize.LMEPersonalizeAlgorithm`
    """

    def __init__(
        self, name: str, with_random_slope_age: Optional[bool] = True, **kwargs
    ):
        super().__init__(name, **kwargs)
        self.with_random_slope_age = with_random_slope_age
        self.dimension = 1

    @property
    def hyperparameters(self) -> DictParamsTorch:
        """Dictionary of values for model hyperparameters."""
        return {}

    def compute_individual_trajectory(
        self,
        timepoints: list[float],
        individual_parameters: IndividualParameters,
    ) -> torch.Tensor:
        """Compute scores values at the given time-point(s) given a subject's individual parameters.

        Parameters
        ----------
        timepoints : array-like of ages (not normalized)
            Timepoints to compute individual trajectory at.

        individual_parameters : :obj:`dict`
            Individual parameters:
                * random_intercept
                * random_slope_age (if ``with_random_slope_age == True``)

        Returns
        -------
        :class:`torch.Tensor` of :obj:`float` :
            The individual trajectories. The shape of the tensor is
            ``(n_individuals == 1, n_tpts == len(timepoints), n_features == 1)``.
        """
        # normalize ages (np.ndarray of float, 1D)
        ages_norm = (
            np.array(timepoints).reshape(-1) - self.parameters["ages_mean"]
        ) / self.parameters["ages_std"]

        # design matrix (same for fixed and random effects)
        X = sm.add_constant(ages_norm, prepend=True, has_constant="add")

        # assert 'random_intercept' in individual_parameters
        if not self.with_random_slope_age:
            # no random slope on ages (fixed effect only)
            re_params = np.array([individual_parameters["random_intercept"].item(), 0])
        else:
            # assert 'random_slope_age' in individual_parameters
            re_params = np.array(
                [
                    individual_parameters["random_intercept"].item(),
                    individual_parameters["random_slope_age"].item(),
                ]
            )
        y = X @ (self.parameters["fe_params"] + re_params)

        return torch.tensor(y, dtype=torch.float32).reshape((1, -1, 1))
