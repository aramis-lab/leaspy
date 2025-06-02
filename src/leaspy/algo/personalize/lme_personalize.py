import warnings

import numpy as np
import statsmodels.api as sm

from leaspy.io.data import Dataset
from leaspy.io.outputs.individual_parameters import IndividualParameters
from leaspy.models import LMEModel

from ..base import AlgorithmName
from .base import PersonalizeAlgorithm

__all__ = ["LMEPersonalizeAlgorithm"]


class LMEPersonalizeAlgorithm(PersonalizeAlgorithm[LMEModel, IndividualParameters]):
    r"""Personalization algorithm associated to :class:`~leaspy.models.LMEModel`.

    Parameters
    ----------
    settings : :class:`.AlgorithmSettings`
        Algorithm settings (none yet).
        Most LME parameters are defined within LME model and LME fit algorithm.

    Attributes
    ----------
    name : ``'lme_personalize'``
    """

    name: AlgorithmName = AlgorithmName.PERSONALIZE_LME
    deterministic: bool = True

    def _compute_individual_parameters(
        self, model: LMEModel, dataset: Dataset, **kwargs
    ) -> IndividualParameters:
        individual_parameters = IndividualParameters()
        residuals = []
        if model.features != dataset.headers:
            raise ValueError(
                "Your data and the model you are using for personalisation do not have the same headers."
            )
        for individual in range(dataset.n_individuals):
            idx = dataset.indices[individual]
            times = dataset.get_times_patient(individual)
            values = dataset.get_values_patient(individual).numpy()
            ind_ip, ind_residuals = self._get_individual_random_effects_and_residuals(
                model, times, values
            )
            residuals.append(ind_residuals)
            individual_parameters.add_individual_parameters(str(idx), ind_ip)

        return individual_parameters

    @staticmethod
    def _remove_nans(values, times):
        values = values.flatten()
        mask = ~np.isnan(values)
        values = values[mask]
        times = times[mask]
        return values, times

    @classmethod
    def _get_individual_random_effects_and_residuals(
        cls, model: LMEModel, times, values
    ):
        values, times = cls._remove_nans(values, times)

        ages_norm = (times - model.parameters["ages_mean"]) / model.parameters[
            "ages_std"
        ]

        X = sm.add_constant(ages_norm, prepend=True, has_constant="add")
        residuals = values - X @ model.parameters["fe_params"]

        cov_re_unscaled_inv = model.parameters["cov_re_unscaled_inv"]

        if not model.with_random_slope_age:
            # only valid with random intercept ("Z"=[1,...,1] and cov_re is a scalar)
            n = len(values)  # number of effective observations
            random_intercept = np.sum(residuals) / (n + cov_re_unscaled_inv.item())

            re_d = {"random_intercept": random_intercept}

            residuals = residuals - random_intercept
        else:
            # valid anytime (exog_re = X)
            re = cls._generic_get_random_effects(
                residuals, X, cov_re_unscaled_inv
            ).squeeze()

            re_d = {"random_intercept": re[0], "random_slope_age": re[1]}

            residuals = residuals - X @ re

        return re_d, residuals

    @staticmethod
    def _generic_get_random_effects(resid, Z, cov_re_unscaled_inv):
        """
        The conditional means of random effects given the data.
        cf. http://sia.webpopix.org/lme.html#estimation-of-the-random-effects

        Parameters
        ----------
        resid : :class:`numpy.ndarray` (n_i,)
            endog - fixed_effects * exog
        Z : :class:`numpy.ndarray` (n_i, k_re)
            exog_re
        cov_re_unscaled_inv : :class:`numpy.ndarray` (k_re, k_re)
            inverse

        Returns
        -------
        random_effects : :class:`numpy.ndarray` (k_re,)
            For a given individual
        """

        tZZ = np.dot(Z.T, Z)
        G = np.linalg.inv(tZZ + cov_re_unscaled_inv)
        return np.dot(G, np.dot(Z.T, resid))  # less costly to multiply in this order

    def set_output_manager(self, output_settings):
        """
        Not implemented.
        """
        if output_settings is not None:
            warnings.warn(
                "Settings logs in lme personalize algorithm is not supported."
            )
        return
