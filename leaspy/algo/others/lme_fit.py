from __future__ import annotations
from typing import TYPE_CHECKING
import warnings

import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM, MixedLMParams
import torch
import numpy as np

from leaspy.algo.abstract_algo import AbstractAlgo
from leaspy.models.lme import LMEModel
from leaspy.exceptions import LeaspyDataInputError

if TYPE_CHECKING:
    from leaspy.io.data.dataset import Dataset


class LMEFitAlgorithm(AbstractAlgo): # AbstractFitAlgo not so generic (EM)
    """
    Calibration algorithm associated to :class:`~.models.lme_model.LMEModel`

    Parameters
    ----------
    settings : :class:`.AlgorithmSettings`
        * with_random_slope_age : bool
            If False: only varying intercepts
            If True: random intercept & random slope w.r.t ages

            .. deprecated:: 1.2

            You should rather define this directly as an hyperparameter of LME model.
        * force_independent_random_effects : bool
            Force independence of random intercept & random slope
        * other keyword arguments passed to :meth:`statsmodels.regression.mixed_linear_model.MixedLM.fit`

    See Also
    --------
    :class:`statsmodels.regression.mixed_linear_model.MixedLM`
    """

    name = 'lme_fit'
    family = 'fit'

    def __init__(self, settings):

        super().__init__(settings)

        params = settings.parameters.copy()

        # Get model hyperparameters from fit algorithm parameters
        # deprecated only for backward-compat (to be removed soon...),
        self._model_hyperparams_to_set = {
            hp_name: params.pop(hp_name)
            for hp_name in LMEModel._hyperparameters.keys()
            if hp_name in params
        }

        # Algorithm true parameters
        self.force_independent_random_effects = params.pop('force_independent_random_effects')

        # Remaining parameters are parameters of statsmodels `fit` method
        self.sm_fit_parameters = params # popped from other params

    def run_impl(self, model: LMEModel, dataset: Dataset):
        """
        Main method, refer to abstract definition in :meth:`~.algo.fit.abstract_fit_algo.AbstractFitAlgo.run`.

        TODO fix proper inheritance

        Parameters
        ----------
        model : :class:`~.LMEModel`
            A subclass object of leaspy `LMEModel`.
        dataset : :class:`.Dataset`
            Dataset object build with leaspy class objects Data, algo & model

        Returns
        -------
        2-tuple:
            * None
            * noise scale (std-dev), scalar
        """

        # DEPRECATED - TO BE REMOVED: Store some algo "hyperparameters" for the model
        model_hps_in_algo_settings = {
            hp: v for hp, v in self._model_hyperparams_to_set.items()
            if hp in model._hyperparameters.keys() and v is not None
        }
        if len(model_hps_in_algo_settings) != 0:
            warnings.warn(f'You should define {model_hps_in_algo_settings} directly as hyperparameters of LME model. '
                          'The current behaviour will soon be dropped.', FutureWarning)
            model.load_hyperparameters(model_hps_in_algo_settings)
        # END DEPRECATED
        if model.dimension != len(dataset.headers):
            raise ValueError('LME is only univariate')

        # get data
        ages = self._get_reformated(dataset, 'timepoints')
        ages_mean, ages_std = np.mean(ages).item(), np.std(ages).item()
        ages_norm = (ages - ages_mean) / ages_std

        y = self._get_reformated(dataset, 'values')
        subjects_with_repeat = self._get_reformated_subjects(dataset)

        # model
        X = sm.add_constant(ages_norm, prepend=True, has_constant='add')

        if model.with_random_slope_age:
            exog_re = X

            if self.force_independent_random_effects:
                free = MixedLMParams.from_components(
                    fe_params=np.ones(2),
                    cov_re=np.eye(2)
                )
                self.sm_fit_parameters['free'] = free
                methods_not_compat_with_free = {'powell','nm'}.intersection(self.sm_fit_parameters['method']) # cf. statsmodels doc
                if len(methods_not_compat_with_free) > 0:
                    warnings.warn("<!> Methods {'powell','nm'} are not compatible with `force_independent_random_effects`")
        else:
            exog_re = None # random_intercept only

        lme = MixedLM(y, X, subjects_with_repeat, exog_re, missing='raise')
        fitted_lme = lme.fit(**self.sm_fit_parameters)

        try:
            cov_re_unscaled_inv = np.linalg.inv(fitted_lme.cov_re_unscaled)
        except np.linalg.LinAlgError:
            raise LeaspyDataInputError("Cannot predict random effects from "
                                       "singular covariance structure.")

        parameters = {
            "ages_mean": ages_mean,
            "ages_std": ages_std,
            "fe_params": fitted_lme.fe_params,
            "cov_re": fitted_lme.cov_re,
            "cov_re_unscaled_inv": cov_re_unscaled_inv, # really useful for personalization
            "noise_std": fitted_lme.scale ** .5, # statsmodels scale is variance
            "bse_fe": fitted_lme.bse_fe,
            "bse_re": fitted_lme.bse_re
        }

        # update model parameters
        model.load_parameters(parameters)

        # return `(fitted_lme.resid ** 2).mean() ** .5` instead of scale?
        return None, parameters['noise_std']

    @staticmethod
    def _get_reformated(dataset, elem):
        # reformat ages
        dataset_elem = getattr(dataset, elem)
        # flatten
        flat_elem = torch.flatten(dataset_elem).numpy()
        # remove padding & nans
        final_elem = flat_elem[torch.flatten(dataset.mask > 0)]
        return final_elem

    @staticmethod
    def _get_reformated_subjects(dataset):
        subjects_with_repeat = []
        for ind, subject in enumerate(dataset.indices):
            subjects_with_repeat += [subject]*max(dataset.n_visits_per_individual) #[ind]
        subjects_with_repeat = np.array(subjects_with_repeat)
        # remove padding & nans
        subjects_with_repeat = subjects_with_repeat[torch.flatten(dataset.mask > 0)]
        return subjects_with_repeat

    def set_output_manager(self, output_settings):
        """
        Not implemented.
        """
        if output_settings is not None:
            warnings.warn('Settings logs in lme fit algorithm is not supported.')
        return
