from __future__ import annotations
from typing import TYPE_CHECKING
from collections.abc import Sized, Callable
from dataclasses import dataclass

import numpy as np
import torch
from scipy import stats
from sklearn.preprocessing import StandardScaler

from leaspy.algo.abstract_algo import AbstractAlgo
from leaspy.io.data.data import Data
from leaspy.io.data.dataset import Dataset
from leaspy.io.outputs.result import Result
from leaspy.io.outputs.individual_parameters import IndividualParameters
from leaspy.models.utils.noise_model import NoiseModel

from leaspy.exceptions import LeaspyAlgoInputError
from leaspy.utils.typing import List, Tuple, Optional, DictParamsTorch

if TYPE_CHECKING:
    from leaspy.models.abstract_model import AbstractModel


class SimulationAlgorithm(AbstractAlgo):
    r"""
    To simulate new data given existing one by learning the individual parameters joined distribution.

    You can choose to only learn the distribution of a group of patient.
    To do so, choose the cofactor(s) and the cofactor(s) state of the wanted patient in the settings.
    For instance, for an Alzheimer's disease patient, you can load a genetic cofactor informative of the APOE4 carriers.
    Choose cofactor ['genetic'] and cofactor_state ['APOE4'] to simulate only APOE4 carriers.

    Parameters
    ----------
    settings : :class:`.AlgorithmSettings`
        The algorithm settings.
        They may include the following parameters, described in __Attributes__ section:
            * `noise`
            * `bandwidth_method`
            * `cofactor`
            * `cofactor_state`
            * `number_of_subjects`
            * `mean_number_of_visits`, `std_number_of_visits`, `min_number_of_visits`, `max_number_of_visits`
            * `delay_btw_visits`
            * `reparametrized_age_bounds`
            * `sources_method`
            * `prefix`
            * `features_bounds`
            * `features_bounds_nb_subjects_factor`

    Attributes
    ----------
    name : ``'simulation'``
        Algorithm's name.
    seed : int
        Used by :mod:`numpy.random` & :mod:`torch.random` for reproducibility.
    algo_parameters : dict
        Contains the algorithm's parameters.

    bandwidth_method : float or str or callable, optional
        Bandwidth argument used in :class:`scipy.stats.gaussian_kde` in order to learn the patients' distribution.
    cofactor : list[str], optional (default = None)
        The list of cofactors included used to select the wanted group of patients (ex - ['genetic']).
        All of them must correspond to an existing cofactor in the attribute `Data`
        of the input `result` of the :meth:`~.run` method.
        TODO? should we allow to learn joint distribution of individual parameters and numeric/categorical cofactors (not fixed)?
    cofactor_state : list[str], optional (default None)
        The cofactors states used to select the wanted group of patients (ex - ['APOE4']).
        There is exactly one state per cofactor in `cofactor` (same order).
        It must correspond to an existing cofactor state in the attribute `Data`
        of the input `result` of the :meth:`~.run` method.
        TODO? it could be replaced by methods to easily sub-select individual having certain cofactors PRIOR to running
        this algorithm + the functionality described just above (included varying cofactors as part of the distribution to estimate).
    features_bounds : bool or dict[str, (float, float)] (default False)
        Specify if the scores of the generated subjects must be bounded.
        This parameter can express in two way:
            * `bool` : the bounds are the maximum and minimum scores observed in the baseline data (TODO: "baseline" instead?).
            * `dict` : the user has to set the min and max bounds for every features. For example:
              ``{'feature1': (score_min, score_max), 'feature2': (score_min, score_max), ...}``
    features_bounds_nb_subjects_factor : float > 1 (default 10)
        Only used if `features_bounds` is not False.
        The ratio of simulated subjects (> 1) so that there is at least `number_of_subjects` that comply to features bounds constraint.
    mean_number_of_visits : int or float (default 6)
        Average number of visits of the simulated patients.
        Examples - choose 5 => in average, a simulated patient will have 5 visits.
    std_number_of_visits : int or float > 0, or None (default 3)
        Standard deviation used into the generation of the number of visits per simulated patient.
        If <= 0 or None: number of visits will be deterministic
    min_number_of_visits, max_number_of_visits : int (optional for max)
        Minimum (resp. maximum) number of visits.
        Only used when `std_number_of_visits` > 0.
        `min_number_of_visits` should be >= 1 (default), `max_number_of_visits` can be None (no limit, default).
    delay_btw_visits :
        Control by how many years consecutive visits of a patient are delayed. Multiple options are possible:
            * float > 0 : regular spacing between all visits
            * dictionary : {'min': float > 0, 'mean': float >= min, 'std': float > 0 [, 'max': float >= mean]}
            Specify a Gaussian random spacing (truncated between min, and max if given)
            * function : n (int >= 1) => 1D numpy.ndarray[float > 0] of length `n` giving delay between visits (e.g.: 3 => [0.5, 1.5, 1.])
    noise : str or float or array-like[float], optional
        Wanted level of gaussian noise in the generated scores:
            * Set noise to ``None`` will lead to patients follow the model exactly (no noise added).
            * Set to ``'inherit_struct'`` (or deprecated ``'default'``), the noise added will follow the model noise structure
              and for Gaussian noise it will be computed from reconstruction errors on data & individual parameters provided.
            * Set noise to ``'model'``, the noise added will follow the model noise structure as well as its values.
            * Set to ``'bernoulli'``, to simulate Bernoulli realizations.
            * Set a float will add for each feature's scores a noise of standard deviation the given float ('gaussian_scalar' noise).
            * Set an array-like[float] (1D of length `n_features`) will add for the feature `j` a noise of standard deviation ``noise[j]`` ('gaussian_diagonal' noise).
        <!> When you simulate data from an ordinal model, you HAVE to keep the default noise='inherit_struct'
            (or use 'model', which is the same in this case since there are no scaling parameter for ordinal noise)
    number_of_subjects : int > 0
        Number of subject to simulate.
    reparametrized_age_bounds : tuple[float, float], optional (default None)
        Define the minimum and maximum reparametrized ages of subjects included in the kernel estimation. See Notes section.
        Example: reparametrized_age_bounds = (65, 70)
    sources_method : str in {'full_kde', 'normal_sources'}
        * ``'full_kde'`` : the sources are also learned with the gaussian kernel density estimation.
        * ``'normal_sources'`` : the sources are generated as multivariate normal distribution linked with the other individual parameters.

    prefix : str
        Prefix appended to simulated patients' identifiers

    Raises
    ------
    :exc:`.LeaspyAlgoInputError`
        If algorithm parameters are of bad type or do not comply to detailed requirements.

    Notes
    -----
    The baseline ages are no more jointly learnt with individual parameters. Instead, we jointly learn
    the _reparametrized_ baseline ages, together with individual parameters. The baseline ages are then
    reconstructed from the simulated reparametrized baseline ages and individual parameters.

    By definition, the relation between age and reparametrized age is:

    .. math:: \psi_i (t) = e^{\xi_i} (t - \tau_i) + \bar{\tau}

    with :math:`t` the real age, :math:`\psi_i (t)` the reparametrized age, :math:`\xi_i` the individual
    log-acceleration parameter, :math:`\tau_i` the individual time-shift parameter and :math:`\bar{\tau}` the mean
    conversion age derived by the `model` object.

    One can restrict the interval of the baseline reparametrized age to be _learnt_ in kernel,
    by setting bounds in `reparametrized_age_bounds`. Note that the simulated reparametrized baseline ages
    are unconstrained and thus could, theoretically (but very unlikely), be out of these prescribed bounds.
    """

    name = 'simulation'
    family = 'simulate'

    def __init__(self, settings):

        super().__init__(settings)

        self.prefix = settings.parameters['prefix']
        self.number_of_subjects = settings.parameters['number_of_subjects']
        self.noise = settings.parameters['noise']
        self.bandwidth_method = settings.parameters['bandwidth_method']
        self.sources_method = settings.parameters['sources_method']

        # TODO? refact params: dict {cofactor_1: forced_state_1, ...}
        self.cofactor = settings.parameters['cofactor']
        self.cofactor_state = settings.parameters['cofactor_state']

        self.reparametrized_age_bounds = settings.parameters['reparametrized_age_bounds']
        self.features_bounds = settings.parameters['features_bounds']
        self.features_bounds_nb_subjects_factor = settings.parameters['features_bounds_nb_subjects_factor']

        self.mean_number_of_visits = settings.parameters['mean_number_of_visits']
        self.std_number_of_visits = settings.parameters['std_number_of_visits']
        self.min_number_of_visits = settings.parameters['min_number_of_visits']
        self.max_number_of_visits = settings.parameters['max_number_of_visits']

        self.delay_btw_visits = settings.parameters['delay_btw_visits']

        # random variables generators
        self.number_of_visits_gen = None
        self.delay_btw_visits_gen = None

        ### Validation of algo parameters
        self._validate_algo_parameters()

    ## HELPERS FOR INPUT VALIDATION  ##
    @staticmethod
    def _check_parameter_has_type(label: str, value, type_or_klass, type_desc: str, *, optional: bool = False):
        if not ((optional and value is None) or isinstance(value, type_or_klass)):
            raise LeaspyAlgoInputError(f'The "{label}" should be {type_desc}{" or None" if optional else ""}, not {type(value)}.')

    def _validate_parameter_has_type(self, param_name: str, *args, **kwargs):
        param_value = getattr(self, param_name)
        self._check_parameter_has_type(param_name, param_value, *args, **kwargs)

    @staticmethod
    def _check_mean_min_max_order(param: str, min, mean, max):
        if not (min <= mean and (max is None or mean <= max)):
            raise LeaspyAlgoInputError(f'Inconsistent "min/mean/max" inequality for "{param}": '
                                       f'min <= mean <= max does not hold.')
    ## END HELPERS ##

    def _validate_number_of_subjects_and_visits(self):

        ## Type checks
        # integers
        for param, is_optional in {
            'number_of_subjects': False,
            'min_number_of_visits': False,
            'max_number_of_visits': True,
        }.items():
            self._validate_parameter_has_type(param, int, 'an integer', optional=is_optional)

        # floats (or integers)
        for param, is_optional in {
            'mean_number_of_visits': False,
            'std_number_of_visits': True,
        }.items():
            self._validate_parameter_has_type(param, (float, int), 'a float or an integer', optional=is_optional)

        ## Non-type conditions (bounds, ...)
        if self.number_of_subjects < 1:
            raise LeaspyAlgoInputError('The "number_of_subjects" should be >= 1')

        if self.min_number_of_visits < 1:
            raise LeaspyAlgoInputError('The "min_number_of_visits" should be an integer >= 1')

        self._check_mean_min_max_order('number_of_visits', self.min_number_of_visits, self.mean_number_of_visits, self.max_number_of_visits)

        if self.std_number_of_visits is not None and self.std_number_of_visits > 0:
            # set generator of random visits in-place
            self.number_of_visits_gen = torch.distributions.normal.Normal(loc=float(self.mean_number_of_visits),
                                                                          scale=float(self.std_number_of_visits))

    def _validate_delay_btw_visits(self):

        self._validate_parameter_has_type('delay_btw_visits', (dict, float, int, Callable),
                "dict{'min', 'mean', 'std'[, 'max']}, float (constant), callable (n -> numpy.array<n>[float])")

        if isinstance(self.delay_btw_visits, dict):
            mandatory_keys = ('min', 'mean', 'std')
            missing_keys = [k for k in mandatory_keys if k not in self.delay_btw_visits]
            expected_keys = mandatory_keys + ('max',)
            unknown_keys = [k for k in self.delay_btw_visits if k not in expected_keys]
            if missing_keys or unknown_keys:
                raise LeaspyAlgoInputError('The "delay_btw_visits" dictionary, defining the random delay distribution, should have: '
                                           '"min", "mean" and "std" keys, and possibly "max" key.')

            # check all floats in dict
            for k, v in self.delay_btw_visits.items():
                self._check_parameter_has_type(f"delay_btw_visits['{k}']", v, (float, int), 'a float or an integer',
                                               optional=k == 'max')

            # check 0 < min <= mean <= max (if set)
            if self.delay_btw_visits['min'] <= 0:
                raise LeaspyAlgoInputError('The "delay_btw_visits.min" should be > 0.')

            self._check_mean_min_max_order('delay_btw_visits',
                    self.delay_btw_visits['min'], self.delay_btw_visits['mean'], self.delay_btw_visits.get('max', None))

            if self.delay_btw_visits['std'] <= 0:
                raise LeaspyAlgoInputError('The "delay_btw_visits.std" should be > 0.')

            # set generator of random delays in-place
            self.delay_btw_visits_gen = torch.distributions.normal.Normal(loc=float(self.delay_btw_visits['mean']),
                                                                          scale=float(self.delay_btw_visits['std']))

        elif isinstance(self.delay_btw_visits, (float, int)):
            if self.delay_btw_visits <= 0:
                raise LeaspyAlgoInputError('The "delay_btw_visits" constant delay between consecutive visits should be > 0 (years).')
        else:
            # callable
            try:
                test_delays = self.delay_btw_visits(self.mean_number_of_visits - 1)
                if not isinstance(test_delays, torch.Tensor):
                    test_delays = torch.tensor(test_delays, dtype=torch.float32)
                assert test_delays.shape == (self.mean_number_of_visits - 1,)
                assert (test_delays > 0).all()
            except Exception as e:
                raise LeaspyAlgoInputError('The "delay_btw_visits" function input n:int and return a numpy.ndarray<n>[float > 0]') from e

    def _validate_cofactors(self):

        if int(self.cofactor is None) ^ int(self.cofactor_state is None):
            raise LeaspyAlgoInputError("`cofactor` and `cofactor_state` should be None or not None simultaneously!")

        if self.cofactor is not None:
            # TODO: check that the loaded cofactors states are strings?
            if not isinstance(self.cofactor, list):
                raise LeaspyAlgoInputError("`cofactor` should be a list of cofactors whose states want to be fixed.")
            if not isinstance(self.cofactor_state, list):
                raise LeaspyAlgoInputError("`cofactor_state` should be the list of cofactors states to fix (same order as `cofactor` list).")
            if len(self.cofactor) != len(self.cofactor_state):
                raise LeaspyAlgoInputError("`cofactor` and `cofactor_state` should have equal length (exactly 1 state per cofactor)")

    def _validate_algo_parameters(self):

        # complex checks in separate methods for clarity
        self._validate_number_of_subjects_and_visits()
        self._validate_delay_btw_visits()
        self._validate_cofactors()

        # other simpler checks
        self._validate_parameter_has_type('prefix', str, 'a string')
        self._validate_parameter_has_type('reparametrized_age_bounds', Sized, 'a array-like', optional=True)
        self._validate_parameter_has_type('features_bounds', (bool, dict), 'a bool or a dictionary')
        self._validate_parameter_has_type('features_bounds_nb_subjects_factor', (float, int), 'a float')

        ## Non-type checks
        if self.features_bounds_nb_subjects_factor <= 1:
            raise LeaspyAlgoInputError('The "features_bounds_nb_subjects_factor" parameter should be > 1 so to simulate extra subjects to be filtered out.')

        if self.sources_method not in ("full_kde", "normal_sources"):
            raise LeaspyAlgoInputError('The "sources_method" parameter must be "full_kde" or "normal_sources"!')

        if self.reparametrized_age_bounds and len(self.reparametrized_age_bounds) != 2:
            raise LeaspyAlgoInputError("The parameter 'reparametrized_age_bounds' must contain exactly two elements, "
                                      f"its lower bound and its upper bound. You gave {self.reparametrized_age_bounds}")

        #self._check_parameter_has_type('noise', (), ..., optional=True)  # to be checked by `_get_noise_model`
        #self._check_parameter_has_type('bandwidth_method', ...)  # error message to be raised by scipy if bad input

    def _check_cofactors(self, data):
        """
        Check the coherence of cofactors given with respect to data object.

        Parameters
        ----------
        data : :class:`.Data`
            Contains the cofactors and cofactors' states.

        Raises
        ------
        :exc:`.LeaspyAlgoInputError`
            Raised if the parameters "cofactor" and "cofactor_state" do not receive a valid value.
        """
        cofactors = {}
        for ind in data.individuals.values():
            if bool(ind.cofactors):
                for key, val in ind.cofactors.items():
                    if key in cofactors.keys():
                        cofactors[key].add(val)
                    else:
                        # set (unique vals)
                        cofactors[key] = {val}

        unknown_cofactors = [cof_ft for cof_ft in self.cofactor if cof_ft not in cofactors.keys()]
        if len(unknown_cofactors) > 0:
            raise LeaspyAlgoInputError(
                f'The `cofactor` parameter has cofactors unknown in your data: {unknown_cofactors}. '
                f'The available cofactor(s) are {list(cofactors.keys())}.')

        invalid_cofactors = dict([(cof_ft, cof_val) for cof_ft, cof_val in zip(self.cofactor, self.cofactor_state)
                             if cof_val not in cofactors[cof_ft]])
        if len(invalid_cofactors) > 0:
            raise LeaspyAlgoInputError(
                f'The `cofactor_state` parameter is invalid for cofactors {invalid_cofactors}. '
                f'The available cofactor states for those are: { {k: cofactors[k] for k in invalid_cofactors} }.')

    @staticmethod
    def _get_mean_and_covariance_matrix(m):
        """
        Compute the empirical mean and covariance matrix of the input. Twice faster than `numpy.cov`.

        Parameters
        ----------
        m : :class:`torch.Tensor`, shape = (n_individual_parameters, n_subjects)
            Input matrix - one row per individual parameter distribution (xi, tau etc).

        Returns
        -------
        mean : :class:`torch.Tensor`
            Mean by variable, shape = (n_individual_parameters,).
        covariance :  :class:`torch.Tensor`
            Covariance matrix, shape = (n_individual_parameters, n_individual_parameters).
        """
        m_exp = torch.mean(m, dim=0)
        x = m - m_exp[None, :]
        cov = 1 / (x.size(0) - 1) * x.t() @ x
        return m_exp, cov

    @staticmethod
    def _sample_sources(bl, tau, xi, source_dimension, df_mean, df_cov):
        """
        Simulate individual sources given baseline age bl, time-shift tau, log-acceleration xi & sources dimension.

        Parameters
        ----------
        bl : float
            Baseline age of the simulated patient.
        tau : float
            Time-shift of the simulated patient.
        xi : float
            Log-acceleration of the simulated patient.
        source_dimension : int
            Sources' dimension of the simulated patient.
        df_mean : :class:`torch.Tensor`, shape = (n_individual_parameters,)
            Mean values per individual parameter type (bl_mean, tau_mean, xi_mean & sources_means) (1-dimensional).
        df_cov : :class:`torch.Tensor`, shape = (n_individual_parameters, n_individual_parameters)
            Empirical covariance matrix of the individual parameters (2-dimensional).

        Returns
        -------
        :class:`torch.Tensor`
            Sources of the simulated patient, shape = (n_sources, ).
        """
        x_1 = torch.tensor([bl, tau, xi], dtype=torch.float32)

        mu_1 = df_mean[:3].clone()
        mu_2 = df_mean[3:].clone()

        sigma_11 = df_cov.narrow(0, 0, 3).narrow(1, 0, 3).clone()
        sigma_22 = df_cov.narrow(0, 3, source_dimension).narrow(1, 3, source_dimension).clone()
        sigma_12 = df_cov.narrow(0, 3, source_dimension).narrow(1, 0, 3).clone()

        mean_cond = mu_2 + sigma_12 @ sigma_11.inverse() @ (x_1 - mu_1)
        cov_cond = sigma_22 - sigma_12 @ sigma_11.inverse() @ sigma_12.transpose(0, -1)

        return torch.distributions.multivariate_normal.MultivariateNormal(mean_cond, cov_cond).sample()

    def _get_number_of_visits(self) -> int:
        """
        Simulate number of visits for a new simulated patient based of attributes 'mean_number_of_visits' &
        'std_number_of_visits'.

        TODO simulate many at once

        Returns
        -------
        number_of_visits : int
            Number of visits.
        """

        # Generate a number of visit around the mean_number_of_visits
        if self.number_of_visits_gen is not None:
            # round before int conversion otherwise values are biased towards lower part
            return int(torch.round(self.number_of_visits_gen.sample().clip(min=self.min_number_of_visits,
                                                                           max=self.max_number_of_visits)).item())
        else:
            return int(self.mean_number_of_visits)

    def _get_features_bounds(self, data):
        """
        Get the bound of the baseline scores of the generated patients. Each generated patient whose baseline is outside
        these bounds are discarded.

        Parameters
        ----------
        data : :class:`~.Data`
            Contains the data to extract features bounds from.

        Returns
        -------
        features_min, features_max : :class:`torch.Tensor`
            Lowest (resp. highest) score allowed per feature - sorted accordingly to the features in ``data.headers``.
        """
        if isinstance(self.features_bounds, dict):
            features = data.headers
            dimension = len(features)

            if features != list(self.features_bounds.keys()):
                raise LeaspyAlgoInputError('The keys of your input "features_bounds" do not match the headers of your data!'
                                          f'\nThe data headers: {features}'
                                          f'\nYour "features_bounds" input: {list(self.features_bounds.keys())}')

            features_min = torch.zeros(dimension)
            features_max = torch.ones(dimension)

            for i, (ft, bounds) in enumerate(self.features_bounds.items()):
                features_min[i] = min(bounds)
                features_max[i] = max(bounds)

            return features_min, features_max
        else:
            # feature_bounds is a bool (True)
            # They are automatically computed from BASELINE (first-available) scores
            df_scores = data.to_dataframe().groupby('ID').first()
            return torch.tensor(df_scores.iloc[:, 1:].min()), torch.tensor(df_scores.iloc[:, 1:].max())

    def _get_timepoints(self, bl: float) -> list:
        """
        Generate the time points of a subject given his baseline age.

        Parameters
        ----------
        bl : float
            The subject's baseline age.

        Returns
        -------
        ages : list[float]
            Contains the subject's time points.
        """
        number_of_visits = self._get_number_of_visits()

        # pathological cases
        if number_of_visits <= 0:
            return []
        elif number_of_visits == 1:
            return [bl]

        # number_of_visits >= 2 (at least 1 delay between visits)
        if isinstance(self.delay_btw_visits, (float, int)):
            # Regular spacing between visits (in years)
            yrs_since_bl = torch.tensor([self.delay_btw_visits * i for i in range(0, number_of_visits)], dtype=torch.float32)
        else:
            if isinstance(self.delay_btw_visits, dict):
                # Random spacing between visits (in years)
                rel_delays = self.delay_btw_visits_gen.sample([number_of_visits - 1]).clip(min=self.delay_btw_visits.get('min'),
                                                                                           max=self.delay_btw_visits.get('max', None))
            else:
                # callable: e.g.: n => [0.5] * min(2, n) + [1.]*max(0, n-2) (old default)
                rel_delays = self.delay_btw_visits(number_of_visits - 1)
                if not isinstance(rel_delays, torch.Tensor):
                    rel_delays = torch.tensor(rel_delays, dtype=torch.float32)

            yrs_since_bl = torch.cat((torch.zeros(1), rel_delays.cumsum(dim=0)))

        return (bl + yrs_since_bl).tolist()

    def _get_noise_model(self, model: AbstractModel, dataset: Dataset, individual_params: DictParamsTorch) -> NoiseModel:

        # special compatibility check for ordinal model
        if getattr(model, 'is_ordinal', False) and self.noise not in ('inherit_struct', 'default', 'model'):
            raise LeaspyAlgoInputError("For an ordinal model, you HAVE to simulate observations with `noise=inherit_struct`"
                                       "(or `noise=model` which is the same in this case).")

        if self.noise is None:
            # no noise at all (will send back raw values upon call)
            return NoiseModel(None)

        elif isinstance(self.noise, str):

            noise_kws = {}
            if self.noise in ['inherit_struct', 'default'] and 'gaussian' in model.noise_model:
                # 'inherit_struct' (and deprecated 'default') option: compute from model reconstruction rmse
                noise_kws['scale'] = NoiseModel.rmse_model(model, dataset, individual_params)

            return NoiseModel.from_model(model, self.noise, **noise_kws)

        else:
            # float or array-like[float] --> gaussian noise
            try:
                noise_scale = torch.tensor(self.noise, dtype=torch.float32).view(-1)
            except Exception:
                raise LeaspyAlgoInputError('The "noise" parameter should be a float or array-like[float] when neither a string nor None.')

            noise_struct = 'gaussian_scalar' if noise_scale.numel() == 1 else 'gaussian_diagonal'
            return NoiseModel.from_model(model, noise_struct, scale=noise_scale)

    @staticmethod
    def _get_reparametrized_age(timepoints, tau, xi, tau_mean):
        """
        Returns the subjects' reparametrized ages.

        Parameters
        ----------
        timepoints : :class:`numpy.ndarray`, shape = (n_subjects,)
            Real ages of the subjects.
        tau : :class:`numpy.ndarray`, shape = (n_subjects,)
            Individual time-shifts.
        xi : :class:`numpy.ndarray`, shape = (n_subjects,)
            Individual log-acceleration.
        tau_mean : float
            The mean conversion age derived by the model.

        Returns
        -------
        :class:`numpy.ndarray`, shape = (n_subjects,)
        """
        return np.exp(xi) * (timepoints - tau) + tau_mean

    @staticmethod
    def _get_real_age(reparam_ages, tau, xi, tau_mean):
        """
        Returns the subjects' real ages.

        Parameters
        ----------
        reparam_ages : :class:`numpy.ndarray`, shape = (n_subjects,)
            Reparametrized ages of the subjects.
        tau : :class:`numpy.ndarray`, shape = (n_subjects,)
            Individual time-shifts.
        xi : :class:`numpy.ndarray`, shape = (n_subjects,)
            Individual log-acceleration.
        tau_mean : float
            The mean conversion age derived by the model.

        Returns
        -------
        :class:`numpy.ndarray`, shape = (n_subjects,)
        """
        return np.exp(-xi) * (reparam_ages - tau_mean) + tau

    def _simulate_individual_parameters(self, model, number_of_simulated_subjects, kernel, ss,
                                        df_mean, df_cov, *, get_sources: bool):
        """
        Compute the simulated individual parameters and timepoints.

        Parameters
        ----------
        model : :class:`~.models.abstract_model.AbstractModel`
            A subclass object of leaspy `AbstractModel`.
        number_of_simulated_subjects : int
        kernel : :class:`scipy.stats.gaussian_kde`
        ss : :class:`sklearn.preprocessing.StandardScaler`
        df_mean : :class:`torch.Tensor`, shape = (n_individual_parameters,)
            Mean values per individual parameter type.
            Only used if `get_sources` and `sources_method` = 'normal_sources'.
        df_cov : :class:`torch.Tensor`, shape = (n_individual_parameters, n_individual_parameters)
            Empirical covariance matrix of the individual parameters.
            Only used if `get_sources` and `sources_method` = 'normal_sources'.

        Returns
        -------
        _SimulatedSubjects
            Simulated subjects (no values simulated yet)
        """
        samples = kernel.resample(number_of_simulated_subjects).T
        samples = ss.inverse_transform(samples)  # A np.ndarray of shape (n_subjects, n_features)

        # Transform reparametrized baseline age into baseline real age
        samples[:, 0] = self._get_real_age(reparam_ages=samples[:, 0],
                                           tau=samples[:, 1],
                                           xi=samples[:, 2],
                                           tau_mean=model.parameters['tau_mean'].item())

        # timepoints is a list of 1D-tensors (varying length) - one tensor per simulated subject
        timepoints = list(map(self._get_timepoints, samples[:, 0]))

        # xi & tau are 1D array - one value per simulated subject
        simulated_parameters = {'tau': torch.from_numpy(samples[:, 1]).view(-1, 1),
                                'xi': torch.from_numpy(samples[:, 2]).view(-1, 1)}

        if get_sources:
            if self.sources_method == "full_kde":
                sources = samples[:, 3:]
            else:
                # Preconditions
                # assert self.sources_method == "normal_sources"
                assert df_mean is not None
                assert df_cov is not None

                # Generate sources
                def simulate_sources(x: np.ndarray) -> np.ndarray:
                    return self._sample_sources(x[0], x[1], x[2], model.source_dimension, df_mean, df_cov).cpu().numpy()

                # sources is np.ndarray of shape (n_subjects, n_sources)
                sources = np.apply_along_axis(simulate_sources, axis=1, arr=samples)

            # torch.tensor
            simulated_parameters['sources'] = torch.from_numpy(sources)

        return _SimulatedSubjects(simulated_parameters, timepoints)

    @staticmethod
    def _simulate_subjects_values(subjects: _SimulatedSubjects, model, noise_model: NoiseModel):
        """
        Compute the simulated scores given the simulated individual parameters, timepoints & noise model.

        Parameters
        ----------
        subjects : _SimulatedSubjects
            Helper class to store simulated individual parameters and timepoints
        model : :class:`~.models.abstract_model.AbstractModel`
            A subclass object of leaspy `AbstractModel`.
        noise_model : :class:`.NoiseModel`
            The noise model that is able to sample realizations around model mean values.

        Returns
        -------
        features_values : list [:class:`torch.Tensor` 2D (n_visits_i, n_features)]
            Contains the scores of all the subjects for all their visits.
            One entry per subject, each of them is a 2D `torch.Tensor` of shape (n_visits, n_features).
        """
        features_values = []
        # TODO : parallelize this for loop on individuals
        for i in range(subjects.n):
            indiv_param = {key: val[[i], :] for key, val in subjects.individual_parameters.items()}
            mean_observations = model.compute_individual_trajectory(subjects.timepoints[i], indiv_param)
            # Sample observations as realizations of the noise model
            observations = noise_model.sample_around(mean_observations)
            # Clip in 0-1 for logistic models (could be out because of noise!), except for ordinal case
            if 'logistic' in model.name and not getattr(model, 'is_ordinal', False):
                observations = observations.clamp(0, 1)

            observations = observations.squeeze(0).detach()
            features_values.append(observations)

        return features_values

    @staticmethod
    def _get_subjects_in_features_bounds(features_values: List[torch.Tensor], features_min: torch.Tensor, features_max: torch.Tensor):
        """
        Select the subject whose scores are within the features boundaries.

        Parameters
        ----------
        features_values : list [:class:`torch.Tensor`]
            Contains the scores of all the subjects of all their visits. Each element correspond to a simulated
            subject, these elements are of shape n_visits x n_features.
        features_min, features_max : :class:`torch.Tensor`
            Lowest (resp. highest) score allowed per feature - sorted accordingly to the features in ``data.headers``.

        Returns
        -------
        list [int]
            Indices of accepted simulated subjects
        """

        def _test_subject(bl_scores: torch.Tensor, features_min: torch.Tensor, features_max: torch.Tensor) -> bool:
            return ((features_min <= bl_scores) & (bl_scores <= features_max)).all()

        baseline_scores = torch.stack([scores[0] for scores in features_values])
        indices_of_accepted_simulated_subjects = [i for i, bl_scores in enumerate(baseline_scores)
                                                  if _test_subject(bl_scores, features_min, features_max)]
        return indices_of_accepted_simulated_subjects

    def _subselect_individuals_in_features_bounds(self, subjects: _SimulatedSubjects, data: Data, n_target: int) -> float:
        # <!> in-place

        # Fetch bounds on the features
        features_min, features_max = self._get_features_bounds(data)

        # Population before filtering
        n_before = subjects.n

        #  Test the  conditions & filter subjects with features' scores outside the bounds.
        indices_of_accepted_subjects = self._get_subjects_in_features_bounds(subjects.values, features_min, features_max)

        for key, val in subjects.individual_parameters.items():
            subjects.individual_parameters[key] = val[indices_of_accepted_subjects, :]
        subjects.timepoints = [v for i, v in enumerate(subjects.timepoints)
                               if i in indices_of_accepted_subjects]
        subjects.values = [v for i, v in enumerate(subjects.values)
                           if i in indices_of_accepted_subjects]

        # Compute the ratio of selected subjects
        ratio_selected = subjects.n / n_before

        # If not enough subjects after filtering raise
        if subjects.n < n_target:
            raise LeaspyAlgoInputError(f'Your features bounds are too stringent: only {subjects.n}/{n_target} ({ratio_selected:.1%} subjects '
                    'were in bounds. Please remove `features_bounds` or increase `features_bounds_nb_subjects_factor` to simulate more subjects.')

        # --------- Take only the `n_target` first generated subjects
        # Needed because of feature_bounds trick...
        subjects.values = subjects.values[:n_target]
        subjects.timepoints = subjects.timepoints[:n_target]
        for key, val in subjects.individual_parameters.items():
            subjects.individual_parameters[key] = val[:n_target, :]

        return ratio_selected

    def run_impl(self, model: AbstractModel, individual_parameters: IndividualParameters, data: Data) -> Tuple[Result, Optional[torch.FloatTensor]]:
        """
        Run simulation - learn joined distribution of patients' individual parameters and return a results object
        containing the simulated individual parameters and the simulated scores.

        <!> The `AbstractAlgo.run` signature is not respected for simulation algorithm...
        TODO: respect it... at least use (model, dataset, individual_parameters) signature...

        Parameters
        ----------
        model : :class:`~.models.abstract_model.AbstractModel`
            Subclass object of `AbstractModel`. Model used to compute the population & individual parameters.
            It contains the population parameters.
        individual_parameters : :class:`.IndividualParameters`
            Object containing the computed individual parameters.
        data : :class:`.Data`
            The data object.

        Returns
        -------
        :class:`~.io.outputs.result.Result`
            Contains the simulated individual parameters & individual scores.

        Notes
        -----
        In simulation_settings, one can specify in the parameters the cofactor & cofactor_state. By doing so,
        one can simulate based only on the subject for the given cofactor & cofactor's state.

        By default, all the subjects provided are used to estimate the joined distribution.
        """
        get_sources = 'univariate' not in model.name and getattr(model, 'source_dimension', 0)

        _, dict_pytorch = individual_parameters.to_pytorch()
        results = Result(data, dict_pytorch)

        # tmp while data argument is not a dataset (cf. signature remark)
        dataset = Dataset(data, model=model, algo=self)

        # validate and get noise model for simulation
        noise_model = self._get_noise_model(model, dataset, dict_pytorch)

        if self.cofactor is not None:
            self._check_cofactors(data)

        # --------- Get individual parameters & reparametrized baseline ages - for joined density estimation
        # Get individual parameters (optional - & the cofactor states)
        df_ind_param = results.get_dataframe_individual_parameters(cofactors=self.cofactor)
        if self.cofactor_state:
            for cof, cof_state in zip(self.cofactor, self.cofactor_state):
                # Select only subjects with the given cofactor state and remove the associated column
                df_ind_param = df_ind_param[df_ind_param[cof] == cof_state].drop(columns=cof)

        # Add the baseline ages
        df_ind_param = data.to_dataframe().groupby('ID').first()[['TIME']].join(df_ind_param, how='right')
        # At this point, df_ind_param.columns = ['TIME', 'tau', 'xi', 'sources_0', 'sources_1', ..., 'sources_n']
        distribution = df_ind_param.values
        # force order TIME tau xi
        distribution[:, 1] = df_ind_param['tau'].values
        distribution[:, 2] = df_ind_param['xi'].values
        # Transform baseline age into reparametrized baseline age
        distribution[:, 0] = self._get_reparametrized_age(timepoints=distribution[:, 0],
                                                          tau=distribution[:, 1],
                                                          xi=distribution[:, 2],
                                                          tau_mean=model.parameters['tau_mean'].item())
        # If constraints on baseline reparametrized age have been set
        # Select only the subjects who satisfy the constraints
        if self.reparametrized_age_bounds:
            distribution = np.array([ind for ind in distribution if
                                     min(self.reparametrized_age_bounds) < ind[0] < max(self.reparametrized_age_bounds)])

        # Get sources according the selected sources_method
        if get_sources and self.sources_method == "normal_sources":
            # Sources are not learned with a kernel density estimator
            distribution = distribution[:, :3]
            # Get mean by variable & covariance matrix
            # Needed to sample new sources from simulated bl, tau & xi
            df_mean, df_cov = self._get_mean_and_covariance_matrix(torch.from_numpy(df_ind_param.values))
        else:
            # full kde with sources
            df_mean, df_cov = None, None

        # --------- Get joined density estimation of reparametrized bl age, tau, xi (and sources if the model has some)
        # Normalize by variable then transpose to learn the joined distribution
        ss = StandardScaler()

        # fit_transform receive an numpy array of shape (n_samples, n_features)
        distribution = ss.fit_transform(distribution).T

        # gaussian_kde receive an numpy array of shape (n_features, n_samples)
        kernel = stats.gaussian_kde(distribution, bw_method=self.bandwidth_method)

        # --------- Simulate new subjects - individual parameters, timepoints and features' scores
        n_target = self.number_of_subjects  # target number of simulated subjects

        # ~Trick: Simulate more subjects in order to have enough of them after filtering that respect features bounds
        number_of_simulated_subjects = self.number_of_subjects
        if self.features_bounds:
            number_of_simulated_subjects *= self.features_bounds_nb_subjects_factor

        simulated_subjects = self._simulate_individual_parameters(
            model, number_of_simulated_subjects, kernel, ss, df_mean, df_cov, get_sources=get_sources)

        simulated_subjects.values = self._simulate_subjects_values(simulated_subjects, model, noise_model)

        # --------- If one wants to constrain baseline scores of generated subjects
        if self.features_bounds:
            self._subselect_individuals_in_features_bounds(simulated_subjects, data, n_target)

        # --------- Generate results object
        # Ex - for 10 subjects, indices = ["Generated_subject_01", "Generated_subject_02", ..., "Generated_subject_10"]
        len_subj_id = len(str(n_target))
        indices = [self.prefix + str(i).rjust(len_subj_id, '0') for i in range(1, n_target + 1)]

        simulated_data = Data.from_individual_values(indices=indices,
                                                     timepoints=simulated_subjects.timepoints,
                                                     observations=[ind_obs.tolist() for ind_obs in simulated_subjects.values],
                                                     headers=data.headers)

        # Output of simulation algorithm
        noise_std_used = noise_model.scale  # will be not None iff Gaussian noise model

        result_obj = Result(data=simulated_data,
                            individual_parameters=simulated_subjects.individual_parameters,
                            noise_std=noise_std_used)

        return (result_obj, noise_std_used)


@dataclass
class _SimulatedSubjects:
    """
    Helper private class to store outputs needed for simulated subjects

    Attributes
    ----------
    individual_parameters : dict [str, :class:`torch.Tensor`]
        Contains the simulated individual parameters.
    timepoints : list[ list[float] ]
        Contains the ages of the subjects for all their visits - 2D list with one row per simulated subject.
    values : list [:class:`torch.Tensor`]
        Contains the scores of all the subjects for all their visits.
        One entry per subject, each of them is a 2D `torch.Tensor` of shape (n_visits, n_features).
    """
    individual_parameters: dict #: DictParamsTorch
    timepoints: list
    values: list = None

    @property
    def n(self) -> int:
        """Number of subjects."""
        return len(self.timepoints)
