import warnings
from operator import itemgetter
from typing import Dict, List

import torch
from scipy import stats
import pandas as pd

# <!> circular imports
import leaspy
from leaspy.exceptions import LeaspyInputError, LeaspyModelInputError
from leaspy.models.noise_models import (
    AbstractGaussianNoiseModel,
    GaussianScalarNoiseModel,
)

XI_STD = .5
TAU_STD = 5.
NOISE_STD = .1
SOURCES_STD = 1.


def _torch_round(t: torch.FloatTensor, *, tol: float = 1 << 16) -> torch.FloatTensor:
    # Round values to ~ 10**-4.8
    return (t * tol).round() * (1./tol)


def initialize_parameters(model, dataset, method="default") -> tuple:
    """
    Initialize the model's group parameters given its name & the scores of all subjects.

    Under-the-hood it calls an initialization function dedicated for the `model`:
        * :func:`.initialize_linear` (including when `univariate`)
        * :func:`.initialize_logistic` (including when `univariate`)
        * :func:`.initialize_logistic_parallel`

    It is automatically called during :meth:`.Leaspy.fit`.

    Parameters
    ----------
    model : :class:`.AbstractModel`
        The model to initialize.
    dataset : :class:`.Dataset`
        Contains the individual scores.
    method : str
        Must be one of:
            * ``'default'``: initialize at mean.
            * ``'random'``:  initialize with a gaussian realization with same mean and variance.

    Returns
    -------
    parameters : dict [str, :class:`torch.Tensor`]
        Contains the initialized model's group parameters.

    Raises
    ------
    :exc:`.LeaspyInputError`
        If no initialization method is known for model type / method
    """

    # we convert once for all dataset to pandas dataframe for convenience
    df = dataset.to_pandas().dropna(how='all').sort_index()
    assert df.index.is_unique
    assert df.index.to_frame().notnull().all(axis=None)
    if model.features != df.columns.tolist():
        raise LeaspyInputError(f"Features mismatch between model and dataset: {model.features} != {df.columns}")

    if method == 'lme':
        raise NotImplementedError("legacy")
        return lme_init(model, df) # support kwargs?

    name = model.name
    if name in ['logistic', 'univariate_logistic']:
        parameters = initialize_logistic(model, df, method)
    elif name == 'logistic_parallel':
        parameters = initialize_logistic_parallel(model, df, method)
    elif name in ['linear', 'univariate_linear']:
        parameters = initialize_linear(model, df, method)
    elif name == 'mixed_linear-logistic':
        raise NotImplementedError("legacy")
    else:
        raise LeaspyInputError(f"There is no initialization method for the parameters of the model '{name}'")

    # convert to float 32 bits & add a rounding step on the initial parameters to ensure full reproducibility
    rounded_parameters = {
        str(p): _torch_round(v.to(torch.float32))
        for p, v in parameters.items()
    }

    # for noise model
    noise_model_params = None
    if isinstance(model.noise_model, AbstractGaussianNoiseModel):
        #noise_scale = NOISE_STD if isinstance(model.noise_model, GaussianScalarNoiseModel) else [NOISE_STD]*model.dimension
        noise_model_params = {"scale": NOISE_STD}

    return rounded_parameters, noise_model_params


def get_lme_results(df: pd.DataFrame, n_jobs=-1, *,
                    with_random_slope_age=True, **lme_fit_kwargs):
    r"""
    Fit a LME on univariate (per feature) time-series (feature vs. patients' ages with varying intercept & slope)

    Parameters
    ----------
    df : :class:`pd.DataFrame`
        Contains all the data (with nans)
    n_jobs : int
        Number of jobs in parallel when multiple features to init
        Not used for now, buggy
    with_random_slope_age : bool (default True)
        Has LME model a random slope per age (otherwise only a random intercept).
    **lme_fit_kwargs
        Kwargs passed to 'lme_fit' (such as `force_independent_random_effects`, default True)

    Returns
    -------
    dict
        {param: str -> param_values_for_ft: torch.Tensor(nb_fts, \*shape_param)}
    """

    # defaults for LME Fit algorithm settings
    lme_fit_kwargs = {
        'force_independent_random_effects': True,
        **lme_fit_kwargs
    }

    #@delayed
    def fit_one_ft(df_ft):
        data_ft = leaspy.Data.from_dataframe(df_ft)
        lsp_lme_ft = leaspy.Leaspy('lme', with_random_slope_age=with_random_slope_age)
        algo = leaspy.AlgorithmSettings('lme_fit', **lme_fit_kwargs) # seed=seed

        lsp_lme_ft.fit(data_ft, algo)

        return lsp_lme_ft.model.parameters

    #res = Parallel(n_jobs=n_jobs)(delayed(fit_one_ft)(s.dropna().to_frame()) for ft, s in df.items())
    res = list(fit_one_ft(s.dropna().to_frame()) for ft, s in df.items())

    # output a dict of tensor stacked by feature, indexed by param
    param_names = next(iter(res)).keys()

    return {
        param_name: torch.stack([torch.tensor(res_ft[param_name])
                                 for res_ft in res])
        for param_name in param_names
    }


def lme_init(model, df: pd.DataFrame, fact_std=1., **kwargs):
    """
    Initialize the model's group parameters.

    Parameters
    ----------
    model : :class:`.AbstractModel`
        The model to initialize (must be an univariate or multivariate linear or logistic manifold model).
    df : :class:`pd.DataFrame`
        Contains the individual scores (with nans).
    fact_std : float
        Multiplicative factor to apply on std-dev (tau, xi, noise) found naively with LME
    **kwargs
        Additional kwargs passed to :func:`.get_lme_results`

    Returns
    -------
    parameters : dict [str, `torch.Tensor`]
        Contains the initialized model's group parameters.

    Raises
    ------
    :exc:`.LeaspyInputError`
        If model is not supported for this initialization
    """
    name = model.name
    noise_model = model.noise_model # has to be set directly at model init and not in algo settings step to be available here

    if not isinstance(noise_model, AbstractGaussianNoiseModel):
        raise LeaspyModelInputError(
            f'`lme` initialization is only compatible with Gaussian noise models, not {noise_model}.'
        )

    multiv = 'univariate' not in name

    #print('Initialization with linear mixed-effects model...')
    lme = get_lme_results(df, **kwargs)
    #print()

    # init
    params = {}

    v0_lin = (lme['fe_params'][:, 1] / lme['ages_std']).clamp(min=1e-2) # > exp(-4.6)

    if 'linear' in name:
        # global tau mean (arithmetic mean of ages mean)
        params['tau_mean'] = lme['ages_mean'].mean()

        params['g'] = lme['fe_params'][:, 0] + v0_lin * (params['tau_mean'] - lme['ages_mean'])
        params['v0' if multiv else 'xi_mean'] = v0_lin.log()

    #elif name in ['logistic_parallel']:
    #    # deltas = torch.zeros((model.dimension - 1,)) ?
    #    pass # TODO...
    elif name in ['logistic', 'univariate_logistic']:

        """
        # global tau mean (arithmetic mean of inflexion point per feature)
        t0_ft = lme['ages_mean'] + (.5 - lme['fe_params'][:, 0]) / v0_lin # inflexion pt
        params['tau_mean'] = t0_ft.mean()
        """

        # global tau mean (arithmetic mean of ages mean)
        params['tau_mean'] = lme['ages_mean'].mean()

        # positions at this tau mean
        pos_ft = lme['fe_params'][:, 0] + v0_lin * (params['tau_mean'] - lme['ages_mean'])

        # parameters under leaspy logistic formulation
        g = 1/pos_ft.clamp(min=1e-2, max=1-1e-2) - 1
        params['g'] = g.log() # -4.6 ... 4.6

        v0 = g/(1+g)**2 * 4 * v0_lin # slope from lme at inflexion point

        #if name == 'logistic_parallel':
        #    # a common speed for all features!
        #    params['xi_mean'] = v0.log().mean() # or log of fts-mean?
        #else:
        params['v0' if multiv else 'xi_mean'] = v0.log()

    else:
        raise LeaspyInputError(f"Model '{name}' is not supported in `lme` initialization.")

    ## Dispersion of individual parameters
    # approx. dispersion on tau (-> on inflexion point when logistic)
    tau_var_ft = lme['cov_re'][:, 0,0] / v0_lin ** 2
    params['tau_std'] = fact_std * (1/tau_var_ft).mean() ** -.5  # harmonic mean on variances per ft

    # approx dispersion on alpha and then xi
    alpha_var_ft = lme['cov_re'][:, 1,1] / lme['fe_params'][:, 1]**2
    xi_var_ft = (1/2+(1/4+alpha_var_ft)**.5).log() # because alpha = exp(xi) so var(alpha) = exp(2*var_xi) - exp(var_xi)
    params['xi_std'] = fact_std * (1/xi_var_ft).mean() ** -.5

    # Residual gaussian noise
    if isinstance(noise_model, GaussianScalarNoiseModel):
        # arithmetic mean on variances
        params['noise_std'] = fact_std * (lme['noise_std'] ** 2).mean().reshape((1,)) ** .5  # 1D tensor
    else:
        # one noise-std per feature
        params['noise_std'] = fact_std * lme['noise_std']

    # For multivariate models, xi_mean == 0.
    if name in ['linear', 'logistic']: # isinstance(model, MultivariateModel)
        params['xi_mean'] = torch.tensor(0.)

    if multiv: # including logistic_parallel
        params['betas'] = torch.zeros((model.dimension - 1, model.source_dimension))
        params['sources_mean'] = torch.tensor(0.)
        params['sources_std'] = torch.tensor(SOURCES_STD)

    return params


def initialize_deltas_ordinal(model, df: pd.DataFrame, parameters: dict) -> None:
    """
    Updates in-place initial values for the ordinal deltas parameters and initializes ordinal noise_model attributes.

    Parameters
    ----------
    model : :class:`.AbstractModel`
        The model to initialize.
    df : :class:`pd.DataFrame`
        Contains the individual scores (with nans).
    parameters : dict[str, `torch.Tensor`]
        The parameters coming from the standard initialization of the logistic model

    Returns
    -------
    parameters : dict[str, `torch.Tensor`]
        The updated parameters initialization, with new parameter deltas
    """

    max_levels = {}

    deltas = {}
    for ft, s in df.items():  # preserve feature order
        max_lvl = int(s.max())  # possible levels not observed in calibration data do not exist for us
        max_levels[ft] = max_lvl
        # we do not model P >= 0 (since constant = 1)
        # we compute stats on P(Y >= k) in our data
        first_age_gte = {}
        for k in range(1, max_lvl + 1):
            s_gte_k = (s >= k).groupby('ID')
            first_age_gte[k] = s_gte_k.idxmax().map(itemgetter(1)).where(s_gte_k.any()) # (ID, TIME) tuple -> TIME or nan
        # we do not need a delta for our anchor curve P >= 1
        # so we start at k == 2
        delays = [(first_age_gte[k] - first_age_gte[k-1]).mean(skipna=True).item()
                  for k in range(2, max_lvl + 1)]
        deltas[ft] = torch.log(torch.clamp(torch.tensor(delays), min=0.1))

    # We store the properties of levels (per feature) directly in the noise-model
    # It will compute the max-level and mask automatically
    model.noise_model.max_levels = max_levels

    if model.batch_deltas:
        # we set the undefined deltas to be infinity to extend validity of formulas for them as well (and to avoid computations)
        deltas_ = float('inf') * torch.ones((len(deltas), model.noise_model.max_level - 1))
        for i, name in enumerate(deltas):
            deltas_[i, :len(deltas[name])] = deltas[name]
        parameters["deltas"] = deltas_
    else:
        for ft in model.features:
            parameters["deltas_" + ft] = deltas[ft]

    # Changes the meaning of v0 # How do we initialize this ?
    #parameters['v0'] = torch.zeros_like(parameters['v0'])


def linregress_against_time(s: pd.Series) -> Dict[str, float]:
    """Return intercept & slope of a linear regression of series values against time (present in series index)."""
    y = s.values
    t = s.index.get_level_values('TIME').values
    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(t, y)
    return {'intercept': intercept, 'slope': slope}


def get_log_velocities(velocities: torch.Tensor, features: List[str], *, min: float = 1e-2) -> torch.Tensor:
    """Warn if some negative velocities are provided, clamp them to `min` and return their log."""
    neg_velocities = velocities <= 0
    if neg_velocities.any():
        warnings.warn(f"Mean slope of individual linear regressions made at initialization is negative for "
                      f"{[f for f, vel in zip(features, velocities) if vel <= 0]}: not properly handled in model...")
    return velocities.clamp(min=min).log()


def initialize_logistic(model, df: pd.DataFrame, method):
    """
    Initialize the logistic model's group parameters.

    Parameters
    ----------
    model : :class:`.AbstractModel`
        The model to initialize.
    df : :class:`pd.DataFrame`
        Contains the individual scores (with nans).
    method : str
        Must be one of:
            * ``'default'``: initialize at mean.
            * ``'random'``:  initialize with a gaussian realization with same mean and variance.

    Returns
    -------
    parameters : dict [str, `torch.Tensor`]
        Contains the initialized model's group parameters.
        The parameters' keys are 'g', 'v0', 'betas', 'tau_mean',
        'tau_std', 'xi_mean', 'xi_std', 'sources_mean', 'sources_std'.

    Raises
    ------
    :exc:`.LeaspyInputError`
        If method is not handled
    """

    # Get the slopes / values / times mu and sigma
    slopes_mu, slopes_sigma = compute_patient_slopes_distribution(df)
    values_mu, values_sigma = compute_patient_values_distribution(df)
    time_mu, time_sigma = compute_patient_time_distribution(df)

    # Method
    if method == "default":
        slopes = slopes_mu
        values = values_mu
        t0 = time_mu
        betas = torch.zeros((model.dimension - 1, model.source_dimension))
    elif method == "random":
        slopes = torch.normal(slopes_mu, slopes_sigma)
        values = torch.normal(values_mu, values_sigma)
        t0 = torch.normal(time_mu, time_sigma)
        betas = torch.distributions.normal.Normal(loc=0., scale=1.).sample(sample_shape=(model.dimension - 1, model.source_dimension))
    else:
        raise LeaspyInputError("Initialization method not supported, must be in {'default', 'random'}")

    # Enforce values are between 0 and 1
    values = values.clamp(min=1e-2, max=1-1e-2)  # always "works" for ordinal (values >= 1)

    # Do transformations
    v0_array = get_log_velocities(slopes, model.features)
    g_array = torch.log(1. / values - 1.)  # cf. Igor thesis; <!> exp is done in Attributes class for logistic models

    # Create smart initialization dictionary
    parameters = {
        "g": g_array,
        "v0": v0_array,
        "betas": betas,
        "tau_mean": t0,
        "tau_std": torch.tensor(TAU_STD),
        "xi_mean": torch.tensor(0.),
        "xi_std": torch.tensor(XI_STD),
        "sources_mean": torch.tensor(0.),
        "sources_std": torch.tensor(SOURCES_STD),
    }

    if model.is_ordinal:
        initialize_deltas_ordinal(model, df, parameters)

    return parameters


def initialize_logistic_parallel(model, df, method):
    """
    Initialize the logistic parallel model's group parameters.

    Parameters
    ----------
    model : :class:`.AbstractModel`
        The model to initialize.
    df : :class:`pd.DataFrame`
        Contains the individual scores (with nans).
    method : str
        Must be one of:
            * ``'default'``: initialize at mean.
            * ``'random'``:  initialize with a gaussian realization with same mean and variance.

    Returns
    -------
    parameters : dict [str, `torch.Tensor`]
        Contains the initialized model's group parameters. The parameters' keys are 'g',  'tau_mean',
        'tau_std', 'xi_mean', 'xi_std', 'sources_mean', 'sources_std', 'deltas' and 'betas'.

    Raises
    ------
    :exc:`.LeaspyInputError`
        If method is not handled
    """

    # Get the slopes / values / times mu and sigma
    slopes_mu, slopes_sigma = compute_patient_slopes_distribution(df)
    values_mu, values_sigma = compute_patient_values_distribution(df)
    time_mu, time_sigma = compute_patient_time_distribution(df)

    if method == 'default':
        slopes = slopes_mu
        values = values_mu
        t0 = time_mu
        betas = torch.zeros((model.dimension - 1, model.source_dimension))
    elif method == 'random':
        # Get random variations
        slopes = torch.normal(slopes_mu, slopes_sigma)
        values = torch.normal(values_mu, values_sigma)
        t0 = torch.normal(time_mu, time_sigma)
        betas = torch.distributions.normal.Normal(loc=0., scale=1.).sample(sample_shape=(model.dimension - 1, model.source_dimension))
    else:
        raise LeaspyInputError("Initialization method not supported, must be in {'default', 'random'}")

    # Enforce values are between 0 and 1
    values = values.clamp(min=1e-2, max=1-1e-2)

    # Do transformations
    v0_array = get_log_velocities(slopes, model.features)
    v0 = v0_array.mean()
    #v0 = slopes.mean().log() # mean before log
    g = torch.log(1. / values - 1.).mean() # cf. Igor thesis; <!> exp is done in Attributes class for logistic models
    #g = torch.log(1. / values.mean() - 1.) # mean before transfo

    parameters = {
        'g': g,
        'deltas': torch.zeros((model.dimension - 1,)),
        'betas': betas,
        'tau_mean': t0,
        'tau_std': torch.tensor(TAU_STD),
        'xi_mean': v0,
        'xi_std': torch.tensor(XI_STD),
        'sources_mean': torch.tensor(0.),
        'sources_std': torch.tensor(SOURCES_STD),
    }

    return parameters


def initialize_linear(model, df: pd.DataFrame, method):
    """
    Initialize the linear model's group parameters.

    Parameters
    ----------
    model : :class:`.AbstractModel`
        The model to initialize.
    df : :class:`pd.DataFrame`
        Contains the individual scores (with nans).
    method : str
        not used for now

    Returns
    -------
    parameters : dict [str, `torch.Tensor`]
        Contains the initialized model's group parameters. The parameters' keys are 'g', 'v0', 'betas', 'tau_mean',
        'tau_std', 'xi_mean', 'xi_std', 'sources_mean', 'sources_std'.
    """
    times = df.index.get_level_values('TIME').values
    t0 = times.mean()

    d_regress_params = compute_linregress_subjects(df, max_inds=None)
    df_all_regress_params = pd.concat(d_regress_params, names=['feature'])
    df_all_regress_params['position'] = df_all_regress_params['intercept'] + t0 * df_all_regress_params['slope']

    df_grp = df_all_regress_params.groupby('feature', sort=False)
    positions = torch.tensor(df_grp['position'].mean().values)
    velocities = torch.tensor(df_grp['slope'].mean().values)

    # always take the log (even in non univariate model!)
    velocities = get_log_velocities(velocities, model.features)

    parameters = {
        "g": positions,
        "v0": velocities,
        "betas": torch.zeros((model.dimension - 1, model.source_dimension)),
        "tau_mean": torch.tensor(t0),
        "tau_std": torch.tensor(TAU_STD),
        "xi_mean": torch.tensor(0.),
        "xi_std": torch.tensor(XI_STD),
        "sources_mean": torch.tensor(0.),
        "sources_std": torch.tensor(SOURCES_STD),
    }

    return parameters


def compute_linregress_subjects(df: pd.DataFrame, *, max_inds: int = None) -> Dict[str, pd.DataFrame]:
    """
    Linear Regression on each feature to get intercept & slopes

    Parameters
    ----------
    df : :class:`pd.DataFrame`
        Contains the individual scores (with nans).
    max_inds : int, optional (default None)
        Restrict computation to first `max_inds` individuals.

    Returns
    -------
    dict[feat_name: str, regress_params_per_subj: pandas.DataFrame]
    """

    d_regress_params = {}

    for ft, s in df.items():
        s = s.dropna()
        nvis = s.groupby('ID').size()
        inds_train = nvis[nvis >= 2].index
        if max_inds is not None:
            inds_train = inds_train[:max_inds]
        s_train = s.loc[inds_train]
        d_regress_params[ft] = s_train.groupby('ID').apply(linregress_against_time).unstack(-1)

    return d_regress_params


def compute_patient_slopes_distribution(df: pd.DataFrame, *, max_inds: int = None):
    """
    Linear Regression on each feature to get slopes

    Parameters
    ----------
    df : :class:`pd.DataFrame`
        Contains the individual scores (with nans).
    max_inds : int, optional (default None)
        Restrict computation to first `max_inds` individuals.

    Returns
    -------
    slopes_mu : :class:`torch.Tensor` [n_features,]
    slopes_sigma : :class:`torch.Tensor` [n_features,]
    """

    d_regress_params = compute_linregress_subjects(df, max_inds=max_inds)
    slopes_mu, slopes_sigma = [], []

    for ft, df_regress_ft in d_regress_params.items():
        slopes_mu.append(df_regress_ft['slope'].mean())
        slopes_sigma.append(df_regress_ft['slope'].std())

    return torch.tensor(slopes_mu), torch.tensor(slopes_sigma)


def compute_patient_values_distribution(df: pd.DataFrame):
    """
    Returns means and standard deviations for the features of the given dataset values.

    Parameters
    ----------
    df : :class:`pd.DataFrame`
        Contains the individual scores (with nans).

    Returns
    -------
    means : :class:`torch.Tensor` [n_features,]
        One mean per feature.
    std : :class:`torch.Tensor` [n_features,]
        One standard deviation per feature.
    """
    return torch.tensor(df.mean().values), torch.tensor(df.std().values)


def compute_patient_time_distribution(df: pd.DataFrame):
    """
    Returns mu / sigma of given dataset times.

    Parameters
    ----------
    df : :class:`pd.DataFrame`
        Contains the individual scores (with nans).

    Returns
    -------
    mean : :class:`torch.Tensor` scalar
    sigma : :class:`torch.Tensor` scalar
    """
    times = df.index.get_level_values('TIME').values
    return torch.tensor(times.mean()), torch.tensor(times.std())
