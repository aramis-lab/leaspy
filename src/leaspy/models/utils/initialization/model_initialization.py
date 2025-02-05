from operator import itemgetter

import pandas as pd
import torch

import leaspy
from leaspy.exceptions import LeaspyInputError, LeaspyModelInputError

XI_STD = 0.5
TAU_STD = 5.0
NOISE_STD = 0.1
SOURCES_STD = 1.0


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
    df = dataset.to_pandas().dropna(how="all").sort_index()
    assert df.index.is_unique
    assert df.index.to_frame().notnull().all(axis=None)
    if model.features != df.columns.tolist():
        raise LeaspyInputError(
            f"Features mismatch between model and dataset: {model.features} != {df.columns}"
        )

    if method == "lme":
        raise NotImplementedError("legacy")
        # return lme_init(model, df) # support kwargs?


def get_lme_results(
    df: pd.DataFrame, n_jobs=-1, *, with_random_slope_age=True, **lme_fit_kwargs
):
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
    lme_fit_kwargs = {"force_independent_random_effects": True, **lme_fit_kwargs}

    # @delayed
    def fit_one_ft(df_ft):
        data_ft = leaspy.Data.from_dataframe(df_ft)
        lsp_lme_ft = leaspy.Leaspy("lme", with_random_slope_age=with_random_slope_age)
        algo = leaspy.AlgorithmSettings("lme_fit", **lme_fit_kwargs)  # seed=seed

        lsp_lme_ft.fit(data_ft, algo)

        return lsp_lme_ft.model.parameters

    # res = Parallel(n_jobs=n_jobs)(delayed(fit_one_ft)(s.dropna().to_frame()) for ft, s in df.items())
    res = list(fit_one_ft(s.dropna().to_frame()) for ft, s in df.items())

    # output a dict of tensor stacked by feature, indexed by param
    param_names = next(iter(res)).keys()

    return {
        param_name: torch.stack([torch.tensor(res_ft[param_name]) for res_ft in res])
        for param_name in param_names
    }


def lme_init(model, df: pd.DataFrame, fact_std=1.0, **kwargs):
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
    raise NotImplementedError("OLD")

    name = model.name
    noise_model = model.noise_model  # has to be set directly at model init and not in algo settings step to be available here

    if not isinstance(noise_model, AbstractGaussianNoiseModel):
        raise LeaspyModelInputError(
            f"`lme` initialization is only compatible with Gaussian noise models, not {noise_model}."
        )

    multiv = "univariate" not in name

    # print('Initialization with linear mixed-effects model...')
    lme = get_lme_results(df, **kwargs)
    # print()

    # init
    params = {}

    v0_lin = (lme["fe_params"][:, 1] / lme["ages_std"]).clamp(min=1e-2)  # > exp(-4.6)

    if "linear" in name:
        # global tau mean (arithmetic mean of ages mean)
        params["tau_mean"] = lme["ages_mean"].mean()

        params["g"] = lme["fe_params"][:, 0] + v0_lin * (
            params["tau_mean"] - lme["ages_mean"]
        )
        params["v0" if multiv else "xi_mean"] = v0_lin.log()

    # elif name in ['logistic_parallel']:
    #    # deltas = torch.zeros((model.dimension - 1,)) ?
    #    pass # TODO...
    elif name in ["logistic", "univariate_logistic"]:
        """
        # global tau mean (arithmetic mean of inflexion point per feature)
        t0_ft = lme['ages_mean'] + (.5 - lme['fe_params'][:, 0]) / v0_lin # inflexion pt
        params['tau_mean'] = t0_ft.mean()
        """

        # global tau mean (arithmetic mean of ages mean)
        params["tau_mean"] = lme["ages_mean"].mean()

        # positions at this tau mean
        pos_ft = lme["fe_params"][:, 0] + v0_lin * (
            params["tau_mean"] - lme["ages_mean"]
        )

        # parameters under leaspy logistic formulation
        g = 1 / pos_ft.clamp(min=1e-2, max=1 - 1e-2) - 1
        params["g"] = g.log()  # -4.6 ... 4.6

        v0 = g / (1 + g) ** 2 * 4 * v0_lin  # slope from lme at inflexion point

        # if name == 'logistic_parallel':
        #    # a common speed for all features!
        #    params['xi_mean'] = v0.log().mean() # or log of fts-mean?
        # else:
        params["v0" if multiv else "xi_mean"] = v0.log()

    else:
        raise LeaspyInputError(
            f"Model '{name}' is not supported in `lme` initialization."
        )

    ## Dispersion of individual parameters
    # approx. dispersion on tau (-> on inflexion point when logistic)
    tau_var_ft = lme["cov_re"][:, 0, 0] / v0_lin**2
    params["tau_std"] = (
        fact_std * (1 / tau_var_ft).mean() ** -0.5
    )  # harmonic mean on variances per ft

    # approx dispersion on alpha and then xi
    alpha_var_ft = lme["cov_re"][:, 1, 1] / lme["fe_params"][:, 1] ** 2
    xi_var_ft = (
        1 / 2 + (1 / 4 + alpha_var_ft) ** 0.5
    ).log()  # because alpha = exp(xi) so var(alpha) = exp(2*var_xi) - exp(var_xi)
    params["xi_std"] = fact_std * (1 / xi_var_ft).mean() ** -0.5

    # Residual gaussian noise
    if isinstance(noise_model, GaussianScalarNoiseModel):
        # arithmetic mean on variances
        params["noise_std"] = (
            fact_std * (lme["noise_std"] ** 2).mean().reshape((1,)) ** 0.5
        )  # 1D tensor
    else:
        # one noise-std per feature
        params["noise_std"] = fact_std * lme["noise_std"]

    # For multivariate models, xi_mean == 0.
    if name in ["linear", "logistic"]:  # isinstance(model, MultivariateModel)
        params["xi_mean"] = torch.tensor(0.0)

    if multiv:  # including logistic_parallel
        params["betas"] = torch.zeros((model.dimension - 1, model.source_dimension))
        params["sources_mean"] = torch.tensor(0.0)
        params["sources_std"] = torch.tensor(SOURCES_STD)

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
        max_lvl = int(
            s.max()
        )  # possible levels not observed in calibration data do not exist for us
        max_levels[ft] = max_lvl
        # we do not model P >= 0 (since constant = 1)
        # we compute stats on P(Y >= k) in our data
        first_age_gte = {}
        for k in range(1, max_lvl + 1):
            s_gte_k = (s >= k).groupby("ID")
            first_age_gte[k] = (
                s_gte_k.idxmax().map(itemgetter(1)).where(s_gte_k.any())
            )  # (ID, TIME) tuple -> TIME or nan
        # we do not need a delta for our anchor curve P >= 1
        # so we start at k == 2
        delays = [
            (first_age_gte[k] - first_age_gte[k - 1]).mean(skipna=True).item()
            for k in range(2, max_lvl + 1)
        ]
        deltas[ft] = torch.log(torch.clamp(torch.tensor(delays), min=0.1))

    # We store the properties of levels (per feature) directly in the noise-model
    # It will compute the max-level and mask automatically
    model.noise_model.max_levels = max_levels

    if model.batch_deltas:
        # we set the undefined deltas to be infinity to extend validity of formulas for them as well (and to avoid computations)
        deltas_ = float("inf") * torch.ones(
            (len(deltas), model.noise_model.max_level - 1)
        )
        for i, name in enumerate(deltas):
            deltas_[i, : len(deltas[name])] = deltas[name]
        parameters["deltas"] = deltas_
    else:
        for ft in model.features:
            parameters["deltas_" + ft] = deltas[ft]

    # Changes the meaning of v0 # How do we initialize this ?
    # parameters['v0'] = torch.zeros_like(parameters['v0'])
