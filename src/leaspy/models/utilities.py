import warnings
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import torch

from leaspy.exceptions import LeaspyConvergenceError
from leaspy.utils.weighted_tensor import WeightedTensor
from leaspy.utils.functional import (
    Identity,
    Mean,
    Prod,
    NamedInputFunction,
    Sqr,
    Std,
    Sum,
    SumDim,
    MatMul,
    get_named_parameters,
)

__all__ = [
    "tensor_to_list",
    "compute_std_from_variance",
    "compute_ind_param_std_from_suff_stats",
    "compute_patient_slopes_distribution",
    "compute_linear_regression_subjects",
    "compute_patient_values_distribution",
    "compute_patient_time_distribution",
    "get_log_velocities",
    "torch_round",
    "compute_ind_param_mean_from_suff_stats_mixture",
    "compute_ind_param_std_from_suff_stats_mixture",
    "compute_ind_param_std_from_suff_stats_mixture_burn_in",
    "compute_probs_from_state",
]


def tensor_to_list(x: Union[list, torch.Tensor]) -> list:
    """
    Convert input tensor to list.
    """
    if isinstance(x, torch.Tensor):
        return x.tolist()
    if isinstance(x, WeightedTensor):
        raise NotImplementedError("TODO")
    return x


def compute_std_from_variance(
    variance: torch.Tensor,
    varname: str,
    tol: float = 1e-5,
) -> torch.Tensor:
    """
    Check that variance is strictly positive and return its square root, otherwise fail with a convergence error.

    If variance is multivariate check that all components are strictly positive.

    TODO? a full Bayesian setting with good priors on all variables should prevent such convergence issues.

    Parameters
    ----------
    variance : :class:`torch.Tensor`
        The variance we would like to convert to a std-dev.
    varname : str
        The name of the variable - to display a nice error message.
    tol : float
        The lower bound on variance, under which the converge error is raised.

    Returns
    -------
    torch.Tensor

    Raises
    ------
    :exc:`.LeaspyConvergenceError`
    """
    if (variance < tol).any():
        raise LeaspyConvergenceError(
            f"The parameter '{varname}' collapsed to zero, which indicates a convergence issue.\n"
            "Start by investigating what happened in the logs of your calibration and try to double check:"
            "\n- your training dataset (not enough subjects and/or visits? too much missing data?)"
            "\n- the hyperparameters of your Leaspy model (`source_dimension` too low or too high? "
            "observation model not suited to your data?)"
            "\n- the hyperparameters of your calibration algorithm"
        )

    return variance.sqrt()


def compute_ind_param_std_from_suff_stats(
    state: Dict[str, torch.Tensor],
    ip_values: torch.Tensor,
    ip_sqr_values: torch.Tensor,
    *,
    ip_name: str,
    dim: int,
    **kws,
):
    """
    Maximization rule, from the sufficient statistics, of the standard-deviation
    of Gaussian prior for individual latent variables.

    Parameters
    ----------
    state : Dict[str, torch.Tensor]
    ip_values : torch.Tensor
    ip_sqr_values : torch.Tensor
    ip_name : str
    dim : int
    """
    ip_old_mean = state[f"{ip_name}_mean"]
    ip_cur_mean = torch.mean(ip_values, dim=dim)
    ip_var_update = torch.mean(ip_sqr_values, dim=dim) - 2 * ip_old_mean * ip_cur_mean
    ip_var = ip_var_update + ip_old_mean**2
    return compute_std_from_variance(ip_var, varname=f"{ip_name}_std", **kws)

def compute_ind_param_mean_from_suff_stats_mixture(
        state: Dict[str, torch.Tensor],
        *,
        ip_name: str,
) -> torch.Tensor:

    ind_var = state[f"{ip_name}"]
    nll_regul_ind_sum_ind = state["nll_regul_ind_sum_ind"].value
    nll_cluster = - nll_regul_ind_sum_ind

    probs_ind = torch.nn.Softmax(dim=1)(torch.clamp(nll_cluster, -100.))

    if ip_name == 'sources' : #special treatement due to the extra dimension
        ind_var_expanded = ind_var.unsqueeze(-1)
        probs_expanded = probs_ind.unsqueeze(1)
        result = ind_var_expanded * probs_expanded
    else:
        result = probs_ind * ind_var

    result = result.sum(dim=0) / probs_ind.sum(dim=0)

    return result

def compute_ind_param_std_from_suff_stats_mixture(
    state: Dict[str, torch.Tensor],
    ip_values: torch.Tensor,
    ip_sqr_values: torch.Tensor,
    *,
    ip_name: str,
    dim: int,
    **kws,
):

    ip_old_mean = state[f"{ip_name}_mean"]
    ip_cur_mean = torch.mean(ip_values, dim=0)
    ip_var_update = torch.mean(ip_sqr_values, dim=0) - 2 * ip_old_mean * ip_cur_mean
    ip_var = ip_var_update + ip_old_mean**2
    std = ip_var.sqrt()

    nll_regul_ind_sum_ind = state["nll_regul_ind_sum_ind"].value
    nll_cluster = - nll_regul_ind_sum_ind

    probs_ind = torch.nn.Softmax(dim=1)(torch.clamp(nll_cluster, -100.))

    result = (probs_ind * std).sum(dim=0) /probs_ind.sum(dim=0)

    return result

def compute_ind_param_std_from_suff_stats_mixture_burn_in(
        state: Dict[str, torch.Tensor],
        *,
        ip_name: str,
) -> torch.Tensor:

    ind_var = state[f"{ip_name}"].std(dim=0)
    nll_regul_ind_sum_ind = state["nll_regul_ind_sum_ind"].value
    nll_cluster = - nll_regul_ind_sum_ind

    probs_ind = torch.nn.Softmax(dim=1)(torch.clamp(nll_cluster, -100.))

    result = (probs_ind * ind_var).sum(dim=0) / probs_ind.sum(dim=0)

    return result

def compute_probs_from_state(
        state: Dict[str, torch.Tensor],
) -> torch.Tensor:

    #probs = state["probs"]
    #n_clusters = probs.shape[0]
    #nll_attach_ind = state["nll_attach_ind"]
    nll_regul_ind_sum_ind = state["nll_regul_ind_sum_ind"].value
    n_inds = nll_regul_ind_sum_ind.shape[0]
    #n_clusters = nll_regul_ind_sum_ind.shape[1]

    #probs = probs.view(1, n_clusters)
    #nll_attach_ind = nll_attach_ind.view(n_inds, 1)
    nll_cluster = - nll_regul_ind_sum_ind

    #nominator = nll_cluster
    #denominator = nll_cluster.sum(dim=1)  # sum for all the clusters
    #probs_list = []

    #for id_cluster in range(nominator.shape[1]):
    #    probs_ind_cluster = nominator[:, id_cluster] / denominator
    #    probs_list.append(probs_ind_cluster)

    #probs_ind = torch.stack(probs_list, dim=1)
    probs_ind = torch.nn.Softmax(dim=1)(torch.clamp(nll_cluster, -100.))

    return probs_ind.sum(dim=0) / n_inds


def compute_patient_slopes_distribution(
    df: pd.DataFrame,
    *,
    max_inds: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
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
    d_regress_params = compute_linear_regression_subjects(df, max_inds=max_inds)
    slopes_mu, slopes_sigma = [], []

    for ft, df_regress_ft in d_regress_params.items():
        slopes_mu.append(df_regress_ft["slope"].mean())
        slopes_sigma.append(df_regress_ft["slope"].std())

    return torch.tensor(slopes_mu), torch.tensor(slopes_sigma)


def compute_linear_regression_subjects(
    df: pd.DataFrame,
    *,
    max_inds: Optional[int] = None,
) -> Dict[str, pd.DataFrame]:
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
    regression_parameters = {}

    for feature, data in df.items():
        data = data.dropna()
        n_visits = data.groupby("ID").size()
        indices_train = n_visits[n_visits >= 2].index
        if max_inds:
            indices_train = indices_train[:max_inds]
        data_train = data.loc[indices_train]
        regression_parameters[feature] = (
            data_train.groupby("ID").apply(_linear_regression_against_time).unstack(-1)
        )

    return regression_parameters


def _linear_regression_against_time(data: pd.Series) -> Dict[str, float]:
    """
    Return intercept & slope of a linear regression of series values
    against time (present in series index).

    Parameters
    ----------
    data : pd.Series

    Returns
    -------
    Dict[str, float]
    """
    from scipy.stats import linregress

    y = data.values
    t = data.index.get_level_values("TIME").values
    slope, intercept, r_value, p_value, std_err = linregress(t, y)
    return {"intercept": intercept, "slope": slope}


def compute_patient_values_distribution(
    df: pd.DataFrame,
) -> Tuple[torch.Tensor, torch.Tensor]:
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


def compute_patient_time_distribution(
    df: pd.DataFrame,
) -> Tuple[torch.Tensor, torch.Tensor]:
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
    times = df.index.get_level_values("TIME").values
    return torch.tensor([times.mean()]), torch.tensor([times.std()])


def get_log_velocities(
    velocities: torch.Tensor, features: List[str], *, min: float = 1e-2
) -> torch.Tensor:
    """Warn if some negative velocities are provided, clamp them to `min` and return their log."""
    neg_velocities = velocities <= 0
    if neg_velocities.any():
        warnings.warn(
            f"Mean slope of individual linear regressions made at initialization is negative for "
            f"{[f for f, vel in zip(features, velocities) if vel <= 0]}: not properly handled in model..."
        )
    return velocities.clamp(min=min).log()


def torch_round(t: torch.FloatTensor, *, tol: float = 1 << 16) -> torch.FloatTensor:
    """Round provided tensor."""
    # Round values to ~ 10**-4.8
    return (t * tol).round() * (1.0 / tol)
