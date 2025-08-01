import re
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
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
    "tensorize_2D",
    "val_to_tensor",
    "serialize_tensor",
    "is_array_like",
    "tensor_to_list",
    "compute_std_from_variance",
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

def tensorize_2D(x, unsqueeze_dim: int, dtype=torch.float32) -> torch.Tensor:
    """Convert a scalar or array_like into an, at least 2D, dtype tensor.

    Parameters
    ----------
    x : scalar or array_like
        Element to be tensorized.

    unsqueeze_dim : :obj:`int`
        Dimension to be unsqueezed (0 or -1).
        Meaningful for 1D array-like only (for scalar or vector
        of length 1 it has no matter).

    Returns
    -------
    :class:`torch.Tensor`, at least 2D

    Examples
    --------
    >>> tensorize_2D([1, 2], 0) == tensor([[1, 2]])
    >>> tensorize_2D([1, 2], -1) == tensor([[1], [2])
    """
    # convert to torch.Tensor if not the case
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=dtype)
    # convert dtype if needed
    if x.dtype != dtype:
        x = x.to(dtype)
    # if tensor is less than 2-dimensional add dimensions
    while x.dim() < 2:
        x = x.unsqueeze(dim=unsqueeze_dim)
    # postcondition: x.dim() >= 2
    return x


def val_to_tensor(val, shape: Optional[tuple] = None):
    if not isinstance(val, (torch.Tensor, WeightedTensor)):
        val = torch.tensor(val)
    if shape is not None:
        val = val.view(shape)  # no expansion here
    return val


def serialize_tensor(v, *, indent: str = "", sub_indent: str = "") -> str:
    """Nice serialization of floats, torch tensors (or numpy arrays)."""
    from torch._tensor_str import PRINT_OPTS as torch_print_opts

    if isinstance(v, (str, bool, int)):
        return str(v)
    if isinstance(v, np.ndarray):
        return str(v.tolist())
    if isinstance(v, float) or getattr(v, "ndim", -1) == 0:
        # for 0D tensors / arrays the default behavior is to print all digits...
        # change this!
        return f"{v:.{1 + torch_print_opts.precision}g}"
    if isinstance(v, (list, frozenset, set, tuple)):
        try:
            return serialize_tensor(
                torch.tensor(list(v)), indent=indent, sub_indent=sub_indent
            )
        except Exception:
            return str(v)
    if isinstance(v, dict):
        if not len(v):
            return ""
        subs = [
            f"{p} : "
            + serialize_tensor(vp, indent="  ", sub_indent=" " * len(f"{p} : ["))
            for p, vp in v.items()
        ]
        lines = [indent + _ for _ in "\n".join(subs).split("\n")]
        return "\n" + "\n".join(lines)
    # torch.tensor, np.array, ...
    # in particular you may use `torch.set_printoptions` and `np.set_printoptions` globally
    # to tune the number of decimals when printing tensors / arrays
    v_repr = str(v)
    # remove tensor prefix & possible device/size/dtype suffixes
    v_repr = re.sub(r"^[^\(]+\(", "", v_repr)
    v_repr = re.sub(r"(?:, device=.+)?(?:, size=.+)?(?:, dtype=.+)?\)$", "", v_repr)
    # adjust justification
    return re.sub(r"\n[ ]+([^ ])", rf"\n{sub_indent}\1", v_repr)


def is_array_like(v: Any) -> bool:
    try:
        len(v)  # exclude np.array(scalar) or torch.tensor(scalar)
        return hasattr(v, "__getitem__")  # exclude set
    except Exception:
        return False


def tensorize_2D(x, unsqueeze_dim: int, dtype=torch.float32) -> torch.Tensor:
    """Convert a scalar or array_like into an, at least 2D, dtype tensor.

    Parameters
    ----------
    x : scalar or array_like
        Element to be tensorized.

    unsqueeze_dim : :obj:`int`
        Dimension to be unsqueezed (0 or -1).
        Meaningful for 1D array-like only (for scalar or vector
        of length 1 it has no matter).

    Returns
    -------
    :class:`torch.Tensor`, at least 2D

    Examples
    --------
    >>> tensorize_2D([1, 2], 0) == tensor([[1, 2]])
    >>> tensorize_2D([1, 2], -1) == tensor([[1], [2])
    """
    # convert to torch.Tensor if not the case
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=dtype)
    # convert dtype if needed
    if x.dtype != dtype:
        x = x.to(dtype)
    # if tensor is less than 2-dimensional add dimensions
    while x.dim() < 2:
        x = x.unsqueeze(dim=unsqueeze_dim)
    # postcondition: x.dim() >= 2
    return x


def val_to_tensor(val, shape: Optional[tuple] = None):
    if not isinstance(val, (torch.Tensor, WeightedTensor)):
        val = torch.tensor(val)
    if shape is not None:
        val = val.view(shape)  # no expansion here
    return val


def serialize_tensor(v, *, indent: str = "", sub_indent: str = "") -> str:
    """Nice serialization of floats, torch tensors (or numpy arrays)."""
    from torch._tensor_str import PRINT_OPTS as torch_print_opts

    if isinstance(v, (str, bool, int)):
        return str(v)
    if isinstance(v, np.ndarray):
        return str(v.tolist())
    if isinstance(v, float) or getattr(v, "ndim", -1) == 0:
        # for 0D tensors / arrays the default behavior is to print all digits...
        # change this!
        return f"{v:.{1 + torch_print_opts.precision}g}"
    if isinstance(v, (list, frozenset, set, tuple)):
        try:
            return serialize_tensor(
                torch.tensor(list(v)), indent=indent, sub_indent=sub_indent
            )
        except Exception:
            return str(v)
    if isinstance(v, dict):
        if not len(v):
            return ""
        subs = [
            f"{p} : "
            + serialize_tensor(vp, indent="  ", sub_indent=" " * len(f"{p} : ["))
            for p, vp in v.items()
        ]
        lines = [indent + _ for _ in "\n".join(subs).split("\n")]
        return "\n" + "\n".join(lines)
    # torch.tensor, np.array, ...
    # in particular you may use `torch.set_printoptions` and `np.set_printoptions` globally
    # to tune the number of decimals when printing tensors / arrays
    v_repr = str(v)
    # remove tensor prefix & possible device/size/dtype suffixes
    v_repr = re.sub(r"^[^\(]+\(", "", v_repr)
    v_repr = re.sub(r"(?:, device=.+)?(?:, size=.+)?(?:, dtype=.+)?\)$", "", v_repr)
    # adjust justification
    return re.sub(r"\n[ ]+([^ ])", rf"\n{sub_indent}\1", v_repr)


def is_array_like(v: Any) -> bool:
    try:
        len(v)  # exclude np.array(scalar) or torch.tensor(scalar)
        return hasattr(v, "__getitem__")  # exclude set
    except Exception:
        return False


def tensor_to_list(x: Union[list, torch.Tensor]) -> list:
    """
    Convert input tensor to list.

    Parameters
    ----------
    x : :obj:`list` or :obj:`torch.Tensor`
        Input tensor or list to be converted.

    Returns
    -------
    :obj:`list`
        List converted from tensor input, or original list if input was not a tensor.

    Raises
    ------
    :exc:`NotImplementedError`
        If the input is a `WeightedTensor`, as this functionality is not yet implemented.
    """
    if isinstance(x, (np.ndarray, torch.Tensor)):
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
    variance : :obj:`torch.Tensor`
        The variance we would like to convert to a std-dev.
    varname : :obj:`str`
        The name of the variable.
    tol : :obj:`float`, optional
        The lower bound on variance, under which the converge error is raised.
        Default=1e-5.

    Returns
    -------
    :obj: `torch.Tensor` :
        The standard deviation from the variance.

    Raises
    ------
    :exc:`.LeaspyConvergenceError`
        If the variance is less than the specified tolerance, indicating a convergence issue.
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
    nll_cluster = -nll_regul_ind_sum_ind

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
    nll_cluster = -nll_regul_ind_sum_ind

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
    nll_cluster = -nll_regul_ind_sum_ind

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
    nll_cluster = -nll_regul_ind_sum_ind

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
    Compute linear regression slopes and their standard deviations for each feature.

    Parameters
    ----------
    df : :obj:`pd.DataFrame`
        DataFrame containing individual scores.
    max_inds : :obj:`int`, optional
        Restrict computation to first `max_inds` individuals.
        Default="None"

    Returns
    -------
    :obj:`Tuple`[:obj:`torch.Tensor`, :obj:`torch.Tensor`]:
        Tuple with :
        - [0] : torch.Tensor of shape (n_features,) - Regression slopes
        - [1] : torch.Tensor of shape (n_features,) - Standard deviation of the slopes
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
    df : :obj:`pd.DataFrame`
        Contains the individual scores.
    max_inds : :obj:`int`, optional
        Restrict computation to first `max_inds` individuals.
        Default="None"

    Returns
    -------
    :obj: `Dict`[:obj:`str`, :obj:`pd.DataFrame`]:
        Dictionary with :
        - keys : feature names
        - values : DataFrame with :
            - index : Individual IDs
            - columns : 'intercept', 'slope'

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
    data : :obj:`pd.Series`
        Series with time index and values to regress

    Returns
    -------
    :obj: `Dict`[:obj:`str`, :obj: `float`]:
        Dictionary with:
        - keys : 'intercept', 'slope'
        - values : intercept & slope of the linear regression
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
    df : :obj:`pd.DataFrame`
        Contains the individual scores.

    Returns
    -------
    :obj: Tuple[:obj:`torch.Tensor`, :obj:`torch.Tensor`]:
        Tuple with:
        - [0] : torch.Tensor of shape (n_features,) - Means of the features
        - [1] : torch.Tensor of shape (n_features,) - Standard deviations of the features
    """
    return torch.tensor(df.mean().values), torch.tensor(df.std().values)


def compute_patient_time_distribution(
    df: pd.DataFrame,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns mu / sigma of given dataset times.

    Parameters
    ----------
    df : :obj:`pd.DataFrame`
        Contains the individual scores.

    Returns
    -------
    :obj:`Tuple`[:obj:`torch.Tensor`, :obj:`torch.Tensor`]:
        Tuple with:
        - [0] : torch.Tensor - Mean of the times
        - [1] : torch.Tensor - Standard deviation of the times
    """
    times = df.index.get_level_values("TIME").values
    return torch.tensor([times.mean()]), torch.tensor([times.std()])


def get_log_velocities(
    velocities: torch.Tensor, features: List[str], *, min: float = 1e-2
) -> torch.Tensor:
    """
    Get the log of the velocities, clamping them to `min` if negative.

    Parameters
    ----------
    velocities : :obj:`torch.Tensor`
        The velocities to be clamped and logged.
    features : :obj:`List`[:obj:`str`]
        The names of the features corresponding to the velocities.
    min : :obj:`float`, optional
        The minimum value to clamp the velocities to.
        Default=1e-2

    Returns
    -------
    :obj:`torch.Tensor` :
        The log of the clamped velocities.

    Raises
    ------
    :obj:`Warning`
        If some negative velocities are provided.
    """

    neg_velocities = velocities <= 0
    if neg_velocities.any():
        warnings.warn(
            f"Mean slope of individual linear regressions made at initialization is negative for "
            f"{[f for f, vel in zip(features, velocities) if vel <= 0]}: not properly handled in model..."
        )
    return velocities.clamp(min=min).log()


def torch_round(t: torch.FloatTensor, *, tol: float = 1 << 16) -> torch.FloatTensor:
    """
    Multiplies the tensor by `tol`, applies standard rounding, then scales back.
    This effectively rounds values to the nearest multiple of `1.0 / tol`.

    Parameters
    ----------
    t : :obj:`torch.FloatTensor`
        The tensor to be rounded.

    tol : :obj:`float`, optional
        The tolerance factor controlling rounding precision (higher = finer rounding).
        Default=1 << 16 (65536).
        This corresponds to rounding to ~ 10**-4.8.

    Returns
    -------
    :obj:`torch.FloatTensor` :
        The rounded tensor with the same shape as input `t`.
    """
    return (t * tol).round() * (1.0 / tol)
