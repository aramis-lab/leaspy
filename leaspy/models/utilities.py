from typing import Union, Dict, Optional, Tuple, List

import torch
import warnings
import pandas as pd

from leaspy.utils.weighted_tensor import WeightedTensor, TensorOrWeightedTensor
from leaspy.exceptions import LeaspyConvergenceError


def tensor_to_list(x: Union[list, torch.Tensor]) -> list:
    """
    Convert input tensor to list.
    """
    if isinstance(x, torch.Tensor):
        return x.tolist()
    if isinstance(x, WeightedTensor):
        raise NotImplementedError("TODO")
    return x


def cast_value_to_tensor(value, shape: Optional[tuple] = None) -> TensorOrWeightedTensor:
    """
    Cast provided value to a tensor if not a tensor or a WeightedTensor.

    Parameters
    ----------
    value : Any
        The value which should be casted to a tensor.

    shape : tuple, optional
        If a specific shape is expected.

    Returns
    -------
    value : TensorOrWeightedTensor
        The casted value.
    """
    if not isinstance(value, (torch.Tensor, WeightedTensor)):
        value = torch.tensor(value)
    if shape is not None:
        try:
            value = value.view(shape)  # no expansion here
        except RuntimeError:
            value = torch.broadcast_to(value, shape)
    return value


def cast_value_to_2d_tensor(value, unsqueeze_dim: int, dtype=torch.float32) -> torch.Tensor:
    """
    Convert a scalar or array_like into an, at least 2D, dtype tensor.

    Parameters
    ----------
    value : scalar or array_like
        Element to be tensorized.
    unsqueeze_dim : :obj:`int`
        Dimension to be unsqueezed (0 or -1).
        Meaningful for 1D array-like only (for scalar or vector
        of length 1 it has no matter).
    dtype : torch dtype

    Returns
    -------
    :class:`torch.Tensor`, at least 2D

    Examples
    --------
    >>> cast_value_to_2d_tensor([1, 2], 0) == tensor([[1, 2]])
    >>> cast_value_to_2d_tensor([1, 2], -1) == tensor([[1], [2])
    """
    if not isinstance(value, torch.Tensor):
        value = torch.tensor(value, dtype=dtype)
    if value.dtype != dtype:
        value = value.to(dtype)
    while value.dim() < 2:
        value = value.unsqueeze(dim=unsqueeze_dim)
    return value


def serialize_tensor(value, *, indent: str = "", sub_indent: str = "") -> str:
    """Serialization of floats, torch tensors (or numpy arrays)."""
    import re
    from torch._tensor_str import PRINT_OPTS as torch_print_opts

    if isinstance(value, (str, bool, int)):
        return str(value)
    if isinstance(value, float) or getattr(value, 'ndim', -1) == 0:
        # for 0D tensors / arrays the default behavior is to print all digits...
        # change this!
        return f'{value:.{1 + torch_print_opts.precision}g}'
    if isinstance(value, (list, frozenset, set, tuple)):
        try:
            return serialize_tensor(
                torch.tensor(list(value)),
                indent=indent,
                sub_indent=sub_indent,
            )
        except Exception:
            return str(value)
    if isinstance(value, dict):
        if not len(value):
            return ""
        subs = [
            f"{p} : " + serialize_tensor(vp, indent="  ", sub_indent=" "*len(f"{p} : ["))
            for p, vp in value.items()
        ]
        lines = [indent + _ for _ in "\n".join(subs).split("\n")]
        return "\n" + "\n".join(lines)
    # torch.tensor, np.array, ...
    # in particular you may use `torch.set_printoptions` and `np.set_printoptions` globally
    # to tune the number of decimals when printing tensors / arrays
    value_repr = str(value)
    # remove tensor prefix & possible device/size/dtype suffixes
    value_repr = re.sub(r'^[^\(]+\(', '', value_repr)
    value_repr = re.sub(r'(?:, device=.+)?(?:, size=.+)?(?:, dtype=.+)?\)$', '', value_repr)
    # adjust justification
    return re.sub(r'\n[ ]+([^ ])', rf'\n{sub_indent}\1', value_repr)


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
        slopes_mu.append(df_regress_ft['slope'].mean())
        slopes_sigma.append(df_regress_ft['slope'].std())

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
        regression_parameters[feature] = data_train.groupby('ID').apply(
            _linear_regression_against_time
        ).unstack(-1)

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
    return {'intercept': intercept, 'slope': slope}


def compute_patient_values_distribution(df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
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


def compute_patient_time_distribution(df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
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


def get_log_velocities(velocities: torch.Tensor, features: List[str], *, min: float = 1e-2) -> torch.Tensor:
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
    return (t * tol).round() * (1. / tol)

