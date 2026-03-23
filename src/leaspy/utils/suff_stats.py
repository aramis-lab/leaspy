from typing import Dict

import torch

from leaspy.exceptions import LeaspyConvergenceError
from leaspy.utils.weighted_tensor import WeightedTensor

__all__ = [
    "compute_std_from_variance",
    "compute_ind_param_std_from_suff_stats",
    "compute_ind_param_mean_from_suff_stats_mixture",
    "compute_ind_param_std_from_suff_stats_mixture",
    "compute_ind_param_std_from_suff_stats_mixture_burn_in",
    "compute_probs_from_state",
]


def compute_std_from_variance(
    variance: torch.Tensor,
    varname: str,
    tol: float = 1e-5,
) -> torch.Tensor:
    """
    Check that variance is strictly positive and return its square root, otherwise fail with a convergence error.
    If variance is multivariate check that all components are strictly positive.

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
    :obj:`torch.Tensor`
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

    probs_ind = torch.nn.Softmax(dim=1)(torch.clamp(nll_cluster, -100.0))

    if ip_name == "sources":  # special treatment due to the extra dimension
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

    probs_ind = torch.nn.Softmax(dim=1)(torch.clamp(nll_cluster, -100.0))

    result = (probs_ind * std).sum(dim=0) / probs_ind.sum(dim=0)

    return result


def compute_ind_param_std_from_suff_stats_mixture_burn_in(
    state: Dict[str, torch.Tensor],
    *,
    ip_name: str,
) -> torch.Tensor:
    ind_var = state[f"{ip_name}"].std(dim=0)
    nll_regul_ind_sum_ind = state["nll_regul_ind_sum_ind"].value
    nll_cluster = -nll_regul_ind_sum_ind

    probs_ind = torch.nn.Softmax(dim=1)(torch.clamp(nll_cluster, -100.0))

    result = (probs_ind * ind_var).sum(dim=0) / probs_ind.sum(dim=0)

    return result


def compute_probs_from_state(
    state: Dict[str, torch.Tensor],
) -> torch.Tensor:
    nll_regul_ind_sum_ind = state["nll_regul_ind_sum_ind"].value
    n_inds = nll_regul_ind_sum_ind.shape[0]
    nll_cluster = -nll_regul_ind_sum_ind
    probs_ind = torch.nn.Softmax(dim=1)(torch.clamp(nll_cluster, -100.0))

    return probs_ind.sum(dim=0) / n_inds
