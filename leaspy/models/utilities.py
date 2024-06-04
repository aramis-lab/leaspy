import torch
from typing import Union
from leaspy.exceptions import LeaspyConvergenceError


def tensor_to_list(x: Union[list, torch.Tensor]) -> list:
    """
    Convert input tensor to list.
    """
    if isinstance(x, torch.Tensor):
        return x.tolist()
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
            "`noise_model` not suited to your data?)"
            "\n- the hyperparameters of your calibration algorithm"
        )

    return variance.sqrt()
