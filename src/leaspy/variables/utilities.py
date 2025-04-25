import torch

__all__ = [
    "compute_individual_parameter_std_from_sufficient_statistics",
    "compute_correlation_from_sufficient_statistics",
]


def compute_individual_parameter_std_from_sufficient_statistics(
    state: dict[str, torch.Tensor],
    individual_parameter_values: torch.Tensor,
    individual_parameter_sqr_values: torch.Tensor,
    *,
    individual_parameter_name: str,
    dim: int,
    **kws,
):
    """Maximization rule, from the sufficient statistics, of the standard-deviation of Gaussian prior for individual latent variables.

    Parameters
    ----------
    state : Dict[str, torch.Tensor]

    individual_parameter_values : torch.Tensor

    individual_parameter_sqr_values : torch.Tensor

    individual_parameter_name : str
        The name of the individual parameter for which to compute the std.

    dim : int
    """
    from leaspy.models.utilities import compute_std_from_variance

    individual_parameter_old_mean = state[f"{individual_parameter_name}_mean"]
    individual_parameter_current_mean = torch.mean(individual_parameter_values, dim=dim)
    individual_parameter_variance_update = (
        torch.mean(individual_parameter_sqr_values, dim=dim)
        - 2 * individual_parameter_old_mean * individual_parameter_current_mean
    )
    individual_parameter_variance = (
        individual_parameter_variance_update + individual_parameter_old_mean**2
    )
    return compute_std_from_variance(
        individual_parameter_variance, varname=f"{individual_parameter_name}_std", **kws
    )


def compute_correlation_from_sufficient_statistics(
    state: dict[str, torch.Tensor],
    parameters_values: torch.Tensor,
    *,
    parameters_name: str,
    dim: int,
):
    """
    Estimates the correlation coefficient (rho) from sufficient statistics.

    Parameters
    ----------
    state : dict[str, torch.Tensor]
        Dictionary containing model state variables. Should include fixed stds under 'phi_tau_std'.

    parameters_values : torch.Tensor
        Tensor of shape (N, 2), containing values of the individual parameter (e.g., phi_tau),
        where the two columns correspond to two components (e.g., phi_mod and phi_ref).

    parameters_name : str
        Name of the variable (e.g., 'phi_tau').

    Returns
    -------
    torch.Tensor
        Estimated correlation coefficient (rho), shape (1,)
    """
    parameters_mean = torch.mean(parameters_values, dim=dim)

    phi_mod = parameters_values[:, 0]
    phi_ref = parameters_values[:, 1]

    phi_mod_mean = parameters_mean[0]
    phi_ref_mean = parameters_mean[1]

    phi_mod_centered = phi_mod - phi_mod_mean
    phi_ref_centered = phi_ref - phi_ref_mean

    covariance = torch.mean(phi_mod_centered * phi_ref_centered)

    phi_tau_std = state[f"{parameters_name}_std"]  # shape: (2,)
    std_mod = phi_tau_std[0]
    std_ref = phi_tau_std[1]

    if std_mod == 0 or std_ref == 0:
        raise ValueError(f"Standard deviation is zero for parameter {parameters_name}")

    rho = covariance / (std_mod * std_ref)  # or add epsilon for stability :  + 1e-8

    return rho.unsqueeze(0)  # shape = (1,)
