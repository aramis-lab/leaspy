import torch

__all__ = [
    "compute_individual_parameter_std_from_sufficient_statistics",
    "compute_correlation_ind",
    "compute_correlation_pop",
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


def compute_correlation_ind(
    state: dict[str, torch.Tensor],
    parameters_values: torch.Tensor,
    *,
    parameters_name: str,
    dim: int,
):
    parameters_mean = state[f"{parameters_name}_mean"]  # shape: (2,)
    parameters_std = state[f"{parameters_name}_std"]  # shape: (2,)

    if parameters_mean.ndim != 1 or parameters_mean.shape[0] != 2:
        raise ValueError(
            f"Expected mean shape (2,) for individual parameter {parameters_name}"
        )

    phi_mod_mean, phi_ref_mean = parameters_mean
    phi_mod = parameters_values[:, 0]
    phi_ref = parameters_values[:, 1]

    phi_mod_centered = phi_mod - phi_mod_mean
    phi_ref_centered = phi_ref - phi_ref_mean

    covariance = torch.mean(phi_mod_centered * phi_ref_centered)

    std_mod, std_ref = parameters_std
    if std_mod == 0 or std_ref == 0:
        raise ValueError(f"Standard deviation is zero for parameter {parameters_name}")

    rho = covariance / (std_mod * std_ref)
    rho = torch.clamp(rho, -0.9999, 0.9999)
    # print("[DEBUG] rho", rho)
    # print(f"[DEBUG] rho_{parameters_name} update:", rho.item(), flush=True)

    return rho  # shape: (1,)


def compute_correlation_pop(
    state: dict[str, torch.Tensor],
    parameters_values: torch.Tensor,
    *,
    parameters_name: str,
    dim: int,
):
    parameters_mean = state[f"{parameters_name}_mean"]  # shape: (K, 2)
    parameters_std = state[f"{parameters_name}_std"]  # shape: (2,)

    if parameters_mean.ndim != 2 or parameters_mean.shape[1] != 2:
        raise ValueError(
            f"Expected mean shape (K, 2) for population parameter {parameters_name}"
        )

    phi_mod_mean = parameters_mean[:, 0]  # shape: (K,)
    phi_ref_mean = parameters_mean[:, 1]  # shape: (K,)

    phi_mod = parameters_values[:, 0]  # shape: (K,)
    phi_ref = parameters_values[:, 1]  # shape: (K,)

    phi_mod_centered = phi_mod - phi_mod_mean
    phi_ref_centered = phi_ref - phi_ref_mean

    covariance = phi_mod_centered * phi_ref_centered  # shape: (K,)

    std_mod, std_ref = parameters_std
    if torch.any(std_mod == 0) or torch.any(std_ref == 0):
        raise ValueError(f"Standard deviation is zero for parameter {parameters_name}")

    rho = covariance / (std_mod * std_ref)  # shape: (K,)
    # print("[DEBUG] rho", rho)
    # print(f"[DEBUG] rho_{parameters_name} update:", rho.tolist(), flush=True)

    return rho  # shape: (K,)
