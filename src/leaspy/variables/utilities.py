import torch

__all__ = [
    "compute_individual_parameter_std_from_sufficient_statistics",
    "compute_population_covariance_from_sufficient_statistics",
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
    """
    Maximization rule, from the sufficient statistics, of the standard-deviation of Gaussian prior for individual latent variables.

    Parameters
    ----------
    state : :obj:`dict`[:obj:`str`, :class:`torch.Tensor`]
        The current state object that holds all the variables
    individual_parameter_values : :class:`torch.Tensor`
        Tensor containing individual parameter values, used to compute current means.
    individual_parameter_sqr_values : :class:`torch.Tensor`
        Tensor containing squared individual parameter values, used to compute variances.
    individual_parameter_name : :obj:`str`
        The name of the individual parameter for which to compute the std.
    dim : :obj:`int`
        The dimension along which to compute the mean and variance

    Returns
    -------
    :class:`torch.Tensor`
        The updated standard deviation of the Gaussian prior for the individual parameter
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


def compute_population_covariance_from_sufficient_statistics(
    state: dict[str, torch.Tensor],
    population_parameter_values: torch.Tensor,
    population_parameter_outer_values: torch.Tensor,
    *,
    population_parameter_name: str,
    dim: int,
    **kws,
):
    # # Sigma = S_16 - S_17 @ S_17^T
    # # mais S_17 est E[delta_t0], pas overline{delta_t0}
    # cov = population_parameter_outer_values - torch.outer(
    #     population_parameter_values, population_parameter_values
    # )

    # outer product par feature : (K, N_c) -> (K, N_c, N_c)
    means_outer = torch.einsum(
        "ki,kj->kij", population_parameter_values, population_parameter_values
    )
    # moyenne sur les K features
    cov = (population_parameter_outer_values - means_outer).mean(dim=0)  # (N_c, N_c)
    return cov
