import torch

__all__ = [
    "compute_individual_parameter_std_from_sufficient_statistics",
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
