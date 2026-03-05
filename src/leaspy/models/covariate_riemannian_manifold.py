from abc import abstractmethod
from typing import Iterable, Optional

import torch

from leaspy.utils.functional import AffineMatrix, Exp, OrthoBasis, Sqr
from leaspy.variables.distributions import MultivariateNormal, Normal
from leaspy.variables.specs import (
    Hyperparameter,
    LinkedVariable,
    ModelParameter,
    NamedVariables,
    PopulationLatentVariable,
    SuffStatsRW,
    VariableName,
)
from leaspy.variables.state import State

from .covariate_time_reparametrized import CovariateTimeReparametrizedModel

# TODO refact? implement a single function
# compute_individual_tensorized(..., with_jacobian: bool) -> returning either
# model values or model values + jacobians wrt individual parameters

# TODO refact? subclass or other proper code technique to extract model's concrete
#  formulation depending on if linear, logistic, mixed log-lin, ...


__all__ = [
    "CovariateRiemannianManifoldModel",
]


class CovariateRiemannianManifoldModel(CovariateTimeReparametrizedModel):
    """Manifold model for multiple variables of interest (logistic or linear formulation).

    Parameters
    ----------
    name : :obj:`str`
        The name of the model.
    **kwargs
        Hyperparameters of the model (including `noise_model`)

    Raises
    ------
    :exc:`.LeaspyModelInputError`

        * If hyperparameters are inconsistent
    """

    def __init__(
        self,
        name: str,
        variables_to_track: Optional[Iterable[VariableName]] = None,
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        default_variables_to_track = [
            "t0",
            "g",
            "v0",
            "delta_t0",
            "delta_g",
            "delta_v0",
            "noise_std",
            "tau_std",
            "xi_mean",
            "xi_std",
            "nll_attach",
            "nll_regul_log_g",
            "nll_regul_log_v0",
            "xi",
            "tau",
            "nll_regul_pop_sum",
            "nll_regul_all_sum",
            "nll_tot",
        ]
        if self.source_dimension:
            default_variables_to_track += [
                "sources",
                "betas",
                "mixing_matrix",
                "space_shifts",
            ]
        self.track_variables(variables_to_track or default_variables_to_track)

    @classmethod
    def _center_xi_realizations(cls, state: State) -> None:
        """
        Center the ``xi`` realizations in place.

        Parameters
        ----------
        state : :class:`.State`
            The dictionary-like object representing current model state, which
            contains keys such as``"xi"`` and ``"log_v0"``.

        Notes
        -----
        This transformation preserves the orthonormal basis since the new ``v0`` remains
        collinear to the previous one. It is a purely internal operation meant to reduce
        redundancy in the parameter space (i.e., improve identifiability and stabilize
        inference).
        """
        mean_xi = torch.mean(state["xi"])
        state["xi"] = state["xi"] - mean_xi
        state["log_v0"] = state["log_v0"] + mean_xi

        # TODO: find a way to prevent re-computation of orthonormal basis since it should
        #  not have changed (v0_collinear update)
        # self.update_MCMC_toolbox({'v0_collinear'}, realizations)

    @classmethod
    def _center_tau_realizations(cls, state: State) -> None:
        """
        Center the ``tau`` realizations in place.

        Parameters
        ----------
        state : :class:`.State`
            The dictionary-like object representing current model state, which
            contains keys such as``"tau"`` and ``"t0"``.

        Notes
        -----
        This transformation preserves the orthonormal basis since the new ``t0`` remains
        collinear to the previous one. It is a purely internal operation meant to reduce
        redundancy in the parameter space (i.e., improve identifiability and stabilize
        inference).
        """
        mean_tau = torch.mean(state["tau"])
        state["tau"] = state["tau"] - mean_tau
        state["t0"] = state["t0"] + mean_tau

    @classmethod
    def compute_sufficient_statistics(cls, state: State) -> SuffStatsRW:
        """
        Compute the model's :term:`sufficient statistics`.

        Parameters
        ----------
        state : :class:`.State`
            The state to pick values from.

        Returns
        -------
        SuffStatsRW :
            The computed sufficient statistics.
        """
        # <!> modify 'xi' and 'log_v0' realizations in-place
        # TODO: what theoretical guarantees for this custom operation?
        cls._center_xi_realizations(state)
        cls._center_tau_realizations(state)

        return super().compute_sufficient_statistics(state)

    def get_variables_specs(self) -> NamedVariables:
        """
        Return the specifications of the variables (latent variables, derived variables,
        model 'parameters') that are part of the model.

        Returns
        -------
        NamedVariables :
            A dictionary-like object mapping variable names to their specifications.
            These include `ModelParameter`, `Hyperparameter`, `PopulationLatentVariable`,
            and `LinkedVariable` instances.
        """
        d = super().get_variables_specs()
        d.update(
            # PRIORS
            log_v0_mean=ModelParameter.for_pop_mean(
                "log_v0",
                shape=(self.dimension,),
            ),
            log_v0_std=Hyperparameter(0.01),
            delta_v0_mean=ModelParameter.for_pop_mean(
                "delta_v0", shape=(self.dimension, self.nb_cov)
            ),
            delta_v0_sigma=ModelParameter.for_pop_cov_matrix(
                "delta_v0", shape=(self.dimension, self.nb_cov, self.nb_cov)
            ),
            xi_mean=Hyperparameter(0.0),
            # LATENT VARS
            log_v0=PopulationLatentVariable(
                Normal("log_v0_mean", "log_v0_std"),
            ),
            delta_v0=PopulationLatentVariable(
                MultivariateNormal("delta_v0_mean", "delta_v0_sigma")
            ),
            # DERIVED VARS
            v0=LinkedVariable(
                Exp("log_v0"),
            ),
            v0_patient=LinkedVariable(AffineMatrix("v0", "delta_v0", "covariates")),
            metric=LinkedVariable(
                self.metric
            ),  # for linear model: metric & metric_sqr are fixed = 1.
        )
        if self.source_dimension >= 1:
            d.update(
                model=LinkedVariable(self.model_with_sources),
                metric_sqr=LinkedVariable(Sqr("metric")),
                orthonormal_basis=LinkedVariable(OrthoBasis("v0", "metric_sqr")),
            )
        else:
            d["model"] = LinkedVariable(self.model_no_sources)

        # TODO: WIP
        # variables_info.update(self.get_additional_ordinal_population_random_variable_information())
        # self.update_ordinal_population_random_variable_information(variables_info)

        return d

    @staticmethod
    @abstractmethod
    def metric(*, g: torch.Tensor) -> torch.Tensor:
        pass

    @classmethod
    def model_no_sources(cls, *, rt: torch.Tensor, metric, v0, g) -> torch.Tensor:
        """
        Return the model output when sources(spatial components) are not present.

        Parameters
        ----------
        rt : :class:`torch.Tensor`
            The reparametrized time.
        metric : Any
            The metric tensor used for computing the spatial/temporal influence.
        v0 : Any
            The values of the population parameter `v0` for each feature.
        g : Any
            The values of the population parameter `g` for each feature.

        Returns
        -------
        :class:`torch.Tensor`
            The model output without contribution from source shifts.

        Notes
        -----
        This implementation delegates to `model_with_sources` with `space_shifts`
        set to a zero tensor of shape (1, 1), effectively removing source effects.
        """
        return cls.model_with_sources(
            rt=rt,
            metric=metric,
            v0=v0,
            g=g,
            space_shifts=torch.zeros((1, 1)),
        )

    @classmethod
    @abstractmethod
    def model_with_sources(
        cls,
        *,
        rt: torch.Tensor,
        space_shifts: torch.Tensor,
        metric,
        v0,
        g,
    ) -> torch.Tensor:
        pass
