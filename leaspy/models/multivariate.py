import pandas as pd
import torch
from abc import abstractmethod

from typing import Iterable, Optional
from leaspy.utils.weighted_tensor import WeightedTensor, TensorOrWeightedTensor
from leaspy.models.base import InitializationMethod
from leaspy.models.abstract_multivariate_model import AbstractMultivariateModel
from leaspy.io.data.dataset import Dataset

from leaspy.utils.docs import doc_with_super

from leaspy.variables.state import State
from leaspy.variables.specs import (
    NamedVariables,
    ModelParameter,
    PopulationLatentVariable,
    LinkedVariable,
    Hyperparameter,
    SuffStatsRW,
    VariablesValuesRO,
)
from leaspy.variables.distributions import Normal
from leaspy.utils.functional import Exp, Sqr, OrthoBasis
from leaspy.utils.weighted_tensor import unsqueeze_right

from leaspy.models.obs_models import FullGaussianObservationModel


# TODO refact? implement a single function
# compute_individual_tensorized(..., with_jacobian: bool) -> returning either
# model values or model values + jacobians wrt individual parameters

# TODO refact? subclass or other proper code technique to extract model's concrete
#  formulation depending on if linear, logistic, mixed log-lin, ...


@doc_with_super()
class MultivariateModel(AbstractMultivariateModel):
    """
    Manifold model for multiple variables of interest (logistic or linear formulation).

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

    def __init__(self, name: str, variables_to_track: Optional[Iterable[str]] = None, **kwargs):
        super().__init__(name, **kwargs)

        variables_to_track = variables_to_track or (
            "g",
            "v0",
            "noise_std",
            "tau_mean",
            "tau_std",
            "xi_mean",
            "xi_std",
            "nll_attach",
            "nll_regul_log_g",
            "nll_regul_log_v0",
            "xi",
            "tau"
        )
        self.tracked_variables = self.tracked_variables.union(set(variables_to_track))

    """
    @suffixed_method
    def compute_individual_tensorized(
        self,
        timepoints: torch.Tensor,
        individual_parameters: dict,
        *,
        attribute_type=None,
    ) -> torch.Tensor:
        pass

    def compute_individual_tensorized_linear(
        self,
        timepoints: torch.Tensor,
        individual_parameters: dict,
        *,
        attribute_type=None,
    ) -> torch.Tensor:
        # Population parameters
        positions, velocities, mixing_matrix = self._get_attributes(attribute_type)
        xi, tau = individual_parameters['xi'], individual_parameters['tau']

        # TODO: use rt instead
        reparametrized_time = self.time_reparametrization(timepoints, xi, tau)

        # Reshaping
        reparametrized_time = reparametrized_time.unsqueeze(-1)  # for automatic broadcast on n_features (last dim)

        # Model expected value
        model = positions + velocities * reparametrized_time

        if self.source_dimension != 0:
            sources = individual_parameters['sources']
            wi = sources.matmul(mixing_matrix.t())
            model += wi.unsqueeze(-2)

        return model  # (n_individuals, n_timepoints, n_features)

    def compute_individual_tensorized_logistic(
        self,
        timepoints: torch.Tensor,
        individual_parameters: dict,
        *,
        attribute_type=None,
    ) -> torch.Tensor:
        # Population parameters
        g, v0, a_matrix = self._get_attributes(attribute_type)
        g_plus_1 = 1. + g
        b = g_plus_1 * g_plus_1 / g

        # Individual parameters
        xi, tau = individual_parameters['xi'], individual_parameters['tau']
        reparametrized_time = self.time_reparametrization(timepoints, xi, tau)

        # Reshaping
        reparametrized_time = reparametrized_time.unsqueeze(-1)  # (n_individuals, n_timepoints, n_features)

        if self.is_ordinal:
            # add an extra dimension for the levels of the ordinal item
            reparametrized_time = reparametrized_time.unsqueeze(-1)
            g = g.unsqueeze(-1)
            b = b.unsqueeze(-1)
            v0 = v0.unsqueeze(-1)
            deltas = self._get_deltas(attribute_type)  # (features, max_level)
            deltas = deltas.unsqueeze(0).unsqueeze(0)  # add (ind, timepoints) dimensions
            # infinite deltas (impossible ordinal levels) will induce model = 0 which is intended
            reparametrized_time = reparametrized_time - deltas.cumsum(dim=-1)

        LL = v0 * reparametrized_time

        if self.source_dimension != 0:
            sources = individual_parameters['sources']
            wi = sources.matmul(a_matrix.t()).unsqueeze(-2)  # unsqueeze for (n_timepoints)
            if self.is_ordinal:
                wi = wi.unsqueeze(-1)
            LL += wi

        # TODO? more efficient & accurate to compute `torch.exp(-t*b + log_g)` since we
        #  directly sample & stored log_g
        LL = 1. + g * torch.exp(-LL * b)
        model = 1. / LL

        # For ordinal models, compute pdf instead of survival function if needed
        model = self.compute_appropriate_ordinal_model(model)

        return model  # (n_individuals, n_timepoints, n_features [, extra_dim_ordinal_models])

    @suffixed_method
    def compute_individual_ages_from_biomarker_values_tensorized(
        self,
        value: torch.Tensor,
        individual_parameters: dict,
        feature: str,
    ) -> torch.Tensor:
        pass

    def compute_individual_ages_from_biomarker_values_tensorized_logistic(
        self,
        value: torch.Tensor,
        individual_parameters: dict,
        feature: str,
    ) -> torch.Tensor:
        if value.dim() != 2:
            raise LeaspyModelInputError(f"The biomarker value should be dim 2, not {value.dim()}!")

        if self.is_ordinal:
            return self._compute_individual_ages_from_biomarker_values_tensorized_logistic_ordinal(
                value, individual_parameters, feature
            )
        # avoid division by zero:
        value = value.masked_fill((value == 0) | (value == 1), float('nan'))

        # 1/ get attributes
        g, v0, a_matrix = self._get_attributes(None)
        xi, tau = individual_parameters['xi'], individual_parameters['tau']
        if self.source_dimension != 0:
            sources = individual_parameters['sources']
            wi = sources.matmul(a_matrix.t())
        else:
            wi = 0

        # get feature value for g, v0 and wi
        feat_ind = self.features.index(feature)  # all consistency checks were done in API layer
        g = torch.tensor([g[feat_ind]])  # g and v0 were shape: (n_features in the multivariate model)
        v0 = torch.tensor([v0[feat_ind]])
        if self.source_dimension != 0:
            wi = wi[0, feat_ind].item()  # wi was shape (1, n_features)

        # 2/ compute age
        ages = tau + (torch.exp(-xi) / v0) * ((g / (g + 1) ** 2) * torch.log(g/(1 / value - 1)) - wi)
        # assert ages.shape == value.shape

        return ages

    def _compute_individual_ages_from_biomarker_values_tensorized_logistic_ordinal(
        self,
        value: torch.Tensor,
        individual_parameters: dict,
        feature: str,
    ) -> torch.Tensor:
        ""
        For one individual, compute age(s) breakpoints at which the given features
        levels are the most likely (given the subject's individual parameters).

        Consistency checks are done in the main :term:`API` layer.

        Parameters
        ----------
        value : :class:`torch.Tensor`
            Contains the :term:`biomarker` level value(s) of the subject.

        individual_parameters : :obj:`dict`
            Contains the individual parameters.
            Each individual parameter should be a scalar or array_like.

        feature : :obj:`str`
            Name of the considered :term:`biomarker`

            .. note::
                Optional for :class:`.UnivariateModel`, compulsory
                for :class:`.MultivariateModel`.

        Returns
        -------
        :class:`torch.Tensor`
            Contains the subject's ages computed at the given values(s)
            Shape of tensor is ``(1, n_values)``.

        Raises
        ------
        :exc:`.LeaspyModelInputError`
            if computation is tried on more than 1 individual
        ""
        # 1/ get attributes
        g, v0, a_matrix = self._get_attributes(None)
        xi, tau = individual_parameters['xi'], individual_parameters['tau']
        if self.source_dimension != 0:
            sources = individual_parameters['sources']
            wi = sources.matmul(a_matrix.t())
        else:
            wi = 0

        # get feature value for g, v0 and wi
        feat_ind = self.features.index(feature)  # all consistency checks were done in API layer
        g = torch.tensor([g[feat_ind]])  # g and v0 were shape: (n_features in the multivariate model)
        v0 = torch.tensor([v0[feat_ind]])
        if self.source_dimension != 0:
            wi = wi[0, feat_ind].item()  # wi was shape (1, n_features)

        # 2/ compute age
        ages_0 = tau + (torch.exp(-xi) / v0) * ((g / (g + 1) ** 2) * torch.log(g) - wi)
        deltas_ft = self._get_deltas(None)[feat_ind]
        delta_max = deltas_ft[torch.isfinite(deltas_ft)].sum()
        ages_max = tau + (torch.exp(-xi) / v0) * ((g / (g + 1) ** 2) * torch.log(g) - wi + delta_max)

        grid_timepoints = torch.linspace(ages_0.item(), ages_max.item(), 1000)

        return self._ordinal_grid_search_value(
            grid_timepoints,
            value,
            individual_parameters=individual_parameters,
            feat_index=feat_ind,
        )

    @suffixed_method
    def compute_jacobian_tensorized(
        self,
        timepoints: torch.Tensor,
        individual_parameters: dict,
        *,
        attribute_type=None,
    ) -> DictParamsTorch:
        pass

    def compute_jacobian_tensorized_linear(
        self,
        timepoints: torch.Tensor,
        individual_parameters: dict,
        *,
        attribute_type=None,
    ) -> DictParamsTorch:
        # Population parameters
        _, v0, mixing_matrix = self._get_attributes(attribute_type)

        # Individual parameters
        xi, tau = individual_parameters['xi'], individual_parameters['tau']
        reparametrized_time = self.time_reparametrization(timepoints, xi, tau)

        # Reshaping
        reparametrized_time = reparametrized_time.unsqueeze(-1)  # (n_individuals, n_timepoints, n_features)

        alpha = torch.exp(xi).reshape(-1, 1, 1)
        dummy_to_broadcast_n_ind_n_tpts = torch.ones_like(reparametrized_time)

        # Jacobian of model expected value w.r.t. individual parameters
        derivatives = {
            'xi': (v0 * reparametrized_time).unsqueeze(-1),  # add a last dimension for len param
            'tau': (v0 * -alpha * dummy_to_broadcast_n_ind_n_tpts).unsqueeze(-1),  # same
        }

        if self.source_dimension > 0:
            derivatives['sources'] = (
                mixing_matrix.expand((1, 1, -1, -1)) * dummy_to_broadcast_n_ind_n_tpts.unsqueeze(-1)
            )

        # dict[param_name: str, torch.Tensor of shape(n_ind, n_tpts, n_fts, n_dims_param)]
        return derivatives

    def compute_jacobian_tensorized_logistic(
        self,
        timepoints: torch.Tensor,
        individual_parameters: dict,
        *,
        attribute_type=None,
    ) -> DictParamsTorch:
        # TODO: refact highly inefficient (many duplicated code from `compute_individual_tensorized_logistic`)

        # Population parameters
        g, v0, a_matrix = self._get_attributes(attribute_type)
        g_plus_1 = 1. + g
        b = g_plus_1 * g_plus_1 / g

        # Individual parameters
        xi, tau = individual_parameters['xi'], individual_parameters['tau']
        reparametrized_time = self.time_reparametrization(timepoints, xi, tau)

        # Reshaping
        reparametrized_time = reparametrized_time.unsqueeze(-1)  # (n_individuals, n_timepoints, n_features)
        alpha = torch.exp(xi).reshape(-1, 1, 1)

        if self.is_ordinal:
            # add an extra dimension for the levels of the ordinal item
            LL = reparametrized_time.unsqueeze(-1)
            g = g.unsqueeze(-1)
            b = b.unsqueeze(-1)
            deltas = self._get_deltas(attribute_type)  # (features, max_level)
            deltas = deltas.unsqueeze(0).unsqueeze(0)  # add (ind, timepoints) dimensions
            LL = LL - deltas.cumsum(dim=-1)
            LL = v0.unsqueeze(-1) * LL

        else:
            LL = v0 * reparametrized_time

        if self.source_dimension != 0:
            sources = individual_parameters['sources']
            wi = sources.matmul(a_matrix.t()).unsqueeze(-2)  # unsqueeze for (n_timepoints)
            if self.is_ordinal:
                wi = wi.unsqueeze(-1)
            LL += wi
        LL = 1. + g * torch.exp(-LL * b)
        model = 1. / LL

        # Jacobian of model expected value w.r.t. individual parameters
        c = model * (1. - model) * b

        derivatives = {
            'xi': (v0 * reparametrized_time).unsqueeze(-1),
            'tau': (-v0 * alpha).unsqueeze(-1),
        }
        if self.source_dimension > 0:
            derivatives['sources'] = a_matrix.expand((1, 1, -1, -1))

        if self.is_ordinal:
            ordinal_lvls_shape = c.shape[-1]
            for param in derivatives:
                derivatives[param] = derivatives[param].unsqueeze(-2).repeat(1, 1, 1, ordinal_lvls_shape, 1)

        # Multiply by common constant, and post-process derivative for ordinal models if needed
        for param in derivatives:
            derivatives[param] = self.compute_appropriate_ordinal_model(
                c.unsqueeze(-1) * derivatives[param]
            )

        # dict[param_name: str, torch.Tensor of shape(n_ind, n_tpts, n_fts [, extra_dim_ordinal_models], n_dims_param)]
        return derivatives
    """

    @classmethod
    def _center_xi_realizations(cls, state: State) -> None:
        """
        Center the ``xi`` realizations in place.

        .. note::
            This operation does not change the orthonormal basis
            (since the resulting ``v0`` is collinear to the previous one)
            Nor all model computations (only ``v0 * exp(xi_i)`` matters),
            it is only intended for model identifiability / ``xi_i`` regularization
            <!> all operations are performed in "log" space (``v0`` is log'ed)

        Parameters
        ----------
        realizations : :class:`.CollectionRealization`
            The realizations to use for updating the :term:`MCMC` toolbox.
        """
        mean_xi = torch.mean(state['xi'])
        state["xi"] = state["xi"] - mean_xi
        state["log_v0"] = state["log_v0"] + mean_xi

        # TODO: find a way to prevent re-computation of orthonormal basis since it should
        #  not have changed (v0_collinear update)
        #self.update_MCMC_toolbox({'v0_collinear'}, realizations)

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

        return super().compute_sufficient_statistics(state)

    def get_variables_specs(self) -> NamedVariables:
        """
        Return the specifications of the variables (latent variables, derived variables,
        model 'parameters') that are part of the model.

        Returns
        -------
        NamedVariables :
            The specifications of the model's variables.
        """
        d = super().get_variables_specs()
        d.update(
            # PRIORS
            log_v0_mean=ModelParameter.for_pop_mean(
                "log_v0",
                shape=(self.dimension,),
            ),
            log_v0_std=Hyperparameter(0.01),
            xi_mean=Hyperparameter(0.),
            # LATENT VARS
            log_v0=PopulationLatentVariable(
                Normal("log_v0_mean", "log_v0_std"),
            ),
            # DERIVED VARS
            v0=LinkedVariable(
                Exp("log_v0"),
            ),
            metric=LinkedVariable(self.metric),  # for linear model: metric & metric_sqr are fixed = 1.
        )
        if self.source_dimension >= 1:
            d.update(
                model=LinkedVariable(self.model_with_sources),
                metric_sqr=LinkedVariable(Sqr("metric")),
                orthonormal_basis=LinkedVariable(OrthoBasis("v0", "metric_sqr")),
            )
        else:
            d['model'] = LinkedVariable(self.model_no_sources)

        # TODO: WIP
        #variables_info.update(self.get_additional_ordinal_population_random_variable_information())
        #self.update_ordinal_population_random_variable_information(variables_info)

        return d

    @staticmethod
    @abstractmethod
    def metric(*, g: torch.Tensor) -> torch.Tensor:
        pass

    @classmethod
    def model_no_sources(cls, *, rt: torch.Tensor, metric, v0, g) -> torch.Tensor:
        """Returns a model without source. A bit dirty?"""
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


class LinearMultivariateInitializationMixin:
        """Compute initial values for model parameters"""
    def _compute_initial_values_for_model_parameters(
        self,
        dataset: Dataset,
        method: InitializationMethod,
    ) -> VariablesValuesRO:
        from leaspy.models.utilities import (
            compute_linear_regression_subjects,
            get_log_velocities,
            torch_round,
        )

        df = self._get_dataframe_from_dataset(dataset)
        times = df.index.get_level_values("TIME").values
        t0 = times.mean()

        d_regress_params = compute_linear_regression_subjects(df, max_inds=None)
        df_all_regress_params = pd.concat(d_regress_params, names=['feature'])
        df_all_regress_params['position'] = (
            df_all_regress_params['intercept'] + t0 * df_all_regress_params['slope']
        )
        df_grp = df_all_regress_params.groupby('feature', sort=False)
        positions = torch.tensor(df_grp['position'].mean().values)
        velocities = torch.tensor(df_grp['slope'].mean().values)

        parameters = {
            "g_mean": positions,
            "log_v0_mean": get_log_velocities(velocities, self.features),
            # "betas": torch.zeros((self.dimension - 1, self.source_dimension)),
            "tau_mean": torch.tensor(t0),
            "tau_std": self.tau_std,
            # "xi_mean": torch.tensor(0.),
            "xi_std": self.xi_std,
            # "sources_mean": torch.tensor(0.),
            # "sources_std": torch.tensor(SOURCES_STD),
        }
        if self.source_dimension >= 1:
            parameters["betas_mean"] = torch.zeros((self.dimension - 1, self.source_dimension))
        rounded_parameters = {
            str(p): torch_round(v.to(torch.float32)) for p, v in parameters.items()
        }
        obs_model = next(iter(self.obs_models))  # WIP: multiple obs models...
        if isinstance(obs_model, FullGaussianObservationModel):
            rounded_parameters["noise_std"] = self.noise_std.expand(
                obs_model.extra_vars['noise_std'].shape
            )
        return rounded_parameters


class LinearMultivariateModel(LinearMultivariateInitializationMixin, MultivariateModel):
    """Manifold model for multiple variables of interest (linear formulation)."""
    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)

    def get_variables_specs(self) -> NamedVariables:
        """
        Return the specifications of the variables (latent variables, derived variables,
        model 'parameters') that are part of the model.

        Returns
        -------
        NamedVariables :
            The specifications of the model's variables.
        """
        d = super().get_variables_specs()
        d.update(
            g_mean=ModelParameter.for_pop_mean("g", shape=(self.dimension,)),
            g_std=Hyperparameter(0.01),
            g=PopulationLatentVariable(
                Normal("g_mean", "g_std")
            ),
        )

        return d
    @staticmethod
    def metric(*, g: torch.Tensor) -> torch.Tensor:
        """Used to define the corresponding variable."""
        return torch.ones_like(g)

    @classmethod
    def model_with_sources(
        cls,
        *,
        rt: torch.Tensor,
        space_shifts: torch.Tensor,
        metric,
        v0,
        g,
    ) -> torch.Tensor:
        """Returns a model with sources."""
        pop_s = (None, None, ...)
        rt = unsqueeze_right(rt, ndim=1)  # .filled(float('nan'))
        return (g[pop_s] + v0[pop_s] * rt + space_shifts[:, None, ...]).weighted_value


class LogisticMultivariateInitializationMixin:
    def _compute_initial_values_for_model_parameters(
        self,
        dataset: Dataset,
        method: InitializationMethod,
    ) -> VariablesValuesRO:
        """Compute initial values for model parameters."""
        from leaspy.models.utilities import (
            compute_patient_slopes_distribution,
            compute_patient_values_distribution,
            compute_patient_time_distribution,
            get_log_velocities,
            torch_round,
        )

        df = self._get_dataframe_from_dataset(dataset)
        slopes_mu, slopes_sigma = compute_patient_slopes_distribution(df)
        values_mu, values_sigma = compute_patient_values_distribution(df)
        time_mu, time_sigma = compute_patient_time_distribution(df)

        if method == InitializationMethod.DEFAULT:
            slopes = slopes_mu
            values = values_mu
            t0 = time_mu
            betas = torch.zeros((self.dimension - 1, self.source_dimension))

        if method == InitializationMethod.RANDOM:
            slopes = torch.normal(slopes_mu, slopes_sigma)
            values = torch.normal(values_mu, values_sigma)
            t0 = torch.normal(time_mu, time_sigma)
            betas = torch.distributions.normal.Normal(loc=0., scale=1.).sample(
                sample_shape=(self.dimension - 1, self.source_dimension)
            )

        # Enforce values are between 0 and 1
        values = values.clamp(min=1e-2, max=1 - 1e-2)  # always "works" for ordinal (values >= 1)

        parameters = {
            "log_g_mean": torch.log(1. / values - 1.),
            "log_v0_mean": get_log_velocities(slopes, self.features),
            "tau_mean": t0,
            "tau_std": self.tau_std,
            "xi_std": self.xi_std,
        }
        if self.source_dimension >= 1:
            parameters["betas_mean"] = betas
        rounded_parameters = {
            str(p): torch_round(v.to(torch.float32)) for p, v in parameters.items()
        }
        obs_model = next(iter(self.obs_models))  # WIP: multiple obs models...
        if isinstance(obs_model, FullGaussianObservationModel):
            rounded_parameters["noise_std"] = self.noise_std.expand(
                obs_model.extra_vars['noise_std'].shape
            )
        return rounded_parameters


class LogisticMultivariateModel(LogisticMultivariateInitializationMixin, MultivariateModel):
    """Manifold model for multiple variables of interest (logistic formulation)."""
    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)

    def get_variables_specs(self) -> NamedVariables:
        """
        Return the specifications of the variables (latent variables, derived variables,
        model 'parameters') that are part of the model.

        Returns
        -------
        NamedVariables :
            The specifications of the model's variables.
        """
        d = super().get_variables_specs()
        d.update(
            log_g_mean=ModelParameter.for_pop_mean("log_g", shape=(self.dimension,)),
            log_g_std=Hyperparameter(0.01),
            log_g=PopulationLatentVariable(
                Normal("log_g_mean", "log_g_std")
            ),
            g=LinkedVariable(Exp("log_g")),

        )

        return d

    @staticmethod
    def metric(*, g: torch.Tensor) -> torch.Tensor:
        """Used to define the corresponding variable."""
        return (g + 1) ** 2 / g

    @classmethod
    def model_with_sources(
        cls,
        *,
        rt: TensorOrWeightedTensor[float],
        space_shifts: TensorOrWeightedTensor[float],
        metric: TensorOrWeightedTensor[float],
        v0: TensorOrWeightedTensor[float],
        g: TensorOrWeightedTensor[float],
    ) -> torch.Tensor:
        """Returns a model with sources."""
        # Shape: (Ni, Nt, Nfts)
        pop_s = (None, None, ...)
        rt = unsqueeze_right(rt, ndim=1)  # .filled(float('nan'))
        w_model_logit = metric[pop_s] * (v0[pop_s] * rt + space_shifts[:, None, ...]) - torch.log(g[pop_s])
        model_logit, weights = WeightedTensor.get_filled_value_and_weight(w_model_logit, fill_value=0.)
        return WeightedTensor(torch.sigmoid(model_logit), weights).weighted_value



"""
# document some methods (we cannot decorate them at method creation since they are
# not yet decorated from `doc_with_super`)
doc_with_(MultivariateModel.compute_individual_tensorized_linear,
          MultivariateModel.compute_individual_tensorized,
          mapping={'the model': 'the model (linear)'})
doc_with_(MultivariateModel.compute_individual_tensorized_logistic,
          MultivariateModel.compute_individual_tensorized,
          mapping={'the model': 'the model (logistic)'})

# doc_with_(MultivariateModel.compute_individual_tensorized_mixed,
#          MultivariateModel.compute_individual_tensorized,
#          mapping={'the model': 'the model (mixed logistic-linear)'})

doc_with_(MultivariateModel.compute_jacobian_tensorized_linear,
          MultivariateModel.compute_jacobian_tensorized,
          mapping={'the model': 'the model (linear)'})
doc_with_(MultivariateModel.compute_jacobian_tensorized_logistic,
          MultivariateModel.compute_jacobian_tensorized,
          mapping={'the model': 'the model (logistic)'})
# doc_with_(MultivariateModel.compute_jacobian_tensorized_mixed,
#          MultivariateModel.compute_jacobian_tensorized,
#          mapping={'the model': 'the model (mixed logistic-linear)'})

# doc_with_(MultivariateModel.compute_individual_ages_from_biomarker_values_tensorized_linear,
#          MultivariateModel.compute_individual_ages_from_biomarker_values_tensorized,
#          mapping={'the model': 'the model (linear)'})
doc_with_(MultivariateModel.compute_individual_ages_from_biomarker_values_tensorized_logistic,
          MultivariateModel.compute_individual_ages_from_biomarker_values_tensorized,
          mapping={'the model': 'the model (logistic)'})
# doc_with_(MultivariateModel.compute_individual_ages_from_biomarker_values_tensorized_mixed,
#          MultivariateModel.compute_individual_ages_from_biomarker_values_tensorized,
#          mapping={'the model': 'the model (mixed logistic-linear)'})
"""
