import pandas as pd
import torch
import warnings
from abc import abstractmethod
from typing import Iterable, Optional

from torch.distributions import MixtureSameFamily

from leaspy.utils.weighted_tensor import WeightedTensor, TensorOrWeightedTensor, unsqueeze_right
from leaspy.utils.docs import doc_with_super
from leaspy.utils.functional import Exp, Sqr, OrthoBasis, MatMul, Sum
from leaspy.utils.typing import KwargsType, Optional

from leaspy.exceptions import LeaspyModelInputError, LeaspyInputError

from leaspy.models.base import InitializationMethod
from leaspy.models.abstract_multivariate_model import AbstractMultivariateModel
from leaspy.models.abstract_model import AbstractModel, InitializationMethod
from leaspy.models.obs_models import observation_model_factory, MixtureGaussianObservationModel, FullGaussianObservationModel
from leaspy.models.multivariate import LogisticMultivariateModel

from leaspy.io.data.dataset import Dataset

from leaspy.variables.distributions import MultinomialDistribution as Multinomial
from leaspy.variables.distributions import Normal, MixtureNormal, Categorical
from leaspy.variables.state import State
from leaspy.variables.specs import (
    NamedVariables,
    ModelParameter,
    PopulationLatentVariable,
    LinkedVariable,
    Hyperparameter,
    SuffStatsRW,
    IndividualLatentVariable,
    VariablesValuesRO,
)

@doc_with_super()
class LogisticMixtureModel(LogisticMultivariateModel):

    """Mixture Manifold model for multiple variables of interest (logistic formulation)."""

    #_xi_std = .5
    #_tau_std = 5.
    #_noise_std = .1
    #_sources_std = 1.

    @property
    def xi_std(self) -> torch.Tensor:
        return torch.tensor([self._xi_std] * self.n_clusters)

    @property
    def tau_std(self) -> torch.Tensor:
        return torch.tensor([self._tau_std] * self.n_clusters)

    @property
    def noise_std(self) -> torch.Tensor:
        return torch.tensor(self._noise_std)

    @property
    def sources_std(self) -> torch.Tensor:
        return torch.tensor([self._sources_std] * self.n_clusters)

    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)

        self.source_dimension: Optional[int] = None
        self.n_clusters: Optional[int] = None
        self.probs: Optional[torch.Tensor] = None # not sure if it has to be there
        #self.probs = torch.ones(self.n_clusters) / self.n_clusters

        # TODO / WIP / TMP: dirty for now...
        # Should we:
        # - use factory of observation models instead? dataset -> ObservationModel
        # - or refact a bit `ObservationModel` structure? (lazy init of its variables...)
        # (cf. note in AbstractModel as well)
        dimension = kwargs.get('dimension', None)
        if 'features' in kwargs:
            dimension = len(kwargs['features'])
        observation_models = kwargs.get("obs_models", None)
        if observation_models is None:
            observation_models = "mixture-gaussian"
        if isinstance(observation_models, (list, tuple)):
            kwargs["obs_models"] = tuple(
                [observation_model_factory(obs_model, **kwargs)
                 for obs_model in observation_models]
            )
        elif isinstance(observation_models, (dict)):
            # Not really satisfied... Used for api load
            kwargs["obs_models"] = tuple(
                [observation_model_factory(observation_models['y'], dimension=dimension)]
            )
        else:
            kwargs["obs_models"] = (observation_model_factory(observation_models, dimension=dimension),)

        variables_to_track = [
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
            "tau",
            "nll_regul_pop_sum",
            "nll_regul_all_sum",
            "nll_tot"]

        if self.source_dimension:
            variables_to_track += ['sources', 'sources_mean', 'betas', 'mixing_matrix', 'space_shifts']

        if self.n_clusters:
            variables_to_track += ['probs']

        self.tracked_variables = self.tracked_variables.union(set(variables_to_track))


    def get_variables_specs(self) -> NamedVariables:
        """
        Return the specifications of the variables (latent variables,
        derived variables, model 'parameters') that are part of the model.

        Returns
        -------
        NamedVariables :
            The specifications of the model's variables.
        """
        d = super().get_variables_specs()

        n_clusters = self.n_clusters
        #probs = torch.ones(n_clusters)
        #probs = probs / n_clusters
        probs = self.probs

        d.update(

            # PRIORS
            log_g_mean=ModelParameter.for_pop_mean("log_g", shape=(self.dimension,)),
            log_v0_mean=ModelParameter.for_pop_mean("log_v0",shape=(self.dimension,)),

            tau_mean=ModelParameter.for_ind_mean("tau", shape=(n_clusters,)),
            tau_std=ModelParameter.for_ind_std("tau", shape=(n_clusters,)),
            xi_mean =ModelParameter.for_ind_mean("xi", shape=(n_clusters,)),
            xi_std=ModelParameter.for_ind_std("xi", shape=(n_clusters,)),

            # LATENT VARS
            log_g=PopulationLatentVariable(Normal("log_g_mean", "log_g_std")),
            log_v0=PopulationLatentVariable(Normal("log_v0_mean", "log_v0_std")),

            xi=IndividualLatentVariable(
                MixtureNormal(mixture_distribution = Categorical(probs),
                                    component_distribution = Normal("xi_mean", "xi_std")
                                    )
            ),
            tau=IndividualLatentVariable(
                MixtureNormal(mixture_distribution=Categorical(probs),
                                    component_distribution=Normal("tau_mean", "tau_std"))
            ),

            # DERIVED VARS
            g=LinkedVariable(Exp("log_g")),
            v0=LinkedVariable(Exp("log_v0")),
            alpha=LinkedVariable(Exp("xi")),
            metric=LinkedVariable(self.metric),  # for linear model: metric & metric_sqr are fixed = 1.

            #HYPERPARAMETERS
            log_g_std=Hyperparameter(0.01),
            log_v0_std=Hyperparameter(0.01),

        )

        if self.source_dimension >= 1:
            d.update(

                # PRIORS
                betas_mean=ModelParameter.for_pop_mean(
                    "betas",
                    shape=(self.dimension - 1, self.source_dimension),
                ),
                betas_std=Hyperparameter(0.01),
                sources_mean=ModelParameter.for_pop_mean(
                    pop_var_name = "sources",
                    shape=(self.source_dimension, self.n_clusters),
                ),
                sources_std=Hyperparameter(torch.ones(self.n_clusters)),

                # LATENT VARS
                betas=PopulationLatentVariable(
                    Normal("betas_mean", "betas_std"),
                    sampling_kws={"scale": .5},   # cf. GibbsSampler (for retro-compat)
                ),
                sources=IndividualLatentVariable(
                    MixtureNormal(mixture_distribution=Categorical(probs),
                                  component_distribution=Normal("sources_mean", "sources_std"))
                ),

                # DERIVED VARS
                mixing_matrix=LinkedVariable(
                    MatMul("orthonormal_basis", "betas").then(torch.t)
                ),  # shape: (Ns, Nfts)
                space_shifts=LinkedVariable(
                    MatMul("sources", "mixing_matrix")
                ),  # shape: (Ni, Nfts)

                model=LinkedVariable(self.model_with_sources),
                metric_sqr=LinkedVariable(Sqr("metric")),
                orthonormal_basis=LinkedVariable(OrthoBasis("v0", "metric_sqr"))
            )
        else:
            d.update(model = LinkedVariable(self.model_no_sources))

        return d

    def _validate_compatibility_of_dataset(self, dataset: Optional[Dataset] = None) -> None:
        super()._validate_compatibility_of_dataset(dataset)
        if not dataset:
            return
        if self.source_dimension is None:
            self.source_dimension = int(dataset.dimension ** .5)
            warnings.warn(
                "You did not provide `source_dimension` hyperparameter for multivariate model, "
                f"setting it to ⌊√dimension⌋ = {self.source_dimension}."
            )
        elif not (isinstance(self.source_dimension, int) and 0 <= self.source_dimension < dataset.dimension):
            raise LeaspyModelInputError(
                f"Sources dimension should be an integer in [0, dimension - 1[ "
                f"but you provided `source_dimension` = {self.source_dimension} "
                f"whereas `dimension` = {dataset.dimension}."
            )
        if self.n_clusters is None:
            warnings.warn("You did not provide `n_clusters` hyperparameter for mixture model")
        elif not (isinstance(self.n_clusters, int) and 2 > self.n_clusters ):
            raise LeaspyModelInputError(
                f"Number of clusters should be an integer greater than 2 "
                f"but you provided `n_clusters` = {self.n_clusters} "
            )

    def _load_hyperparameters(self, hyperparameters: KwargsType) -> None:
        """
        Updates all model hyperparameters from the provided hyperparameters.

        Parameters
        ----------
        hyperparameters : KwargsType
            The hyperparameters to be loaded.
        """
        expected_hyperparameters = ('features', 'dimension', 'source_dimension', "n_clusters")

        if 'features' in hyperparameters:
            self.features = hyperparameters['features']

        if 'dimension' in hyperparameters:
            if self.features and hyperparameters['dimension'] != len(self.features):
                raise LeaspyModelInputError(
                    f"Dimension provided ({hyperparameters['dimension']}) does not match "
                    f"features ({len(self.features)})"
                )
            self.dimension = hyperparameters['dimension']

        if 'source_dimension' in hyperparameters:
            if not (
                isinstance(hyperparameters['source_dimension'], int)
                and (hyperparameters['source_dimension'] >= 0)
                and (self.dimension is None or hyperparameters['source_dimension'] <= self.dimension - 1)
            ):
                raise LeaspyModelInputError(
                    f"Source dimension should be an integer in [0, dimension - 1], "
                    f"not {hyperparameters['source_dimension']}"
                )
            self.source_dimension = hyperparameters['source_dimension']

        if 'n_clusters' in hyperparameters:
            if not (
                isinstance(hyperparameters['n_clusters'], int)
                and (hyperparameters['n_clusters'] >= 2)
                and (self.n_clusters is None)
            ):
                raise LeaspyModelInputError(
                    f"Number of clusters should be an integer greater than 2 , "
                    f"not {hyperparameters['n_clusters']}"
                )
            self.n_clusters = hyperparameters['n_clusters']

        # WIP
        ## special hyperparameter(s) for ordinal model
        #expected_hyperparameters += self._handle_ordinal_hyperparameters(hyperparameters)

        self._raise_if_unknown_hyperparameters(expected_hyperparameters, hyperparameters)

    def to_dict(self, *, with_mixing_matrix: bool = True) -> KwargsType:
        """
        Export ``Leaspy`` object as dictionary ready for :term:`JSON` saving.

        Parameters
        ----------
        with_mixing_matrix : :obj:`bool` (default ``True``)
            Save the :term:`mixing matrix` in the exported file in its 'parameters' section.

            .. warning::
                It is not a real parameter and its value will be overwritten at model loading
                (orthonormal basis is recomputed from other "true" parameters and mixing matrix
                is then deduced from this orthonormal basis and the betas)!
                It was integrated historically because it is used for convenience in
                browser webtool and only there...

        Returns
        -------
        KwargsType :
            The object as a dictionary.
        """
        model_settings = super().to_dict()
        model_settings['source_dimension'] = self.source_dimension

        model_settings['n_clusters'] = self.n_clusters

        if with_mixing_matrix and self.source_dimension >= 1:
            # transposed compared to previous version
            model_settings['parameters']['mixing_matrix'] = self.state['mixing_matrix'].tolist()

        # self._export_extra_ordinal_settings(model_settings)

        return model_settings

    def _compute_initial_values_for_model_parameters(
            #Taken from class LogisticMultivariateInitializationMixin
            #Should I create a Mixture Mixin or leave it here with ++probs
            #do we need to change t0?

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

        n_clusters = self.n_clusters
        for c in range(n_clusters):
            #modify as needed to specify the parameters at each cluster
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
                "tau_mean": self.tau_mean,
                "tau_std": self.tau_std,
                "xi_mean": self.xi_mean,
                "xi_std": self.xi_std,
                "probs": torch.ones(self.n_clusters) / self.n_clusters
            }
            if self.source_dimension >= 1:
                parameters["betas_mean"] = betas
                parameters["sources_mean"] = self.sources_mean
            rounded_parameters = {
                str(p): torch_round(v.to(torch.float32)) for p, v in parameters.items()
            }
            obs_model = next(iter(self.obs_models))  # WIP: multiple obs models...
            #if isinstance(obs_model, FullGaussianObservationModel):
            rounded_parameters["noise_std"] = self.noise_std.expand(
                obs_model.extra_vars['noise_std'].shape
            )
            return rounded_parameters

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

        # is it ok for all the clusters?
        mean_xi = torch.mean(state['xi'])
        state["xi"] = state["xi"] - mean_xi
        state["log_v0"] = state["log_v0"] + mean_xi

    @classmethod
    def _center_sources_realizations(cls, state: State) -> None:
        """
        Center the ``sources`` realizations in place.

        """
        # is it ok for all the clusters?
        mean_sources = torch.mean(state['sources'])
        state["sources"] = state["sources"] - mean_sources

    @classmethod
    def compute_probs(cls, state: State) -> torch.Tensor:
        """
        Compute the probability that each individual belongs to each cluster
        """
        # to complete

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
        cls._center_sources_realizations(state)

        return super().compute_sufficient_statistics(state)

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

    #maybe need to change to take into account all the clusters? also model_no_sources
