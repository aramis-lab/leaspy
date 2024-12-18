import pandas as pd
import numpy as np
import math
import torch
import warnings
from abc import abstractmethod
from typing import Iterable, Optional

from sympy import Product
from sympy.codegen.cnodes import union
from torch.distributions import MixtureSameFamily

from leaspy.utils.weighted_tensor import WeightedTensor, TensorOrWeightedTensor, unsqueeze_right
from leaspy.utils.docs import doc_with_super
from leaspy.utils.functional import Exp, Sqr, OrthoBasis, MatMul, Sum, Prod
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
    _xi_mean = 0
    _sources_mean = 0

    @property
    def xi_mean(self) -> torch.Tensor:
        return torch.tensor(self._xi_mean)

    @property
    def sources_mean(self) -> float:
        return self._sources_mean

    #@property
    #def xi_std(self) -> torch.Tensor:
    #    return torch.tensor([self._xi_std] * self.n_clusters)

    #@property
    #def tau_std(self) -> torch.Tensor:
    #    return torch.tensor([self._tau_std] * self.n_clusters)

    #@property
    #def noise_std(self) -> torch.Tensor:
    #    return torch.tensor(self._noise_std)

    #@property
    #def sources_std(self) -> torch.Tensor:
    #    return torch.tensor([self._sources_std] * self.n_clusters)

    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        obs_models_to_string = [o.to_string() for o in self.obs_models]

        n_clusters = kwargs.get('n_clusters', None)
        observation_models = kwargs.get("obs_models", None)
        kwargs["obs_models"] = (observation_model_factory(observation_models, n_clusters=n_clusters),)

        if (self.dimension == 1) or (self.source_dimension == 0):
            if "mixture-gaussian" in obs_models_to_string:
                raise LeaspyInputError("Mixture does not work for now with a univariate model")
        else:
            if "mixture-gaussian" not in obs_models_to_string:
                self.obs_models += (
                    observation_model_factory(
                        "mixture-gaussian",
                        n_clusters="n_clusters",
                        xi='xi',
                        tau='tau',
                        sources='sources'
                    ),
                )
                obs_models_to_string += ["mixture-gaussian"]

        variables_to_track = [
            "probs_ind",
            "xi_mean",
            #"nll_attach_xi_ind_cluster",
            #"nll_attach_tau_ind_cluster",
            #"nll_attach_y_ind_cluster",
        ]

        if self.source_dimension:
            variables_to_track += ['sources_mean'] #, 'nll_attach_sources_ind_cluster']

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

        d.update(

            # PRIORS
            log_g_mean=ModelParameter.for_pop_mean("log_g", shape=(self.dimension,)),
            log_v0_mean=ModelParameter.for_pop_mean("log_v0",shape=(self.dimension,)),

            tau_mean=ModelParameter.for_ind_mean("tau", shape=(self.n_clusters,)),
            tau_std=ModelParameter.for_ind_std("tau", shape=(self.n_clusters,)),
            xi_mean =ModelParameter.for_ind_mean("xi", shape=(self.n_clusters,)),
            xi_std=ModelParameter.for_ind_std("xi", shape=(self.n_clusters,)),

            # LATENT VARS
            log_g=PopulationLatentVariable(Normal("log_g_mean", "log_g_std")),
            log_v0=PopulationLatentVariable(Normal("log_v0_mean", "log_v0_std")),

            xi=IndividualLatentVariable(
                MixtureNormal(mixture_distribution = Categorical("probs"),
                                    component_distribution = Normal("xi_mean", "xi_std")
                                    )
            ),
            tau=IndividualLatentVariable(
                MixtureNormal(mixture_distribution=Categorical("probs"),
                                    component_distribution=Normal("tau_mean", "tau_std"))
            ),

            # DERIVED VARS
            g=LinkedVariable(Exp("log_g")),
            v0=LinkedVariable(Exp("log_v0")),
            alpha=LinkedVariable(Exp("xi")),
            metric=LinkedVariable(self.metric),
            probs=LinkedVariable(self.compute_probs_ind),

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
                    MixtureNormal(mixture_distribution=Categorical("probs"),
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

        #d.update() #something with the total likelihood and the probabilities #TODO

        return d

    def _validate_compatibility_of_dataset(self, dataset: Optional[Dataset] = None) -> None:
        """
        Check that a valid number of clusters is provided
        """

        super()._validate_compatibility_of_dataset(dataset)

        if self.n_clusters is None:
            warnings.warn("You did not provide `n_clusters` hyperparameter for mixture model")
        elif not (isinstance(self.n_clusters, int) and 2 > self.n_clusters ):
            raise LeaspyModelInputError(
                f"Number of clusters should be an integer greater than 2 "
                f"but you provided `n_clusters` = {self.n_clusters} "
            )

    def _load_hyperparameters(self, hyperparameters: KwargsType) -> None:
        """
        Updates n_clusters along with the other hyperparameters.
        """
        super()._load_hyperparameters(hyperparameters)
        expected_hyperparameters = ('n_clusters')

        if 'n_clusters' in hyperparameters:
            if not (
                    isinstance(hyperparameters['n_clusters'], int)
                    and (hyperparameters['n_clusters'] >= 2)
            ):
                raise LeaspyModelInputError(
                    f"Source dimension should be an integer in greater than 2 , "
                    f"not {hyperparameters['n_clusters']} "
                )
            self.n_clusters = hyperparameters['n_clusters']

        self._raise_if_unknown_hyperparameters(expected_hyperparameters, hyperparameters)

    def to_dict(self, *, with_mixing_matrix: bool = True) -> KwargsType:
        """
        Pass n_clusters to dictionary for consistency
        """
        dict_params = super().to_dict(with_mixing_matrix=with_mixing_matrix)
        dict_params['n_clusters'] = self.n_clusters
        return dict_params

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

        #initialize a df with the probabilities of each individual belonging to each cluster
        n_inds = dataset.to_pandas().reset_index('TIME').groupby('ID').min().shape[0]
        n_clusters = self.n_clusters
        probs_ind = torch.ones(n_inds,n_clusters)/n_clusters
        probs = probs_ind.sum(axis=0)/n_inds

        df = self._get_dataframe_from_dataset(dataset)
        probs_ind_df = pd.concat([pd.DataFrame({"ID": np.arrange(1, n_inds+1,1)}),
                                  pd.DataFrame(probs_ind)], axis=1, join="outer")
        for c in range(n_clusters):
            print(c)
            probs_ind_df = probs_ind_df.rename(columns={c: 'prob_cluster_' + str(c + 1)})

        df = pd.concat([df, probs_ind_df], axis=1, join="outer")

        step = math.ceil(n_inds / n_clusters)
        start = 0
        for c in range(n_clusters):
            df_cluster = df[start:step * (c + 1)]
            start = step * (c + 1)
            slopes_mu, slopes_sigma = compute_patient_slopes_distribution(df_cluster)
            values_mu, values_sigma = compute_patient_values_distribution(df_cluster)
            time_mu, time_sigma = compute_patient_time_distribution(df_cluster)

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
                "xi_mean": self.xi_mean,
                "xi_std": self.xi_std,
            }
            if self.source_dimension >= 1:
                parameters["betas_mean"] = betas
                parameters["sources_mean"] = self.sources_mean
            rounded_parameters_cluster = {
                str(p): torch_round(v.to(torch.float32)) for p, v in parameters.items()
            }
            obs_model_full = FullGaussianObservationModel
            #dirty hack to use the noise as in FullGaussianObservationModel
            rounded_parameters_cluster["noise_std"] = self.noise_std.expand(
                obs_model_full.extra_vars['noise_std'].shape)

            if c==0:
                rounded_parameters = rounded_parameters_cluster
            else:
                rounded_parameters.update(rounded_parameters_cluster)

        rounded_parameters.update(dict({"probs":probs}))
        return rounded_parameters

    @classmethod
    def _center_xi_realizations(cls, state: State) -> None:
        """
        Center the ``xi`` realizations in place
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
    def compute_nll_cluster_random_effects(
            cls,
            probs_ind: torch.Tensor,
            tau: torch.Tensor,
            xi: torch.Tensor,
            sources: torch.Tensor) -> torch.Tensor:

        # the estimated for every cluster
        tau_std = cls.tau_std
        tau_mean = cls.tau_mean
        xi_std = cls.xi_std
        xi_mean = cls.xi_mean
        sources_mean = cls.sources_mean
        n_sources = sources_mean.size()[0]
        nll_constant_standard = 0.5 * torch.log(2 * torch.tensor(math.pi))

        nll_tau = (probs_ind * torch.log(tau_std) + probs_ind * nll_constant_standard +
                   (0.5 * probs_ind * ((tau - tau_mean) / tau_std) ** 2))
        nll_xi = (probs_ind * torch.log(xi_std) + probs_ind * nll_constant_standard +
                  (0.5 * probs_ind * ((xi - xi_mean) / xi_std) ** 2))
        nll_sources = n_sources * nll_constant_standard + (
                0.5 * probs_ind * (sources - sources_mean) ** 2)

        return - nll_tau - nll_xi - nll_sources

    @classmethod
    def compute_nll_cluster_ind(
            cls,
            x: WeightedTensor,
            probs: torch.Tensor,
            loc: torch.Tensor,
            scale: torch.Tensor, ) -> torch.Tensor:

        nll_constant_standard = 0.5 * torch.log(2 * torch.tensor(math.pi))
        nll_ind = probs * torch.log(scale) + probs * nll_constant_standard + (
                    0.5 * probs * ((x.value - loc) / scale) ** 2)

        return -nll_ind

    @classmethod
    def compute_probs_ind(cls,
                          x: WeightedTensor,
                          probs: torch.Tensor,
                          loc: torch.Tensor,
                          scale: torch.Tensor,
                          tau: torch.Tensor,
                          xi: torch.Tensor,
                          sources: torch.Tensor,
                          n_clusters: int,
                          probs_ind=None) -> torch.Tensor:

        nll_ind = cls.compute_nll_cluster_ind(x, probs, loc, scale)
        nll_random = cls.compute_nll_cluster_random_effects(probs, tau, xi, sources)

        denominator = (probs * nll_ind * nll_random).sum(dim=1)  # sum for all the clusters
        nominator = probs * nll_ind * nll_random
        for c in range(n_clusters):
            probs_ind[:, c] = nominator[:, c] / denominator

        return probs_ind


    @classmethod
    def compute_sufficient_statistics(cls, state: State) -> SuffStatsRW:
        """
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
