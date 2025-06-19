import math
import warnings
from abc import abstractmethod
from typing import Iterable, Optional, Dict

import numpy as np
import pandas as pd
import torch

from leaspy.exceptions import LeaspyInputError, LeaspyModelInputError
from leaspy.io.data.dataset import Dataset
from leaspy.models.abstract_model import AbstractModel, InitializationMethod
from leaspy.models.abstract_multivariate_model import AbstractMultivariateModel
from leaspy.models.base import InitializationMethod
from leaspy.models.multivariate import LogisticMultivariateModel
from pandas import Categorical

from leaspy.models.obs_models import (
    FullGaussianObservationModel,
    MixtureGaussianObservationModel,
    observation_model_factory,
)
from leaspy.utils.docs import doc_with_super
from leaspy.utils.functional import Exp, MatMul, OrthoBasis, Prod, Sqr, Sum
from leaspy.utils.typing import KwargsType, Optional

# from sympy import Product
# from sympy.codegen.cnodes import union
# from torch.distributions import MixtureSameFamily
from leaspy.utils.weighted_tensor import (
    TensorOrWeightedTensor,
    WeightedTensor,
    unsqueeze_right,
)
from leaspy.variables.distributions import MixtureNormal, Normal, MultivariateNormal
#from torch.distributions import Categorical as TorchCategorical
from leaspy.variables.distributions import MultinomialDistribution as Multinomial
from leaspy.variables.specs import (#
    Hyperparameter,
    IndividualLatentVariable,
    LinkedVariable,
    ModelParameter,
    NamedVariables,
    PopulationLatentVariable,
    SuffStatsRW,
    VariablesValuesRO,
)

from leaspy.variables.specs import (
    LVL_FT,
    Collect,
    VariableInterface,
    VarName,
)
from leaspy.variables.state import State


@doc_with_super()
class AbstractMultivariateMixtureModel(AbstractModel):
    """
    Contains the common attributes & methods of the mixture models.
    Developed according to AbstractMultivariateModel.
    Modified accordingly to handle the n_clusters parameter and model parameters as vectors with n_cluster items.
    """

    _xi_mean = 0
    _xi_std = 0.5
    _tau_std = 5.0
    _noise_std = 0.1
    _sources_mean = 0
    _sources_std = 1.0

    @property
    def xi_mean(self) -> torch.Tensor:
        #return torch.tensor([self._xi_mean] * self.n_clusters)
        return torch.tensor([2 if i % 2 == 0 else -2 for i in range(self.n_clusters)])

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
    def sources_mean(self) -> torch.Tensor:
        #return torch.zeros(self.source_dimension, self.n_clusters)  # be careful with the dimensions
        return torch.tensor([[1 if (i + j) % 2 == 0 else -1 for j in range(self.n_clusters)]
                             for i in range(self.source_dimension)])

    @property
    def sources_std(self) -> torch.Tensor:
        return torch.ones(
            self.source_dimension, self.n_clusters
        )

        #@property
    #def sources_std(self) -> torch.Tensor:
    #    return torch.Tensor(
    #        [self._sources_std] * self.n_clusters
    #    )  # not sure it's working, it was float before

    #@property
    #def sources_std(self) -> float:
    #    return self._sources_std


    def __init__(self, name: str, **kwargs):
        # n_clusters = kwargs.get('n_clusters', None)
        # kwargs["obs_models"] = (observation_model_factory(observation_models, n_clusters=n_clusters, dimension=dimension),)
        # not sure how to treat n_clusters

        self.source_dimension: Optional[int] = None

        dimension = kwargs.get("dimension", None)
        n_clusters = kwargs.get("n_clusters", None)
        if "features" in kwargs:
            dimension = len(kwargs["features"])
        observation_models = kwargs.get("obs_models", None)
        if observation_models is None:
            #observation_models = "mixture-gaussian"
            observation_models = "gaussian-diagonal"
        #if observation_models == "mixture-gaussian":
        if observation_models == "gaussian-diagonal":
            if n_clusters < 2:
                raise LeaspyInputError(
                    "Number of clusters should be at least 2 to fit a mixture model"
                )
            if dimension == 1:
                raise LeaspyInputError(
                    "You cannot use a multivariate model with 1 feature"
                )
        if isinstance(observation_models, (list, tuple)):
            kwargs["obs_models"] = tuple(
                [
                    observation_model_factory(obs_model, **kwargs)
                    for obs_model in observation_models
                ]
            )
        elif isinstance(observation_models, (dict)):
            # Not really satisfied... Used for api load
            kwargs["obs_models"] = tuple(
                [
                    observation_model_factory(
                        observation_models["y"],
                        dimension=dimension,
                        n_clusters=n_clusters,
                    )
                ]
            )
        else:
            kwargs["obs_models"] = (
                observation_model_factory(
                    observation_models, dimension=dimension, n_clusters=n_clusters
                ),
            )
        super().__init__(name, **kwargs)

    def get_variables_specs(self) -> NamedVariables:
        """
        Return the specifications of the variables (latent variables,
        derived variables, model 'parameters') that are part of the model.
        """
        d = super().get_variables_specs()

        d.update(
            # PRIORS
            tau_mean=ModelParameter.for_ind_mean_mixture("tau", shape=(self.n_clusters,)),
            tau_std=ModelParameter.for_ind_std_mixture("tau", shape=(self.n_clusters,)),
            xi_mean=ModelParameter.for_ind_mean_mixture("xi", shape=(self.n_clusters,)),
            xi_std=ModelParameter.for_ind_std_mixture("xi", shape=(self.n_clusters,)),
            probs = ModelParameter.for_probs(shape=self.n_clusters),
            # LATENT VARS
            xi=IndividualLatentVariable(MixtureNormal("xi_mean", "xi_std", "probs"),
                                        sampling_kws={"scale": 10},),
            tau=IndividualLatentVariable(MixtureNormal("tau_mean", "tau_std", "probs"),
                                         sampling_kws={"scale": 10},),
            # DERIVED VARS
            alpha=LinkedVariable(Exp("xi")),
            #probs=LinkedVariable(self.compute_probs),
        )

        if self.source_dimension >= 1:
            d.update(
                # PRIORS
                betas_mean=ModelParameter.for_pop_mean(
                    "betas",
                    shape=(self.dimension - 1, self.source_dimension),
                ),
                betas_std=Hyperparameter(0.01),
                sources_mean=ModelParameter.for_ind_mean_mixture(
                    "sources",
                    shape=(self.source_dimension, self.n_clusters,),
                ),
                #sources_std=Hyperparameter(self.sources_std),
                sources_std=Hyperparameter(1.0),
                # LATENT VARS
                betas=PopulationLatentVariable(
                    Normal("betas_mean", "betas_std"),
                    sampling_kws={"scale": 0.5},
                ),
                sources=IndividualLatentVariable(MixtureNormal("sources_mean", "sources_std", "probs"),
                                                 sampling_kws={"scale": 10}),
                #sources=IndividualLatentVariable(MultivariateNormal(
                #    "sources_mean", "sources_std"
                #)
                #),
                # DERIVED VARS
                mixing_matrix=LinkedVariable(
                    MatMul("orthonormal_basis", "betas").then(torch.t)
                ),  # shape: (Ns, Nfts)
                space_shifts=LinkedVariable(
                    MatMul("sources", "mixing_matrix")
                ),  # shape: (Ni, Nfts)
            )

        return d

    def _get_dataframe_from_dataset(self, dataset: Dataset) -> pd.DataFrame:
        """
        Returns a pands dataframe from the given dataset.
        """
        # exact same function as in the AbstractMultivariateModel

        df = dataset.to_pandas().dropna(how="all").sort_index()[dataset.headers]
        if not df.index.is_unique:
            raise LeaspyInputError("Index of DataFrame is not unique.")
        if not df.index.to_frame().notnull().all(axis=None):
            raise LeaspyInputError("Index of DataFrame contains unvalid values.")
        if self.features != df.columns.tolist():
            raise LeaspyInputError(
                f"Features mismatch between model and dataset: {self.features} != {df.columns}"
            )
        return df

    def _validate_compatibility_of_dataset(
        self, dataset: Optional[Dataset] = None
    ) -> None:
        """
        Checks compatibility of dataset.
        Raises input errors if hyperparameters are not valid.
        """
        super()._validate_compatibility_of_dataset(dataset)

        if not dataset:
            return
        if self.source_dimension is None:
            self.source_dimension = int(dataset.dimension**0.5)
            warnings.warn(
                "You did not provide `source_dimension` hyperparameter for multivariate model, "
                f"setting it to ⌊√dimension⌋ = {self.source_dimension}."
            )
        elif not (
            isinstance(self.source_dimension, int)
            and 0 <= self.source_dimension < dataset.dimension
        ):
            raise LeaspyModelInputError(
                f"Sources dimension should be an integer in [0, dimension - 1[ "
                f"but you provided `source_dimension` = {self.source_dimension} "
                f"whereas `dimension` = {dataset.dimension}."
            )

        # add n_clusters
        if self.n_clusters is None:
            warnings.warn(
                "You did not provide `n_clusters` hyperparameter for mixture model"
            )
        elif not (isinstance(self.n_clusters, int) and self.n_clusters >= 2):
            raise LeaspyModelInputError(
                f"Number of clusters should be an integer greater than 2 "
                f"but you provided `n_clusters` = {self.n_clusters} "
            )

    def put_individual_parameters(self, state: State, dataset: Dataset):
        df = dataset.to_pandas().reset_index("TIME").groupby("ID").min()

        # Initialise individual parameters if they are not already initialised
        if not state.are_variables_set(("xi", "tau")):
            df_ind = df["TIME"].to_frame(name="tau")
            df_ind["xi"] = 0.0
        else:
            #df_ind = pd.DataFrame(
            #    torch.concat([state["xi"], state["tau"]], axis=1),
            #    columns=["xi", "tau"],
            #    index=df.index,
            #)
            df_ind = pd.DataFrame(
                torch.concat([state["xi"], state["tau"]], axis=1).detach().numpy(),
                columns=["xi", "tau"],
                index=np.arange(state["xi"].shape[0]),  # use correct number of rows
            )

        # Set the right initialisation point fpr barrier methods -JOINTMODEL
        #df_inter = pd.concat(
        #    [df["EVENT_TIME"] - self.init_tolerance, df_ind["tau"]], axis=1
        #)
        #df_ind["tau"] = df_inter.min(axis=1)

        if self.source_dimension > 0:
            for i in range(self.source_dimension):
                df_ind[f"sources_{i}"] = 0.0

        #if self.n_clusters > 0:
        #    for i in range(self.n_clusters):
        #        df_ind[f"probs_{i}"] = 1/self.n_clusters

        with state.auto_fork(None):
            state.put_individual_latent_variables(df=df_ind)

    def _load_hyperparameters(self, hyperparameters: KwargsType) -> None:
        """
        Updates all model hyperparameters from the provided hyperparameters.
        """
        # add n_clusters
        expected_hyperparameters = (
            "features",
            "dimension",
            "source_dimension",
            "n_clusters",
        )

        if "features" in hyperparameters:
            self.features = hyperparameters["features"]

        if "dimension" in hyperparameters:
            if self.features and hyperparameters["dimension"] != len(self.features):
                raise LeaspyModelInputError(
                    f"Dimension provided ({hyperparameters['dimension']}) does not match "
                    f"features ({len(self.features)})"
                )
            self.dimension = hyperparameters["dimension"]

        if "source_dimension" in hyperparameters:
            if not (
                isinstance(hyperparameters["source_dimension"], int)
                and (hyperparameters["source_dimension"] >= 0)
                and (
                    self.dimension is None
                    or hyperparameters["source_dimension"] <= self.dimension - 1
                )
            ):
                raise LeaspyModelInputError(
                    f"Source dimension should be an integer in [0, dimension - 1], "
                    f"not {hyperparameters['source_dimension']}"
                )
            self.source_dimension = hyperparameters["source_dimension"]

            if "n_clusters" in hyperparameters:
                if not (
                    isinstance(hyperparameters["n_clusters"], int)
                    and (hyperparameters["n_clusters"] >= 2)
                ):
                    raise LeaspyModelInputError(
                        f"Number of clusters should be an integer greater than 2, "
                        f"not {hyperparameters['n_clusters']} "
                    )
                self.n_clusters = hyperparameters["n_clusters"]

        self._raise_if_unknown_hyperparameters(
            expected_hyperparameters, hyperparameters
        )

    def to_dict(self, *, with_mixing_matrix: bool = True) -> KwargsType:
        """
        Export ``Leaspy`` object as dictionary ready for :term:`JSON` saving.
        """
        # add n_clusters
        model_settings = super().to_dict()

        model_settings["n_clusters"] = self.n_clusters
        model_settings["source_dimension"] = self.source_dimension

        if with_mixing_matrix and self.source_dimension >= 1:
            # transposed compared to previous version
            model_settings["parameters"]["mixing_matrix"] = self.state[
                "mixing_matrix"
            ].tolist()

        return model_settings


@doc_with_super()
class MultivariateMixtureModel(AbstractMultivariateMixtureModel):
    """
    Manifold mixture model for multiple variables of interest (logistic or linear formulation).
    """

    def __init__(
        self, name: str, variables_to_track: Optional[Iterable[str]] = None, **kwargs
    ):
        super().__init__(name, **kwargs)

        default_variables_to_track = [
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
            "nll_tot",
            # specific to the mixture model :
            #"probs",
            #"probs_ind",
            ##"nll_attach_ind",
            ##"nll_regul_tau",
            ##"nll_regul_tau_ind",
            ##"nll_regul_xi",
            ##"nll_regul_xi_ind",
        ]

        if self.source_dimension:
            default_variables_to_track += [
                "sources",
                "betas",
                "mixing_matrix",
                "space_shifts",
                "sources_mean",
                ##"nll_regul_sources",
                ##"nll_regul_sources_ind",
            ]  # specific to the mixture model

        variables_to_track = variables_to_track or default_variables_to_track
        self.tracked_variables = self.tracked_variables.union(set(variables_to_track))

        self.tracked_variables_ordered = variables_to_track

    @classmethod
    def _center_xi_realizations(cls, state: State) -> None:
        """
        Center the ``xi`` realizations in place
        """
        mean_xi = torch.mean(state["xi"])
        state["xi"] = state["xi"] - mean_xi
        state["log_v0"] = state["log_v0"] + mean_xi

    @classmethod
    def _center_sources_realizations(cls, state: State) -> None:
        """
        Center the ``sources`` realizations in place.
        """
        mean_sources = torch.mean(state["sources"])
        #std_sources = torch.std(state['sources'])
        state["sources"] = state["sources"] - mean_sources
        #state["sources"] = (state["sources"] - mean_sources) / std_sources

    @classmethod
    def compute_sufficient_statistics(cls, state: State) -> SuffStatsRW:
        """ """
        cls._center_xi_realizations(state)
        cls._center_sources_realizations(state)

        return super().compute_sufficient_statistics(state)

    def get_variables_specs(self) -> NamedVariables:
        """
        Return the specifications of the variables (latent variables, derived variables,
        model 'parameters') that are part of the model.
        """
        d = super().get_variables_specs()
        d.update(
            # PRIORS
            log_v0_mean=ModelParameter.for_pop_mean(
                "log_v0",
                shape=(self.dimension,),
            ),
            log_v0_std=Hyperparameter(0.01),
            # no xi_mean as hyperaparameter
            # LATENT VARS
            log_v0=PopulationLatentVariable(Normal("log_v0_mean", "log_v0_std")),
            # DERIVED VARS
            v0=LinkedVariable(
                Exp("log_v0"),
            ),
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


class LogisticMultivariateMixtureInitializationMixin:
    def _compute_initial_values_for_model_parameters(
        self,
        dataset: Dataset,
        method: InitializationMethod,
    ) -> VariablesValuesRO:
        """Compute initial values for model parameters."""
        from leaspy.models.utilities import (
            compute_patient_slopes_distribution,
            compute_patient_time_distribution,
            compute_patient_values_distribution,
            get_log_velocities,
            torch_round,
        )

        # initialize a df with the probabilities of each individual belonging to each cluster
        n_inds = dataset.to_pandas().reset_index("TIME").groupby("ID").min().shape[0]
        n_clusters = self.n_clusters
        #probs_ind = torch.ones(n_inds, n_clusters) / n_clusters
        #probs = probs_ind.sum(axis=0) / n_inds
        probs = torch.ones(n_clusters) / n_clusters

        df = self._get_dataframe_from_dataset(dataset)
        slopes_mu, slopes_sigma = compute_patient_slopes_distribution(df)
        values_mu, values_sigma = compute_patient_values_distribution(df)

        if method == InitializationMethod.DEFAULT:
            slopes = slopes_mu
            values = values_mu
            betas = torch.zeros((self.dimension - 1, self.source_dimension))

        if method == InitializationMethod.RANDOM:
            slopes = torch.normal(slopes_mu, slopes_sigma)
            values = torch.normal(values_mu, values_sigma)
            betas = torch.distributions.normal.Normal(loc=0.0, scale=1.0).sample(
                sample_shape=(self.dimension - 1, self.source_dimension)
            )

        #probs_ind_df = pd.concat(
        #    [
        #        pd.DataFrame({"ID": np.arange(1, n_inds + 1, 1)}),
        #        pd.DataFrame(probs_ind),
        #    ],
        #    axis=1,
        #    join="outer",
        #)
        #for c in range(n_clusters):
        #    probs_ind_df = probs_ind_df.rename(
        #        columns={c: "prob_cluster_" + str(c + 1)}
        #    )

        # df = pd.concat([df, probs_ind_df], axis=1, join="outer")

        step = math.ceil(n_inds / n_clusters)
        start = 0
        ids = pd.DataFrame(
            df.index.get_level_values("ID").unique()
        )  # get the values of the IDs

        for c in range(n_clusters):
            ids_cluster = ids.loc[
                start : step * (c + 1), "ID"
            ]  # get the IDs of the cluster
            df_cluster = df.loc[
                ids_cluster.values
            ]  # get all the dataframe for the cluster
            time_mu, time_sigma = compute_patient_time_distribution(df_cluster)

            if method == InitializationMethod.DEFAULT:
                t0_c = time_mu

            if method == InitializationMethod.RANDOM:
                t0_c = torch.normal(time_mu, time_sigma)

            start = step * (c + 1) + 1

            # stock the values for all the clusters
            if c == 0:
                t0 = t0_c.unsqueeze(-2)
            else:
                t0 = torch.tensor(np.append(t0, t0_c.item()))

        # Enforce values are between 0 and 1
        values = values.clamp(
            min=1e-2, max=1 - 1e-2
        )  # always "works" for ordinal (values >= 1)

        parameters = {
            "log_g_mean": torch.log(1.0 / values - 1.0),
            "log_v0_mean": get_log_velocities(slopes, self.features),
            # "log_v0_mean": log_velocities,
            "tau_mean": t0,
            "tau_std": self.tau_std,
            "xi_mean": self.xi_mean,
            "xi_std": self.xi_std,
            #"probs_ind": probs_ind,
            "probs": probs,
        }
        if self.source_dimension >= 1:
            parameters["betas_mean"] = betas
            parameters["sources_mean"] = self.sources_mean
            rounded_parameters = {
                str(p): torch_round(v.to(torch.float32)) for p, v in parameters.items()
            }
            obs_model = next(iter(self.obs_models))  # WIP: multiple obs models...
            #if isinstance(obs_model, MixtureGaussianObservationModel):
            if isinstance(obs_model, FullGaussianObservationModel):
                rounded_parameters["noise_std"] = self.noise_std.expand(
                    obs_model.extra_vars["noise_std"].shape
                )
            return rounded_parameters


class LogisticMultivariateMixtureModel(
    LogisticMultivariateMixtureInitializationMixin, MultivariateMixtureModel
):
    """Mixture Manifold model for multiple variables of interest (logistic formulation)."""

    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)

    def get_variables_specs(self) -> NamedVariables:
        """
        Return the specifications of the variables (latent variables,
        derived variables, model 'parameters') that are part of the model.
        """
        d = super().get_variables_specs()
        d.update(
            log_g_mean=ModelParameter.for_pop_mean("log_g", shape=(self.dimension,)),
            log_g_std=Hyperparameter(0.01),
            log_g=PopulationLatentVariable(Normal("log_g_mean", "log_g_std")),
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
        w_model_logit = metric[pop_s] * (
            v0[pop_s] * rt + space_shifts[:, None, ...]
        ) - torch.log(g[pop_s])
        model_logit, weights = WeightedTensor.get_filled_value_and_weight(
            w_model_logit, fill_value=0.0
        )
        return WeightedTensor(torch.sigmoid(model_logit), weights).weighted_value


