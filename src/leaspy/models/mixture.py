import math
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import torch

from leaspy.exceptions import LeaspyInputError, LeaspyModelInputError
from leaspy.io.data.dataset import Dataset
from leaspy.utils.typing import KwargsType

from leaspy.variables.distributions import MixtureNormal
from leaspy.variables.specs import (
    Hyperparameter,
    IndividualLatentVariable,
    LinkedVariable,
    ModelParameter,
    NamedVariables,
    SuffStatsRW,
    VariablesLazyValuesRO,
)
from leaspy.variables.state import State

from leaspy.models.base import InitializationMethod
from leaspy.models.obs_models import FullGaussianObservationModel
from .logistic import LogisticModel

from torch.distributions import Normal as TorchNormal

class MixtureInitializationMixin:
    def _compute_initial_values_for_model_parameters(
        self,
        dataset: Dataset,
        ) -> VariablesLazyValuesRO:
        """
        Compute initial values for model parameters.

        Parameters
        ----------
        dataset : ::class:`~leaspy.io.data.Data.Dataset`
            The dataset from which to extract observations and masks.

        Returns
        -------
        :class:`~leaspy.variables.specs.VariablesLazyValuesRO`
            A dictionary mapping parameter names (as strings) to their initialized
            torch.Tensor values.

        Notes
        -----
        - If the initialization method is `DEFAULT`, patient means are used.
        - If `RANDOM`, parameters are sampled from normal distributions
        centered at patient means with estimated standard deviations.
        - `values` are clamped between 0.01 and 0.99 to avoid boundary issues.
        - If the model includes sources (source_dimension >= 1),
        regression coefficients `betas_mean` are initialized accordingly.
        - If the observation model is a `FullGaussianObservationModel`,
        the noise standard deviation parameter is expanded to the correct shape.
        """
        from leaspy.models.utilities import (
            compute_patient_slopes_distribution,
            compute_patient_time_distribution,
            compute_patient_values_distribution,
            get_log_velocities,
            torch_round,
        )

        # initialize a df with the probabilities of each individual belonging to each cluster
        df = dataset.to_pandas(apply_headers=True)
        n_inds = df.reset_index("TIME").groupby("ID").min().shape[0]
        n_clusters = self.n_clusters
        probs = torch.ones(n_clusters) / n_clusters

        slopes_mu, slopes_sigma = compute_patient_slopes_distribution(df)
        values_mu, values_sigma = compute_patient_values_distribution(df)

        if self.initialization_method == InitializationMethod.DEFAULT:
            slopes = slopes_mu
            values = values_mu
            betas = torch.zeros((self.dimension - 1, self.source_dimension))

        if self.initialization_method == InitializationMethod.RANDOM:
            slopes = torch.normal(slopes_mu, slopes_sigma)
            values = torch.normal(values_mu, values_sigma)
            betas = torch.distributions.normal.Normal(loc=0.0, scale=1.0).sample(
                sample_shape=(self.dimension - 1, self.source_dimension)
            )

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

            if self.initialization_method == InitializationMethod.DEFAULT:
                t0_c = time_mu

            if self.initialization_method == InitializationMethod.RANDOM:
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
            "tau_mean": t0,
            "tau_std": self.tau_std,
            "xi_mean": self.xi_mean,
            "xi_std": self.xi_std,
            "probs": probs,
        }
        if self.source_dimension >= 1:
            parameters["betas_mean"] = betas
            parameters["sources_mean"] = self.sources_mean
            rounded_parameters = {
                str(p): torch_round(v.to(torch.float32)) for p, v in parameters.items()
            }
            obs_model = next(iter(self.obs_models))  # WIP: multiple obs models...
            if isinstance(obs_model, FullGaussianObservationModel):
                rounded_parameters["noise_std"] = self.noise_std.expand(
                    obs_model.extra_vars["noise_std"].shape
                )
            return rounded_parameters


class MixtureModel(
    MixtureInitializationMixin, LogisticModel
):
    """Mixture Manifold model for multiple variables of interest (logistic formulation)."""
    type = "mixture"

    _xi_mean = 0
    _xi_std = 0.5
    _tau_std = 5.0
    _noise_std = 0.1
    _sources_mean = 0
    _sources_std = 1.0
    
    @property
    def xi_mean(self) -> torch.Tensor:
        """Return the mean of xi as a tensor."""
        return torch.tensor([2 if i % 2 == 0 else -2 for i in range(self.n_clusters)])
    
    @property
    def xi_std(self) -> torch.Tensor:
        """Return the standard deviation of xi as a tensor."""
        return torch.tensor([self._xi_std] * self.n_clusters)
    
    @property
    def tau_std(self) -> torch.Tensor:
        """Return the standard deviation of tau as a tensor."""
        return torch.tensor([self._tau_std] * self.n_clusters)
    
    @property
    def noise_std(self) -> torch.Tensor:
        """Return the standard deviation of the model as a tensor."""
        return torch.tensor(self._noise_std)
    
    @property
    def sources_mean(self) -> torch.Tensor:
        """Return the mean of the sources as a tensor."""
        return torch.tensor([[1 if (i + j) % 2 == 0 else -1 for j in range(self.n_clusters)]
                            for i in range(self.source_dimension)])
    
    @property
    def sources_std(self) -> torch.Tensor:
        """Return the standard deviation of the sources as a tensor."""
        return torch.ones(
            self.source_dimension, self.n_clusters
        )

    def __init__(self, name: Optional[str] = None, **kwargs):
        
        dimension = kwargs.get("dimension", None)
        n_clusters = kwargs.get("n_clusters", None)

        if "features" in kwargs:
            dimension = len(kwargs["features"])

        observation_models = kwargs.get("obs_models", None)
        if observation_models is None:
            observation_models = "gaussian-diagonal"
            kwargs["obs_models"] = observation_models

        if observation_models == "gaussian-diagonal":
            if n_clusters < 2:
                raise LeaspyInputError(
                    "Number of clusters should be at least 2 to fit a mixture model"
                )
            if dimension == 1:
                raise LeaspyInputError(
                    "You cannot use a multivariate model with 1 feature"
                )
        
        super().__init__(self.type, **kwargs)
        self._configure_variables_to_track()

    def _configure_variables_to_track(self):
        self.track_variables(["probs"])

    def get_variables_specs(self) -> NamedVariables:
        """
        Return the specifications of the variables (latent variables, derived variables,
        model 'parameters') that are part of the model.

        Returns
        -------
        :class:`~leaspy.variables.specs.NamedVariables`
            A dictionary-like object mapping variable names to their specifications.
            These include `ModelParameter`, `Hyperparameter`, `PopulationLatentVariable`,
            and `LinkedVariable` instances.
        """
        d = super().get_variables_specs()

        conflicting_keys = ['tau_mean', 'tau_std', 'tau_sqr', 'xi_mean', 'xi_std', 'xi_sqr',
                            'probs', 'xi', 'tau', 'nll_regul_xi_ind', 'nll_regul_xi', 
                            'nll_regul_tau_ind', 'nll_regul_tau', 'nll_regul_sources_ind', 'nll_regul_sources',
                            'sources_mean', 'sources_std', 'sources']
        for key in conflicting_keys:
            d.pop(key, None)

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
        )
        
        if self.source_dimension >= 1:
            d.update(
            # PRIORS
                sources_mean=ModelParameter.for_ind_mean_mixture(
                    "sources",
                    shape=(self.source_dimension, self.n_clusters,),
                ),
                sources_std=Hyperparameter(1.0),
                # LATENT VARS
                sources=IndividualLatentVariable(MixtureNormal("sources_mean", "sources_std", "probs"),
                                                        sampling_kws={"scale": 10}),
            )
        else:
            d["model"] = LinkedVariable(self.model_no_sources)
        return d

    def _validate_compatibility_of_dataset(
        self, dataset: Optional[Dataset] = None
    ) -> None:
        """
        Validate the compatibility of the provided dataset with the model's configuration.
    
        Parameters
        ----------
        dataset : :class:`~leaspy.io.data.dataset.Dataset`, optional
            The dataset to validate against, by default None.
    
        Raises
        ------
        :exc: `.LeaspyModelInputError`
            If `source_dimension` is provided but not an integer in the valid range
            [0, dataset.dimension - 1), or if `n_clusters` is provided but is not an integer ≥ 2.
        """
        super()._validate_compatibility_of_dataset(dataset)
    
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
        """
        Initialize individual latent parameters in the given state if not already set.
            
        Parameters
        ----------
        state : :class:`~leaspy.variables.state.State`
            The current state object that holds all the variables
        dataset : :class:`~leaspy.io.data.Data.Dataset`
            Dataset used to initialize latent variables accordingly.
        """
        df = dataset.to_pandas().reset_index("TIME").groupby("ID").min()
    
        # Initialise individual parameters if they are not already initialised
        if not state.are_variables_set(("xi", "tau")):
            df_ind = df["TIME"].to_frame(name="tau")
            df_ind["xi"] = 0.0
        else:
            for k in ["xi", "tau"]:
                if state[k].ndim != 2:
                    state[k] = state[k].reshape(-1, 1)
            df_ind = pd.DataFrame(
                torch.concat([state["xi"], state["tau"]], axis=1).detach().numpy(),
                columns=["xi", "tau"],
                index=np.arange(state["xi"].shape[0]),  # use correct number of rows
            )
    
        if self.source_dimension > 0:
            for i in range(self.source_dimension):
                df_ind[f"sources_{i}"] = 0.0
    
        with state.auto_fork(None):
            state.put_individual_latent_variables(df=df_ind)

    def _load_hyperparameters(self, hyperparameters: KwargsType) -> None:
        """
        Updates all model hyperparameters from the provided hyperparameters.
    
        Parameters
        ----------
        hyperparameters : :class:`~leaspy.utils.typing.KwargsType`
            Dictionary containing the hyperparameters to be loaded.
            Expected keys include:
            - "n_clusters": Integer, must be ≥ 2
    
        Raises
        ------
        :exc: `LeaspyModelInputError`
            - `n_clusters` is missing or less than 2
        """
        super()._load_hyperparameters(hyperparameters)

        expected_hyperparameters = (
            "features",
            "dimension",
            "source_dimension",
            "n_clusters",
            "obs_models",
        )
    
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

    def to_dict(self) -> KwargsType:
        """
        Export model object as dictionary ready for :term:`JSON` saving.
    
        Returns
        -------
        :class:`~leaspy.utils.typing.KwargsType` :
            The object as a dictionary.
        """
        # add n_clusters
        model_settings = super().to_dict()
        model_settings["n_clusters"] = self.n_clusters
    
        return model_settings

    @classmethod
    def _center_sources_realizations(cls, state: State) -> None:
        """
        Center the ``sources`` realizations in place.
    
        Parameters
        ----------
        state : :class:`~leaspy.variables.state.State`
            The dictionary-like object representing current model state, which
            contains keys such as``"sources"``.
        """
        mean_sources = torch.mean(state["sources"])
        state["sources"] = state["sources"] - mean_sources

    @classmethod
    def compute_sufficient_statistics(cls, state: State) -> SuffStatsRW:
        """
        Compute the model's :term:`sufficient statistics`.
    
        Parameters
        ----------
        state : :class:`~leaspy.variables.state.State`
            The state to pick values from.
    
        Returns
        -------
        SuffStatsRW :
            The computed sufficient statistics.
        """
        cls._center_sources_realizations(state)
    
        return super().compute_sufficient_statistics(state)

    def get_individual_probabilities(self, ip_dataframe: pd.DataFrame):
        """
        Return the dataframe of individual parameters with the probabilities 
        for each individual belonging to each cluster and the cluster labels.

        Parameters
        ----------
        ip_dataframe : :class:`pandas.DataFrame`
            The dataframe of the individual parameters that comes as an output of personalize.

        Returns
        -------
        :class:`pandas.DataFrame`
            The input dataframe with additional columns for the probabilities of each cluster.
        """

        params = self.parameters
        probs = params["probs"]

        n = len(ip_dataframe)
        c = self.n_clusters
        d = self.source_dimension

        means = {
            "tau": params["tau_mean"], 
            "xi": params["xi_mean"],
        }
    
        for s in range(d):
            means[f"sources_{s}"] = params["sources_mean"][s, :]

        stds = {
            "tau": params["tau_std"],
            "xi": params["xi_std"],
        }

        for s in range(d):
            stds[f"sources_{s}"] = torch.ones(c)

        values = {
            "tau": torch.tensor(ip_dataframe["tau"].values),
            "xi": torch.tensor(ip_dataframe["xi"].values),
        }
    
        for s in range(d):
            values[f"sources_{s}"] = torch.tensor(ip_dataframe[f"sources_{s}"].values)

        # Compute log-likelihoods for each variable
        log_likelihoods = torch.zeros((n, c))

        for var in means.keys():
            x = values[var]

            for cluster in range(c):
                dist = TorchNormal(means[var][cluster], stds[var][cluster])
                log_likelihoods[:, cluster] += dist.log_prob(x)

        # Add log-priors
        log_priors = torch.log(probs)
        log_posteriors = log_likelihoods + log_priors

        # Normalize using logsumexp
        log_sum = torch.logsumexp(log_posteriors, dim=1, keepdim=True)
        responsibilities = torch.exp(log_posteriors - log_sum)

        for i in range(responsibilities.shape[1]):
            ip_dataframe[f"prob_cluster_{i}"] = responsibilities[:, i].numpy()

        # Automatically find all probability columns
        prob_cols = [col for col in ip_dataframe.columns if col.startswith("prob_cluster_")]

        # Assign the most likely cluster
        ip_dataframe["cluster_label"] = ip_dataframe[prob_cols].values.argmax(axis=1)

        return ip_dataframe
