import math
import warnings
from abc import abstractmethod
from typing import Iterable, Optional, Dict

import numpy as np
import pandas as pd
import torch

from leaspy.exceptions import LeaspyInputError, LeaspyModelInputError, LeaspyIndividualParamsInputError
from leaspy.io.data.dataset import Dataset
from leaspy.models.base import InitializationMethod

from leaspy.models.obs_models import (
    FullGaussianObservationModel,
    observation_model_factory,
)
from leaspy.utils.docs import doc_with_super
from leaspy.utils.functional import Exp, MatMul, OrthoBasis, Sqr
from leaspy.utils.typing import KwargsType
from leaspy.utils.typing import DictParams, KwargsType

from leaspy.utils.weighted_tensor import (
    TensorOrWeightedTensor,
    WeightedTensor,
    unsqueeze_right,
)
from leaspy.variables.distributions import MixtureNormal, Normal
from leaspy.variables.specs import (
    Hyperparameter,
    IndividualLatentVariable,
    LinkedVariable,
    ModelParameter,
    NamedVariables,
    PopulationLatentVariable,
    SuffStatsRW,
    VariablesLazyValuesRO,
)

from leaspy.variables.specs import (
    LVL_FT,
)
from leaspy.variables.state import State
from .mcmc_saem_compatible import McmcSaemCompatibleModel


@doc_with_super()
class TimeReparametrizedMixtureModel(McmcSaemCompatibleModel):
    """
    A time-reparametrized model tailored to handle mixture models with multiple clusters.
    
    This class extends `TimeReparametrizedModel` to incorporate mixture-specific behaviors,
    including support for multiple clusters (`n_clusters`) and corresponding vectorized parameters.

    Parameters
    ----------
    name : :obj:`str`
        Name of the model.
    source_dimension : Optional[:obj:`int`]
        Number of sources. Dimension of spatial components (default is None).
    **kwargs: :obj:`dict`
       Additional hyperparameters for the model. Must include:
            - 'n_clusters': int
                Number of mixture components (must be ≥ 2).
            - 'dimension' or 'features': int or list
                Dimensionality of the input data.
            - 'obs_models': str, list, or dict (optional)
                Specification of the observation model(s). Defaults to "gaussian-diagonal".

    Raises
    ------
    :exc:`.LeaspyModelInputError`
        If inconsistent hyperparameters.
    """

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

    def __init__(self, name: str, **kwargs):

        self.source_dimension: Optional[int] = None

        dimension = kwargs.get("dimension", None)
        n_clusters = kwargs.get("n_clusters", None)
        if "features" in kwargs:
            dimension = len(kwargs["features"])
        observation_models = kwargs.get("obs_models", None)
        if observation_models is None:
            observation_models = "gaussian-diagonal"
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

        Returns
        -------
        :class:`~leaspy.variables.specs.NamedVariables` :
            A dictionary-like object containing specifications for the variables
        """
        d = super().get_variables_specs()

        d.update(
            rt=LinkedVariable(self.time_reparametrization),
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
                sources_std=Hyperparameter(1.0),
                # LATENT VARS
                betas=PopulationLatentVariable(
                    Normal("betas_mean", "betas_std"),
                    sampling_kws={"scale": 0.5},
                ),
                sources=IndividualLatentVariable(MixtureNormal("sources_mean", "sources_std", "probs"),
                                                 sampling_kws={"scale": 10}),
                # DERIVED VARS
                mixing_matrix=LinkedVariable(
                    MatMul("orthonormal_basis", "betas").then(torch.t)
                ),  # shape: (Ns, Nfts)
                space_shifts=LinkedVariable(
                    MatMul("sources", "mixing_matrix")
                ),  # shape: (Ni, Nfts)
            )

        return d
    
    @property
    def has_sources(self) -> bool:
        """
        Indicates whether the model includes sources.

        Returns
        -------
        :obj:`bool`
            True if `source_dimension` is a positive integer.
            False otherwise.
        """
        return (
            hasattr(self, "source_dimension")
            and isinstance(self.source_dimension, int)
            and self.source_dimension > 0
        )
    
    @staticmethod
    def time_reparametrization(
        *,
        t: TensorOrWeightedTensor[float],
        alpha: torch.Tensor,
        tau: torch.Tensor,
    ) -> TensorOrWeightedTensor[float]:
        """
        Tensorized time reparametrization formula.

        .. warning::
            Shapes of tensors must be compatible between them.

        Parameters
        ----------
        t : :class:`torch.Tensor`
            Timepoints to reparametrize
        alpha : :class:`torch.Tensor`
            Acceleration factors of individual(s)
        tau : :class:`torch.Tensor`
            Time-shift(s) of individual(s)

        Returns
        -------
        :class:`torch.Tensor`
            Reparametrized time of same shape as `timepoints`
        """
        return alpha * (t - tau)


    def _validate_compatibility_of_dataset(
        self, dataset: Optional[Dataset] = None
    ) -> None:
        """
        Validate the compatibility of the provided dataset with the model's configuration.

        Parameters
        ----------
        dataset : Optional[:class:`~leaspy.io.data.Data.Dataset`], optional
            The dataset to validate against, by default None.

        Raises
        ------
        :exc: `.LeaspyModelInputError`
            If `source_dimension` is provided but not an integer in the valid range
            [0, dataset.dimension - 1), or if `n_clusters` is provided but is not an integer ≥ 2.
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
        
    def _audit_individual_parameters(
        self, individual_parameters: DictParams
    ) -> KwargsType:
        """
        Validate and process individual parameter inputs for model compatibility.

        Parameters
        ----------
        individual_parameters : :class:`~leaspy.utils.typing.DictParams`
            A dictionary mapping parameter names (strings) to their values,
            which can be scalars or array-like structures.

        Returns
        -------
        KwargsType: :class:`~leaspy.utils.typing.KwargsType`
            A dictionary with the following keys:
            - "nb_inds": Number of individuals
            - "tensorized_ips": Dictionary of parameters converted to 2D tensors.
            - "tensorized_ips_gen": Generator yielding tensors for each individual,
            each with an added batch dimension.

        Raises
        ------
        :exc: `LeaspyIndividualParamsInputError`
            If the provided dictionary keys do not match the expected parameter names,
            or if the sizes of individual parameters are inconsistent,
            or if `sources` parameter does not meet array-like requirements.
        """
        from .utilities import is_array_like, tensorize_2D

        expected_parameters = set(["xi", "tau"] + int(self.has_sources) * ["sources"])
        given_parameters = set(individual_parameters.keys())
        symmetric_diff = expected_parameters.symmetric_difference(given_parameters)
        if len(symmetric_diff) > 0:
            raise LeaspyIndividualParamsInputError(
                f"Individual parameters dict provided {given_parameters} "
                f"is not compatible for {self.name} model. "
                f"The expected individual parameters are {expected_parameters}."
            )
        ips_is_array_like = {
            k: is_array_like(v) for k, v in individual_parameters.items()
        }
        ips_size = {
            k: len(v) if ips_is_array_like[k] else 1
            for k, v in individual_parameters.items()
        }
        if self.has_sources:
            if not ips_is_array_like["sources"]:
                raise LeaspyIndividualParamsInputError(
                    f"Sources must be an array_like but {individual_parameters['sources']} was provided."
                )
            tau_xi_scalars = all(ips_size[k] == 1 for k in ["tau", "xi"])
            if tau_xi_scalars and (ips_size["sources"] > 1):
                # is 'sources' not a nested array? (allowed iff tau & xi are scalars)
                if not is_array_like(individual_parameters["sources"][0]):
                    # then update sources size (1D vector representing only 1 individual)
                    ips_size["sources"] = 1
            # TODO? check source dimension compatibility?
        uniq_sizes = set(ips_size.values())
        if len(uniq_sizes) != 1:
            raise LeaspyIndividualParamsInputError(
                f"Individual parameters sizes are not compatible together. Sizes are {ips_size}."
            )
        # number of individuals present
        n_individual_parameters = uniq_sizes.pop()
        # properly choose unsqueezing dimension when tensorizing array_like (useful for sources)
        # [1,2] => [[1],[2]] (expected for 2 individuals / 1D sources)
        # [1,2] => [[1,2]] (expected for 1 individual / 2D sources)
        unsqueeze_dim = 0 if n_individual_parameters == 1 else -1
        # tensorized (2D) version of ips
        tensorized_individual_parameters = {
            name: tensorize_2D(value, unsqueeze_dim=unsqueeze_dim)
            for name, value in individual_parameters.items()
        }

        return {
            "nb_inds": n_individual_parameters,
            "tensorized_ips": tensorized_individual_parameters,
            "tensorized_ips_gen": (
                {
                    name: value[individual, :].unsqueeze(0)
                    for name, value in tensorized_individual_parameters.items()
                }
                for individual in range(n_individual_parameters)
            ),
        }

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
            - "features": List or sequence of feature names
            - "dimension": Integer specifying the number of features
            - "source_dimension": Integer specifying the number of sources; must be in
            [0, dimension - 1].
            - "n_clusters": Integer, must be ≥ 2

        Raises
        ------
        :exc: `LeaspyModelInputError`
            - `dimension` does not match the number of `features`
            - `source_dimension` is invalid or out of range
            - `n_clusters` is missing or less than 2
        """
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
        Export model object as dictionary ready for :term:`JSON` saving.

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
        :class:`~leaspy.utils.typing.KwargsType` :
            The object as a dictionary.
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
class RiemanianManifoldMixtureModel(TimeReparametrizedMixtureModel):
    """
    A riemannian manifold model tailored to handle mixture models with multiple clusters.
    
    This class extends `RiemanianManifoldModel` to incorporate mixture-specific behaviors,
    mainly the handling of sources for multiple clusters. 

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
        ]

        if self.source_dimension:
            default_variables_to_track += [
                "sources",
                "betas",
                "mixing_matrix",
                "space_shifts",
                "sources_mean",
            ]

        variables_to_track = variables_to_track or default_variables_to_track
        self.tracked_variables = self.tracked_variables.union(set(variables_to_track))

        self.tracked_variables_ordered = variables_to_track

    @classmethod
    def _center_xi_realizations(cls, state: State) -> None:
        """
        Center the ``xi`` realizations in place.

        Parameters
        ----------
        state : :class:`~leaspy.variables.state.State`
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
        cls._center_xi_realizations(state)
        cls._center_sources_realizations(state)

        return super().compute_sufficient_statistics(state)

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
        """
        Return the model output when sources(spatial components) are not present.

        Parameters
        ----------
        rt :  :class:`torch.Tensor`
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


class LogisticMixtureInitializationMixin:
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


class LogisticMultivariateMixtureModel(
    LogisticMixtureInitializationMixin, RiemanianManifoldMixtureModel
):
    """Mixture Manifold model for multiple variables of interest (logistic formulation)."""

    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)

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
        d.update(
            log_g_mean=ModelParameter.for_pop_mean("log_g", shape=(self.dimension,)),
            log_g_std=Hyperparameter(0.01),
            log_g=PopulationLatentVariable(Normal("log_g_mean", "log_g_std")),
            g=LinkedVariable(Exp("log_g")),
        )
        return d

    @staticmethod
    def metric(*, g: torch.Tensor) -> torch.Tensor:
        """
        Compute the metric tensor from input tensor `g`.
        This function calculates the metric as \((g + 1)^2 / g\) element-wise.

        Parameters
        ----------
        g : t :class:`torch.Tensor`
            Input tensor with values of the population parameter `g` for each feature.

        Returns
        -------
         :class:`torch.Tensor`
            The computed metric tensor, same shape as g(number of features)
        """
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
        """
        Return the model output when sources(spatial components) are present.

        Parameters
        ----------
        rt : :class:`~leaspy.uitls.weighted_tensor._weighted_tensor.TensorOrWeightedTensor`[:obj:`float`]
            Tensor containing the reparametrized time.
        space_shifts : `~leaspy.uitls.weighted_tensor._weighted_tensor.TensorOrWeightedTensor`[:obj:`float`]
            Tensor containing the values of the space-shifts
        metric :`~leaspy.uitls.weighted_tensor._weighted_tensor.TensorOrWeightedTensor`[:obj:`float`]
            Tensor containing the metric tensor used for computing the spatial/temporal influence.
        v0 : `~leaspy.uitls.weighted_tensor._weighted_tensor.TensorOrWeightedTensor`[:obj:`float`]
            Tensor containing the values of the population parameter `v0` for each feature.
        g : `~leaspy.uitls.weighted_tensor._weighted_tensor.TensorOrWeightedTensor`[:obj:`float`]
            Tensor containing the values of the population parameter `g` for each feature.

        Returns
        -------
         :class:`torch.Tensor`
            Weighted value tensor after applying sigmoid transformation,
            representing the model output with sources.
        """
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


