import warnings
from typing import Optional, Union

import torch

from leaspy.exceptions import LeaspyIndividualParamsInputError, LeaspyModelInputError
from leaspy.io.data.dataset import Dataset
from leaspy.utils.functional import Exp, MatMul
from leaspy.utils.typing import DictParams, DictParamsTorch, FeatureType, KwargsType
from leaspy.utils.weighted_tensor import TensorOrWeightedTensor
from leaspy.variables.distributions import Normal
from leaspy.variables.specs import (
    Hyperparameter,
    IndividualLatentVariable,
    LatentVariableInitType,
    LinkedVariable,
    ModelParameter,
    NamedVariables,
    PopulationLatentVariable,
)
from leaspy.variables.state import State

from .mcmc_saem_compatible import McmcSaemCompatibleModel
from .obs_models import observation_model_factory

__all__ = ["TimeReparametrizedModel"]


class TimeReparametrizedModel(McmcSaemCompatibleModel):
    """
    Contains the common attributes & methods of the multivariate time-reparametrized models.

    Parameters
    ----------
    name : :obj:`str`
        Name of the model.
    source_dimension : Optional[int], optional
        Number of sources. Dimension of spatial components (default is None).
    **kwargs
        Hyperparameters for the model (including `obs_models`).

    Raises
    ------
    :exc:`.LeaspyModelInputError`
        If inconsistent hyperparameters.
    """

    _xi_std = 0.5
    _tau_std = 5.0
    _noise_std = 0.1
    _sources_std = 1.0

    def __init__(
        self,
        name: str,
        source_dimension: Optional[int] = None,
        **kwargs,
    ):
        # TODO / WIP / TMP: dirty for now...
        # Should we:
        # - use factory of observation models instead? dataset -> ObservationModel
        # - or refact a bit `ObservationModel` structure? (lazy init of its variables...)
        # (cf. note in AbstractModel as well)
        dimension = kwargs.get("dimension", None)
        if "features" in kwargs:
            dimension = len(kwargs["features"])
        # source_dimension = kwargs.get("source_dimension", None)
        # if dimension == 1 and source_dimension not in {0, None}:
        #    raise LeaspyModelInputError(
        #        "You should not provide `source_dimension` != 0 for univariate model."
        #    )
        # self.source_dimension: Optional[int] = source_dimension
        observation_models = kwargs.get("obs_models", None)
        if observation_models is None:
            observation_models = (
                "gaussian-scalar" if dimension is None else "gaussian-diagonal"
            )
        if isinstance(observation_models, (list, tuple)):
            kwargs["obs_models"] = tuple(
                [
                    observation_model_factory(obs_model, **kwargs)
                    for obs_model in observation_models
                ]
            )
        elif isinstance(observation_models, dict):
            # Not really satisfied... Used for api load
            kwargs["obs_models"] = tuple(
                [
                    observation_model_factory(
                        observation_models["y"], dimension=dimension
                    )
                ]
            )
        else:
            kwargs["obs_models"] = (
                observation_model_factory(observation_models, dimension=dimension),
            )
        super().__init__(name, **kwargs)
        self._source_dimension = self._validate_source_dimension(source_dimension)

    @property
    def xi_std(self) -> torch.Tensor:
        """Return the standard deviation of xi as a tensor."""
        return torch.tensor([self._xi_std])

    @property
    def tau_std(self) -> torch.Tensor:
        """Return the standard deviation of tau as a tensor."""
        return torch.tensor([self._tau_std])

    @property
    def noise_std(self) -> torch.Tensor:
        """Return the standard deviation of the model as a tensor."""
        return torch.tensor(self._noise_std)

    @property
    def sources_std(self) -> float:
        """Return the standard deviation of sources as a float."""
        return self._sources_std

    @property
    def source_dimension(self) -> Optional[int]:
        """Return the number of the sources"""
        return self._source_dimension

    @source_dimension.setter
    def source_dimension(self, source_dimension: Optional[int] = None):
        """Set the dimensionality of the source space for the model."""
        self._source_dimension = self._validate_source_dimension(source_dimension)

    def _validate_source_dimension(self, source_dimension: Optional[int] = None) -> int:
        """
        Validate and sanitize the `source_dimension` parameter.

        Parameters
        ----------
        source_dimension : Optional[int], default=None
            The candidate source dimension to validate.

        Returns
        -------
        Optional[int]
            The validated source dimension value. Returns 0 if the model dimension is 1,
            otherwise returns the validated `source_dimension` or None if not provided.

        Raises
        ------
        LeaspyModelInputError
            If `source_dimension` is not an integer, is negative, or exceeds the allowable range
            based on the model's dimension.
        """
        if self.dimension == 1:
            return 0
        if source_dimension is not None:
            if not isinstance(source_dimension, int):
                raise LeaspyModelInputError(
                    f"`source_dimension` must be an integer, not {type(source_dimension)}"
                )
            if source_dimension < 0:
                raise LeaspyModelInputError(
                    f"`source_dimension` must be >= 0, you provided {source_dimension}"
                )
            if self.dimension is not None and source_dimension > self.dimension - 1:
                raise LeaspyModelInputError(
                    f"Source dimension should be within [0, {self.dimension - 1}], "
                    f"you provided {source_dimension}"
                )
        return source_dimension

    @property
    def has_sources(self) -> bool:
        """
        Indicates whether the model includes sources.

        Returns
        -------
        bool
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
        alpha : torch.Tensor
            Acceleration factors of individual(s)
        tau : torch.Tensor
            Time-shift(s) of individual(s)

        Returns
        -------
        torch.Tensor 
            Reparametrized time of same shape as `timepoints`
        """
        return alpha * (t - tau)

    def get_variables_specs(self) -> NamedVariables:
        """
        Return the specifications of the variables (latent variables,
        derived variables, model 'parameters') that are part of the model.

        Returns
        -------
        NamedVariables :
            A dictionary-like object containing specifications for the variables
        """
        specifications = super().get_variables_specs()
        specifications.update(
            rt=LinkedVariable(self.time_reparametrization),
            # PRIORS
            tau_mean=ModelParameter.for_ind_mean("tau", shape=(1,)),
            tau_std=ModelParameter.for_ind_std("tau", shape=(1,)),
            xi_std=ModelParameter.for_ind_std("xi", shape=(1,)),
            # LATENT VARS
            xi=IndividualLatentVariable(Normal("xi_mean", "xi_std")),
            tau=IndividualLatentVariable(Normal("tau_mean", "tau_std")),
            # DERIVED VARS
            alpha=LinkedVariable(Exp("xi")),
        )
        if self.source_dimension >= 1:
            specifications.update(
                # PRIORS
                betas_mean=ModelParameter.for_pop_mean(
                    "betas",
                    shape=(self.dimension - 1, self.source_dimension),
                ),
                betas_std=Hyperparameter(0.01),
                sources_mean=Hyperparameter(torch.zeros((self.source_dimension,))),
                sources_std=Hyperparameter(1.0),
                # LATENT VARS
                betas=PopulationLatentVariable(
                    Normal("betas_mean", "betas_std"),
                    sampling_kws={"scale": 0.5},  # cf. GibbsSampler (for retro-compat)
                ),
                sources=IndividualLatentVariable(Normal("sources_mean", "sources_std")),
                # DERIVED VARS
                mixing_matrix=LinkedVariable(
                    MatMul("orthonormal_basis", "betas").then(torch.t)
                ),  # shape: (Ns, Nfts)
                space_shifts=LinkedVariable(
                    MatMul("sources", "mixing_matrix")
                ),  # shape: (Ni, Nfts)
            )

        return specifications

    def _validate_compatibility_of_dataset(
        self, dataset: Optional[Dataset] = None
    ) -> None:
        """
        Validate the compatibility of the provided dataset with the model's configuration.

        Parameters
        ----------
        dataset : Optional[Dataset], optional
            The dataset to validate against, by default None.

        Raises
        ------
        LeaspyModelInputError
            If `source_dimension` is provided but not an integer in the valid range
            [0, dataset.dimension - 1).
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

    def _audit_individual_parameters(
        self, individual_parameters: DictParams
    ) -> KwargsType:
        """
        Validate and process individual parameter inputs for model compatibility.

        Parameters
        ----------
        individual_parameters : DictParams
            A dictionary mapping parameter names (strings) to their values,
            which can be scalars or array-like structures.

        Returns
        -------
        KwargsType
            A dictionary with the following keys:
            - "nb_inds": Number of individuals
            - "tensorized_ips": Dictionary of parameters converted to 2D tensors.
            - "tensorized_ips_gen": Generator yielding tensors for each individual,
            each with an added batch dimension.

        Raises
        ------
        LeaspyIndividualParamsInputError
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

    def _load_hyperparameters(self, hyperparameters: KwargsType) -> None:
        """
        Updates all model hyperparameters from the provided dictionary.

        Parameters
        ----------
        hyperparameters : KwargsType
            Dictionary containing the hyperparameters to be loaded.
            Expected keys include:
            - "features": List or sequence of feature names
            - "dimension": Integer specifying the number of features
            - "source_dimension": Integer specifying the number of sources; must be in
            [0, dimension - 1].

        Raises
        ------
        LeaspyModelInputError
            If `dimension` does not match the length of `features`, or if `source_dimension`
            is not an integer within the valid range [0, dimension - 1].
        """
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

    def put_individual_parameters(self, state: State, dataset: Dataset):
        """
        Initialize individual latent parameters in the given state if not already set.
        
        Parameters
        ----------
        state : State
            The current state object that holds all the variables
        dataset : Dataset
            Dataset used to initialize latent variables accordingly.
        """
        if not state.are_variables_set(("xi", "tau")):
            with state.auto_fork(None):
                state.put_individual_latent_variables(
                    LatentVariableInitType.PRIOR_SAMPLES,
                    n_individuals=dataset.n_individuals,
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
        KwargsType :
            The object as a dictionary.
        """
        model_settings = super().to_dict()
        model_settings["source_dimension"] = self.source_dimension

        if with_mixing_matrix and self.source_dimension >= 1:
            # transposed compared to previous version
            model_settings["parameters"]["mixing_matrix"] = self.state[
                "mixing_matrix"
            ].tolist()

        return model_settings

    # TODO: unit tests? (functional tests covered by api.estimate)
    def compute_individual_ages_from_biomarker_values(
        self,
        value: Union[float, list[float]],
        individual_parameters: DictParams,
        feature: Optional[FeatureType] = None,
    ) -> torch.Tensor:
        """
        For one individual, compute age(s) at which the given features values
        are reached (given the subject's individual parameters).

        Consistency checks are done in the main :term:`API` layer.

        Parameters
        ----------
        value : scalar or array_like[scalar] (:obj:`list`, :obj:`tuple`, :class:`numpy.ndarray`)
            Contains the :term:`biomarker` value(s) of the subject.

        individual_parameters : :obj:`dict`
            Contains the individual parameters.
            Each individual parameter should be a scalar or array_like.

        feature : :obj:`str` (or None)
            Name of the considered :term:`biomarker`.

            .. note::
                Optional for :class:`.UnivariateModel`, compulsory
                for :class:`.MultivariateModel`.

        Returns
        -------
        :class:`torch.Tensor`
            Contains the subject's ages computed at the given values(s).
            Shape of tensor is ``(1, n_values)``.

        Raises
        ------
        :exc:`.LeaspyModelInputError`
            If computation is tried on more than 1 individual.
        """
        # value, individual_parameters = self._get_tensorized_inputs(
        #     value, individual_parameters, skip_ips_checks=False
        # )
        # return self.compute_individual_ages_from_biomarker_values_tensorized(
        #     value, individual_parameters, feature
        # )
        raise NotImplementedError("This method is currently not implemented.")

    def compute_individual_ages_from_biomarker_values_tensorized(
        self,
        value: torch.Tensor,
        individual_parameters: DictParamsTorch,
        feature: Optional[FeatureType],
    ) -> torch.Tensor:
        """
        For one individual, compute age(s) at which the given features values are
        reached (given the subject's individual parameters), with tensorized inputs.

        Parameters
        ----------
        value : :class:`torch.Tensor` of shape ``(1, n_values)``
            Contains the :term:`biomarker` value(s) of the subject.

        individual_parameters : DictParamsTorch
            Contains the individual parameters.
            Each individual parameter should be a :class:`torch.Tensor`.

        feature : :obj:`str` (or None)
            Name of the considered :term:`biomarker`.

            .. note::
                Optional for :class:`.UnivariateModel`, compulsory
                for :class:`.MultivariateModel`.

        Returns
        -------
        :class:`torch.Tensor`
            Contains the subject's ages computed at the given values(s).
            Shape of tensor is ``(n_values, 1)``.
        """
        raise NotImplementedError("This method is currently not implemented.")
