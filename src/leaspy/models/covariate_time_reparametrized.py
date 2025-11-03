import warnings
from typing import Optional, Union

import torch

from leaspy.exceptions import LeaspyIndividualParamsInputError, LeaspyModelInputError
from leaspy.io.data.dataset import Dataset
from leaspy.utils.functional import (
    AffineFromVector,
    BatchMatMulByIndex,
    Exp,
    IndexOf,
    MatMul,
)
from leaspy.utils.typing import DictParams, DictParamsTorch, FeatureType, KwargsType
from leaspy.utils.weighted_tensor import TensorOrWeightedTensor, WeightedTensor
from leaspy.variables.distributions import BivariateNormalInd, Normal
from leaspy.variables.specs import (
    DataVariable,
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

__all__ = ["CovariateTimeReparametrizedModel"]


class CovariateTimeReparametrizedModel(McmcSaemCompatibleModel):
    """
    Contains the common attributes & methods of the multivariate models.

    Parameters
    ----------
    name : :obj:`str`
        Name of the model.
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
        return torch.tensor([self._xi_std])

    @property
    def tau_std(self) -> torch.Tensor:
        return torch.tensor([self._tau_std])

    @property
    def noise_std(self) -> torch.Tensor:
        return torch.tensor(self._noise_std)

    @property
    def sources_std(self) -> float:
        return self._sources_std

    @property
    def source_dimension(self) -> Optional[int]:
        return self._source_dimension

    @source_dimension.setter
    def source_dimension(self, source_dimension: Optional[int] = None):
        self._source_dimension = self._validate_source_dimension(source_dimension)

    def _validate_source_dimension(self, source_dimension: Optional[int] = None) -> int:
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
            Time-shift(s).

        Returns
        -------
        :class:`torch.Tensor` of same shape as `timepoints`
        """
        return alpha * (t - tau)

    def get_variables_specs(self) -> NamedVariables:
        """
        Return the specifications of the variables (latent variables,
        derived variables, model 'parameters') that are part of the model.

        Returns
        -------
        NamedVariables :
            The specifications of the model's variables.
        """
        specifications = super().get_variables_specs()
        specifications.update(
            rt=LinkedVariable(self.time_reparametrization),
            # INPUTS
            covariates=DataVariable(),
            # PRIORS
            phi_tau_mean=ModelParameter.for_ind_mean(
                ("phi_tau"), shape=(2,)
            ),  # (phi_mod_tau_mean, phi_ref_tau_mean)
            phi_tau_std=Hyperparameter((1, 10)),
            rho_tau=ModelParameter.for_ind_coeff_corr(("phi_tau"), shape=(1,)),
            xi_std=ModelParameter.for_ind_std("xi", shape=(1,)),
            # LATENT VARS
            xi=IndividualLatentVariable(Normal("xi_mean", "xi_std")),
            phi_tau=IndividualLatentVariable(
                BivariateNormalInd("phi_tau_mean", "phi_tau_std", "rho_tau")
            ),  # phi_tau = (phi_mod_tau, phi_ref_tau)
            # DERIVED VARS
            tau=LinkedVariable(AffineFromVector("phi_tau", "covariates")),
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
                index_cov=LinkedVariable(IndexOf("covariates", "unique_covariates")),
                mixing_matrix=LinkedVariable(
                    MatMul("orthonormal_basis", "betas").then(
                        lambda x: x.permute(0, 2, 1)
                    )
                ),  # shape: (Ns, Nfts)
                space_shifts=LinkedVariable(
                    BatchMatMulByIndex("sources", "mixing_matrix", "index_cov")
                ),  # shape: (Ni, Nfts)
            )

        return specifications

    def put_data_variables(self, state: State, dataset: Dataset) -> None:
        super().put_data_variables(state, dataset)
        covariates_tensor = dataset.covariates.clone().detach().to(torch.int)
        state["covariates"] = WeightedTensor(covariates_tensor)

    def _validate_compatibility_of_dataset(
        self, dataset: Optional[Dataset] = None
    ) -> None:
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
        Updates all model hyperparameters from the provided hyperparameters.

        Parameters
        ----------
        hyperparameters : KwargsType
            The hyperparameters to be loaded.
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
        if not state.are_variables_set(("xi", "tau")):
            with state.auto_fork(None):
                state.put_individual_latent_variables(
                    LatentVariableInitType.PRIOR_SAMPLES,
                    n_individuals=dataset.n_individuals,
                )

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
