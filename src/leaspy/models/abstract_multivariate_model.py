import warnings
from typing import Optional

import torch

from leaspy.exceptions import LeaspyModelInputError
from leaspy.io.data.dataset import Dataset
from leaspy.utils.docs import doc_with_super
from leaspy.utils.functional import Exp, MatMul
from leaspy.utils.typing import KwargsType
from leaspy.utils.weighted_tensor import TensorOrWeightedTensor
from leaspy.variables.distributions import Normal
from leaspy.variables.specs import (
    Hyperparameter,
    IndividualLatentVariable,
    LinkedVariable,
    ModelParameter,
    NamedVariables,
    PopulationLatentVariable,
)

from .abstract_model import AbstractModel
from .obs_models import observation_model_factory

__all__ = ["AbstractMultivariateModel"]


@doc_with_super()
class AbstractMultivariateModel(AbstractModel):
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
