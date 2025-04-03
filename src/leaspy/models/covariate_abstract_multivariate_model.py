import warnings
from typing import Optional

import pandas as pd
import torch

from leaspy.exceptions import LeaspyInputError, LeaspyModelInputError
from leaspy.io.data.dataset import Dataset
from leaspy.utils.docs import doc_with_super
from leaspy.utils.functional import Exp, MatMul, Sum
from leaspy.utils.typing import KwargsType
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

__all__ = ["CovariateAbstractMultivariateModel"]


@doc_with_super()
class CovariateAbstractMultivariateModel(AbstractModel):  # OrdinalModelMixin,
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

    def __init__(self, name: str, **kwargs):
        self.source_dimension: Optional[int] = None

        # TODO / WIP / TMP: dirty for now...
        # Should we:
        # - use factory of observation models instead? dataset -> ObservationModel
        # - or refact a bit `ObservationModel` structure? (lazy init of its variables...)
        # (cf. note in AbstractModel as well)
        dimension = kwargs.get("dimension", None)
        if "features" in kwargs:
            dimension = len(kwargs["features"])
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
            ## POPULATION PARAMETERS
            # PRIORS
            phi_mod_t0_mean=ModelParameter.for_pop_mean("phi_mod_t0", shape=(1,)),
            phi_mod_t0_std=Hyperparameter(0.001),
            phi_ref_t0_mean=ModelParameter.for_pop_mean("phi_ref_t0", shape=(1,)),
            phi_ref_t0_std=Hyperparameter(0.01),
            # LATENT VARS
            phi_mod_t0=PopulationLatentVariable(
                Normal("phi_mod_t0_mean", "phi_mod_t0_std")
            ),
            phi_ref_t0=PopulationLatentVariable(
                Normal("phi_ref_t0_mean", "phi_ref_t0_std")
            ),
            # LINKED VARS
            t0_mean=LinkedVariable(Sum("phi_mod_t0_mean", "phi_ref_t0_mean")),
            ## INDIVIDUAL PARAMETERS
            # PRIORS
            xi_std=ModelParameter.for_ind_std("xi", shape=(1,)),
            tau_std=ModelParameter.for_ind_std("tau", shape=(1,)),
            # LATENT VARS
            xi=IndividualLatentVariable(Normal("xi_mean", "xi_std")),
            tau=IndividualLatentVariable(
                Normal("t0_mean", "tau_std")
            ),  # t0_mean ou log_t0_mean ?
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

        return d

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

    # def load_parameters(self, parameters: KwargsType) -> None:
    #    """
    #    Updates all model parameters from the provided parameters.
    #
    #    Parameters
    #    ----------
    #    parameters : KwargsType
    #        The parameters to be loaded.
    #    """
    #    self.parameters = {}
    #    for k, v in parameters.items():
    #        if k in ('mixing_matrix',):
    #            # The mixing matrix will always be recomputed from `betas`
    #            # and the other needed model parameters (g, v0)
    #            continue
    #        if not isinstance(v, torch.Tensor):
    #            v = torch.tensor(v)
    #        self.parameters[k] = v
    #
    #    self._check_ordinal_parameters_consistency()

    def _load_hyperparameters(self, hyperparameters: KwargsType) -> None:
        """
        Updates all model hyperparameters from the provided hyperparameters.

        Parameters
        ----------
        hyperparameters : KwargsType
            The hyperparameters to be loaded.
        """
        expected_hyperparameters = ("features", "dimension", "source_dimension")

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

        # WIP
        ## special hyperparameter(s) for ordinal model
        # expected_hyperparameters += self._handle_ordinal_hyperparameters(hyperparameters)

        self._raise_if_unknown_hyperparameters(
            expected_hyperparameters, hyperparameters
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

        # self._export_extra_ordinal_settings(model_settings)

        return model_settings
