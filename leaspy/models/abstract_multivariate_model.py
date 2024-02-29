import warnings

import torch

from leaspy.models.obs_models import ObservationModelFactoryInput
from leaspy.models.abstract_model import AbstractModel

from leaspy.io.data.dataset import Dataset
from leaspy.variables.specs import (
    VarName,
    NamedVariables,
    ModelParameter,
    Hyperparameter,
    PopulationLatentVariable,
    IndividualLatentVariable,
    LinkedVariable,
)
from leaspy.variables.distributions import Normal
from leaspy.utils.functional import (
    Exp,
    MatMul,
)
from leaspy.utils.docs import doc_with_super
from leaspy.exceptions import LeaspyModelInputError

from leaspy.utils.typing import (
    FeatureType,
    KwargsType,
    Union,
    List,
    Iterable,
    Optional,
    Dict,
)


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
    _xi_std = .5
    _tau_std = 5.
    _noise_std = .1

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
        *,
        obs_models: Optional[Union[ObservationModelFactoryInput, Iterable[ObservationModelFactoryInput]]] = None,
        dimension: Optional[int] = None,
        source_dimension: Optional[int] = None,
        features: Optional[List[FeatureType]] = None,
        fit_metrics: Optional[Dict[str, float]] = None,
        variables_to_track: Optional[Iterable[VarName]] = None,
        **kwargs,
    ):
        super().__init__(
            name,
            obs_models=obs_models,
            dimension=dimension,
            features=features,
            fit_metrics=fit_metrics,
            variables_to_track=variables_to_track,
        )
        self._source_dimension: Optional[int] = None
        self.source_dimension = source_dimension
        self._log_g_std = kwargs.get("log_g_std", 0.01)
        self._betas_std = kwargs.get("betas_std", 0.01)
        self._sources_std = kwargs.get("sources_std", 1.0)

    #def _get_settable_hyperparameters(self) -> List[str]:
    #    settable_hyperparameters = super()._get_settable_hyperparameters()
    #    settable_hyperparameters.append("source_dimension")
    #    return settable_hyperparameters

    @property
    def source_dimension(self) -> Optional[int]:
        return self._source_dimension

    @source_dimension.setter
    def source_dimension(self, source_dimension: Optional[int] = None):
        if not self._is_source_dimension_valid(source_dimension):
            raise LeaspyModelInputError(
                f"Source dimension should be an integer in [0, dimension - 1], not {source_dimension}"
            )
        self._source_dimension = source_dimension

    def _is_source_dimension_valid(self, source_dimension: Optional[int] = None) -> bool:
        """source_dimension can either be None, or it should be an integer in [0, dimension -1]."""
        if source_dimension is None or (
            isinstance(source_dimension, int)
            and (source_dimension >= 0)
            and (self.dimension is None or source_dimension <= self.dimension - 1)
        ):
            return True
        return False

    def get_variables_specs(self) -> NamedVariables:
        """
        Return the specifications of the variables (latent variables,
        derived variables, model 'parameters') that are part of the model.

        Returns
        -------
        NamedVariables :
            The specifications of the model's variables.
        """
        variable_specs = super().get_variables_specs()
        variable_specs.update(
            # PRIORS
            log_g_mean=ModelParameter.for_pop_mean("log_g", shape=(self.dimension,)),
            log_g_std=Hyperparameter(self._log_g_std),
            tau_mean=ModelParameter.for_ind_mean("tau", shape=(1,)),
            tau_std=ModelParameter.for_ind_std("tau", shape=(1,)),
            xi_std=ModelParameter.for_ind_std("xi", shape=(1,)),
            # LATENT VARS
            log_g=PopulationLatentVariable(
                Normal("log_g_mean", "log_g_std")
            ),
            xi=IndividualLatentVariable(
                Normal("xi_mean", "xi_std")
            ),
            tau=IndividualLatentVariable(
                Normal("tau_mean", "tau_std")
            ),
            # DERIVED VARS
            g=LinkedVariable(Exp("log_g")),
            alpha=LinkedVariable(Exp("xi")),
        )
        if self.source_dimension >= 1:
            variable_specs.update(
                # PRIORS
                betas_mean=ModelParameter.for_pop_mean(
                    "betas",
                    shape=(self.dimension - 1, self.source_dimension),
                ),
                betas_std=Hyperparameter(self._betas_std),
                sources_mean=Hyperparameter(
                    torch.zeros((self.source_dimension,))
                ),
                sources_std=Hyperparameter(self._sources_std),
                # LATENT VARS
                betas=PopulationLatentVariable(
                    Normal("betas_mean", "betas_std"),
                    sampling_kws={"scale": .5},   # cf. GibbsSampler (for retro-compat)
                ),
                sources=IndividualLatentVariable(
                    Normal("sources_mean", "sources_std")
                ),
                # DERIVED VARS
                mixing_matrix=LinkedVariable(
                    MatMul("orthonormal_basis", "betas").then(torch.t)
                ),  # shape: (Ns, Nfts)
                space_shifts=LinkedVariable(
                    MatMul("sources", "mixing_matrix")
                ),  # shape: (Ni, Nfts)
            )

        return variable_specs

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

    #def load_parameters(self, parameters: KwargsType) -> None:
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

    def to_dict(self, *, with_mixing_matrix: bool = True, **kwargs) -> KwargsType:
        """
        Export ``Leaspy`` object as dictionary ready for :term:`JSON` saving.

        Parameters
        ----------
        with_mixing_matrix : :obj:`bool` (default ``True``)
            Save the :term:`mixing matrix` in the exported file in its 'parameters' section.

        Returns
        -------
        KwargsType :
            The object as a dictionary.
        """
        model_settings = super().to_dict(**kwargs)
        model_settings["source_dimension"] = self.source_dimension

        if with_mixing_matrix and self.source_dimension >= 1:
            # transposed compared to previous version
            model_settings["mixing_matrix"] = self.state['mixing_matrix'].tolist()

        # self._export_extra_ordinal_settings(model_settings)

        return model_settings
