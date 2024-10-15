from abc import abstractmethod
import math
import warnings

import torch

from leaspy.models.abstract_model import AbstractModel
from leaspy.models.utils.attributes import AttributesFactory
from leaspy.models.utils.attributes.abstract_manifold_model_attributes import AbstractManifoldModelAttributes
from leaspy.models.utils.initialization.model_initialization import initialize_parameters
from leaspy.models.utils.ordinal import OrdinalModelMixin
from leaspy.io.data.dataset import Dataset
from leaspy.io.realizations import CollectionRealization

from leaspy.utils.typing import KwargsType, Set, Optional, DictParamsTorch
from leaspy.utils.docs import doc_with_super
from leaspy.exceptions import LeaspyModelInputError


@doc_with_super()
class AbstractMultivariateModel(OrdinalModelMixin, AbstractModel):
    """
    Contains the common attributes & methods of the multivariate models.

    Parameters
    ----------
    name : :obj:`str`
        Name of the model.
    **kwargs
        Hyperparameters for the model (including `noise_model`).

    Raises
    ------
    :exc:`.LeaspyModelInputError`
        If inconsistent hyperparameters.
    """
    def __init__(self, name: str, **kwargs):

        self.source_dimension: int = None
        self.bayesian_priors = None
        self.attributes: AbstractManifoldModelAttributes = None

        # MCMC related "parameters"
        self.MCMC_toolbox = {
            'attributes': None,
            'priors': {
                # for logistic: "p0" = 1 / (1+exp(g)) i.e. exp(g) = 1/p0 - 1
                # for linear: "p0" = g
                'g_std': None,
                'betas_std': None
            }
        }

        if kwargs.get('noise_model', None) is None:
            kwargs['noise_model'] = "gaussian-diagonal"

        super().__init__(name, **kwargs)

        self.parameters = {
            "g": None,
            "betas": None,
            "tau_mean": None, "tau_std": None,
            "xi_mean": None, "xi_std": None,
            "sources_mean": None, "sources_std": None
        }

    def initialize(self, dataset: Dataset, method: str = 'default') -> None:
        """
        Overloads base initialization of model (base method takes care of features consistency checks).

        Parameters
        ----------
        dataset : :class:`.Dataset`
            Input :class:`.Dataset` from which to initialize the model.
        method : :obj:`str`, optional
            The initialization method to be used.
            Default='default'.
        """
        super().initialize(dataset, method=method)

        if self.source_dimension is None:
            self.source_dimension = int(math.sqrt(dataset.dimension))
            warnings.warn('You did not provide `source_dimension` hyperparameter for multivariate model, '
                          f'setting it to ⌊√dimension⌋ = {self.source_dimension}.')

        elif not (isinstance(self.source_dimension, int) and 0 <= self.source_dimension < self.dimension):
            raise LeaspyModelInputError(f"Sources dimension should be an integer in [0, dimension - 1[ "
                                        f"but you provided `source_dimension` = {self.source_dimension} whereas `dimension` = {self.dimension}")

        self.parameters, noise_model_parameters = initialize_parameters(self, dataset, method=method)
        if noise_model_parameters is not None:
            self.noise_model.update_parameters(validate=True, **noise_model_parameters)

        self.attributes = AttributesFactory.attributes(
            self.name,
            dimension=self.dimension,
            source_dimension=self.source_dimension,
            **self._attributes_factory_ordinal_kws
        )

        # Postpone the computation of attributes when really needed!
        #self.attributes.update({'all'}, self.parameters)

    @abstractmethod
    def initialize_MCMC_toolbox(self) -> None:
        """
        Initialize :term:`MCMC` toolbox for calibration of model.
        """
        # TODO to move in a "MCMC-model interface"

    @abstractmethod
    def update_MCMC_toolbox(self, vars_to_update: Set[str], realizations: CollectionRealization) -> None:
        """
        Update the :term:`MCMC` toolbox with a :class:`.CollectionRealization` of model population parameters.

        Parameters
        ----------
        vars_to_update : :obj:`set` of :obj:`str`
            Names of the population parameters to update in :term:`MCMC` toolbox.
        realizations : :class:`.CollectionRealization`
            All the realizations to update :term:`MCMC` toolbox with.
        """
        # TODO to move in a "MCMC-model interface"

    def load_parameters(self, parameters: KwargsType) -> None:
        """
        Updates all model parameters from the provided parameters.

        Parameters
        ----------
        parameters : KwargsType
            The parameters to be loaded.
        """
        self.parameters = {}
        for k, v in parameters.items():
            if k in ('mixing_matrix',):
                # The mixing matrix will always be recomputed from `betas`
                # and the other needed model parameters (g, v0)
                continue
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v)
            self.parameters[k] = v

        self._check_ordinal_parameters_consistency()

        # derive the model attributes from model parameters upon reloading of model
        self.attributes = AttributesFactory.attributes(
            self.name,
            dimension=self.dimension,
            source_dimension=self.source_dimension,
            **self._attributes_factory_ordinal_kws
        )
        self.attributes.update({'all'}, self.parameters)

    def load_hyperparameters(self, hyperparameters: KwargsType) -> None:
        """
        Updates all model hyperparameters from the provided hyperparameters.

        Parameters
        ----------
        hyperparameters : KwargsType
            The hyperparameters to be loaded.
        """
        expected_hyperparameters = ('features', 'dimension', 'source_dimension')

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
                and (not self.dimension or hyperparameters['source_dimension'] <= self.dimension - 1)
            ):
                raise LeaspyModelInputError(
                    f"Source dimension should be an integer in [0, dimension - 1], "
                    f"not {hyperparameters['source_dimension']}"
                )
            self.source_dimension = hyperparameters['source_dimension']

        # special hyperparameter(s) for ordinal model
        expected_hyperparameters += self._handle_ordinal_hyperparameters(hyperparameters)

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

        if with_mixing_matrix:
            model_settings['parameters']['mixing_matrix'] = self.attributes.mixing_matrix.tolist()

        self._export_extra_ordinal_settings(model_settings)

        return model_settings

    @abstractmethod
    def compute_individual_tensorized(
        self,
        timepoints: torch.Tensor,
        individual_parameters: DictParamsTorch,
        *,
        attribute_type=None,
    ) -> torch.Tensor:
        """
        Compute the individual trajectories.

        Parameters
        ----------
        timepoints : :class:`torch.Tensor`
            The time points for which to compute the trajectory.
        individual_parameters : DictParamsTorch
            The individual parameters to use.
        attribute_type : Any, optional

        Returns
        -------
        :class:`torch.Tensor` :
            Individual trajectories.
        """
        pass

    def compute_mean_traj(
        self,
        timepoints: torch.Tensor,
        *,
        attribute_type: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Compute trajectory of the model with individual parameters being the group-average ones.

        TODO check dimensions of io?

        Parameters
        ----------
        timepoints : :class:`torch.Tensor` of shape ``(1, n_timepoints)``
        attribute_type : :obj:`str` or None
            If a string, should be "MCMC".

        Returns
        -------
        :class:`torch.Tensor` of shape ``(1, n_timepoints, dimension)``
            The group-average values at given timepoints.
        """
        individual_parameters = {
            'xi': torch.tensor([self.parameters['xi_mean']]),
            'tau': torch.tensor([self.parameters['tau_mean']]),
            'sources': torch.zeros(self.source_dimension)
        }

        return self.compute_individual_tensorized(
            timepoints, individual_parameters, attribute_type=attribute_type
        )

    def _call_method_from_attributes(self, method_name: str, attribute_type: Optional[str], **call_kws):
        # TODO: mutualize with same function in univariate case...
        if attribute_type is None:
            return getattr(self.attributes, method_name)(**call_kws)
        elif attribute_type == 'MCMC':
            return getattr(self.MCMC_toolbox['attributes'], method_name)(**call_kws)
        else:
            raise LeaspyModelInputError(
                f"The specified attribute type does not exist: {attribute_type}. "
                "Should be None or 'MCMC'."
            )

    def _get_attributes(self, attribute_type: Optional[str]):
        return self._call_method_from_attributes('get_attributes', attribute_type)


@doc_with_super()
class AbstractMultivariateMixtureModel(OrdinalModelMixin, AbstractModel):
    """
    Contains the common attributes & methods of the multivariate models.

    Parameters
    ----------
    name : :obj:`str`
        Name of the model.
    **kwargs
        Hyperparameters for the model (including `noise_model`).

    Raises
    ------
    :exc:`.LeaspyModelInputError`
        If inconsistent hyperparameters.
    """
    def __init__(self, name: str, **kwargs):

        self.source_dimension: int = None
        self.dimension: int = None
        self.features = None
        self.bayesian_priors = None
        self.attributes: AbstractManifoldModelAttributes = None

        # MCMC related "parameters"
        self.MCMC_toolbox = {
            'attributes': None,
            'priors': {
                # for logistic: "p0" = 1 / (1+exp(g)) i.e. exp(g) = 1/p0 - 1
                # for linear: "p0" = g
                'g_std': None,
                'betas_std': None
            }
        }

        if kwargs.get('noise_model', None) is None:
            kwargs['noise_model'] = "gaussian-diagonal"

        super().__init__(name, **kwargs)

        self.parameters = {
            "g": None,
            "betas": None,
            "tau_mean": None, "tau_std": None,
            "xi_mean": None, "xi_std": None,
            "wi_mean": None, "wi_std": None
        }

    def initialize(self, dataset: Dataset, method: str = 'default') -> None:
        """
        Overloads base initialization of model (base method takes care of features consistency checks).

        Parameters
        ----------
        dataset : :class:`.Dataset`
            Input :class:`.Dataset` from which to initialize the model.
        method : :obj:`str`, optional
            The initialization method to be used.
            Default='default'.
        """
        super().initialize(dataset, method=method)

        if self.source_dimension is None:
            self.source_dimension = int(math.sqrt(dataset.dimension))
            warnings.warn('You did not provide `source_dimension` hyperparameter for multivariate model, '
                          f'setting it to ⌊√dimension⌋ = {self.source_dimension}.')

        elif not (isinstance(self.source_dimension, int) and 0 <= self.source_dimension < self.dimension):
            raise LeaspyModelInputError(f"Sources dimension should be an integer in [0, dimension - 1[ "
                                        f"but you provided `source_dimension` = {self.source_dimension} whereas `dimension` = {self.dimension}")

        self.parameters, noise_model_parameters = initialize_parameters(self, dataset, method=method)
        if noise_model_parameters is not None:
            self.noise_model.update_parameters(validate=True, **noise_model_parameters)

        ##Here add the attributes I want, wi and orthonormal basis??
        self.attributes = AttributesFactory.attributes(
            self.name,
            dimension=self.dimension,
            source_dimension=self.source_dimension,
            **self._attributes_factory_ordinal_kws
        )

        # Postpone the computation of attributes when really needed!
        #self.attributes.update({'all'}, self.parameters)

    @abstractmethod
    def initialize_MCMC_toolbox(self) -> None:
        """
        Initialize :term:`MCMC` toolbox for calibration of model.
        """
        # TODO to move in a "MCMC-model interface"

    @abstractmethod
    def update_MCMC_toolbox(self, vars_to_update: Set[str], realizations: CollectionRealization) -> None:
        """
        Update the :term:`MCMC` toolbox with a :class:`.CollectionRealization` of model population parameters.

        Parameters
        ----------
        vars_to_update : :obj:`set` of :obj:`str`
            Names of the population parameters to update in :term:`MCMC` toolbox.
        realizations : :class:`.CollectionRealization`
            All the realizations to update :term:`MCMC` toolbox with.
        """
        # TODO to move in a "MCMC-model interface"

    def load_parameters(self, parameters: KwargsType) -> None:
        """
        Updates all model parameters from the provided parameters.

        Parameters
        ----------
        parameters : KwargsType
            The parameters to be loaded.
        """
        self.parameters = {}
        ## Here add something about the orthonormal basis?
        for k, v in parameters.items():
            if k in ('mixing_matrix',):
                # The mixing matrix will always be recomputed from `betas`
                # and the other needed model parameters (g, v0)
                continue
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v)
            self.parameters[k] = v

        self._check_ordinal_parameters_consistency()

        # derive the model attributes from model parameters upon reloading of model
        ## Also here like the attributes before?
        self.attributes = AttributesFactory.attributes(
            self.name,
            dimension=self.dimension,
            source_dimension=self.source_dimension,
            **self._attributes_factory_ordinal_kws
        )
        self.attributes.update({'all'}, self.parameters)

    def load_hyperparameters(self, hyperparameters: KwargsType) -> None:
        """
        Updates all model hyperparameters from the provided hyperparameters.

        Parameters
        ----------
        hyperparameters : KwargsType
            The hyperparameters to be loaded.
        """
        expected_hyperparameters = ('features', 'dimension', 'source_dimension')

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
                and (not self.dimension or hyperparameters['source_dimension'] <= self.dimension - 1)
            ):
                raise LeaspyModelInputError(
                    f"Source dimension should be an integer in [0, dimension - 1], "
                    f"not {hyperparameters['source_dimension']}"
                )
            self.source_dimension = hyperparameters['source_dimension']

        # special hyperparameter(s) for ordinal model
        expected_hyperparameters += self._handle_ordinal_hyperparameters(hyperparameters)

        self._raise_if_unknown_hyperparameters(expected_hyperparameters, hyperparameters)

    def to_dict(self, *, with_mixing_matrix: bool = True, with_orthonormal_basis: bool = True) -> KwargsType:
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
        model_settings['dimension'] = self.dimension

        if with_mixing_matrix:
            model_settings['parameters']['mixing_matrix'] = self.attributes.mixing_matrix.tolist()

        if with_orthonormal_basis :
            model_settings['parameters']['orthonormal_basis'] = self.attributes.orthonormal_basis.tolist()

        self._export_extra_ordinal_settings(model_settings)

        return model_settings

    @abstractmethod
    def compute_individual_tensorized(
        self,
        timepoints: torch.Tensor,
        individual_parameters: DictParamsTorch,
        *,
        attribute_type=None,
    ) -> torch.Tensor:
        """
        Compute the individual trajectories.

        Parameters
        ----------
        timepoints : :class:`torch.Tensor`
            The time points for which to compute the trajectory.
        individual_parameters : DictParamsTorch
            The individual parameters to use.
        attribute_type : Any, optional

        Returns
        -------
        :class:`torch.Tensor` :
            Individual trajectories.
        """
        pass

    def compute_mean_traj(
        self,
        timepoints: torch.Tensor,
        *,
        attribute_type: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Compute trajectory of the model with individual parameters being the group-average ones.

        TODO check dimensions of io?

        Parameters
        ----------
        timepoints : :class:`torch.Tensor` of shape ``(1, n_timepoints)``
        attribute_type : :obj:`str` or None
            If a string, should be "MCMC".

        Returns
        -------
        :class:`torch.Tensor` of shape ``(1, n_timepoints, dimension)``
            The group-average values at given timepoints.
        """
        individual_parameters = {
            'xi': torch.tensor([self.parameters['xi_mean']]),
            'tau': torch.tensor([self.parameters['tau_mean']]),
            'wi': torch.tensor(self.parameters['wi_mean'])
        }

        return self.compute_individual_tensorized(
            timepoints, individual_parameters, attribute_type=attribute_type
        )

    def _call_method_from_attributes(self, method_name: str, attribute_type: Optional[str], **call_kws):
        # TODO: mutualize with same function in univariate case...
        if attribute_type is None:
            return getattr(self.attributes, method_name)(**call_kws)
        elif attribute_type == 'MCMC':
            return getattr(self.MCMC_toolbox['attributes'], method_name)(**call_kws)
        else:
            raise LeaspyModelInputError(
                f"The specified attribute type does not exist: {attribute_type}. "
                "Should be None or 'MCMC'."
            )

    def _get_attributes(self, attribute_type: Optional[str]):
        return self._call_method_from_attributes('get_attributes', attribute_type)
