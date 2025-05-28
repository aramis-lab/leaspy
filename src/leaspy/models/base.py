import warnings
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional

from leaspy.exceptions import LeaspyModelInputError
from leaspy.io.data.dataset import Dataset
from leaspy.utils.typing import FeatureType

__all__ = [
    "InitializationMethod",
    "BaseModel",
]


class InitializationMethod(str, Enum):
    DEFAULT = "default"
    RANDOM = "random"


class BaseModel(ABC):
    """
    Base model class from which all ``Leaspy`` models should inherit.

    It defines the interface that a model should implement to be
    compatible with ``Leaspy``.

    Parameters
    ----------
    name : :obj:`str`
        The name of the model.

    **kwargs
        Hyperparameters of the model

    Attributes
    ----------
    name : :obj:`str`
        The name of the model.

    is_initialized : :obj:`bool`
        ``True``if the model is initialized, ``False`` otherwise.

    features : :obj:`list` of :obj:`str`
        List of model features (``None`` if not initialization).

    dimension : :obj:`int`
        Number of features.
    """

    def __init__(self, name: str, **kwargs):
        self.is_initialized: bool = False
        self.name = name
        self._features: Optional[list[FeatureType]] = None
        self._dimension: Optional[int] = None

    @property
    def features(self) -> Optional[list[FeatureType]]:
        return self._features

    @features.setter
    def features(self, features: Optional[list[FeatureType]]):
        """
        Features setter.
        Ensure coherence between dimension and features attributes.
        """
        if features is None:
            # used to reset features
            self._features = None
            return

        if self.dimension is not None and len(features) != self.dimension:
            raise LeaspyModelInputError(
                f"Cannot set the model's features to {features}, because "
                f"the model has been configured with a dimension of {self.dimension}."
            )
        self._features = features

    @property
    def dimension(self) -> Optional[int]:
        """
        The dimension of the model.
        If the private attribute is defined, then it takes precedence over the feature length.
        The associated setters are responsible for their coherence.
        """
        if self._dimension is not None:
            return self._dimension
        if self.features is not None:
            return len(self.features)
        return None

    @dimension.setter
    def dimension(self, dimension: int):
        """
        Dimension setter.
        Ensures coherence between dimension and feature attributes.
        """
        if self.features is None:
            self._dimension = dimension
        elif len(self.features) != dimension:
            raise LeaspyModelInputError(
                f"Model has {len(self.features)} features. Cannot set the dimension to {dimension}."
            )

    def _validate_compatibility_of_dataset(
        self, dataset: Optional[Dataset] = None
    ) -> None:
        """
        Raise if the given :class:`.Dataset` is not compatible with the current model.

        Parameters
        ----------
        dataset : :class:`.Dataset`, optional
            The :class:`.Dataset` we want to model.

        Raises
        ------
        :exc:`.LeaspyModelInputError` :
            - If the :class:`.Dataset` has a number of dimensions smaller than 2.
            - If the :class:`.Dataset` does not have the same dimensionality as the model.
            - If the :class:`.Dataset`'s headers do not match the model's.
        """
        if not dataset:
            return
        if self.dimension is not None and dataset.dimension != self.dimension:
            raise LeaspyModelInputError(
                f"Unmatched dimensions: {self.dimension} (model) ≠ {dataset.dimension} (data)."
            )
        if self.features is not None and dataset.headers != self.features:
            raise LeaspyModelInputError(
                f"Unmatched features: {self.features} (model) ≠ {dataset.headers} (data)."
            )

    def initialize(
        self,
        dataset: Optional[Dataset] = None,
        method: Optional[InitializationMethod] = None,
    ) -> None:
        """
        Initialize the model given a :class:`.Dataset` and an initialization method.

        After calling this method :attr:`is_initialized` should be ``True`` and model
        should be ready for use.

        Parameters
        ----------
        dataset : :class:`.Dataset`, optional
            The dataset we want to initialize from.
        method : InitializationMethod, optional
            A custom method to initialize the model
        """
        method = InitializationMethod(method or InitializationMethod.DEFAULT)
        if self.is_initialized and self.features is not None:
            # we also test that self.features is not None, since for `ConstantModel`:
            # `is_initialized`` is True but as a mock for being personalization-ready,
            # without really being initialized!
            warn_msg = "<!> Re-initializing an already initialized model."
            if dataset and dataset.headers != self.features:
                warn_msg += (
                    f" Overwriting previous model features ({self.features}) "
                    f"with new ones ({dataset.headers})."
                )
                self.features = (
                    None  # wait validation of compatibility to store new features
                )
            warnings.warn(warn_msg)
        self._validate_compatibility_of_dataset(dataset)
        self.features = dataset.headers if dataset else None
        self.is_initialized = True

    @abstractmethod
    def save(self, path: str, **kwargs) -> None:
        """
        Save ``Leaspy`` object as json model parameter file.

        Parameters
        ----------
        path : :obj:`str`
            Path to store the model's parameters.

        **kwargs
            Additional parameters for writing.
        """
        raise NotImplementedError
