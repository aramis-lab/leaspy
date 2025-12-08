import warnings
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch

from leaspy.algo import AlgorithmName, AlgorithmSettings
from leaspy.exceptions import LeaspyModelInputError
from leaspy.io.data.dataset import Data, Dataset
from leaspy.io.outputs import IndividualParameters
from leaspy.io.outputs.result import Result
from leaspy.utils.typing import DictParamsTorch, FeatureType, IDType, KwargsType

__all__ = [
    "InitializationMethod",
    "BaseModel",
    "ModelInterface",
]


class InitializationMethod(str, Enum):
    """Possible initialization methods for Leaspy models.

    Attributes
    ----------
    DEFAULT : :obj:`str`
        Default initialization method.

    RANDOM : :obj:`str`
        Random initialization method.
    """

    DEFAULT = "default"
    RANDOM = "random"


class ModelInterface(ABC):
    """This is the public interface for Leaspy models.
    It defines the methods and properties that all models should implement.
    It is not meant to be instantiated directly, but rather to be inherited by concrete model classes.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """The name of the model.

        Returns
        -------
        :obj:`str`

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def is_initialized(self) -> bool:
        """True if the model is initialized, False otherwise.

        Returns
        -------
        :obj:`bool`

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Number of features.

        Returns
        -------
        :obj:`int`

        Raises
        ------
        NotImplementedError"""
        raise NotImplementedError

    @property
    @abstractmethod
    def features(self) -> list[FeatureType]:
        """List of model features (`None` if not initialization).

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def parameters(self) -> DictParamsTorch:
        """Dictionary of values for model parameters.

        Returns
        -------
        :class:`~leaspy.utils.typing.DictParamsTorch`

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def hyperparameters(self) -> DictParamsTorch:
        """Dictionary of values for model hyperparameters.

        Returns
        -------
        :class:`~leaspy.utils.typing.DictParamsTorch`

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, path: Union[str, Path], **kwargs) -> None:
        """Save model as json model parameter file.

        Parameters
        ----------
        path : :obj:`str` or :obj:`Path`
            The path to store the model's parameters.

        **kwargs : :obj:`dict`
            Additional parameters for writing.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, path_to_model_settings: Union[str, Path]):
        """Load a model from a json model parameter file.

        Parameters
        ----------
        path_to_model_settings : :obj:`str` or :obj:`Path`
            The path to the model's parameters file.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError

    @abstractmethod
    def fit(
        self,
        data: Optional[Union[pd.DataFrame, Data, Dataset]] = None,
        algorithm: Optional[Union[str, AlgorithmName]] = None,
        algorithm_settings: Optional[AlgorithmSettings] = None,
        algorithm_settings_path: Optional[Union[str, Path]] = None,
        **kwargs,
    ):
        r"""Estimate the model's parameters :math:`\theta` for a given dataset and a given algorithm.

        These model's parameters correspond to the fixed-effects of the mixed-effects model.

        There are three ways to provide parameters to the fitting algorithm:

        1. By providing an instance of :class:`~leaspy.algo.AlgorithmSettings`
        2. By providing a path to a serialized :class:`~leaspy.algo.AlgorithmSettings`
        3. By providing the algorithm name and parameters directly

        If settings are provided in multiple ways, the order above will prevail.

        Parameters
        ----------
        data : :obj:`pd.DataFrame` or :class:`~leaspy.io.Data` or :class:`~leaspy.io.Dataset`, optional
            Contains the information of the individuals, in particular the time-points
            :math:`(t_{i,j})` and the observations :math:`(y_{i,j})`.

        algorithm : :obj:`str` or :class:`~leaspy.algo.base.AlgorithmName`,optional
            The name of the algorithm to use.

            .. note::
                Use this if you want to provide algorithm settings through kwargs.

        algorithm_settings : :class:`~leaspy.algo.AlgorithmSettings`, optional
            The algorithm settings to use.

            .. note::
                Use this if you want to customize algorithm settings through the
                :class:`~leaspy.algo.AlgorithmSettings` class.
                If provided, the fit will rely on these settings.

        algorithm_settings_path : :obj:`str` or :obj:`Path`, optional
            The path to the algorithm settings file.

            .. note::
                If provided, the settings from the file will be used instead of the
                settings provided through kwarsg.

        **kwargs : :obj:`dict`
            Contains the algorithm's settings.

        Raises
        ------
        NotImplementedError

        Examples
        --------
        Fit a logistic model on a longitudinal dataset, display the group parameters

        >>> from leaspy.models import LogisticModel
        >>> from leaspy.datasets import load_dataset
        >>> putamen_df = load_dataset("parkinson-putamen")
        >>> model = LogisticModel(name="test-model-logistic")
        >>> model.fit(putamen_df, "mcmc_saem", seed=0, print_periodicity=50)
        >>> print(model)
        === MODEL ===
        betas_mean : []
        log_g_mean : [-0.8394]
        log_v0_mean : [-3.7930]
        noise_std : 0.021183
        tau_mean : [64.6920]
        tau_std : [10.0864]
        xi_std : [0.5232]
        """
        raise NotImplementedError

    @abstractmethod
    def personalize(
        self,
        data: Optional[Union[pd.DataFrame, Data, Dataset]] = None,
        algorithm: Optional[Union[str, AlgorithmName]] = None,
        algorithm_settings: Optional[AlgorithmSettings] = None,
        algorithm_settings_path: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> IndividualParameters:
        r"""Estimate individual parameters for each `ID` of a given dataset.

        These individual parameters correspond to the random-effects :math:`(z_{i,j})` of the mixed-effects model.

        Parameters
        ----------
        data : :obj:`pd.DataFrame` or :class:`~leaspy.io.Data` or :class:`~leaspy.io.Dataset`, optional
            Contains the information of the individuals, in particular the time-points
            :math:`(t_{i,j})` and the observations :math:`(y_{i,j})`.

        algorithm : :obj:`str` or :class:`~leaspy.algo.base.AlgorithmName`, optional
            The name of the algorithm to use.

        algorithm_settings : :class:`~leaspy.algo.AlgorithmSettings`, optional
            The algorithm settings to use.

            .. note::
                Use this if you want to customize algorithm settings through the
                :class:`~leaspy.algo.AlgorithmSettings` class.
                If provided, the fit will rely on these settings.

        algorithm_settings_path : :obj:`str` or :obj:`Path`, optional
            The path to the algorithm settings file.

            .. note::
                If provided, the settings from the file will be used instead of the settings provided.

        **kwargs : :obj:`dict`
            Contains the algorithm's settings.

        Returns
        -------
        ips : :class:`~leaspy.io.IndividualParameters`
            Individual parameters computed.

        Raises
        ------
        NotImplementedError

        Examples
        --------
        Compute the individual parameters for a given longitudinal dataset and calibrated model, then
        display the histogram of the log-acceleration:

        >>> from leaspy.datasets import load_model, load_dataset
        >>> model = load_model("parkinson-putamen")
        >>> putamen_df = load_dataset("parkinson-putamen")
        >>> individual_parameters = model.personalize(putamen_df, "scipy_minimize", seed=0)
        """
        raise NotImplementedError

    @abstractmethod
    def estimate(
        self,
        timepoints: Union[pd.MultiIndex, dict[IDType, list[float]]],
        individual_parameters: IndividualParameters,
        *,
        to_dataframe: Optional[bool] = None,
    ) -> Union[pd.DataFrame, dict[IDType, np.ndarray]]:
        r"""Return the model values for individuals characterized by their individual parameters :math:`z_i` at time-points :math:`(t_{i,j})_j`.

        Parameters
        ----------
        timepoints : :obj:`pd.MultiIndex` or :obj:`dict` [:obj:`IDType`, :obj:`list` [:obj:`float` ] ]
            Contains, for each individual, the time-points to estimate.
            It can be a unique time-point or a list of time-points.

        individual_parameters : :class:`~leaspy.io.IndividualParameters`
            Corresponds to the individual parameters of individuals.

        to_dataframe : :obj:`bool`, optional
            Whether to output a dataframe of estimations?
            If None: default is to be True if and only if timepoints is a `pandas.MultiIndex`

        Returns
        -------
        individual_trajectory : :obj:`pd.MultiIndex` or :obj:`dict` [:obj:`IDType`, :obj:`list` [:obj:`float`]]
            Key: patient indices.
            Value: :class:`numpy.ndarray` of the estimated value, in the shape (number of timepoints, number of features)

        Raises
        ------
        NotImplementedError

        Examples
        --------
        Given the individual parameters of two subjects, estimate the features of the first
        at 70, 74 and 80 years old and at 71 and 72 years old for the second.

        >>> from leaspy.datasets import load_model, load_individual_parameters, load_dataset
        >>> model = load_model("parkinson-putamen")
        >>> individual_parameters = load_individual_parameters("parkinson-putamen")
        >>> df_train = load_dataset("parkinson-putamen-train_and_test").xs("train", level="SPLIT")
        >>> timepoints = {'GS-001': (70, 74, 80), 'GS-002': (71, 72)}
        >>> estimations = model.estimate(timepoints, individual_parameters)
        """
        raise NotImplementedError

    @abstractmethod
    def simulate(
        self,
        individual_parameters: IndividualParameters,
        data: Optional[Union[pd.DataFrame, Data, Dataset]] = None,
        **kwargs,
    ):
        """Run the simulation pipeline using a leaspy model.
         This method simulates longitudinal data using the given leaspy model.
         It performs the following steps:
         - Retrieves individual parameters (IP) from fixed effects of the model.
         - Loads the specified Leaspy model.
         - Generates visit ages (timepoints) for each individual (based on specifications
           in visits_type).
         - Simulates observations at those visit ages.
         - Packages the result into a `Result` object, including simulated data,
           individual parameters, and the model's noise standard deviation.

         Parameters
         ----------
         individual_parameters : :class:`~leaspy.io.IndividualParameters`
             Individual parameters to use for the simulation.

         data : :obj:`pd.DataFrame` or :class:`~leaspy.io.Data` or :class:`~leaspy.io.Dataset`
             Data object. If None, returns empty Result.

         **kwargs : :obj:`dict`
         raise NotImplementedError
        `  Additional arguments for algorithm settings.

         Returns
         -------
         simulated_data : :class:`~leaspy.io.outputs.result.Result`
             Contains the generated individual parameters & the corresponding generated scores.
             Returns empty Result if any required input is None.

         Raises
         ------
         NotImplementedError
        """
        raise NotImplementedError


class BaseModel(ModelInterface):
    """Base model class from which all ``Leaspy`` models should inherit.

    It implements the :class:`~leaspy.models.ModelInterface`.
    """

    def __init__(self, name: str, **kwargs):
        self._is_initialized: bool = False
        self._name = name
        user_provided_dimension, user_provided_features = (
            self._validate_user_provided_dimension_and_features_at_init(**kwargs)
        )
        self._features: Optional[list[FeatureType]] = user_provided_features
        self._dimension: Optional[int] = user_provided_dimension
        self.initialization_method: InitializationMethod = InitializationMethod.DEFAULT
        if "initialization_method" in kwargs:
            self.initialization_method = InitializationMethod(
                kwargs["initialization_method"]
            )

    @property
    def name(self) -> str:
        """The name of the model.

        Returns
        -------
        :obj:`str`
            The name of the model.
        """

        return self._name

    @property
    def is_initialized(self) -> bool:
        """True if the model is initialized, False otherwise.

        Returns
        -------
        :obj:`bool`
            True if the model is initialized, False otherwise.
        """
        return self._is_initialized

    def _validate_user_provided_dimension_and_features_at_init(
        self,
        **kwargs,
    ) -> tuple[Optional[int], Optional[list[FeatureType]]]:
        """Validate user provided dimension and features at model initialization.

        Parameters
        ----------
        **kwargs : :obj:`dict`
            Keyword arguments that may contain 'features' and 'dimension'.

        Returns
        -------
        :obj:`tuple` [:obj:`int`, :obj:`list` [:obj:`FeatureType`]]
            A tuple containing the validated dimension and features.

        Raises
        ------
        :exc:`.LeaspyModelInputError`
            If the provided dimension is not an integer or if the features are not sizeable.
            If the provided dimension does not match the number of features.
        """
        from collections.abc import Sized

        user_provided_features = kwargs.pop("features", None)
        user_provided_dimension = kwargs.pop("dimension", None)
        if user_provided_dimension is not None and not isinstance(
            user_provided_dimension, int
        ):
            raise LeaspyModelInputError(
                f"{self.__class__.__name__} model '{self.name}' cannot be instantiated with "
                f"dimension = {user_provided_dimension}, of type {type(user_provided_dimension)}. "
                "The number of dimension must be an integer."
            )
        if user_provided_features is not None and not isinstance(
            user_provided_features, Sized
        ):
            raise LeaspyModelInputError(
                f"{self.__class__.__name__} model '{self.name}' cannot be instantiated with "
                f"features = {user_provided_features}. The model's features must be a sizeable object."
            )
        if (
            user_provided_features is not None
            and user_provided_dimension is not None
            and user_provided_dimension != len(user_provided_features)
        ):
            raise LeaspyModelInputError(
                f"{self.__class__.__name__} model '{self.name}' cannot be instantiated with "
                f"dimension = {user_provided_dimension} and features = {user_provided_features}. "
                "The model dimension must match the number of features."
            )
        return user_provided_dimension, user_provided_features

    @property
    def features(self) -> Optional[list[FeatureType]]:
        """List of model features (`None` if not initialization).

        Returns
        -------
        : :obj:`list` [:obj:`FeatureType`], optional
            The features of the model, or None if not initialized.
        """
        return self._features

    @features.setter
    def features(self, features: Optional[list[FeatureType]]):
        """Set the model's features.

        Parameters
        ----------
        features : :obj:`list`[:obj:`FeatureType`], optional
            The features to set for the model. If None, it resets the features.

        Raises
        ------
        :exc:`.LeaspyModelInputError`
            If the provided features do not match the model's dimension.
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
        """Number of features.

        Returns
        -------
        :obj:`int`, optional
            The dimension of the model, or None if not initialized.
        """
        if self._dimension is not None:
            return self._dimension
        if self.features is not None:
            return len(self.features)
        return None

    @dimension.setter
    def dimension(self, dimension: int):
        """Set the model's dimension.

        Parameters
        ----------
        dimension : :obj:`int`
            The dimension to set for the model.

        Raises
        ------
        :exc:`.LeaspyModelInputError`
            If the model has already been initialized with features that do not match the new dimension.
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
        """Raise if the given Dataset is not compatible with the current model.

        Parameters
        ----------
        dataset : :class:`~leaspy.io.data.dataset.Dataset`, optional
            The class we want to model.

        Raises
        ------
        :exc:`.LeaspyModelInputError` :
            - If the Dataset has a number of dimensions smaller than 2.
            - If the Dataset does not have the same dimensionality as the model.
            - If the Dataset's headers do not match the model's.
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

    def initialize(self, dataset: Optional[Dataset] = None) -> None:
        """Initialize the model given a :class:`.Dataset` and an initialization method.

        After calling this method :attr:`is_initialized` should be ``True`` and model
        should be ready for use.

        Parameters
        ----------
        dataset : :class:`~leaspy.io.data.dataset.Dataset`, optional
            The dataset we want to initialize from.
        """
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
                # wait validation of compatibility to store new features
                self.features = None
            warnings.warn(warn_msg)
        self._validate_compatibility_of_dataset(dataset)
        self.features = dataset.headers if dataset else None
        self._is_initialized = True

    def save(self, path: Union[str, Path], **kwargs) -> None:
        """Save model as json model parameter file.

        Parameters
        ----------
        path : :obj:`str` or :obj:`Path`
            The path to store the model's parameters.

        **kwargs : :obj:`dict`
            Additional parameters for writing.

        Raises
        ------
        :exc:`.LeaspyModelInputError`
            If the model is not initialized.
        """
        import json
        from inspect import signature

        export_kws = {
            k: kwargs.pop(k) for k in signature(self.to_dict).parameters if k in kwargs
        }
        model_settings = self.to_dict(**export_kws)
        kwargs = {"indent": 2, **kwargs}
        with open(path, "w") as fp:
            json.dump(model_settings, fp, **kwargs)

    def to_dict(self, **kwargs) -> KwargsType:
        """Export model as a dictionary ready for export.

        Returns
        -------
        :obj:`KwargsType`
            The model instance serialized as a dictionary.
        """
        from leaspy import __version__

        from .utilities import tensor_to_list

        return {
            "leaspy_version": __version__,
            "name": self.name,
            "features": self.features,
            "dimension": self.dimension,
            "hyperparameters": {
                k: tensor_to_list(v) for k, v in (self.hyperparameters or {}).items()
            },
            "parameters": {
                k: tensor_to_list(v) for k, v in (self.parameters or {}).items()
            },
        }

    @classmethod
    def load(cls, path_to_model_settings: Union[str, Path]):
        """Load a model from a json model parameter file.

        Parameters
        ----------
        path_to_model_settings : :obj:`str` or :obj:`Path`
            The path to the model's parameters file.

        Returns
        -------
        :class:`~leasp.models.base.BaseModel`
            An instance of the model loaded from the file.

        Raises
        ------
        :exc:`.LeaspyModelInputError`
            If the model settings file is not found or cannot be read.
        """

        from .factory import model_factory
        from .settings import ModelSettings

        reader = ModelSettings(path_to_model_settings)
        instance = model_factory(reader.name, **reader.hyperparameters)
        instance.load_parameters(reader.parameters)
        instance._is_initialized = True
        return instance

    @abstractmethod
    def load_parameters(self, parameters: KwargsType) -> None:
        """Load model parameters from a dictionary.

        Parameters
        ----------
        parameters : :obj:`KwargsType`
            The parameters to load into the model.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError

    def fit(
        self,
        data: Optional[Union[pd.DataFrame, Data, Dataset]] = None,
        algorithm: Optional[Union[str, AlgorithmName]] = None,
        algorithm_settings: Optional[AlgorithmSettings] = None,
        algorithm_settings_path: Optional[Union[str, Path]] = None,
        **kwargs,
    ):
        """Estimate the model's parameters for a given dataset and a given algorithm.
        These model's parameters correspond to the fixed-effects of the mixed-effects model.

        Parameters
        ----------
        data : :obj:`pd.DataFrame` or :class:`~leaspy.io.Data` or :class:`~leaspy.io.Dataset`, optional
            Contains the information of the individuals, in particular the time-points
            :math:`(t_{i,j})` and the observations :math:`(y_{i,j})`.
        algorithm : :obj:`str` or :class:`~leaspy.algo.base.AlgorithmName`, optional
            The name of the algorithm to use.
            Use this if you want to provide algorithm settings through kwargs.
        algorithm_settings : :class:`~leaspy.algo.AlgorithmSettings`, optional
            The algorithm settings to use.
            Use this if you want to customize algorithm settings through the
            :class:`~leaspy.algo.AlgorithmSettings` class.
            If provided, the fit will rely on these settings.
        algorithm_settings_path : :obj:`str` or :obj:`Path`, optional
            The path to the algorithm settings file.
            If provided, the settings from the file will be used instead of the settings provided.
        **kwargs : :obj:`dict`
            Contains the algorithm's settings.
        """
        if (dataset := BaseModel._get_dataset(data)) is None:
            return
        if not self.is_initialized:
            self.initialize(dataset)
        if (
            algorithm := BaseModel._get_algorithm(
                algorithm, algorithm_settings, algorithm_settings_path, **kwargs
            )
        ) is None:
            return
        algorithm.run(self, dataset)
        
    def summary(self) -> None:
        """Print a formatted summary of the model parameters.

        This method prints a statistical summary of the model. It displays the model's
        population parameters (fixed effects), individual parameter
        distributions (random effects), and noise estimates.

        Examples
        --------
        >>> from leaspy.models import LogisticModel
        >>> model = LogisticModel(name="test")
        >>> model.fit(data, "mcmc_saem", n_iter=100)
        >>> model.summary()
        """
        if not self.is_initialized:
            print("Model is not initialized. Call fit() first.")
            return

        if self.parameters is None or len(self.parameters) == 0:
            print("Model has no parameters. Call fit() first.")
            return

        lines = []
        sep = "=" * 60

        # Header
        lines.append(sep)
        lines.append(f"{'Model Summary':^60}")
        lines.append(sep)
        lines.append(f"Model Name: {self.name}")
        lines.append(f"Model Type: {self.__class__.__name__}")
        if self.features is not None:
            lines.append(f"Features ({self.dimension}): {', '.join(self.features)}")
        if (fm := getattr(self, "fit_metrics", None)) and (nll := fm.get("nll_tot")):
            lines.append(f"Neg. Log-Likelihood: {nll:.4f}")
        lines.append(sep)

        # Get parameter categories (from model or fallback)
        if hasattr(self, "_param_categories"):
            cats = self._param_categories
            pop_params = cats.get("population", [])
            ind_params = cats.get("individual_priors", [])
            noise_params = cats.get("noise", [])
        else:
            # Fallback: all params as population
            pop_params = sorted(self.parameters.keys())
            ind_params = []
            noise_params = []

        # Population Parameters (Fixed Effects)
        if pop_params:
            lines.append("")
            lines.append("Population Parameters")
            lines.append("-" * 60)
            lines.extend(self._format_parameter_group(pop_params))

        # Individual Parameters (Random Effects)
        if ind_params:
            lines.append("")
            lines.append("Individual Parameters")
            lines.append("-" * 60)
            lines.extend(self._format_parameter_group(ind_params))

        # Noise Model
        if noise_params:
            lines.append("")
            lines.append("Noise Model")
            lines.append("-" * 60)
            lines.extend(self._format_parameter_group(noise_params))

        lines.append(sep)

        print("\n".join(lines))

    def _format_parameter_group(self, param_names: list[str]) -> list[str]:
        """Format a group of parameters, consolidating 1D parameters with the same axis.

        This method groups consecutive 1D parameters that share the same axis
        to avoid repeating column headers.

        Parameters
        ----------
        param_names : :obj:`list` [:obj:`str`]
            List of parameter names to format.

        Returns
        -------
        :obj:`list` [:obj:`str`]
            List of formatted lines.
        """
        lines = []
        axes_map = getattr(self, "_param_axes", {})

        # Group consecutive 1D parameters with the same axis
        i = 0
        while i < len(param_names):
            name = param_names[i]
            value = self.parameters[name]
            axes = axes_map.get(name, ())

            # Check if this is a 1D parameter with axis labels
            if value.ndim == 1 and len(axes) >= 1:
                axis_name = axes[0]
                n = len(value)

                # Collect consecutive 1D params with the same axis and size
                group = [(name, value)]
                j = i + 1
                while j < len(param_names):
                    next_name = param_names[j]
                    next_value = self.parameters[next_name]
                    next_axes = axes_map.get(next_name, ())
                    if (next_value.ndim == 1 and
                        len(next_axes) >= 1 and
                        next_axes[0] == axis_name and
                        len(next_value) == n):
                        group.append((next_name, next_value))
                        j += 1
                    else:
                        break

                # Format the group with a single header
                if len(group) > 1:
                    col_labels = self._get_axis_labels(axis_name, n)
                    if col_labels:
                        header = " " * 20 + "  ".join(f"{lbl:>8}" for lbl in col_labels)
                        lines.append(header)
                        for gname, gvalue in group:
                            row = f"  {gname:<18}" + "  ".join(
                                f"{v.item():>8.4f}" for v in gvalue
                            )
                            lines.append(row)
                        i = j
                        continue

            # Default: format individually
            lines.append(self._format_parameter(name, value))
            i += 1

        return lines

    def _format_parameter(self, name: str, value: torch.Tensor) -> str:
        """Format a single parameter for display in the summary.

        Uses axis metadata from `_param_axes` (if defined) to create
        properly labeled tables for multi-dimensional parameters.

        Parameters
        ----------
        name : :obj:`str`
            The name of the parameter.
        value : :class:`torch.Tensor`
            The tensor value of the parameter.

        Returns
        -------
        :obj:`str`
            A formatted string representation of the parameter.
        """
        # Get axis labels if available
        axes = getattr(self, "_param_axes", {}).get(name, ())

        if value.ndim == 0:
            # Scalar
            val_str = f"{value.item():.4f}"
            return f"  {name:<18} {val_str}"

        elif value.ndim == 1:
            # 1D tensor - format as table with column headers
            n = len(value)
            if n > 10:
                return f"  {name:<18} Tensor of shape ({n},)"

            axis_name = axes[0] if len(axes) >= 1 else None
            col_labels = self._get_axis_labels(axis_name, n)

            if col_labels:
                # Create table format for 1D with labeled columns
                header = " " * 20 + "  ".join(f"{lbl:>8}" for lbl in col_labels)
                values = f"  {name:<18}" + "  ".join(
                    f"{v.item():>8.4f}" for v in value
                )
                return header + "\n" + values
            else:
                # Simple list format
                val_str = "[" + ", ".join(f"{v.item():.4f}" for v in value) + "]"
                return f"  {name:<18} {val_str}"

        elif value.ndim == 2:
            rows, cols = value.shape
            if rows > 8 or cols > 8:
                return f"  {name:<18} Tensor of shape {tuple(value.shape)}"

            row_axis = axes[0] if len(axes) >= 1 else None
            col_axis = axes[1] if len(axes) >= 2 else None

            row_labels = self._get_axis_labels(row_axis, rows)
            col_labels = self._get_axis_labels(col_axis, cols)

            lines = [f"  {name}:"]

            # Column headers
            if col_labels:
                header = " " * 12 + "  ".join(f"{lbl:>8}" for lbl in col_labels)
                lines.append(header)

            # Data rows
            for i, row in enumerate(value):
                row_lbl = row_labels[i] if row_labels else f"[{i}]"
                row_str = f"    {row_lbl:<8}" + "  ".join(f"{v.item():>8.4f}" for v in row)
                lines.append(row_str)

            return "\n".join(lines)

        else:
            return f"  {name:<18} Tensor of shape {tuple(value.shape)}"

    def _get_axis_labels(self, axis_name: Optional[str], size: int) -> Optional[list[str]]:
        """Get human-readable labels for an axis dimension.

        Parameters
        ----------
        axis_name : :obj:`str` or None
            The semantic name of the axis (e.g., 'feature', 'source', 'cluster').
        size : :obj:`int`
            The size of the axis.

        Returns
        -------
        :obj:`list` [:obj:`str`] or None
            A list of labels, or None if no meaningful labels available.
        """
        if axis_name is None:
            return None

        if axis_name == "feature":
            if hasattr(self, "features") and self.features is not None:
                # Use actual feature names (truncated), respecting the actual size
                feats = self.features[:size]  # In case size < len(features)
                return [f[:8] if len(f) <= 8 else f[:7] + "." for f in feats]
            return [f"f{i}" for i in range(size)]
        elif axis_name == "source":
            return [f"s{i}" for i in range(size)]
        elif axis_name == "cluster":
            return [f"c{i}" for i in range(size)]
        elif axis_name == "event":
            return [f"e{i}" for i in range(size)]
        elif axis_name == "basis":
            # For basis vectors (e.g., in betas_mean), use generic indices
            return [f"b{i}" for i in range(size)]
        else:
            return [f"{axis_name[:1]}{i}" for i in range(size)]

    @staticmethod
    def _get_dataset(
        data: Optional[Union[pd.DataFrame, Data, Dataset]] = None,
    ) -> Optional[Dataset]:
        """Process user provided data and return a Dataset object.

        Parameters
        ----------
        data : :obj:`pd.DataFrame` or :class:`~leaspy.io.Data` or :class:`~leaspy.io.Dataset`, optional
            The data to process. If None, returns None.

        Returns
        -------
        :class:`~leaspy.io.data.dataset.Dataset`, optional
            A Dataset object if data is provided and valid, otherwise None.
        """
        if data is None:
            return None
        if isinstance(data, pd.DataFrame):
            data = Data.from_dataframe(data)
        return Dataset(data) if isinstance(data, Data) else data

    @staticmethod
    def _get_algorithm(
        algorithm: Optional[Union[str, AlgorithmName]] = None,
        algorithm_settings: Optional[AlgorithmSettings] = None,
        algorithm_settings_path: Optional[Union[str, Path]] = None,
        **kwargs,
    ):
        """Process user provided algorithm and return the corresponding algorithm instance.

        Parameters
        ----------
        algorithm : :obj:`str` or :class:`~leaspy.algo.base.AlgorithmName`, optional
            The name of the algorithm to use. If None, returns None.
        algorithm_settings : :class:`~leaspy.algo.AlgorithmSettings`, optional
            The algorithm settings to use. If None, returns None.
        algorithm_settings_path : :obj:`str` or :obj:`Path`, optional
            The path to the algorithm settings file. If None, returns None.
        **kwargs : :obj:`dict`
            Additional parameters for the algorithm settings.

        Returns
        -------
        :class:`~leaspy.algo.base.AlgorithmInterface`, optional
            An instance of the algorithm if provided, otherwise None."""
        from leaspy.algo import AlgorithmName, AlgorithmSettings, algorithm_factory

        if algorithm_settings is not None:
            settings = algorithm_settings
        elif algorithm_settings_path is not None:
            settings = AlgorithmSettings.load(algorithm_settings_path)
        else:
            algorithm = AlgorithmName(algorithm) if algorithm else None
            if algorithm is None:
                return None
            settings = AlgorithmSettings(algorithm.value, **kwargs)
            settings.set_logs(**kwargs)
        return algorithm_factory(settings)

    def personalize(
        self,
        data: Optional[Union[pd.DataFrame, Data, Dataset]] = None,
        algorithm: Optional[Union[str, AlgorithmName]] = None,
        algorithm_settings: Optional[AlgorithmSettings] = None,
        algorithm_settings_path: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> IndividualParameters:
        """Estimate individual parameters for each `ID` of a given dataset.
        These individual parameters correspond to the random-effects :math:`(z_{i,j})` of the mixed-effects model.

        Parameters
        ----------
        data : :obj:`pd.DataFrame` or :class:`~leaspy.io.Data` or :class:`~leaspy.io.Dataset`, optional
            Contains the information of the individuals, in particular the time-points
            :math:`(t_{i,j})` and the observations :math:`(y_{i,j})`.
        algorithm : :obj:`str` or :class:`~leaspy.algo.base.AlgorithmName`, optional
            The name of the algorithm to use.
        algorithm_settings : :class:`~leaspy.algo.AlgorithmSettings`, optional
            The algorithm settings to use.
            Use this if you want to customize algorithm settings through the
            :class:`~leaspy.algo.AlgorithmSettings` class.
            If provided, the fit will rely on these settings.
        algorithm_settings_path : :obj:`str` or :obj:`Path`, optional
            The path to the algorithm settings file.
            If provided, the settings from the file will be used instead of the settings provided.
        **kwargs : :obj:`dict`
            Contains the algorithm's settings.

        Returns
        -------
        :class:`~leaspy.io.outputs.IndividualParameters`
            Individual parameters computed.
        """

        from leaspy.exceptions import LeaspyInputError

        if not self.is_initialized:
            raise LeaspyInputError("Model has not been initialized")
        if (dataset := BaseModel._get_dataset(data)) is None:
            return IndividualParameters()
        if (
            algorithm := BaseModel._get_algorithm(
                algorithm, algorithm_settings, algorithm_settings_path, **kwargs
            )
        ) is None:
            return IndividualParameters()
        return algorithm.run(self, dataset)

    def estimate(
        self,
        timepoints: Union[pd.MultiIndex, dict[IDType, list[float]]],
        individual_parameters: IndividualParameters,
        *,
        to_dataframe: Optional[bool] = None,
    ) -> Union[pd.DataFrame, dict[IDType, np.ndarray]]:
        """Return the model values for individuals characterized by their individual parameters :math:`z_i` at time-points :math:`(t_{i,j})_j`.

        Parameters
        ----------
        timepoints : :obj:`pd.MultiIndex` or :obj:`dict` [:obj:`IDType`, :obj:`list` [:obj:`float`]]
            Contains, for each individual, the time-points to estimate.
            It can be a unique time-point or a list of time-points.

        individual_parameters : :class:`~leaspy.io.IndividualParameters`
            Corresponds to the individual parameters of individuals.

        to_dataframe : :obj:`bool`, optional
            Whether to output a dataframe of estimations?
            If None: default is to be True if and only if timepoints is a `pandas.MultiIndex`

        Returns
        -------
        individual_trajectory : :obj:`pd.DataFrame` or :obj:`dict` [:obj:`IDType`, :obj:`np.ndarray`]
            Key: patient indices.
            Value: :class:`numpy.ndarray` of the estimated value, in the shape (number of timepoints, number of features)
        """

        estimations = {}
        ix = None
        # get timepoints to estimate from index
        if isinstance(timepoints, pd.MultiIndex):
            # default output is pd.DataFrame when input as pd.MultiIndex
            if to_dataframe is None:
                to_dataframe = True
            ix = timepoints
            timepoints = {
                subj_id: tpts.values
                for subj_id, tpts in timepoints.to_frame()["TIME"].groupby("ID")
            }
        for subj_id, tpts in timepoints.items():
            ip = individual_parameters[subj_id]
            est = self.compute_individual_trajectory(tpts, ip).cpu().numpy()
            # 1 individual at a time --> squeeze the first dimension of the array
            estimations[subj_id] = est[0]

        # convert to proper dataframe
        if to_dataframe:
            estimations = pd.concat(
                {
                    subj_id: pd.DataFrame(  # columns names may be directly embedded in the dictionary after a `postprocess_model_estimation`
                        ests,
                        columns=None if isinstance(ests, dict) else self.features,
                        index=timepoints[subj_id],
                    )
                    for subj_id, ests in estimations.items()
                },
                names=["ID", "TIME"],
            )
            # reindex back to given index being careful to index order (join so to handle multi-levels cases)
            if ix is not None:
                # we need to explicitly pass `on` to preserve order of index levels
                # and to explicitly pass columns to preserve 2D columns when they are
                empty_df_like_ests = pd.DataFrame(
                    [], index=ix, columns=estimations.columns
                )
                estimations = empty_df_like_ests[[]].join(
                    estimations, on=["ID", "TIME"]
                )

        return estimations

    @abstractmethod
    def compute_individual_trajectory(
        self,
        timepoints: list[float],
        individual_parameters: IndividualParameters,
    ) -> torch.Tensor:
        """Compute the model values for an individual characterized by their individual parameters at given time-points.
        Parameters
        ----------
        timepoints : :obj:`list` [:obj:`float`]
            The time-points at which to compute the model values.
        individual_parameters : :class:`~leaspy.io.IndividualParameters`
            The individual parameters of the individual.

        Returns
        -------
        :class:`torch.Tensor`
            The computed model values for the individual at the given time-points.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in the subclass.
        """
        raise NotImplementedError

    def simulate(
        self,
        algorithm: Optional[Union[str, AlgorithmName]] = None,
        algorithm_settings: Optional[AlgorithmSettings] = None,
        algorithm_settings_path: Optional[Union[str, Path]] = None,
        **kwargs,
    ):
        """
        Run the simulation pipeline using a leaspy model.

        This method simulates longitudinal data using the given leaspy model.
        It performs the following steps:
        - Retrieves individual parameters (IP) from fixed effects of the model.
        - Loads the specified Leaspy model.
        - Generates visit ages (timepoints) for each individual (based on specifications
        in visits_type)
        - Simulates observations at those visit ages.
        - Packages the result into a `Result` object, including simulated data,
        individual parameters, and the model's noise standard deviation.

        Parameters
        ----------
        algorithm : :obj:`str` or :class:`~leaspy.algo.base.AlgorithmName`, optional
            The name of the algorithm to use.
            Use this if you want to provide algorithm settings through kwargs.
        algorithm_settings : :class:`~leaspy.algo.AlgorithmSettings`, optional
            The algorithm settings to use.
            Use this if you want to customize algorithm settings through the
            :class:`~leaspy.algo.AlgorithmSettings` class.
            If provided, the fit will rely on these settings.
        algorithm_settings_path : :obj:`str` or :obj:`Path`, optional
            The path to the algorithm settings file.
            If provided, the settings from the file will be used instead of the settings provided.
        **kwargs : :obj:`dict`
            Contains the algorithm's settings.

        Returns
        -------
        simulated_data : :class:`~leaspy.io.outputs.result.Result`
            Contains the generated individual parameters & the corresponding generated scores.
            Returns empty Result if any required input is None.


        Notes
        -----
        To generate a new subject, first we estimate the joined distribution of the individual parameters and the
        reparametrized baseline ages. Then, we randomly pick a new point from this distribution, which define the
        individual parameters & baseline age of our new subjects. Then, we generate the timepoints
        following the baseline age. Then, from the model and the generated timepoints and individual parameters, we
        compute the corresponding values estimations. Then, we add some noise to these estimations, which is the
        same noise-model as the one from your model by default. But, you may customize it by setting the `noise` keyword.

        Examples
        --------
        Use a calibrated model & individual parameters to simulate new subjects similar to the ones you have:

        >>> from leaspy.models import LogisticModel
        >>> from leaspy.io.data import Data
        >>> from leaspy.datasets import load_dataset, load_leaspy_instance, load_individual_parameters
        >>> putamen_df = load_dataset("parkinson-putamen-train_and_test")
        >>> data = Data.from_dataframe(putamen_df.xs('train', level='SPLIT'))
        >>> leaspy_logistic = load_leaspy_instance("parkinson-putamen-train")
        >>> visits_params = {'patient_number':200,
                 'visit_type': "random",
                 'first_visit_mean' : 0.,
                 'first_visit_std' : 0.4,
                 'time_follow_up_mean' : 11,
                 'time_follow_up_std' : 0.5,
                 'distance_visit_mean' : 2/12,
                 'distance_visit_std' : 0.75/12,
                 'min_spacing_between_visits': 1/365
                }
        >>> simulated_data = model.simulate( algorithm="simulate", features=["MDS1_total", "MDS2_total", "MDS3_off_total", 'SCOPA_total','MOCA_total','REM_total','PUTAMEN_R','PUTAMEN_L','CAUDATE_R','CAUDATE_L'],visit_parameters= visits_params  )
        """
        from leaspy.exceptions import LeaspyInputError

        if not self.is_initialized:
            raise LeaspyInputError("Model has not been initialized")

        if (
            algorithm := BaseModel._get_algorithm(
                algorithm, algorithm_settings, algorithm_settings_path, **kwargs
            )
        ) is None:
            # if no algorithm is provided, we cannot simulate anything
            return Result()

        return algorithm.run(self)

    def __str__(self) -> str:
        """String representation of the model.
        Returns
        -------
        :obj:`str`
            A string representation of the model, including its name, dimension, features, and parameters.
        """
        from .utilities import serialize_tensor

        output = f"=== {self.__class__.__name__} {self.name} ==="
        output += f"\ndimension : {self.dimension}\nfeatures : {self.features}"
        output += serialize_tensor(self.parameters)

        # TODO/WIP obs models...
        # nm_props = export_noise_model(self.noise_model)
        # nm_name = nm_props.pop('name')
        # output += f"\nnoise-model : {nm_name}"
        # output += self._serialize_tensor(nm_props, indent="  ")

        return output
