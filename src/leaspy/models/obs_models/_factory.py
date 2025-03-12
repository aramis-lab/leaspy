"""Defines the noise model factory."""

from enum import Enum
from typing import Dict, Type, Union

from leaspy.exceptions import LeaspyModelInputError

from ._base import ObservationModel
from ._bernoulli import BernoulliObservationModel
from ._gaussian import FullGaussianObservationModel
from ._weibull import (
    WeibullRightCensoredObservationModel,
    WeibullRightCensoredWithSourcesObservationModel,
)
from ._mixture_gaussian import MixtureGaussianObservationModel

__all__ = [
    "ObservationModelNames",
    "ObservationModelFactoryInput",
    "OBSERVATION_MODELS",
    "observation_model_factory",
]


class ObservationModelNames(Enum):
    """Enumeration defining the possible names for observation models."""

    GAUSSIAN_DIAGONAL = "gaussian-diagonal"
    GAUSSIAN_SCALAR = "gaussian-scalar"
    BERNOULLI = "bernoulli"
    WEIBULL_RIGHT_CENSORED = "weibull-right-censored"
    WEIBULL_RIGHT_CENSORED_WITH_SOURCES = "weibull-right-censored-with-sources"
    MIXTURE_GAUSSIAN  = "mixture-gaussian"

    @classmethod
    def from_string(cls, model_name: str):
        try:
            return cls(model_name.lower().replace("_", "-"))
        except ValueError:
            raise NotImplementedError(
                f"The requested ObservationModel {model_name} is not implemented. "
                f"Valid observation model names are: {[elt.value for elt in cls]}."
            )


ObservationModelFactoryInput = Union[str, ObservationModelNames, ObservationModel]

OBSERVATION_MODELS: Dict[ObservationModelNames, Type[ObservationModel]] = {
    ObservationModelNames.GAUSSIAN_DIAGONAL: FullGaussianObservationModel,
    ObservationModelNames.GAUSSIAN_SCALAR: FullGaussianObservationModel,
    ObservationModelNames.BERNOULLI: BernoulliObservationModel,
    ObservationModelNames.WEIBULL_RIGHT_CENSORED: WeibullRightCensoredObservationModel,
    ObservationModelNames.WEIBULL_RIGHT_CENSORED_WITH_SOURCES: WeibullRightCensoredWithSourcesObservationModel,
    ObservationModelNames.MIXTURE_GAUSSIAN: MixtureGaussianObservationModel,
}


def observation_model_factory(
    model: ObservationModelFactoryInput, **kwargs
) -> ObservationModel:
    """
    Factory for observation models.

    Parameters
    ----------
    model : :obj:`str` or :class:`.ObservationModel` or :obj:`dict` [ :obj:`str`, ...]
        - If an instance of a subclass of :class:`.ObservationModel`, returns the instance.
        - If a string, then returns a new instance of the appropriate class (with optional parameters `kws`).
        - If a dictionary, it must contain the 'name' key and other initialization parameters.
    **kwargs
        Optional parameters for initializing the requested observation model when a string.

    Returns
    -------
    :class:`.ObservationModel` :
        The desired observation model.

    Raises
    ------
    :exc:`.LeaspyModelInputError` :
        If `model` is not supported.
    """
    dimension = kwargs.pop("dimension", None)
    n_clusters = kwargs.pop("n_clusters", None)
    if isinstance(model, ObservationModel):
        return model
    if isinstance(model, str):
        model = ObservationModelNames.from_string(model)
    if isinstance(model, ObservationModelNames):
        if model == ObservationModelNames.GAUSSIAN_DIAGONAL:
            if dimension is None:
                raise NotImplementedError(
                    "WIP: dimension / features should be provided to "
                    f"init the obs_model = {ObservationModelNames.GAUSSIAN_DIAGONAL}."
                )
            return FullGaussianObservationModel.with_noise_std_as_model_parameter(
                dimension
            )
        if model == ObservationModelNames.GAUSSIAN_SCALAR:
            return FullGaussianObservationModel.with_noise_std_as_model_parameter(1)
        if model == ObservationModelNames.WEIBULL_RIGHT_CENSORED:
            return WeibullRightCensoredObservationModel.default_init(kwargs=kwargs)
        if model == ObservationModelNames.WEIBULL_RIGHT_CENSORED_WITH_SOURCES:
            return WeibullRightCensoredWithSourcesObservationModel.default_init(
                kwargs=kwargs
            )
        if model == ObservationModelNames.MIXTURE_GAUSSIAN:
            #if n_clusters is None:
            #    raise NotImplementedError(
            #        "WIP: number of clusters should be provided to "
            #        f"init the obs_model = {ObservationModelNames.MIXTURE_GAUSSIAN}."
            #    )
            #if dimension is None:
            #    raise NotImplementedError(
            #        "WIP: dimension / features should be provided to "
            #        f"init the obs_model = {ObservationModelNames.GAUSSIAN_DIAGONAL}."
            #    )
            return MixtureGaussianObservationModel.with_probs(dimension, n_clusters)
        return OBSERVATION_MODELS[model](**kwargs)
    raise LeaspyModelInputError(
        "The provided `model` should be a valid instance of `ObservationModel`, "
        f"or a string among {[c.value for c in ObservationModelNames]}."
        f"Instead, {model} of type {type(model)} was provided."
    )
