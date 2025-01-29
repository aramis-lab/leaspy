from enum import Enum
from typing import Type, Union

from .abstract_algo import AbstractAlgo
from .fit import TensorMCMCSAEM
from .others import (
    ConstantPredictionAlgorithm,
    LMEFitAlgorithm,
    LMEPersonalizeAlgorithm,
)
from .personalize import MeanReal, ModeReal, ScipyMinimize
from .settings import AlgorithmSettings

__all__ = [
    "AlgorithmType",
    "AlgorithmName",
    "get_algorithm_type",
    "algorithm_factory",
    "get_algorithm_class",
]


class AlgorithmType(str, Enum):
    FIT = "fit"
    PERSONALIZE = "personalize"
    SIMULATE = "simulate"


class AlgorithmName(str, Enum):
    """The available algorithms in Leaspy."""

    FIT_MCMC_SAEM = "mcmc_saem"
    FIT_LME = "lme_fit"
    PERSONALIZE_SCIPY_MINIMIZE = "scipy_minimize"
    PERSONALIZE_MEAN_REAL = "mean_real"
    PERSONALIZE_MODE_REAL = "mode_real"
    PERSONALIZE_CONSTANT = "constant_prediction"
    PERSONALIZE_LME = "lme_personalize"
    SIMULATE = "simulate"


def get_algorithm_type(name: Union[str, AlgorithmName]) -> AlgorithmType:
    name = AlgorithmName(name)
    if name in (AlgorithmName.FIT_LME, AlgorithmName.FIT_MCMC_SAEM):
        return AlgorithmType.FIT
    if name == AlgorithmName.SIMULATE:
        return AlgorithmType.SIMULATE
    return AlgorithmType.PERSONALIZE


def algorithm_factory(settings: AlgorithmSettings) -> AbstractAlgo:
    """
    Return the wanted algorithm.

    Parameters
    settings : :class:`.AlgorithmSettings`
        The algorithm settings.

    Returns
    -------
    algorithm : child class of :class:`.AbstractAlgo`
        The wanted algorithm if it exists and is compatible with algorithm family.
    """
    algorithm = get_algorithm_class(settings.name)(settings)
    algorithm.set_output_manager(settings.logs)
    return algorithm


def get_algorithm_class(name: Union[str, AlgorithmName]) -> Type[AbstractAlgo]:
    name = AlgorithmName(name)
    if name == AlgorithmName.FIT_MCMC_SAEM:
        return TensorMCMCSAEM
    if name == AlgorithmName.FIT_LME:
        return LMEFitAlgorithm
    if name == AlgorithmName.PERSONALIZE_SCIPY_MINIMIZE:
        return ScipyMinimize
    if name == AlgorithmName.PERSONALIZE_MEAN_REAL:
        return MeanReal
    if name == AlgorithmName.PERSONALIZE_MODE_REAL:
        return ModeReal
    if name == AlgorithmName.PERSONALIZE_CONSTANT:
        return ConstantPredictionAlgorithm
    if name == AlgorithmName.PERSONALIZE_LME:
        return LMEPersonalizeAlgorithm
    if name == AlgorithmName.SIMULATE:
        raise ValueError("The simulation algorithm is currently broken.")
