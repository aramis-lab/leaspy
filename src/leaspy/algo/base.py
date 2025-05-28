"""This module defines the `AlgorithmType`, `AlgorithmName` and `AbstractAlgo` classes"""

import random
import sys
import time
from abc import ABC, abstractmethod
from copy import deepcopy
from enum import Enum
from typing import Generic, Optional, Type, TypeVar, Union

import numpy as np
import torch

from leaspy.exceptions import LeaspyAlgoInputError
from leaspy.io.data import Dataset
from leaspy.models import ModelType

from .settings import AlgorithmSettings, OutputsSettings

__all__ = [
    "BaseAlgo",
    "IterativeAlgo",
    "AlgorithmType",
    "AlgorithmName",
    "get_algorithm_type",
    "get_algorithm_class",
    "algorithm_factory",
    "ReturnType",
]


ReturnType = TypeVar("ReturnType")


class AlgorithmType(str, Enum):
    """The type of the algorithms."""

    FIT = "fit"
    PERSONALIZE = "personalize"
    SIMULATE = "simulate"


class AlgorithmName(str, Enum):
    """The available algorithms in Leaspy."""

    FIT_MCMC_SAEM = "mcmc_saem"
    FIT_LME = "lme_fit"
    PERSONALIZE_SCIPY_MINIMIZE = "scipy_minimize"
    PERSONALIZE_MEAN_POSTERIOR = "mean_posterior"
    PERSONALIZE_MODE_POSTERIOR = "mode_posterior"
    PERSONALIZE_CONSTANT = "constant_prediction"
    PERSONALIZE_LME = "lme_personalize"
    SIMULATE = "simulation"


class BaseAlgo(ABC, Generic[ModelType, ReturnType]):
    """Base class containing common methods for all algorithm classes.

    Parameters
    ----------
    settings : :class:`~leaspy.algo.AlgorithmSettings`
        The specifications of the algorithm as a :class:`~leaspy.algo.AlgorithmSettings` instance.

    Attributes
    ----------
    name : :class:`~leaspy.algo.base.AlgorithmName`
        Name of the algorithm.
    family : :class:`~leaspy.algo.base.AlgorithmType`
        Family of the algorithm.
    deterministic : :obj:`bool`
        True, if and only if algorithm does not involve randomness.
        Setting a seed will have no effect on such algorithms.
    algo_parameters : :obj:`dict`
        Contains the algorithm's parameters. Those are controlled by
        the :attr:`leaspy.algo.AlgorithmSettings.parameters` class attribute.
    seed : :obj:`int`, optional
        Seed used by :mod:`numpy` and :mod:`torch`.
    """

    name: AlgorithmName = None
    family: AlgorithmType = None
    deterministic: bool = False

    def __init__(self, settings: AlgorithmSettings):
        if settings.name != self.name:
            raise LeaspyAlgoInputError(
                f"Inconsistent naming: {settings.name} != {self.name}"
            )
        self.seed = settings.seed
        # we deepcopy the settings.parameters, because those algo_parameters may be
        # modified within algorithm (e.g. `n_burn_in_iter`) and we would not want the original
        # settings parameters to be also modified (e.g. to be able to reuse them without any trouble)
        self.algo_parameters = deepcopy(settings.parameters)
        self.output_manager = None

    @abstractmethod
    def set_output_manager(self, output_settings: OutputsSettings) -> None:
        raise NotImplementedError

    @staticmethod
    def _initialize_seed(seed: Optional[int]):
        """Set :mod:`random`, :mod:`numpy` and :mod:`torch` seeds and display it (static method).

        Notes - numpy seed is needed for reproducibility for the simulation algorithm which use the scipy kernel
        density estimation function. Indeed, scipy use numpy random seed.

        Parameters
        ----------
        seed : int
            The wanted seed
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            # TODO: use logger instead (level=INFO)
            print(f" ==> Setting seed to {seed}")

    def run(self, model: ModelType, dataset: Dataset, **kwargs) -> ReturnType:
        """Main method, run the algorithm.

        Parameters
        ----------
        model : :class:`~leaspy.models.BaseModel`
            The used model.

        dataset : :class:`~leaspy.io.data.Dataset`
            Contains all the subjects' observations with corresponding timepoints, in torch format to speed up computations.

        Returns
        -------
        ReturnType:
            Depends on algorithm class.

        See Also
        --------
        :class:`.AbstractFitAlgo`
        :class:`.AbstractPersonalizeAlgo`
        """
        if self.algo_parameters is None:
            raise LeaspyAlgoInputError(
                f"The `{self.name}` algorithm was not properly created."
            )
        self._initialize_seed(self.seed)
        time_beginning = time.time()
        output = self._run(model, dataset, **kwargs)
        duration_in_seconds = time.time() - time_beginning
        if self.algo_parameters.get("progress_bar"):
            print()
        print(
            f"\n{self.family.value.title()} with `{self.name}` took: {self._duration_to_str(duration_in_seconds)}"
        )
        return output

    @abstractmethod
    def _run(
        self,
        model: ModelType,
        dataset: Dataset,
        **kwargs,
    ) -> ReturnType:
        """Run the algorithm (actual implementation), to be implemented in children classes.

        Parameters
        ----------
        model : :class:`~leaspy.models.BaseModel`
            The used model.

        dataset : :class:`~leaspy.io.data.Dataset`
            Contains all the subjects' observations with corresponding timepoints, in torch format to speed up computations.

        Returns
        -------
        ReturnType :
            Depends on the algorithm.

        See Also
        --------
        :class:`.AbstractFitAlgo`
        :class:`.AbstractPersonalizeAlgo`
        """
        raise NotImplementedError

    def load_parameters(self, parameters: dict):
        """Update the algorithm's parameters by the ones in the given dictionary.

        The keys in the input which does not belong to the algorithm's parameters are ignored.

        Parameters
        ----------
        parameters : :obj:`dict`
            Contains the pairs (key, value) of the requested parameters

        Examples
        --------
        >>> from leaspy.algo import AlgorithmSettings, algorithm_factory, OutputsSettings
        >>> my_algo = algorithm_factory(AlgorithmSettings("mcmc_saem"))
        >>> my_algo.algo_parameters
        {'progress_bar': True,
        'n_iter': 10000,
        'n_burn_in_iter': 9000,
        'n_burn_in_iter_frac': 0.9,
        'burn_in_step_power': 0.8,
        'random_order_variables': True,
        'sampler_ind': 'Gibbs',
        'sampler_ind_params': {'acceptation_history_length': 25,
        'mean_acceptation_rate_target_bounds': [0.2, 0.4],
        'adaptive_std_factor': 0.1},
        'sampler_pop': 'Gibbs',
        'sampler_pop_params': {'random_order_dimension': True,
        'acceptation_history_length': 25,
        'mean_acceptation_rate_target_bounds': [0.2, 0.4],
        'adaptive_std_factor': 0.1},
        'annealing': {'do_annealing': False,
         'initial_temperature': 10,
         'n_plateau': 10,
         'n_iter': None,
         'n_iter_frac': 0.5}}
        >>> parameters = {'n_iter': 5000, 'n_burn_in_iter': 4000}
        >>> my_algo.load_parameters(parameters)
        >>> my_algo.algo_parameters
        {'progress_bar': True,
        'n_iter': 5000,
        'n_burn_in_iter': 4000,
        'n_burn_in_iter_frac': 0.9,
        'burn_in_step_power': 0.8,
        'random_order_variables': True,
        'sampler_ind': 'Gibbs',
        'sampler_ind_params': {'acceptation_history_length': 25,
        'mean_acceptation_rate_target_bounds': [0.2, 0.4],
        'adaptive_std_factor': 0.1},
        'sampler_pop': 'Gibbs',
        'sampler_pop_params': {'random_order_dimension': True,
        'acceptation_history_length': 25,
        'mean_acceptation_rate_target_bounds': [0.2, 0.4],
        'adaptive_std_factor': 0.1},
        'annealing': {'do_annealing': False,
         'initial_temperature': 10,
         'n_plateau': 10,
         'n_iter': None,
         'n_iter_frac': 0.5}}
        """
        for k, v in parameters.items():
            if k in self.algo_parameters.keys():
                previous_v = self.algo_parameters[k]
                # TODO? log it instead (level=INFO or DEBUG)
                print(f"Replacing {k} parameter from value {previous_v} to value {v}")
            self.algo_parameters[k] = v

    @staticmethod
    def _duration_to_str(seconds: float, *, seconds_fmt=".0f") -> str:
        """
        Convert a float representing computation time in seconds to a string giving time in hour, minutes and
        seconds ``%h %min %s``.

        If less than one hour, do not return hours. If less than a minute, do not return minutes.

        Parameters
        ----------
        seconds : :obj:`float`
            Computation time

        Returns
        -------
        str
            Time formatting in hour, minutes and seconds.
        """
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = seconds % 60  # float

        res = ""
        if m:
            if h:
                res += f"{h}h "
            res += f"{m}m "
        res += f"{s:{seconds_fmt}}s"

        return res

    def __str__(self) -> str:
        out = "=== ALGO ===\n"
        out += f"Instance of {self.name} algo"
        if hasattr(self, "algorithm_device"):
            out += f" [{self.algorithm_device.upper()}]"
        return out


class IterativeAlgo(BaseAlgo[ModelType, ReturnType]):
    def __init__(self, settings: AlgorithmSettings):
        super().__init__(settings)
        self.current_iteration: int = 0

    @staticmethod
    def _display_progress_bar(
        iteration: int, n_iter: int, suffix: str, n_step_default: int = 50
    ):
        """
        Display a progression bar while running algorithm, simply based on `sys.stdout`.

        TODO: use tqdm instead?

        Parameters
        ----------
        iteration : :obj:`int` >= 0 or -1
            Current iteration of the algorithm.
            If a positive integer, it is the current iteration of the algorithm. If equals to '-1' then it initialises the bar.
            The final current iteration should be `n_iter - 1`
        n_iter : :obj:`int`
            Total iterations' number of the algorithm.
        suffix : :obj:`str`
            Used to differentiate types of algorithms:
                * for fit algorithms: ``suffix = 'iterations'``
                * for personalization algorithms: ``suffix = 'subjects'``.
        n_step_default : :obj:`int`, default 50
            The size of the progression bar.
        """
        n_step = min(n_step_default, n_iter)
        if iteration == -1:
            sys.stdout.write("\r")
            sys.stdout.write("|" + "-" * n_step + "|   0/%d " % n_iter + suffix)
            sys.stdout.flush()
        else:
            print_every_iter = n_iter // n_step
            iteration_plus_1 = iteration + 1
            display = iteration_plus_1 % print_every_iter
            if display == 0:
                nbar = iteration_plus_1 // print_every_iter
                sys.stdout.write("\r")
                sys.stdout.write(
                    f"|{'#' * nbar}{'-' * (n_step - nbar)}|   {iteration_plus_1}/{n_iter} {suffix}"
                )
                sys.stdout.flush()

    def _get_progress_str(self) -> Optional[str]:
        # TODO in a special mixin for sequential algos with nb of iters (MCMC fit, MCMC personalize)
        if not hasattr(self, "current_iteration"):
            return None
        return f"Iteration {self.current_iteration} / {self.algo_parameters['n_iter']}"

    def __str__(self) -> str:
        out = super().__str__()
        progress_str = self._get_progress_str()
        if progress_str:
            out += "\n" + progress_str
        return out


def get_algorithm_type(name: Union[str, AlgorithmName]) -> AlgorithmType:
    """Return the algorithm type.

    Parameters
    ----------
    name : :obj:`str` or :class:`~leaspy.algo.base.AlgorithmName`
        The name of the algorithm.

    Returns
    -------
    algorithm type: :class:`leaspy.algo.AlgorithmType`
    """
    name = AlgorithmName(name)
    if name in (AlgorithmName.FIT_LME, AlgorithmName.FIT_MCMC_SAEM):
        return AlgorithmType.FIT
    if name == AlgorithmName.SIMULATE:
        return AlgorithmType.SIMULATE
    return AlgorithmType.PERSONALIZE


def get_algorithm_class(name: Union[str, AlgorithmName]) -> Type[BaseAlgo]:
    """Return the algorithm class.

    Parameters
    ----------
    name : :obj:`str` or :class:`~leaspy.algo.base.AlgorithmName`
         The name of the algorithm.

    Returns
    -------
    algorithm class: :class:`leaspy.algo.AbstractAlgo`
    """
    name = AlgorithmName(name)
    if name == AlgorithmName.FIT_MCMC_SAEM:
        from .fit import TensorMCMCSAEM

        return TensorMCMCSAEM
    if name == AlgorithmName.FIT_LME:
        from .fit import LMEFitAlgorithm

        return LMEFitAlgorithm
    if name == AlgorithmName.PERSONALIZE_SCIPY_MINIMIZE:
        from .personalize import ScipyMinimize

        return ScipyMinimize
    if name == AlgorithmName.PERSONALIZE_MEAN_POSTERIOR:
        from .personalize import MeanPosterior

        return MeanPosterior
    if name == AlgorithmName.PERSONALIZE_MODE_POSTERIOR:
        from .personalize import ModePosterior

        return ModePosterior
    if name == AlgorithmName.PERSONALIZE_CONSTANT:
        from .personalize import ConstantPredictionAlgorithm

        return ConstantPredictionAlgorithm
    if name == AlgorithmName.PERSONALIZE_LME:
        from .personalize import LMEPersonalizeAlgorithm

        return LMEPersonalizeAlgorithm
    if name == AlgorithmName.SIMULATE:
        raise ValueError("The simulation algorithm is currently broken.")


def algorithm_factory(settings: AlgorithmSettings) -> BaseAlgo:
    """Return the requested algorithm based on the provided settings.

    Parameters
    ----------
    settings : :class:`leaspy.algo.AlgorithmSettingss`
        The algorithm settings.

    Returns
    -------
    algorithm : child class of :class:`leaspy.algo.AbstractAlgo`
        The requested algorithm. If it exists, it will be compatible with algorithm family.
    """
    algorithm = get_algorithm_class(settings.name)(settings)
    if settings.logs is None and algorithm.family == AlgorithmType.FIT:
        settings.set_logs()
    algorithm.set_output_manager(settings.logs)
    return algorithm
