import random
import sys
import time
from abc import ABC, abstractmethod
from copy import deepcopy
from enum import Enum
from typing import Any, Optional, Tuple, Type, Union

import numpy as np
import torch

from leaspy.exceptions import LeaspyAlgoInputError, LeaspyModelInputError
from leaspy.io.logs.fit_output_manager import FitOutputManager
from leaspy.models import AbstractModel

from .settings import AlgorithmSettings, OutputsSettings

__all__ = [
    "AbstractAlgo",
    "AlgorithmType",
    "AlgorithmName",
    "get_algorithm_type",
    "get_algorithm_class",
    "algorithm_factory",
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
    SIMULATE = "simulation"


class AbstractAlgo(ABC):
    """
    Abstract class containing common methods for all algorithm classes.
    These classes are child classes of `AbstractAlgo`.

    Parameters
    ----------
    settings : :class:`.AlgorithmSettings`
        The specifications of the algorithm as a :class:`.AlgorithmSettings` instance.

    Attributes
    ----------
    name : str
        Name of the algorithm.
    family : str
        Family of the algorithm. For now, valid families are:
            * ``'fit'```
            * ``'personalize'```
            * ``'simulate'``
    deterministic : bool
        True, if and only if algorithm does not involve in randomness.
        Setting a seed and such algorithms will be useless.
    algo_parameters : dict
        Contains the algorithm's parameters. Those are controlled by
        the :attr:`.AlgorithmSettings.parameters` class attribute.
    seed : int, optional
        Seed used by :mod:`numpy` and :mod:`torch`.
    output_manager : :class:`~.io.logs.fit_output_manager.FitOutputManager`
        Optional output manager of the algorithm
    """

    # Identifier of algorithm (classes variables)
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
        # settings parameters to be also modified (e.g. to be able to re-use them without any trouble)
        self.algo_parameters = deepcopy(settings.parameters)
        self.output_manager: Optional[FitOutputManager] = None

    @staticmethod
    def _initialize_seed(seed: Optional[int]):
        """
        Set :mod:`random`, :mod:`numpy` and :mod:`torch` seeds and display it (static method).

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

    @abstractmethod
    def run_impl(
        self,
        model: AbstractModel,
        *args,
        **extra_kwargs,
    ) -> Tuple[Any, Optional[torch.FloatTensor]]:
        """
        Run the algorithm (actual implementation), to be implemented in children classes.

        TODO fix proper abstract class

        Parameters
        ----------
        model : :class:`~.models.abstract_model.AbstractModel`
            The used model.
        dataset : :class:`.Dataset`
            Contains all the subjects' observations with corresponding timepoints, in torch format to speed up computations.

        Returns
        -------
        A 2-tuple containing:
            * the result to send back to user
            * optional float tensor representing loss (to be printed)

        See Also
        --------
        :class:`.AbstractFitAlgo`
        :class:`.AbstractPersonalizeAlgo`
        :class:`.SimulationAlgorithm`
        """

    def run(
        self, model: AbstractModel, *args, return_loss: bool = False, **extra_kwargs
    ) -> Any:
        """
        Main method, run the algorithm.

        TODO fix proper abstract class method: input depends on algorithm... (esp. simulate != from others...)

        Parameters
        ----------
        model : :class:`~.models.abstract_model.AbstractModel`
            The used model.
        dataset : :class:`.Dataset`
            Contains all the subjects' observations with corresponding timepoints, in torch format to speed up computations.
        return_loss : bool (default False), keyword only
            Should the algorithm return main output and optional loss output as a 2-tuple?

        Returns
        -------
        Depends on algorithm class: TODO change?

        See Also
        --------
        :class:`.AbstractFitAlgo`
        :class:`.AbstractPersonalizeAlgo`
        :class:`.SimulationAlgorithm`
        """

        # Check algo is well-defined
        if self.algo_parameters is None:
            raise LeaspyAlgoInputError(
                f"The `{self.name}` algorithm was not properly created."
            )

        self._initialize_seed(self.seed)

        time_beginning = time.time()

        output = self.run_impl(model, *args, **extra_kwargs)

        duration_in_seconds = time.time() - time_beginning
        if self.algo_parameters.get("progress_bar"):
            # new line for clarity
            print()
        print(
            f"\n{self.family.value.title()} with `{self.name}` took: {self._duration_to_str(duration_in_seconds)}"
        )

        return output

    def load_parameters(self, parameters: dict):
        """
        Update the algorithm's parameters by the ones in the given dictionary. The keys in the io which does not
        belong to the algorithm's parameters keys are ignored.

        Parameters
        ----------
        parameters : dict
            Contains the pairs (key, value) of the wanted parameters

        Examples
        --------
        >>> from leaspy.algo import AlgorithmSettings, algorithm_factory, OutputsSettings
        >>> my_algo = algorithm_factory(AlgorithmSettings("mcmc_saem"))
        >>> my_algo.algo_parameters
        {'n_iter': 10000,
         'n_burn_in_iter': 9000,
         'eps': 0.001,
         'L': 10,
         'sampler_ind': 'Gibbs',
         'sampler_pop': 'Gibbs',
         'annealing': {'do_annealing': False,
          'initial_temperature': 10,
          'n_plateau': 10,
          'n_iter': 200}}
        >>> parameters = {'n_iter': 5000, 'n_burn_in_iter': 4000}
        >>> my_algo.load_parameters(parameters)
        >>> my_algo.algo_parameters
        {'n_iter': 5000,
         'n_burn_in_iter': 4000,
         'eps': 0.001,
         'L': 10,
         'sampler_ind': 'Gibbs',
         'sampler_pop': 'Gibbs',
         'annealing': {'do_annealing': False,
          'initial_temperature': 10,
          'n_plateau': 10,
          'n_iter': 200}}
        """
        for k, v in parameters.items():
            if k in self.algo_parameters.keys():
                previous_v = self.algo_parameters[k]
                # TODO? log it instead (level=INFO or DEBUG)
                print(f"Replacing {k} parameter from value {previous_v} to value {v}")
            self.algo_parameters[k] = v

    def set_output_manager(self, output_settings: OutputsSettings) -> None:
        """
        Set a :class:`~.io.logs.fit_output_manager.FitOutputManager` object for the run of the algorithm

        Parameters
        ----------
        output_settings : :class:`~.io.settings.outputs_settings.OutputsSettings`
            Contains the logs settings for the computation run (console print periodicity, plot periodicity ...)

        Examples
        --------
        >>> from leaspy.algo import AlgorithmSettings, algorithm_factory, OutputsSettings
        >>> algo_settings = AlgorithmSettings("mcmc_saem")
        >>> my_algo = algorithm_factory(algo_settings)
        >>> settings = {
            'path': 'brouillons',
            'print_periodicity': 50,
            'plot_periodicity': 100,
            'save_periodicity': 50
        }
        >>> my_algo.set_output_manager(OutputsSettings(settings))
        """
        if output_settings is not None:
            self.output_manager = FitOutputManager(output_settings)

    @staticmethod
    def _display_progress_bar(
        iteration: int, n_iter: int, suffix: str, n_step_default: int = 50
    ):
        """
        Display a progression bar while running algorithm, simply based on `sys.stdout`.

        TODO: use tqdm instead?

        Parameters
        ----------
        iteration : int >= 0 or -1
            Current iteration of the algorithm.
            The final iteration should be `n_iter - 1`
        n_iter : int
            Total iterations' number of the algorithm.
        suffix : str
            Used to differentiate types of algorithms:
                * for fit algorithms: ``suffix = 'iterations'``
                * for personalization algorithms: ``suffix = 'subjects'``.
        n_step_default : int, default 50
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
                    f"|{'#'*nbar}{'-'*(n_step - nbar)}|   {iteration_plus_1}/{n_iter} {suffix}"
                )
                sys.stdout.flush()

    @staticmethod
    def _duration_to_str(seconds: float, *, seconds_fmt=".0f") -> str:
        """
        Convert a float representing computation time in seconds to a string giving time in hour, minutes and
        seconds ``%h %min %s``.

        If less than one hour, do not return hours. If less than a minute, do not return minutes.

        Parameters
        ----------
        seconds : float
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

    def _get_progress_str(self) -> Optional[str]:
        # TODO in a special mixin for sequential algos with nb of iters (MCMC fit, MCMC personalize)
        if not hasattr(self, "current_iteration"):
            return
        return f"Iteration {self.current_iteration} / {self.algo_parameters['n_iter']}"

    def __str__(self):
        out = "=== ALGO ===\n"
        out += f"Instance of {self.name} algo"
        if hasattr(self, "algorithm_device"):
            out += f" [{self.algorithm_device.upper()}]"
        progress_str = self._get_progress_str()
        if progress_str:
            out += "\n" + progress_str
        return out


def get_algorithm_type(name: Union[str, AlgorithmName]) -> AlgorithmType:
    name = AlgorithmName(name)
    if name in (AlgorithmName.FIT_LME, AlgorithmName.FIT_MCMC_SAEM):
        return AlgorithmType.FIT
    if name == AlgorithmName.SIMULATE:
        return AlgorithmType.SIMULATE
    return AlgorithmType.PERSONALIZE


def get_algorithm_class(name: Union[str, AlgorithmName]) -> Type[AbstractAlgo]:
    name = AlgorithmName(name)
    if name == AlgorithmName.FIT_MCMC_SAEM:
        from .fit import TensorMCMCSAEM

        return TensorMCMCSAEM
    if name == AlgorithmName.FIT_LME:
        from .others import LMEFitAlgorithm

        return LMEFitAlgorithm
    if name == AlgorithmName.PERSONALIZE_SCIPY_MINIMIZE:
        from .personalize import ScipyMinimize

        return ScipyMinimize
    if name == AlgorithmName.PERSONALIZE_MEAN_REAL:
        from .personalize import MeanReal

        return MeanReal
    if name == AlgorithmName.PERSONALIZE_MODE_REAL:
        from .personalize import ModeReal

        return ModeReal
    if name == AlgorithmName.PERSONALIZE_CONSTANT:
        from .others import ConstantPredictionAlgorithm

        return ConstantPredictionAlgorithm
    if name == AlgorithmName.PERSONALIZE_LME:
        from .others import LMEPersonalizeAlgorithm

        return LMEPersonalizeAlgorithm
    if name == AlgorithmName.SIMULATE:
        from .simulate import SimulationAlgorithm
        return SimulationAlgorithm


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
    if settings.logs is None and algorithm.family == AlgorithmType.FIT:
        settings.set_logs()
    algorithm.set_output_manager(settings.logs)
    return algorithm
