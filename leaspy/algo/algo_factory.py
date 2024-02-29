from __future__ import annotations
from typing import TYPE_CHECKING, Type

from collections import ChainMap

from leaspy.algo.fit.tensor_mcmcsaem import TensorMCMCSAEM
from leaspy.algo.personalize.scipy_minimize import ScipyMinimize
from leaspy.algo.personalize.mean_realisations import MeanReal
from leaspy.algo.personalize.mode_realisations import ModeReal
from leaspy.algo.others.constant_prediction_algo import ConstantPredictionAlgorithm
from leaspy.algo.simulate.simulate import SimulationAlgorithm
from leaspy.exceptions import LeaspyAlgoInputError

if TYPE_CHECKING:
    from leaspy.algo.abstract_algo import AbstractAlgo


class AlgoFactory:
    """
    Return the wanted algorithm given its name.

    Notes
    -----
    For developers: add your new algorithm in corresponding category of ``_algos`` dictionary.
    """

    _algos = {
        'fit': {
            'mcmc_saem': TensorMCMCSAEM,
        },
        'personalize': {
            'scipy_minimize': ScipyMinimize,
            'mean_real': MeanReal,
            'mode_real': ModeReal,
            'constant_prediction': ConstantPredictionAlgorithm,
        },
        'simulate': {
            'simulation': SimulationAlgorithm
        }
    }

    @classmethod
    def get_class(cls, name: str) -> Type[AbstractAlgo]:
        """Get the class of the algorithm identified as `name`."""
        klasses = ChainMap(*cls._algos.values())
        klass = klasses.get(name, None)

        if klass is None:
            raise LeaspyAlgoInputError(f'Your algorithm "{name}" should be part of the known algorithms: {cls._algos}.')

        return klass

    @classmethod
    def algo(cls, algorithm_family: str, settings) -> AbstractAlgo:
        """
        Return the wanted algorithm given its name.

        Parameters
        ----------
        algorithm_family : str
            Task name, used to check if the algorithm within the input `settings` is compatible with this task.
            Must be one of the following api's name:
                * `fit`
                * `personalize`
                * `simulate`

        settings : :class:`.AlgorithmSettings`
            The algorithm settings.

        Returns
        -------
        algorithm : child class of :class:`.AbstractAlgo`
            The wanted algorithm if it exists and is compatible with algorithm family.

        Raises
        ------
        :exc:`.LeaspyAlgoInputError`
            * if the algorithm family is unknown
            * if the algorithm name is unknown / does not belong to the wanted algorithm family
        """
        if algorithm_family not in cls._algos:
            raise LeaspyAlgoInputError(
                f"Algorithm family '{algorithm_family}' is unknown: "
                f"it must be in {set(cls._algos.keys())}."
            )
        if (name := settings.name) not in cls._algos[algorithm_family]:
            raise LeaspyAlgoInputError(
                f"Algorithm '{name}' is unknown or does not belong to '{algorithm_family} "
                f"algorithms: it must be in {set(cls._algos[algorithm_family].keys())}."
            )
        # instantiate algorithm with settings and set output manager
        algorithm = cls._algos[algorithm_family][name](settings)
        algorithm.set_output_manager(settings.logs)

        return algorithm
