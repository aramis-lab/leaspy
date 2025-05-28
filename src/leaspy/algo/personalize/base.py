"""This module defines the `AbstractPersonalizeAlgo` class used for all personalize algorithms."""

from abc import abstractmethod

from leaspy.io.data import Dataset
from leaspy.io.outputs import IndividualParameters
from leaspy.models import ModelType

from ..base import AlgorithmType, IterativeAlgorithm, ReturnType
from ..settings import OutputsSettings

__all__ = ["PersonalizeAlgorithm"]


class PersonalizeAlgorithm(IterativeAlgorithm[ModelType, ReturnType]):
    """Abstract class for `personalize` algorithm.

    Estimation of individual parameters of a given `Data` file with
    a frozen model (already estimated, or loaded from known parameters).

    Parameters
    ----------
    settings : :class:`.AlgorithmSettings`
        Settings of the algorithm.

    Attributes
    ----------
    name : :obj:`str`
        Algorithm's name.
    seed : :obj:`int`, optional
        Algorithm's seed (default None).
    algo_parameters : :obj:`dict`
        Algorithm's parameters.

    See Also
    --------
    :meth:`.Leaspy.personalize`
    """

    family: AlgorithmType = AlgorithmType.PERSONALIZE

    def set_output_manager(self, output_settings: OutputsSettings) -> None:
        """Set the output manager.

        This is currently not implemented for personalize.
        """
        pass

    def _run(self, model: ModelType, dataset: Dataset, **kwargs) -> ReturnType:
        r"""Main personalize function, wraps the abstract :meth:`._get_individual_parameters` method.

        Parameters
        ----------
        model : :class:`~leaspy.models.McmcSaemCompatibleModel`
            A subclass object of leaspy `McmcSaemCompatibleModel`.

        dataset : :class:`.Dataset`
            Dataset object build with leaspy class objects Data, algo & model

        Returns
        -------
        individual_parameters : :class:`.IndividualParameters`
            Contains individual parameters.
        """
        return self._compute_individual_parameters(model, dataset, **kwargs)

    @abstractmethod
    def _compute_individual_parameters(
        self, model: ModelType, dataset: Dataset, **kwargs
    ) -> IndividualParameters:
        raise NotImplementedError()
