"""This module defines the `AbstractPersonalizeAlgo` class used for all personalize algorithms."""

from abc import abstractmethod

from leaspy.io.data import Dataset
from leaspy.io.outputs.individual_parameters import IndividualParameters
from leaspy.models import McmcSaemCompatibleModel

from ..base import AbstractAlgo, AlgorithmType
from ..settings import OutputsSettings

__all__ = ["AbstractPersonalizeAlgo"]


class AbstractPersonalizeAlgo(
    AbstractAlgo[McmcSaemCompatibleModel, IndividualParameters]
):
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

    def run_impl(
        self, model: McmcSaemCompatibleModel, dataset: Dataset, **kwargs
    ) -> IndividualParameters:
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
        individual_parameters = self._get_individual_parameters(model, dataset)
        local_state = model.state.clone(disable_auto_fork=True)
        model.put_data_variables(local_state, dataset)
        _, pyt_individual_parameters = individual_parameters.to_pytorch()
        for ip, ip_vals in pyt_individual_parameters.items():
            local_state[ip] = ip_vals
        return individual_parameters

    @abstractmethod
    def _get_individual_parameters(
        self, model: McmcSaemCompatibleModel, data: Dataset
    ) -> IndividualParameters:
        """Estimate individual parameters from a `Dataset`.

        Parameters
        ----------
        model : :class:`~leaspy.models.McmcSaemCompatibleModel`
            A subclass object of leaspy McmcSaemCompatibleModel.

        dataset : :class:`.Dataset`
            Dataset object build with leaspy class objects Data, algo & model

        Returns
        -------
        :class:`.IndividualParameters`
        """
        raise NotImplementedError()
