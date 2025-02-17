from abc import abstractmethod
from typing import Tuple

import torch

from leaspy.io.data import Dataset
from leaspy.io.outputs.individual_parameters import IndividualParameters
from leaspy.models import AbstractModel
from leaspy.utils.weighted_tensor import wsum_dim

from ..base import AbstractAlgo, AlgorithmType

__all__ = ["AbstractPersonalizeAlgo"]


class AbstractPersonalizeAlgo(AbstractAlgo):
    """
    Abstract class for `personalize` algorithm.
    Estimation of individual parameters of a given `Data` file with
    a frozen model (already estimated, or loaded from known parameters).

    Parameters
    ----------
    settings : :class:`.AlgorithmSettings`
        Settings of the algorithm.

    Attributes
    ----------
    name : str
        Algorithm's name.
    seed : int, optional
        Algorithm's seed (default None).
    algo_parameters : dict
        Algorithm's parameters.

    See Also
    --------
    :meth:`.Leaspy.personalize`
    """

    family = "personalize"

    def run_impl(
        self, model: AbstractModel, dataset: Dataset
    ) -> Tuple[IndividualParameters, torch.Tensor]:
        r"""
        Main personalize function, wraps the abstract :meth:`._get_individual_parameters` method.

        Parameters
        ----------
        model : :class:`~.models.abstract_model.AbstractModel`
            A subclass object of leaspy `AbstractModel`.
        dataset : :class:`.Dataset`
            Dataset object build with leaspy class objects Data, algo & model

        Returns
        -------
        individual_parameters : :class:`.IndividualParameters`
            Contains individual parameters.
        """

        # Estimate individual parameters
        individual_parameters = self._get_individual_parameters(model, dataset)

        local_state = model.state.clone(disable_auto_fork=True)
        model.put_data_variables(local_state, dataset)
        _, pyt_individual_parameters = individual_parameters.to_pytorch()
        for ip, ip_vals in pyt_individual_parameters.items():
            local_state[ip] = ip_vals

        return individual_parameters

    @abstractmethod
    def _get_individual_parameters(
        self, model: AbstractModel, data: Dataset
    ) -> IndividualParameters:
        """
        Estimate individual parameters from a `Dataset`.

        Parameters
        ----------
        model : :class:`~.models.abstract_model.AbstractModel`
            A subclass object of leaspy AbstractModel.
        dataset : :class:`.Dataset`
            Dataset object build with leaspy class objects Data, algo & model

        Returns
        -------
        :class:`.IndividualParameters`
        """
