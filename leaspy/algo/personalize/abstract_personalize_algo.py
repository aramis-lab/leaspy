from __future__ import annotations
from typing import TYPE_CHECKING, Tuple
from abc import abstractmethod

import torch

from leaspy.algo.abstract_algo import AbstractAlgo

if TYPE_CHECKING:
    from leaspy.io.data.dataset import Dataset
    from leaspy.models.abstract_model import AbstractModel
    from leaspy.io.outputs.individual_parameters import IndividualParameters


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

    family = 'personalize'

    def run_impl(self, model: AbstractModel, dataset: Dataset) -> Tuple[IndividualParameters, torch.Tensor]:
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
        noise_std : float or :class:`torch.FloatTensor`
            The estimated noise (is a tensor if `model.noise_model` is ``'gaussian_diagonal'``)

            .. math:: = \frac{1}{n_{visits} \times n_{dim}} \sqrt{\sum_{i, j \in [1, n_{visits}] \times [1, n_{dim}]} \varepsilon_{i,j}}

            where :math:`\varepsilon_{i,j} = \left( f(\theta, (z_{i,j}), (t_{i,j})) - (y_{i,j}) \right)^2` , where
            :math:`\theta` are the model's fixed effect, :math:`(z_{i,j})` the model's random effects,
            :math:`(t_{i,j})` the time-points and :math:`f` the model's estimator.
        """

        # Estimate individual parameters
        individual_parameters = self._get_individual_parameters(model, dataset)

        # Compute the loss with these estimated individual parameters (RMSE or NLL depending on noise models)
        _, pyt_individual_params = individual_parameters.to_pytorch()
        loss = model.compute_canonical_loss_tensorized(dataset, pyt_individual_params)

        return individual_parameters, loss

    @abstractmethod
    def _get_individual_parameters(self, model: AbstractModel, data: Dataset) -> IndividualParameters:
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
