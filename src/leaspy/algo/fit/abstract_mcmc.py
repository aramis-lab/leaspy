"""This module defines the `AbstractFitMCMC` class."""

from random import shuffle

from leaspy.io.data import Dataset
from leaspy.models import McmcSaemCompatibleModel
from leaspy.variables.specs import (
    IndividualLatentVariable,
    PopulationLatentVariable,
)
from leaspy.variables.state import State

from ..utils import AlgoWithAnnealingMixin, AlgoWithSamplersMixin
from .abstract_fit_algo import AbstractFitAlgo

__all__ = ["AbstractFitMCMC"]


class AbstractFitMCMC(AlgoWithAnnealingMixin, AlgoWithSamplersMixin, AbstractFitAlgo):
    """
    Abstract class containing common method for all `fit` algorithm classes based on `Monte-Carlo Markov Chains` (MCMC).

    Parameters
    ----------
    settings : :class:`~leaspy.algo.AlgorithmSettings`
        MCMC fit algorithm settings

    Attributes
    ----------
    samplers : :obj:`dict` [:obj:`str`, :class:`~leaspy.samplers.AbstractSampler` ]
        Dictionary of samplers per each variable

    random_order_variables : :obj:`bool` (default True)
        This attribute controls whether we randomize the order of variables at each iteration.
        `Article <https://proceedings.neurips.cc/paper/2016/hash/e4da3b7fbbce2345d7772b0674a318d5-Abstract.html>`_
        gives a reason on why we should activate this flag.

    temperature : :obj:`float`
    temperature_inv : :obj:`float`
        Temperature and its inverse are modified during algorithm if annealing is used

    See Also
    --------
    :mod:`leaspy.samplers`
    """

    def _initialize_algo(
        self,
        model: McmcSaemCompatibleModel,
        dataset: Dataset,
    ) -> State:
        # TODO? mutualize with perso mcmc algo?
        state = super()._initialize_algo(model, dataset)
        # Initialize individual latent variables (population ones should be initialized before)
        model.put_individual_parameters(state, dataset)
        self._initialize_samplers(state, dataset)
        self._initialize_annealing()

        return state

    def iteration(
        self,
        model: McmcSaemCompatibleModel,
        state: State,
    ) -> None:
        """
        MCMC-SAEM iteration.

        1. Sample : MC sample successively of the population and individual variables
        2. Maximization step : update model parameters from current population/individual variables values.

        Parameters
        ----------
        model : :class:`~leaspy.models.McmcSaemCompatibleModel`
        state : :class:`~leaspy.variables.state.State`
        """
        variables = sorted(
            list(state.dag.sorted_variables_by_type[PopulationLatentVariable])
            + list(state.dag.sorted_variables_by_type[IndividualLatentVariable])
        )
        if self.random_order_variables:
            shuffle(variables)
        for variable in variables:
            self.samplers[variable].sample(state, temperature_inv=self.temperature_inv)
        self._maximization_step(model, state)
        self._update_temperature()
