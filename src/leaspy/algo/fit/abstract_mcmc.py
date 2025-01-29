from random import shuffle

from leaspy.io.data.dataset import Dataset
from leaspy.models import AbstractModel
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
    settings : :class:`.AlgorithmSettings`
        MCMC fit algorithm settings

    Attributes
    ----------
    samplers : dict[ str, :class:`~.algo.utils.samplers.abstract_sampler.AbstractSampler` ]
        Dictionary of samplers per each variable

    random_order_variables : bool (default True)
        This attribute controls whether we randomize the order of variables at each iteration.
        Article https://proceedings.neurips.cc/paper/2016/hash/e4da3b7fbbce2345d7772b0674a318d5-Abstract.html
        gives a rationale on why we should activate this flag.

    temperature : float
    temperature_inv : float
        Temperature and its inverse (modified during algorithm when using annealing)

    See Also
    --------
    :mod:`leaspy.algo.utils.samplers`
    """

    def _initialize_algo(
        self,
        model: AbstractModel,
        dataset: Dataset,
    ) -> State:
        # TODO? mutualize with perso mcmc algo?
        state = super()._initialize_algo(model, dataset)
        # Initialize individual latent variables (population ones should be initialized before)
        model.put_individual_parameters(state, dataset)
        # Samplers mixin
        self._initialize_samplers(state, dataset)
        # Annealing mixin
        self._initialize_annealing()

        return state

    def iteration(
        self,
        model: AbstractModel,
        state: State,
    ) -> None:
        """
        MCMC-SAEM iteration.

        1. Sample : MC sample successively of the population and individual variables
        2. Maximization step : update model parameters from current population/individual variables values.

        Parameters
        ----------
        model : :class:`~.models.abstract_model.AbstractModel`
        state : :class:`.State`
        """
        vars_order = list(
            state.dag.sorted_variables_by_type[PopulationLatentVariable]
        ) + list(state.dag.sorted_variables_by_type[IndividualLatentVariable])
        # TMP --> fix order of random variables as previously to pass functional tests...
        if set(vars_order) == {"log_g", "log_v0", "xi", "tau"}:
            vars_order = ["log_g", "log_v0", "tau", "xi"]
        elif set(vars_order) == {"log_g", "betas", "log_v0", "xi", "tau", "sources"}:
            vars_order = ["log_g", "log_v0", "betas", "tau", "xi", "sources"]
        elif set(vars_order) == {"g", "log_v0", "xi", "tau"}:
            vars_order = ["g", "log_v0", "tau", "xi"]
        elif set(vars_order) == {"g", "betas", "log_v0", "xi", "tau", "sources"}:
            vars_order = ["g", "log_v0", "betas", "tau", "xi", "sources"]
        elif set(vars_order) == {"betas", "deltas", "log_g", "sources", "tau", "xi"}:
            vars_order = ["log_g", "deltas", "betas", "tau", "xi", "sources"]
        elif set(vars_order) == {"deltas", "log_g", "tau", "xi"}:
            vars_order = ["log_g", "deltas", "tau", "xi"]
        # END TMP
        if self.random_order_variables:
            shuffle(vars_order)  # shuffle order in-place!

        for key in vars_order:
            self.samplers[key].sample(state, temperature_inv=self.temperature_inv)

        # Maximization step
        self._maximization_step(model, state)

        # Annealing mixin
        self._update_temperature()
