"""This module defines the `TensorMCMCSAEM` class."""

from random import shuffle

from leaspy.exceptions import LeaspyAlgoInputError
from leaspy.io.data import Dataset
from leaspy.models import McmcSaemCompatibleModel
from leaspy.variables.specs import (
    IndividualLatentVariable,
    LatentVariableInitType,
    PopulationLatentVariable,
)
from leaspy.variables.state import State

from ..algo_with_annealing import AlgoWithAnnealingMixin
from ..algo_with_device import AlgoWithDeviceMixin
from ..algo_with_samplers import AlgoWithSamplersMixin
from ..base import AlgorithmName
from ..settings import AlgorithmSettings
from .base import FitAlgo

__all__ = ["TensorMCMCSAEM"]


class TensorMCMCSAEM(
    AlgoWithDeviceMixin,
    AlgoWithAnnealingMixin,
    AlgoWithSamplersMixin,
    FitAlgo[McmcSaemCompatibleModel, State],
):
    """Main algorithm for MCMC-SAEM.

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

    name: AlgorithmName = AlgorithmName.FIT_MCMC_SAEM

    def __init__(self, settings: AlgorithmSettings):
        super().__init__(settings)
        if not (0.5 < self.algo_parameters["burn_in_step_power"] <= 1):
            raise LeaspyAlgoInputError(
                "The parameter `burn_in_step_power` should be in ]0.5, 1] in order to "
                "have theoretical guarantees on convergence of MCMC-SAEM algorithm."
            )

    def _run(self, model: McmcSaemCompatibleModel, dataset: Dataset, **kwargs) -> State:
        """Main method to run the algorithm.

        Basically, it initializes the :class:`~leaspy.variables.state.State` object,
        updates it using the :meth:`~leaspy.algo.AbstractFitAlgo.iteration` method then returns it.

        Parameters
        ----------
        model : :class:`~leaspy.models.McmcSaemCompatibleModel`
            The used model. It must be a subclass of :class:`~leaspy.models.McmcSaemCompatibleModel`.

        dataset : :class:`~leaspy.io.data.Dataset`
            Contains the subjects' observations in torch format to speed up computation.

        Returns
        -------
        :class:`~leaspy.variables.state.State` :
            The fitted state.
        """
        with self._device_manager(model, dataset):
            state = self._initialize_algo(model, dataset)
            if self.algo_parameters["progress_bar"]:
                self._display_progress_bar(
                    -1, self.algo_parameters["n_iter"], suffix="iterations"
                )
            for self.current_iteration in range(1, self.algo_parameters["n_iter"] + 1):
                self._iteration(model, state)

                if self.output_manager is not None:
                    # print/plot first & last iteration!
                    # <!> everything that will be printed/saved is AFTER iteration N
                    # (including temperature when annealing...)
                    self.output_manager.iteration(self, model, dataset)

                if self.algo_parameters["progress_bar"]:
                    self._display_progress_bar(
                        self.current_iteration - 1,
                        self.algo_parameters["n_iter"],
                        suffix="iterations",
                    )
        model.fit_metrics = self._get_fit_metrics()
        model_state = state.clone()
        with model_state.auto_fork(None):
            # <!> At the end of the MCMC, population and individual latent variables
            # may have diverged from final model parameters.
            # Thus, we reset population latent variables to their mode
            model_state.put_population_latent_variables(
                LatentVariableInitType.PRIOR_MODE
            )
        model.state = model_state
        return state

    def _initialize_algo(
        self,
        model: McmcSaemCompatibleModel,
        dataset: Dataset,
    ) -> State:
        # TODO? mutualize with perso mcmc algo?
        state = model.state
        with state.auto_fork(None):
            model.put_data_variables(state, dataset)
        # Initialize individual latent variables (population ones should be initialized before)
        model.put_individual_parameters(state, dataset)
        self._initialize_samplers(state, dataset)
        self._initialize_annealing()
        return state

    def _iteration(
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

    def _maximization_step(self, model: McmcSaemCompatibleModel, state: State):
        """Maximization step as in the EM algorithm.

        In practice parameters are set to current state (burn-in phase), or as a barycenter with previous state.

        Parameters
        ----------
        model : :class:`~leaspy.models.McmcSaemCompatibleModel`

        state : :class:`~leaspy.variables.state.State`
        """
        # TODO/WIP: not 100% clear to me whether model methods should take a state param, or always use its internal state...
        sufficient_statistics = model.compute_sufficient_statistics(state)
        if (
            self._is_burn_in()
            or self.current_iteration == 1 + self.algo_parameters["n_burn_in_iter"]
        ):
            # the maximization step is memoryless (or first iteration with memory)
            self.sufficient_statistics = sufficient_statistics
        else:
            burn_in_step = (
                self.current_iteration - self.algo_parameters["n_burn_in_iter"]
            )  # min = 2, max = n_iter - n_burn_in_iter
            burn_in_step **= -self.algo_parameters["burn_in_step_power"]

            # this new formulation (instead of v + burn_in_step*(sufficient_statistics[k] - v))
            # enables to keep `inf` deltas
            self.sufficient_statistics = {
                k: v * (1.0 - burn_in_step) + burn_in_step * sufficient_statistics[k]
                for k, v in self.sufficient_statistics.items()
            }
        # TODO: use the same method in both cases (<!> very minor differences that might break
        #  exact reproducibility in tests)
        model.update_parameters(
            state, self.sufficient_statistics, burn_in=self._is_burn_in()
        )

    def log_current_iteration(self, state: State):
        if (
            self.is_current_iteration_in_last_n()
            or self.should_current_iteration_be_saved()
        ):
            state.save(
                self.logs.parameter_convergence_path,
                iteration=self.current_iteration,
            )

    def is_current_iteration_in_last_n(self):
        """Return True if current iteration is within the last n realizations defined in logging settings."""
        return (
            self.current_iteration
            > self.algo_parameters["n_iter"] - self.logs.save_last_n_realizations
        )

    def should_current_iteration_be_saved(self):
        """Return True if current iteration should be saved based on log saving periodicity."""
        return (
            self.logs.save_periodicity
            and self.current_iteration % self.logs.save_periodicity == 0
        )
