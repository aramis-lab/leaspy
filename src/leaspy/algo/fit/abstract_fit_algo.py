"""This module defines the `AbstractFitAlgo` class used for fitting algorithms."""

from abc import abstractmethod
from typing import Dict, Optional

from leaspy.exceptions import LeaspyAlgoInputError
from leaspy.io.data.dataset import Dataset
from leaspy.models import McmcSaemCompatibleModel
from leaspy.utils.typing import DictParamsTorch
from leaspy.variables.specs import LatentVariableInitType
from leaspy.variables.state import State

from ..base import AbstractAlgo, AlgorithmType
from ..settings import AlgorithmSettings, OutputsSettings
from ..utils import AlgoWithDeviceMixin

__all__ = ["AbstractFitAlgo"]


class AbstractFitAlgo(AlgoWithDeviceMixin, AbstractAlgo):
    r"""
    Abstract class containing common method for all `fit` algorithm classes.

    The algorithm is proven to converge if the sequence `burn_in_step` is positive, with an
    infinite sum :math:`\sum_k \epsilon_k = +\infty` and a finite sum of the squares
    :math:`\sum_k \epsilon_k^2 < \infty` (see following paper).

    `Construction of Bayesian Deformable Models via a Stochastic Approximation Algorithm: A Convergence Study <https://arxiv.org/abs/0706.0787>`_

    Parameters
    ----------
    settings : :class:`~leaspy.algo.AlgorithmSettings`
        The specifications of the algorithm as a :class:`~leaspy.algo.AlgorithmSettings` instance.

    Attributes
    ----------
    algorithm_device : :obj:`str`
        Valid :class:`torch.device`
    current_iteration : :obj:`int`, default 0
        The number of the current iteration.
        The first iteration will be 1 and the last one `n_iter`.
    sufficient_statistics : :obj:`dict` [:obj:`str`, :class:`torch.Tensor`] or None
        Sufficient statistics of the previous step.
        It is None during all the burn-in phase.
    output_manager : :class:`~leaspy.io.logs.fit_output_manager.FitOutputManager`
        Optional output manager of the algorithm
    Inherited attributes
        From :class:`~leaspy.algo.AbstractAlgo`

    See Also
    --------
    :meth:`leaspy.api.Leaspy.fit`
    """

    family = AlgorithmType.FIT

    def __init__(self, settings: AlgorithmSettings):
        super().__init__(settings)
        self.logs = settings.logs
        if not (0.5 < self.algo_parameters["burn_in_step_power"] <= 1):
            raise LeaspyAlgoInputError(
                "The parameter `burn_in_step_power` should be in ]0.5, 1] in order to "
                "have theoretical guarantees on convergence of MCMC-SAEM algorithm."
            )
        self.current_iteration: int = 0
        self.sufficient_statistics: Optional[DictParamsTorch] = None

    def set_output_manager(self, output_settings: OutputsSettings) -> None:
        """
        Set a :class:`~leaspy.io.logs.FitOutputManager` object for the run of the algorithm

        Parameters
        ----------
        output_settings : :class:`~leaspy.algo.OutputsSettings`
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
            from .fit_output_manager import FitOutputManager

            self.output_manager = FitOutputManager(output_settings)

    def run_impl(self, model: McmcSaemCompatibleModel, dataset: Dataset):
        """
        Main method to run the algorithm.

        Basically, it initializes the :class:`~leaspy.variables.state.State` object,
        updates it using the :meth:`~leaspy.algo.AbstractFitAlgo.iteration` method then returns it.

        Parameters
        ----------
        model : :class:`~leaspy.models.McmcSaemCompatibleModel`
            The used model.
        dataset : :class:`~leaspy.io.data.Dataset`
            Contains the subjects' observations in torch format to speed up computation.

        Returns
        -------
        2-tuple:
            * state : :class:`~leaspy.variables.state.State`
        """
        with self._device_manager(model, dataset):
            state = self._initialize_algo(model, dataset)

            if self.algo_parameters["progress_bar"]:
                self._display_progress_bar(
                    -1, self.algo_parameters["n_iter"], suffix="iterations"
                )

            for self.current_iteration in range(1, self.algo_parameters["n_iter"] + 1):
                self.iteration(model, state)

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
            # <!> At the end of the MCMC, population and individual latent variables may have diverged from final model parameters
            # Thus we reset population latent variables to their mode
            model_state.put_population_latent_variables(
                LatentVariableInitType.PRIOR_MODE
            )
        model.state = model_state

        return state

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

    def _get_fit_metrics(self) -> Optional[Dict[str, float]]:
        # TODO: finalize metrics handling, a bit dirty to place them in sufficient stats, only with a prefix...
        if self.sufficient_statistics is None:
            return
        return {
            # (scalars only)
            k: v.item()
            for k, v in self.sufficient_statistics.items()
            if k.startswith("nll_")
        }

    def __str__(self) -> str:
        out = super().__str__()
        # add the fit metrics after iteration number (included the sufficient statistics for now...)
        fit_metrics = self._get_fit_metrics() or {}
        if len(fit_metrics):
            out += "\n= Metrics ="
            for m, v in fit_metrics.items():
                out += f"\n    {m} : {v:.5g}"

        return out

    @abstractmethod
    def iteration(self, model: McmcSaemCompatibleModel, state: State):
        """
        Update the model parameters (abstract method).

        Parameters
        ----------
        model : :class:`~leaspy.models.McmcSaemCompatibleModel`
            The used model.
        state : ::class:`~leaspy.variables.state.State`
            During the fit, this state holds all model variables, together with dataset observations.
        """

    def _initialize_algo(
        self, model: McmcSaemCompatibleModel, dataset: Dataset
    ) -> State:
        """
        Initialize the fit algorithm (abstract method) and return the state to work on.

        Parameters
        ----------
        model : :class:~leaspy.models.McmcSaemCompatibleModel
        dataset : :class:`~leaspy.io.data.Dataset`

        Returns
        -------
        state : :class:`~leaspy.variables.state.State`
        """
        # WIP: Would it be relevant to fit on a dedicated algo state?
        state = model.state
        with state.auto_fork(None):
            # Set data variables
            model.put_data_variables(state, dataset)

        return state

    def _maximization_step(self, model: McmcSaemCompatibleModel, state: State):
        """
        Maximization step as in the EM algorithm. In practice parameters are set to current state (burn-in phase),
        or as a barycenter with previous state.

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
