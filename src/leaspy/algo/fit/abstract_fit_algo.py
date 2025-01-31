from abc import abstractmethod
from typing import Any, Dict, Optional

from leaspy.exceptions import LeaspyAlgoInputError
from leaspy.io.data.dataset import Dataset
from leaspy.models import AbstractModel
from leaspy.utils.typing import DictParamsTorch
from leaspy.variables.specs import LatentVariableInitType
from leaspy.variables.state import State

from ..base import AbstractAlgo, AlgorithmType
from ..settings import AlgorithmSettings
from ..utils import AlgoWithDeviceMixin

__all__ = ["AbstractFitAlgo"]


class AbstractFitAlgo(AlgoWithDeviceMixin, AbstractAlgo):
    """
    Abstract class containing common method for all `fit` algorithm classes.

    Parameters
    ----------
    settings : :class:`.AlgorithmSettings`
        The specifications of the algorithm as a :class:`.AlgorithmSettings` instance.

    Attributes
    ----------
    algorithm_device : str
        Valid torch device
    current_iteration : int, default 0
        The number of the current iteration.
        The first iteration will be 1 and the last one `n_iter`.
    sufficient_statistics : dict[str, `torch.FloatTensor`] or None
        The previous step sufficient statistics.
        It is None during all the burn-in phase.
    Inherited attributes
        From :class:`.AbstractAlgo`

    See Also
    --------
    :meth:`.Leaspy.fit`
    """

    family = AlgorithmType.FIT

    def __init__(self, settings: AlgorithmSettings):
        super().__init__(settings)
        self.logs = settings.logs
        # The algorithm is proven to converge if the sequence `burn_in_step` is positive, with an infinite sum \sum
        # (\sum_k \epsilon_k = + \infty) but a finite sum of the squares (\sum_k \epsilon_k^2 < \infty )
        # cf page 657 of the book that contains the paper
        # "Construction of Bayesian deformable models via a stochastic approximation algorithm: a convergence study"
        if not (0.5 < self.algo_parameters["burn_in_step_power"] <= 1):
            raise LeaspyAlgoInputError(
                "The parameter `burn_in_step_power` should be in ]0.5, 1] in order to "
                "have theoretical guarantees on convergence of MCMC-SAEM algorithm."
            )
        self.current_iteration: int = 0
        self.sufficient_statistics: Optional[DictParamsTorch] = None

    def run_impl(self, model: AbstractModel, dataset: Dataset):
        """
        Main method, run the algorithm.

        Basically, it initializes the :class:`~.io.realizations.collection_realization.CollectionRealization` object,
        updates it using the `iteration` method then returns it.

        TODO fix proper abstract class

        Parameters
        ----------
        model : :class:`~.models.abstract_model.AbstractModel`
            The used model.
        dataset : :class:`.Dataset`
            Contains the subjects' observations in torch format to speed up computation.

        Returns
        -------
        2-tuple:
            * state : :class:`.State`
            * loss : Any
        """
        with self._device_manager(model, dataset):
            state = self._initialize_algo(model, dataset)

            if self.algo_parameters["progress_bar"]:
                self._display_progress_bar(
                    -1, self.algo_parameters["n_iter"], suffix="iterations"
                )
            if self.logs:
                state.save(self.logs.parameter_convergence_path, iteration=0)

            for self.current_iteration in range(1, self.algo_parameters["n_iter"] + 1):
                self.iteration(model, state)
                if self.logs:
                    self.log_current_iteration(state)

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

        loss = self._terminate_algo(model, state)
        return state, loss

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
    def iteration(self, model: AbstractModel, state: State):
        """
        Update the model parameters (abstract method).

        Parameters
        ----------
        model : :class:`~.models.abstract_model.AbstractModel`
            The used model.
        state : :class:`.State`
            During the fit, this state holds all model variables, together with dataset observations.
        """

    def _initialize_algo(self, model: AbstractModel, dataset: Dataset) -> State:
        """
        Initialize the fit algorithm (abstract method) and return the state to work on.

        Parameters
        ----------
        model : :class:`.AbstractModel`
        dataset : :class:`.Dataset`

        Returns
        -------
        state : :class:`.State`
        """

        # WIP: Would it be relevant to fit on a dedicated algo state?
        state = model.state
        with state.auto_fork(None):
            # Set data variables
            model.put_data_variables(state, dataset)

        return state

    def _terminate_algo(
        self, model: AbstractModel, state: State
    ) -> Any:  # torch.Tensor?
        """
        Perform the last steps upon terminaison of algorithm (cleaning stuff, ...).

        Parameters
        ----------
        model : :class:`.AbstractModel`
        state : :class:`.State`

        Returns
        -------
        loss : Any
        """

        # TODO: finalize metrics handling
        # we store metrics after the fit so they can be exported along with model
        # parameters & hyper-parameters for archive...
        model.fit_metrics = self._get_fit_metrics()

        ## TODO: Shouldn't we always return (nll_tot, nll_attach, nll_regul_tot or nll_regul_{ind_param},
        ##  and parameters of noise-model if any)
        ## If noise-model is a 1-parameter distribution family final loss is the value of this parameter
        ## Otherwise we use the negative log-likelihood as measure of goodness-of-fit
        # if len(model.noise_model.free_parameters) == 1:
        #    loss = next(iter(model.noise_model.parameters.values()))
        # else:
        #    # TODO? rather return nll_tot (unlike previously)
        #    loss = self.sufficient_statistics.get("nll_attach", -1.)
        #
        loss = -1.0

        # WIP: cf. interrogation about internal state in model or not...
        model_state = state.clone()
        # TODO? Should those cleaning steps be performed here, or in the model.state setter instead?
        with model_state.auto_fork(None):
            # model.reset_data_variables(model_state)
            # <!> At the end of the MCMC, population and individual latent variables may have diverged from final model parameters
            # Thus we reset population latent variables to their mode, and we remove individual latent variables
            model_state.put_population_latent_variables(
                LatentVariableInitType.PRIOR_MODE
            )
            # model_state.put_individual_latent_variables(None)
        model.state = model_state

        return loss

    def _maximization_step(self, model: AbstractModel, state: State):
        """
        Maximization step as in the EM algorithm. In practice parameters are set to current realizations (burn-in phase),
        or as a barycenter with previous realizations.

        Parameters
        ----------
        dataset : :class:`.Dataset`
        model : :class:`.AbstractModel`
        realizations : :class:`.CollectionRealization`
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
