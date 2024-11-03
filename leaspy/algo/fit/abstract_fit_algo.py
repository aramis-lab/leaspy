from __future__ import annotations
from typing import TYPE_CHECKING, Dict, Optional
from abc import abstractmethod

from leaspy.algo.abstract_algo import AbstractAlgo
from leaspy.algo.utils.algo_with_device import AlgoWithDeviceMixin

from leaspy.utils.typing import DictParamsTorch
from leaspy.exceptions import LeaspyAlgoInputError

from leaspy.io.realizations import CollectionRealization

if TYPE_CHECKING:
    from leaspy.io.data.dataset import Dataset
    from leaspy.models.abstract_model import AbstractModel


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

    family = "fit"

    def __init__(self, settings):

        super().__init__(settings)

        # The algorithm is proven to converge if the sequence `burn_in_step` is positive, with an infinite sum \sum
        # (\sum_k \epsilon_k = + \infty) but a finite sum of the squares (\sum_k \epsilon_k^2 < \infty )
        # cf page 657 of the book that contains the paper
        # "Construction of Bayesian deformable models via a stochastic approximation algorithm: a convergence study"
        if not (0.5 < self.algo_parameters['burn_in_step_power'] <= 1):
            raise LeaspyAlgoInputError("The parameter `burn_in_step_power` should be in ]0.5, 1] in order to "
                                       "have theoretical guarantees on convergence of MCMC-SAEM algorithm.")

        self.current_iteration: int = 0

        self.sufficient_statistics: DictParamsTorch = None

    ###########################
    # Core
    ###########################

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
            * realizations : :class:`~.io.realizations.collection_realization.CollectionRealization`
                The optimized parameters.
            * None : placeholder for noise-std
        """

        with self._device_manager(model, dataset):
            realizations = CollectionRealization()
            realizations.initialize(model, n_individuals=dataset.n_individuals, **self.algo_parameters.get('init_params', {}))
            self._initialize_algo(dataset, model)

            if self.algo_parameters['progress_bar']:
                self._display_progress_bar(-1, self.algo_parameters['n_iter'], suffix='iterations')

            # Iterate
            for self.current_iteration in range(1, self.algo_parameters['n_iter']+1):

                self.iteration(dataset, model, realizations)

                if self.output_manager is not None:
                    # print/plot first & last iteration!
                    # <!> everything that will be printed/saved is AFTER iteration N (including
                    # temperature when annealing...)
                    self.output_manager.iteration(self, dataset, model, realizations)

                if self.algo_parameters['progress_bar']:
                    self._display_progress_bar(
                        self.current_iteration - 1,
                        self.algo_parameters['n_iter'],
                        suffix='iterations',
                    )

            # Finally we compute model attributes once converged
            model.attributes.update({'all'}, model.parameters)

        # TODO: finalize metrics handling
        # we store metrics after the fit so they can be exported along with model
        # parameters & hyper-parameters for archive...
        model.fit_metrics = self._get_fit_metrics()

        # TODO: Shouldn't we always return (nll_tot, nll_attach, nll_regul_tot or nll_regul_{ind_param},
        #  and parameters of noise-model if any)
        # If noise-model is a 1-parameter distribution family final loss is the value of this parameter
        # Otherwise we use the negative log-likelihood as measure of goodness-of-fit
        if len(model.noise_model.free_parameters) == 1:
            loss = next(iter(model.noise_model.parameters.values()))
        else:
            # TODO? rather return nll_tot (unlike previously)
            loss = self.sufficient_statistics.get("nll_attach", -1.)

        return realizations, loss

    def _get_fit_metrics(self) -> Optional[Dict[str, float]]:
        # TODO: finalize metrics handling, a bit dirty to place them in sufficient stats, only with a prefix...
        if self.sufficient_statistics is None:
            return
        return {
            # (scalars only)
            k: v.item() for k, v in self.sufficient_statistics.items()
            if k.startswith('nll_')
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
    def iteration(self, dataset: Dataset, model: AbstractModel, realizations: CollectionRealization):
        """
        Update the parameters (abstract method).

        Parameters
        ----------
        dataset : :class:`.Dataset`
            Contains the subjects' observations in torch format to speed-up computation.
        model : :class:`~.models.abstract_model.AbstractModel`
            The used model.
        realizations : :class:`~.io.realizations.collection_realization.CollectionRealization`
            The parameters.
        """

    @abstractmethod
    def _initialize_algo(self, dataset: Dataset, model: AbstractModel) -> None:
        """
        Initialize the fit algorithm (abstract method).

        Parameters
        ----------
        dataset : :class:`.Dataset`
        model : :class:`~.models.abstract_model.AbstractModel`
        """

    def _maximization_step(self, dataset: Dataset, model: AbstractModel, realizations: CollectionRealization):
        """
        Maximization step as in the EM algorithm. In practice parameters are set to current realizations (burn-in phase),
        or as a barycenter with previous realizations.

        Parameters
        ----------
        dataset : :class:`.Dataset`
        model : :class:`.AbstractModel`
        realizations : :class:`.CollectionRealization`
        """
        sufficient_statistics = model.compute_sufficient_statistics(dataset, realizations)

        if self._is_burn_in() or self.current_iteration == 1 + self.algo_parameters['n_burn_in_iter']:
            # the maximization step is memoryless (or first iteration with memory)
            self.sufficient_statistics = sufficient_statistics
        else:
            burn_in_step = self.current_iteration - self.algo_parameters['n_burn_in_iter'] # min = 2, max = n_iter - n_burn_in_iter
            burn_in_step **= -self.algo_parameters['burn_in_step_power']

            # this new formulation (instead of v + burn_in_step*(sufficient_statistics[k] - v))
            # enables to keep `inf` deltas
            self.sufficient_statistics = {
                k: v * (1. - burn_in_step) + burn_in_step * sufficient_statistics[k]
                for k, v in self.sufficient_statistics.items()
            }

        # TODO: use the same method in both cases (<!> very minor differences that might break
        #  exact reproducibility in tests)
        if self._is_burn_in():
            model.update_parameters_burn_in(dataset, self.sufficient_statistics)
        else:
            model.update_parameters_normal(dataset, self.sufficient_statistics)

        # No need to update model attributes (derived from model parameters)
        # since all model computations are done with the MCMC toolbox during calibration
