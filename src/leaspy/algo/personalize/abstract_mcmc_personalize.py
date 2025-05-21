"""This module defines the `AbstractMCMCPersonalizeAlgo` class used for sampler based personalize algorithms."""

from abc import abstractmethod
from random import shuffle

import torch

from leaspy.io.data import Dataset
from leaspy.io.outputs.individual_parameters import IndividualParameters
from leaspy.models import AbstractModel
from leaspy.utils.typing import DictParamsTorch
from leaspy.variables.specs import IndividualLatentVariable, LatentVariableInitType
from leaspy.variables.state import State

from ..utils import AlgoWithAnnealingMixin, AlgoWithDeviceMixin, AlgoWithSamplersMixin
from .abstract_personalize_algo import AbstractPersonalizeAlgo

__all__ = ["AbstractMCMCPersonalizeAlgo"]


class AbstractMCMCPersonalizeAlgo(
    AlgoWithAnnealingMixin,
    AlgoWithSamplersMixin,
    AlgoWithDeviceMixin,
    AbstractPersonalizeAlgo,
):
    """
    Base class for MCMC-based personalization algorithms.

    Individual parameters are derived from values of individual variables of the model.

    Parameters
    ----------
    settings : :class:`.AlgorithmSettings`
        Settings of the algorithm.
    """

    @abstractmethod
    def _compute_individual_parameters_from_samples_torch(
        self,
        values: DictParamsTorch,
        attachments: torch.Tensor,
        regularities: torch.Tensor,
    ) -> DictParamsTorch:
        """
        Compute dictionary of individual parameters from stacked values, attachments and regularities.

        Parameters
        ----------
        values : dict[ind_var_name: str, `torch.Tensor[float]` of shape (n_iter, n_individuals, *ind_var.shape)]
            The stacked history of values for individual latent variables.
        attachments : `torch.Tensor[float]` of shape (n_iter, n_individuals)
            The stacked history of attachments (per individual).
        regularities : `torch.Tensor[float]` of shape (n_iter, n_individuals)
            The stacked history of regularities (per individual; but summed on all individual variables and all of their dimensions).

        Returns
        -------
        dict[ind_var_name: str, `torch.Tensor[float]` of shape (n_individuals, *ind_var.shape)]
        """

    def _initialize_algo(
        self,
        model: AbstractModel,
        dataset: Dataset,
    ) -> State:
        """
        Initialize the individual latent variables in state, the algo samplers & the annealing.

        TODO? mutualize some code with leaspy.algo.fit.abstract_mcmc? (<!> `LatentVariableInitType` is different in personalization)

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
            model.put_data_variables(state, dataset)
            # Initialize individual latent variables at their mode
            # (population ones should be initialized before)
            state.put_individual_latent_variables(
                LatentVariableInitType.PRIOR_MODE, n_individuals=dataset.n_individuals
            )
        self._initialize_samplers(state, dataset)
        self._initialize_annealing()

        return state

    def _terminate_algo(self, model: AbstractModel, state: State) -> None:
        """Clean-up of state at end of algorithm."""
        # WIP: cf. interrogation about internal state in model or not...
        model_state = state.clone()
        with model_state.auto_fork(None):
            model.reset_data_variables(model_state)
            model_state.put_individual_latent_variables(None)
        model.state = model_state

    def _get_individual_parameters(
        self,
        model: AbstractModel,
        dataset: Dataset,
    ) -> IndividualParameters:
        individual_variable_names = sorted(
            list(model.dag.sorted_variables_by_type[IndividualLatentVariable])
        )
        values_history = {name: [] for name in individual_variable_names}
        attachment_history = []
        regularity_history = []
        with self._device_manager(model, dataset):
            state = self._initialize_algo(model, dataset)
            n_iter = self.algo_parameters["n_iter"]
            if self.algo_parameters.get("progress_bar", True):
                self._display_progress_bar(-1, n_iter, suffix="iterations")
            # Gibbs sample `n_iter` times (only individual parameters)
            for self.current_iteration in range(1, n_iter + 1):
                if self.random_order_variables:
                    shuffle(individual_variable_names)
                for individual_variable_name in individual_variable_names:
                    self.samplers[individual_variable_name].sample(
                        state, temperature_inv=self.temperature_inv
                    )
                # Append current values if "burn-in phase" is finished
                if not self._is_burn_in():
                    for individual_variable_name in individual_variable_names:
                        values_history[individual_variable_name].append(
                            state[individual_variable_name]
                        )
                    attachment_history.append(state.get_tensor_value("nll_attach_ind"))
                    regularity_history.append(
                        state.get_tensor_value("nll_regul_ind_sum_ind")
                    )
                self._update_temperature()
                # TODO? print(self) periodically? or refact OutputManager for not fit algorithms...
                if self.algo_parameters.get("progress_bar", True):
                    self._display_progress_bar(
                        self.current_iteration - 1, n_iter, suffix="iterations"
                    )
            # Stack tensor values as well as attachments and tot_regularities
            torch_values = {
                individual_variable_name: torch.stack(individual_variable_values)
                for individual_variable_name, individual_variable_values in values_history.items()
            }
            torch_attachments = torch.stack(attachment_history)
            torch_tot_regularities = torch.stack(regularity_history)

            # TODO? we could also return the full posterior when credible intervals are needed
            # (but currently it would not fit with `IndividualParameters` structure, which expects point-estimates)
            # return torch_values, torch_attachments, torch_tot_regularities
            # Derive individual parameters from `values_history` list
            individual_parameters_torch = (
                self._compute_individual_parameters_from_samples_torch(
                    torch_values, torch_attachments, torch_tot_regularities
                )
            )
        self._terminate_algo(model, state)
        # Create the IndividualParameters object
        return IndividualParameters.from_pytorch(
            dataset.indices, individual_parameters_torch
        )
