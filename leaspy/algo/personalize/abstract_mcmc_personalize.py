from __future__ import annotations
from typing import TYPE_CHECKING
from random import shuffle
from copy import copy
from abc import abstractmethod

import torch

from leaspy.algo.personalize.abstract_personalize_algo import AbstractPersonalizeAlgo
from leaspy.algo.utils.algo_with_samplers import AlgoWithSamplersMixin
from leaspy.algo.utils.algo_with_device import AlgoWithDeviceMixin
from leaspy.algo.utils.algo_with_annealing import AlgoWithAnnealingMixin
from leaspy.io.outputs.individual_parameters import IndividualParameters
from leaspy.utils.typing import DictParamsTorch

if TYPE_CHECKING:
    from leaspy.io.data.dataset import Dataset
    from leaspy.models.abstract_model import AbstractModel
from leaspy.io.realizations import CollectionRealization


class AbstractMCMCPersonalizeAlgo(AlgoWithAnnealingMixin, AlgoWithSamplersMixin, AlgoWithDeviceMixin, AbstractPersonalizeAlgo):
    """
    Base class for MCMC-based personalization algorithms.

    Individual parameters are derived from realizations of individual variables of the model.

    Parameters
    ----------
    settings : :class:`.AlgorithmSettings`
        Settings of the algorithm.
    """

    @abstractmethod
    def _compute_individual_parameters_from_samples_torch(self,
            realizations: DictParamsTorch,
            attachments: torch.FloatTensor,
            regularities: torch.FloatTensor) -> DictParamsTorch:
        """
        Compute dictionary of individual parameters from stacked realizations, attachments and regularities.

        Parameters
        ----------
        realizations : dict[ind_var_name: str, `torch.FloatTensor` of shape (n_iter, n_individuals, *ind_var.shape)]
            The stacked history of realizations for individual variables.
        attachments : `torch.FloatTensor` of shape (n_iter, n_individuals)
            The stacked history of attachments.
        regularities : `torch.FloatTensor` of shape (n_iter, n_individuals)
            The stacked history of regularities (sum on all individual variables / dimensions).

        Returns
        -------
        dict[ind_var_name: str, `torch.FloatTensor` of shape (n_individuals, *ind_var.shape)]
        """

    def _get_individual_parameters(self, model: AbstractModel, dataset: Dataset):

        # Initialize realizations storage object
        realizations_history = []
        attachment_history = []
        regularity_history = []

        # We are not in a calibration any more so attribute_type=None (NOT using MCMC toolbox, since it is undefined!)
        computation_kws = dict(attribute_type=None)

        with self._device_manager(model, dataset):
            # Initialize samplers
            self._initialize_samplers(model, dataset)

            # Initialize Annealing
            self._initialize_annealing()

            # Initialize realizations, only for individual variables, and at their mean values
            realizations = CollectionRealization()
            realizations.initialize_individuals(
                model, n_individuals=dataset.n_individuals, init_at_mean=True,
            )
            ind_vars_names = copy(realizations.individual.names)

            n_iter = self.algo_parameters['n_iter']
            if self.algo_parameters.get('progress_bar', True):
                self._display_progress_bar(-1, n_iter, suffix='iterations')

            # Gibbs sample `n_iter` times (only individual parameters)
            for self.current_iteration in range(1, n_iter+1):

                if self.random_order_variables:
                    shuffle(ind_vars_names)  # shuffle in-place!

                last_attachment = None
                tot_regularities = 0.
                for ind_var_name in ind_vars_names:
                    last_attachment, regularity_var = self.samplers[ind_var_name].sample(
                        dataset, model, realizations, self.temperature_inv, **computation_kws
                    )
                    tot_regularities += regularity_var

                # Append current realizations if "burn-in phase" is finished
                if not self._is_burn_in():
                    realizations_history.append(realizations.clone())
                    attachment_history.append(last_attachment.clone().detach())
                    regularity_history.append(tot_regularities.clone().detach())

                # Annealing
                self._update_temperature()

                # TODO? print(self) periodically? or refact OutputManager for not fit algorithms...

                if self.algo_parameters.get('progress_bar', True):
                    self._display_progress_bar(self.current_iteration - 1, n_iter, suffix='iterations')

            # Stack tensor realizations as well as attachments and tot_regularities
            torch_realizations = {
                ind_var_name: torch.stack(
                    [
                        realizations[ind_var_name].tensor for realizations in realizations_history
                    ]
                ) for ind_var_name in realizations.individual.names
            }
            torch_attachments = torch.stack(attachment_history)
            torch_tot_regularities = torch.stack(regularity_history)

            # Derive individual parameters from `realizations_history` list
            individual_parameters_torch = self._compute_individual_parameters_from_samples_torch(
                torch_realizations, torch_attachments, torch_tot_regularities
            )

            # Create the IndividualParameters object
            return IndividualParameters.from_pytorch(dataset.indices, individual_parameters_torch)
