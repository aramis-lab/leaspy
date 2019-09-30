from .abstract_personalize_algo import AbstractPersonalizeAlgo
import torch


class GradientDescentPersonalize(AbstractPersonalizeAlgo):

    def __init__(self, settings):

        # Algorithm parameters
        super().__init__(settings)

    def _initialize_torchvariables(self, realizations):
        for name, realization in realizations.realizations.items():
            realization.to_torch_Variable()

    def _get_individual_parameters(self, model, data):

        # TODO : warnings / stop if nans in the gradient

        # initialize realizations
        realizations = model.get_realization_object(data.n_individuals)

        # To torch vraibles
        self._initialize_torchvariables(realizations)

        # Do n_iter gradient update
        for iteration in range(self.algo_parameters['n_iter']):

            # Compute loss
            previous_attachment = model.compute_individual_attachment_tensorized_mcmc(data, realizations).sum()
            previous_regularity = 0
            for key in realizations.keys():
                previous_regularity += model.compute_regularity_realization(realizations[key]).sum()
            loss = previous_attachment + previous_regularity

            # Do backward and backprop on realizations
            loss.backward()

            # Update ind
            with torch.no_grad():
                for key in realizations.reals_ind_variable_names:
                    eps = self.algo_parameters['learning_rate']
                    realizations[key].tensor_realizations -= eps * realizations[key].tensor_realizations.grad
                    realizations[key].tensor_realizations.grad.zero_()

        # Get individual realizations from realizations object
        param_ind = model.get_param_from_real(realizations)

        return param_ind
