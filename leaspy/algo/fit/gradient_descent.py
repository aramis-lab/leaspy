import torch
from leaspy.algo.fit.abstract_fit_algo import AbstractFitAlgo


class GradientDescent(AbstractFitAlgo):

    def __init__(self, settings):

        super().__init__()

        self.name = "Gradient Descent"
        self.realizations = None
        self.task = None
        self.algo_parameters = settings.parameters
        self.current_iteration = 0
        self.path_output = 'output/'

    ###########################
    ## Initialization
    ###########################

    def _initialize_algo(self, data, model, realizations):
        # MCMC toolbox (cache variables for speed-ups + tricks)
        model.initialize_MCMC_toolbox()
        self._initialize_torchvariables(realizations)
        return realizations

    def _initialize_torchvariables(self, realizations):
        for name, realization in realizations.realizations.items():
            realization.to_torch_Variable()

    ###########################
    ## Core
    ###########################

    def iteration(self, data, model, realizations):

        # Update intermediary model variables if necessary
        model.update_MCMC_toolbox(["all"], realizations)

        # Compute loss
        previous_attachment = model.compute_individual_attachment_tensorized_mcmc(data, realizations).sum()
        previous_regularity = 0
        for key in realizations.keys():
            previous_regularity += model.compute_regularity_realization(realizations[key]).sum()
        loss = previous_attachment + previous_regularity

        # Do backward and backprop on realizations
        loss.backward()

        # Update pop
        with torch.no_grad():
            for key in realizations.reals_pop_variable_names:
                eps = self.algo_parameters['learning_rate'] / data.n_individuals
                realizations[key].tensor_realizations -= eps * realizations[key].tensor_realizations.grad
                realizations[key].tensor_realizations.grad.zero_()

        # Update ind
        with torch.no_grad():
            for key in realizations.reals_ind_variable_names:
                eps = self.algo_parameters['learning_rate'] / data.n_individuals
                realizations[key].tensor_realizations -= eps * realizations[key].tensor_realizations.grad
                realizations[key].tensor_realizations.grad.zero_()

        # Update the sufficient statistics
        self._maximization_step(data, model, realizations)
