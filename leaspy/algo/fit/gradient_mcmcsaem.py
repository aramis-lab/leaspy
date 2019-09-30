import torch
from .abstract_mcmc import AbstractFitMCMC


class GradientMCMCSAEM(AbstractFitMCMC):

    def __init__(self, settings):
        super().__init__(settings)
        self.name = "MCMC_SAEM (tensor)"

    def iteration(self, data, model, realizations):

        # Sample step
        if int(self.current_iteration) % 2 == 0:
            self._sample_population_realizations(data, model, realizations)
            self._sample_individual_realizations(data, model, realizations)

            # Annealing
            if self.algo_parameters['annealing']['do_annealing']:
                self._update_temperature()
        else:
            # Torch variable
            for key in realizations.keys():
                realizations[key].to_torch_Variable()
            self._gradient_population_update(data, model, realizations)

            # back to tensors
            for key in realizations.keys():
                realizations[key].to_torch_Tensor()

        # Maximization step
        self._maximization_step(data, model, realizations)
        model.update_MCMC_toolbox(['all'], realizations)

    def _gradient_population_update(self, data, model, realizations):

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

        """
        # Update ind
        with torch.no_grad():
            for key in realizations.reals_ind_variable_names:
                eps = self.algo_parameters['learning_rate'] 
                realizations[key].tensor_realizations -= eps * realizations[key].tensor_realizations.grad
                realizations[key].tensor_realizations.grad.zero_()"""
