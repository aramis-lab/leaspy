import warnings

import torch

from leaspy.algo.fit.abstract_mcmc import AbstractFitMCMC


class GradientMCMCSAEM(AbstractFitMCMC):
    """
    .. deprecated:: 1.0
    """

    def __init__(self, settings):
        warnings.warn(
            "Gradient MCMC SAEM algorithm is deprecated. Please use `mcmc_saem` for your fits.",
            DeprecationWarning,
        )

        super().__init__(settings)
        self.name = "MCMC_SAEM (tensor)"

    def iteration(self, data, model, realizations):
        # Sample step
        if int(self.current_iteration) % 2 == 0:
            self._sample_population_realizations(data, model, realizations)
            self._sample_individual_realizations(data, model, realizations)

            # Annealing
            if self.algo_parameters["annealing"]["do_annealing"]:
                self._update_temperature()
        else:
            # set autograd
            for key in realizations.keys():
                realizations[key].set_autograd()
            self._gradient_population_update(data, model, realizations)

            # unset autograd
            for key in realizations.keys():
                realizations[key].unset_autograd()

        # Maximization step
        self._maximization_step(data, model, realizations)
        model.update_MCMC_toolbox(["all"], realizations)

    def _gradient_population_update(self, data, model, realizations):
        previous_attachment = model.compute_individual_attachment_tensorized_mcmc(
            data, realizations
        ).sum()
        previous_regularity = 0
        for key in realizations.keys():
            previous_regularity += model.compute_regularity_realization(
                realizations[key]
            ).sum()
        loss = previous_attachment + previous_regularity

        # Do backward and backprop on realizations
        loss.backward()

        # Update pop
        with torch.no_grad():
            for key in realizations.reals_pop_variable_names:
                eps = self.algo_parameters["learning_rate"] / data.n_individuals
                realizations[key].tensor_realizations -= (
                    eps * realizations[key].tensor_realizations.grad
                )
                realizations[key].tensor_realizations.grad.zero_()

        """
        # Update ind
        with torch.no_grad():
            for key in realizations.reals_ind_variable_names:
                eps = self.algo_parameters['learning_rate']
                realizations[key].tensor_realizations -= eps * realizations[key].tensor_realizations.grad
                realizations[key].tensor_realizations.grad.zero_()"""
