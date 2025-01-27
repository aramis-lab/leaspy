import warnings

import torch

from leaspy.algo.personalize.abstract_personalize_algo import AbstractPersonalizeAlgo
from leaspy.io.outputs.individual_parameters import IndividualParameters


class GradientDescentPersonalize(AbstractPersonalizeAlgo):
    def __init__(self, settings):
        # Algorithm parameters
        super().__init__(settings)

        # Deprecation warning: but with nans
        warnings.warn(
            "Personalization algorithm 'gradient_descent_personalize' is deprecated "
            "due to issues with nans in multivariate logistic models. "
            "Try another personalization algorithm such as 'scipy_minimize'.",
            DeprecationWarning,
        )

    def _initialize_torchvariables(self, realizations):
        for name, realization in realizations.realizations.items():
            realization.set_autograd()

    def _get_individual_parameters(self, model, data):
        # TODO : warnings / stop if nans in the gradient

        # initialize realizations
        realizations = model.initialize_realizations_for_model(data.n_individuals)

        # To torch vraibles
        self._initialize_torchvariables(realizations)

        # Do n_iter gradient update
        for iteration in range(self.algo_parameters["n_iter"]):
            # Compute loss
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

            # Update ind
            with torch.no_grad():
                for key in realizations.reals_ind_variable_names:
                    eps = self.algo_parameters["learning_rate"]
                    realizations[key].tensor_realizations -= (
                        eps * realizations[key].tensor_realizations.grad
                    )
                    realizations[key].tensor_realizations.grad.zero_()

        # Get individual realizations from realizations object
        param_ind = model.get_param_from_real(realizations)

        ### TODO : The following was adding for the conversion from Results to IndividualParameters. Everything should be changed

        individual_parameters = IndividualParameters()
        p_names = list(param_ind.keys())
        n_sub = len(param_ind[p_names[0]])

        for i in range(n_sub):
            p_dict = {k: param_ind[k][i].detach().numpy() for k in p_names}
            p_dict = {k: v[0] if v.shape[0] == 1 else v for k, v in p_dict.items()}
            individual_parameters.add_individual_parameters(str(i), p_dict)

        return individual_parameters
