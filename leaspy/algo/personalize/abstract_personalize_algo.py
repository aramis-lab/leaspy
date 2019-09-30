from ..abstract_algo import AbstractAlgo
import numpy as np
import time


class AbstractPersonalizeAlgo(AbstractAlgo):
    """
    Abstract class for "personalize" algorithm.
    Estimation of individual parameters of a given "Data" file with
    a freezed model (already estimated, or loaded from known parameters)
    """

    def __init__(self, settings):
        """
        Initialize from settings object
        :param settings:
        """

        def __init__(self):
            super().__init__()

        # Name
        self.name = settings.name

        # Seed
        self.seed = settings.seed

        # Algorithm parameters
        self.algo_parameters = settings.parameters

    def _get_individual_parameters(self, model, data):
        """
        Estimate individual parameters from a data using a model.
        :param model:
        :param data:
        :return: tuple of tensors of indvidual variables
        """
        raise NotImplementedError('This algorithm does not present a personalization procedure')

    def run(self, model, data):
        """
        Main personalize function, wraps the _get_individual_parameters
        :param model:
        :param data:
        :return:
        """

        # Set seed
        self._initialize_seed(self.seed)

        # Init the run
        print("Beginning personalization : std error of the model is {0}".format(model.parameters['noise_std']))
        time_beginning = time.time()

        # Estimate individual parametersabstr
        individual_parameters = self._get_individual_parameters(model, data)

        # Compute the noise with the estimated individual paraeters
        squared_diff = model.compute_sum_squared_tensorized(data, individual_parameters).sum()
        noise_std = np.sqrt(float(squared_diff.detach().numpy()) / (data.n_visits * data.dimension))

        # Print run infos
        time_end = time.time()
        diff_time = (time_end - time_beginning) / 1000
        print("The standard deviation of the noise at the end of the personalization is of {:.4f}".format(noise_std))
        print("Personalization {1} took : {0}s".format(diff_time, self.name))

        # Transform individual parameters to dictinnary ID / variable_ind
        # indices = data.indices
        # new_individual_parameters = dict.fromkeys(indices)
        # for i, idx in enumerate(indices):
        #    new_individual_parameters[idx] = {}
        #    for j, variable_ind_name in enumerate(model.get_individual_variable_name()):
        #        new_individual_parameters[idx][variable_ind_name] = individual_parameters[j][i].detach().numpy()

        return individual_parameters, noise_std
