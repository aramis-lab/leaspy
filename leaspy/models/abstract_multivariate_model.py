import json
import torch
import numpy as np

from .abstract_model import AbstractModel
from leaspy.utils.realizations.realization import Realization
from leaspy.models.utils.attributes.attributes_factory import AttributesFactory
from leaspy.models.utils.initialization.model_initialization import initialize_parameters


class AbstractMultivariateModel(AbstractModel):
    def __init__(self, name):
        super(AbstractMultivariateModel, self).__init__(name)
        self.source_dimension = None
        self.dimension = None
        self.parameters = {
            "g": None,
            "betas": None,
            "tau_mean": None, "tau_std": None,
            "xi_mean": None, "xi_std": None,
            "sources_mean": None, "sources_std": None,
            "noise_std": None
        }
        self.bayesian_priors = None
        self.attributes = None

        # MCMC related "parameters"
        self.MCMC_toolbox = {
            'attributes': None,
            'priors': {
                'g_std': None,  # tq p0 = 1 / (1+exp(g)) i.e. g = 1/p0 - 1
                'betas_std': None
            }
        }

    def smart_initialization_realizations(self, data, realizations):
        # TODO : Qui a fait ça? A quoi ça sert?
        # means_time = torch.Tensor([torch.mean(data.get_times_patient(i)) for i in range(data.n_individuals)]).reshape(realizations['tau'].tensor_realizations.shape)
        # realizations['tau'].tensor_realizations = means_time
        return realizations

    def initialize(self, dataset, method="default"):
        self.dimension = dataset.dimension

        if self.source_dimension is None:
            self.source_dimension = int(np.sqrt(dataset.dimension))

        self.parameters = initialize_parameters(self, dataset, method)

        self.attributes = AttributesFactory.attributes(self.name, self.dimension, self.source_dimension)
        self.attributes.update(['all'], self.parameters)
        self.is_initialized = True

    def load_hyperparameters(self, hyperparameters):
        if 'dimension' in hyperparameters.keys():
            self.dimension = hyperparameters['dimension']
        if 'source_dimension' in hyperparameters.keys():
            self.source_dimension = hyperparameters['source_dimension']

    def save(self, path):
        model_parameters_save = self.parameters.copy()
        model_parameters_save['mixing_matrix'] = self.attributes.mixing_matrix
        for key, value in model_parameters_save.items():
            if type(value) in [torch.Tensor]:
                model_parameters_save[key] = value.tolist()
        model_settings = {
            'name': self.name,
            'dimension': self.dimension,
            'source_dimension': self.source_dimension,
            'parameters': model_parameters_save
        }
        with open(path, 'w') as fp:
            json.dump(model_settings, fp)

    def compute_mean_traj(self, timepoints):
        individual_parameters = {
            'xi': torch.tensor([self.parameters['xi_mean']], dtype=torch.float32),
            'tau': torch.tensor([self.parameters['tau_mean']], dtype=torch.float32),
            'sources': torch.zeros(self.source_dimension)
        }

        return self.compute_individual_tensorized(timepoints, individual_parameters)

    def _get_attributes(self, attribute_type):
        if attribute_type is None:
            return self.attributes.get_attributes()
        elif attribute_type == 'MCMC':
            return self.MCMC_toolbox['attributes'].get_attributes()
        else:
            raise ValueError("The specified attribute type does not exist : {}".format(attribute_type))
