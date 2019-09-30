import torch
from leaspy.utils.realizations.collection_realization import CollectionRealization
import math

from leaspy.utils.realizations.realization import Realization

TWO_PI = 2 * math.pi


class AbstractModel():
    def __init__(self, name):
        self.is_initialized = False
        self.name = name
        self.parameters = None
        self.distribution = torch.distributions.normal.Normal(loc=0., scale=0.)

    def load_parameters(self, parameters):
        self.parameters = {}
        for k in parameters.keys():
            self.parameters[k] = parameters[k]

    def load_hyperparameters(self, hyperparameters):
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError

    def get_individual_variable_name(self):
        """
        Return list of names of the individual variables from the model
        :return:
        """

        individual_variable_name = []

        infos = self.random_variable_informations()
        for name, info in infos.items():
            if info['type'] == 'individual':
                individual_variable_name.append(name)

        return individual_variable_name

    def compute_sum_squared_tensorized(self, data, param_ind, attribute_type=None):
        res = self.compute_individual_tensorized(data.timepoints, param_ind, attribute_type)
        res *= data.mask
        return torch.sum((res * data.mask - data.values) ** 2, dim=(1, 2))

    def compute_individual_attachment_tensorized_mcmc(self, data, realizations):
        param_ind = self.get_param_from_real(realizations)
        attachment = self.compute_individual_attachment_tensorized(data, param_ind, attribute_type='MCMC')
        return attachment

    def compute_individual_attachment_tensorized(self, data, param_ind, attribute_type):
        res = self.compute_individual_tensorized(data.timepoints, param_ind, attribute_type)
        # res *= data.mask

        r1 = res * data.mask - data.values
        squared_sum = torch.sum(r1 * r1, dim=(1, 2))

        # noise_var = self.parameters['noise_std'] ** 2
        noise_var = self.parameters['noise_std'] * self.parameters['noise_std']
        attachment = 0.5 * (1 / noise_var) * squared_sum

        attachment += torch.log(torch.sqrt(TWO_PI * noise_var))
        return attachment

    def update_model_parameters(self, data, suff_stats, burn_in_phase=True):
        # Memoryless part of the algorithm
        if burn_in_phase:
            self.update_model_parameters_burn_in(data, suff_stats)
        # Stochastic sufficient statistics used to update the parameters of the model
        else:
            self.update_model_parameters_normal(data, suff_stats)
        self.attributes.update(['all'], self.parameters)

    def update_model_parameters_burn_in(self, data, realizations):
        raise NotImplementedError

    def get_population_realization_names(self):
        return [name for name, value in self.random_variable_informations().items() if value['type'] == 'population']

    def get_individual_realization_names(self):
        return [name for name, value in self.random_variable_informations().items() if value['type'] == 'individual']

    def __str__(self):
        output = "=== MODEL ===\n"
        for key in self.parameters.keys():
            # if type(self.parameters[key]) == float:
            #    output += "{} : {:.5f}\n".format(key, self.parameters[key])
            # else:
            output += "{} : {}\n".format(key, self.parameters[key])
        return output

    def compute_regularity_realization(self, realization):
        # Instanciate torch distribution
        if realization.variable_type == 'population':
            mean = self.parameters[realization.name]
            # TODO : Sure it is only MCMC_toolbox?
            std = self.MCMC_toolbox['priors']['{0}_std'.format(realization.name)]
        elif realization.variable_type == 'individual':
            mean = self.parameters["{0}_mean".format(realization.name)]
            std = self.parameters["{0}_std".format(realization.name)]
        else:
            raise ValueError("Variable type not known")

        return self.compute_regularity_variable(realization.tensor_realizations, mean, std)

    def compute_regularity_variable(self, value, mean, std):
        # TODO change to static ???
        # Instanciate torch distribution
        # distribution = torch.distributions.normal.Normal(loc=mean, scale=std)

        self.distribution.loc = mean
        self.distribution.scale = std
        return -self.distribution.log_prob(value)

    def get_realization_object(self, n_individuals):
        ### TODO : CollectionRealizations should probably get self.get_info_var rather than all self
        realizations = CollectionRealization()
        realizations.initialize(n_individuals, self)
        return realizations

    def random_variable_informations(self):
        raise NotImplementedError

    def smart_initialization_realizations(self, data, realizations):
        return realizations

    def _create_dictionary_of_population_realizations(self):
        pop_dictionary = {}
        for name_var, info_var in self.random_variable_informations().items():
            if info_var['type'] != "population":
                continue
            real = Realization.from_tensor(name_var, info_var['shape'], info_var['type'], self.parameters[name_var])
            pop_dictionary[name_var] = real

        return pop_dictionary

    def time_reparametrization(self, timepoints, xi, tau):
        return torch.exp(xi) * (timepoints - tau)

    def get_param_from_real(self, realizations):

        individual_parameters = dict.fromkeys(self.get_individual_variable_name())

        for variable_ind in self.get_individual_variable_name():
            if variable_ind == "sources" and self.source_dimension == 0:
                individual_parameters[variable_ind] = None
            else:
                individual_parameters[variable_ind] = realizations[variable_ind].tensor_realizations

        return individual_parameters
