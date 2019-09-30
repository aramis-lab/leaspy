import torch

from .abstract_multivariate_model import AbstractMultivariateModel
from .utils.attributes.attributes_factory import AttributesFactory


class MultivariateModel(AbstractMultivariateModel):
    def __init__(self, name):
        super(MultivariateModel, self).__init__(name)
        self.parameters["v0"] = None
        self.MCMC_toolbox['priors']['v0_std'] = None  # Value, Coef

    def load_parameters(self, parameters):
        self.parameters = {}
        for k in parameters.keys():
            if k in ['mixing_matrix']:
                continue
            self.parameters[k] = torch.tensor(parameters[k])
        self.attributes = AttributesFactory.attributes(self.name, self.dimension, self.source_dimension)
        self.attributes.update(['all'], self.parameters)

    def compute_individual_tensorized(self, timepoints, ind_parameters, attribute_type=None):
        if self.name == 'logistic':
            return self.compute_individual_tensorized_logistic(timepoints, ind_parameters, attribute_type)
        elif self.name == 'linear':
            return self.compute_individual_tensorized_linear(timepoints, ind_parameters, attribute_type)
        else:
            raise ValueError("Mutivariate model > Compute individual tensorized")

    def compute_individual_tensorized_linear(self, timepoints, ind_parameters, attribute_type=None):
        positions, velocities, mixing_matrix = self._get_attributes(attribute_type)
        xi, tau, sources = ind_parameters['xi'], ind_parameters['tau'], ind_parameters['sources']
        reparametrized_time = self.time_reparametrization(timepoints, xi, tau)

        # Reshaping
        a = tuple([1] * reparametrized_time.ndimension())
        velocities = velocities.unsqueeze(0).repeat(*tuple(reparametrized_time.shape), 1)
        positions = positions.unsqueeze(0).repeat(*tuple(reparametrized_time.shape), 1)
        reparametrized_time = reparametrized_time.unsqueeze(-1).repeat(*a, velocities.shape[-1])

        # Computation
        LL = velocities * reparametrized_time + positions
        if self.source_dimension != 0:
            wi = torch.nn.functional.linear(sources, mixing_matrix, bias=None)
            LL += wi.unsqueeze(-2)
        return LL

    def compute_individual_tensorized_logistic(self, timepoints, ind_parameters, attribute_type=None):
        # Population parameters
        g, v0, a_matrix = self._get_attributes(attribute_type)
        b = (1. + g) * (1. + g) / g

        # Individual parameters
        xi, tau, sources = ind_parameters['xi'], ind_parameters['tau'], ind_parameters['sources']
        reparametrized_time = self.time_reparametrization(timepoints, xi, tau)

        # Log likelihood computation
        a = tuple([1] * reparametrized_time.ndimension())
        v0 = v0.unsqueeze(0).repeat(*tuple(reparametrized_time.shape), 1)
        reparametrized_time = reparametrized_time.unsqueeze(-1).repeat(*a, v0.shape[-1])

        LL = v0 * reparametrized_time
        if self.source_dimension != 0:
            wi = torch.nn.functional.linear(sources, a_matrix, bias=None)
            LL += wi.unsqueeze(-2)
        LL = 1. + g * torch.exp(-LL * b)
        model = 1. / LL
        return model

    ##############################
    ### MCMC-related functions ###
    ##############################

    def initialize_MCMC_toolbox(self):
        self.MCMC_toolbox = {
            'priors': {'g_std': 0.01, 'v0_std': 0.01, 'betas_std': 0.01},
            'attributes': AttributesFactory.attributes(self.name, self.dimension, self.source_dimension)
        }

        population_dictionary = self._create_dictionary_of_population_realizations()
        self.update_MCMC_toolbox(["all"], population_dictionary)

        # TODO maybe not here
        # Initialize priors
        self.MCMC_toolbox['priors']['v0_mean'] = self.parameters['v0'].clone().detach()
        self.MCMC_toolbox['priors']['s_v0'] = 0.1

    def update_MCMC_toolbox(self, name_of_the_variables_that_have_been_changed, realizations):
        L = name_of_the_variables_that_have_been_changed
        values = {}
        if any(c in L for c in ('g', 'all')):
            values['g'] = realizations['g'].tensor_realizations
        if any(c in L for c in ('v0', 'all')):
            values['v0'] = realizations['v0'].tensor_realizations
        if any(c in L for c in ('betas', 'all')) and self.source_dimension != 0:
            values['betas'] = realizations['betas'].tensor_realizations

        self.MCMC_toolbox['attributes'].update(name_of_the_variables_that_have_been_changed, values)

    def _center_xi_realizations(self, realizations):
        mean_xi = torch.mean(realizations['xi'].tensor_realizations)
        realizations['xi'].tensor_realizations = realizations['xi'].tensor_realizations - mean_xi
        realizations['v0'].tensor_realizations = realizations['v0'].tensor_realizations + mean_xi

        self.update_MCMC_toolbox(['all'], realizations)
        return realizations

    def compute_sufficient_statistics(self, data, realizations):
        # if self.name == 'logistic':
        realizations = self._center_xi_realizations(realizations)

        sufficient_statistics = {
            'g': realizations['g'].tensor_realizations.detach(),
            'v0': realizations['v0'].tensor_realizations.detach(),
            'tau': realizations['tau'].tensor_realizations,
            'tau_sqrd': torch.pow(realizations['tau'].tensor_realizations, 2),
            'xi': realizations['xi'].tensor_realizations,
            'xi_sqrd': torch.pow(realizations['xi'].tensor_realizations, 2)
        }
        if self.source_dimension != 0:
            sufficient_statistics['betas'] = realizations['betas'].tensor_realizations.detach()

        data_reconstruction = self.compute_individual_tensorized(data.timepoints,
                                                                 self.get_param_from_real(realizations),
                                                                 attribute_type='MCMC')

        # TODO : Remove norm_0 to directly get data.L2_norm in update_model_parameters
        norm_0 = data.values * data.values * data.mask
        norm_1 = data.values * data_reconstruction * data.mask
        norm_2 = data_reconstruction * data_reconstruction * data.mask
        sufficient_statistics['obs_x_obs'] = torch.sum(norm_0, dim=2)
        sufficient_statistics['obs_x_reconstruction'] = torch.sum(norm_1, dim=2)
        sufficient_statistics['reconstruction_x_reconstruction'] = torch.sum(norm_2, dim=2)

        return sufficient_statistics

    def update_model_parameters_burn_in(self, data, realizations):
        # if self.name == 'logistic':
        realizations = self._center_xi_realizations(realizations)

        # Memoryless part of the algorithm
        self.parameters['g'] = realizations['g'].tensor_realizations.detach()

        if self.MCMC_toolbox['priors']['v0_mean'] is not None:
            v0_mean = self.MCMC_toolbox['priors']['v0_mean']
            v0_emp = realizations['v0'].tensor_realizations.detach()
            s_v0 = self.MCMC_toolbox['priors']['s_v0']
            sigma_v0 = self.MCMC_toolbox['priors']['v0_std']
            self.parameters['v0'] = (1 / (1 / (s_v0 ** 2) + 1 / (sigma_v0 ** 2))) * (
                        v0_emp / (sigma_v0 ** 2) + v0_mean / (s_v0 ** 2))
        else:
            self.parameters['v0'] = realizations['v0'].tensor_realizations.detach()

        if self.source_dimension != 0:
            self.parameters['betas'] = realizations['betas'].tensor_realizations.detach()
        xi = realizations['xi'].tensor_realizations.detach()
        # self.parameters['xi_mean'] = torch.mean(xi)
        self.parameters['xi_std'] = torch.std(xi)
        tau = realizations['tau'].tensor_realizations.detach()
        self.parameters['tau_mean'] = torch.mean(tau)
        self.parameters['tau_std'] = torch.std(tau)

        param_ind = self.get_param_from_real(realizations)
        # TODO : Why is it MCMC-SAEM? SHouldn't it be computed with the parameters?
        squared_diff = self.compute_sum_squared_tensorized(data, param_ind, attribute_type='MCMC').sum()
        self.parameters['noise_std'] = torch.sqrt(squared_diff / (data.n_visits * data.dimension))

        # TODO : This is just for debugging of linear
        data_reconstruction = self.compute_individual_tensorized(data.timepoints,
                                                                 self.get_param_from_real(realizations),
                                                                 attribute_type='MCMC')
        norm_0 = data.values * data.values * data.mask
        norm_1 = data.values * data_reconstruction * data.mask
        norm_2 = data_reconstruction * data_reconstruction * data.mask
        S1 = torch.sum(torch.sum(norm_0, dim=2))
        S2 = torch.sum(torch.sum(norm_1, dim=2))
        S3 = torch.sum(torch.sum(norm_2, dim=2))

        # print("During burn-in : ", torch.sqrt((S1 - 2. * S2 + S3) / (data.dimension * data.n_visits)), torch.sqrt(squared_diff / (data.n_visits * data.dimension)))

        # Stochastic sufficient statistics used to update the parameters of the model

    def update_model_parameters_normal(self, data, suff_stats):
        # TODO with Raphael : check the SS, especially the issue with mean(xi) and v_k
        # TODO : 1. Learn the mean of xi and v_k
        # TODO : 2. Set the mean of xi to 0 and add it to the mean of V_k
        self.parameters['g'] = suff_stats['g']
        self.parameters['v0'] = suff_stats['v0']
        if self.source_dimension != 0:
            self.parameters['betas'] = suff_stats['betas']

        tau_mean = self.parameters['tau_mean'].detach().clone()
        tau_std_updt = torch.mean(suff_stats['tau_sqrd']) - 2 * tau_mean * torch.mean(suff_stats['tau'])
        self.parameters['tau_std'] = torch.sqrt(tau_std_updt + self.parameters['tau_mean'] ** 2)
        self.parameters['tau_mean'] = torch.mean(suff_stats['tau'])

        xi_mean = torch.tensor(self.parameters['xi_mean'])
        xi_std_updt = torch.mean(suff_stats['xi_sqrd']) - 2 * xi_mean * torch.mean(suff_stats['xi'])
        self.parameters['xi_std'] = torch.sqrt(xi_std_updt + self.parameters['xi_mean'] ** 2)
        # self.parameters['xi_mean'] = torch.mean(suff_stats['xi'])

        S1 = torch.sum(suff_stats['obs_x_obs'])
        S2 = torch.sum(suff_stats['obs_x_reconstruction'])
        S3 = torch.sum(suff_stats['reconstruction_x_reconstruction'])

        self.parameters['noise_std'] = torch.sqrt((S1 - 2. * S2 + S3) / (data.dimension * data.n_visits))

        # print("After burn-in : ", torch.sqrt((S1 - 2. * S2 + S3) / (data.dimension * data.n_visits)))

    ###################################
    ### Random Variable Information ###
    ###################################

    def random_variable_informations(self):

        ## Population variables
        g_infos = {
            "name": "g",
            "shape": torch.Size([self.dimension]),
            "type": "population",
            "rv_type": "multigaussian"
        }

        v0_infos = {
            "name": "v0",
            "shape": torch.Size([self.dimension]),
            "type": "population",
            "rv_type": "multigaussian"
        }

        betas_infos = {
            "name": "betas",
            "shape": torch.Size([self.dimension - 1, self.source_dimension]),
            "type": "population",
            "rv_type": "multigaussian"
        }

        ## Individual variables
        tau_infos = {
            "name": "tau",
            "shape": torch.Size([1]),
            "type": "individual",
            "rv_type": "gaussian"
        }

        xi_infos = {
            "name": "xi",
            "shape": torch.Size([1]),
            "type": "individual",
            "rv_type": "gaussian"
        }

        sources_infos = {
            "name": "sources",
            "shape": torch.Size([self.source_dimension]),
            "type": "individual",
            "rv_type": "gaussian"
        }

        variables_infos = {
            "g": g_infos,
            "v0": v0_infos,
            "tau": tau_infos,
            "xi": xi_infos,
        }

        if self.source_dimension != 0:
            variables_infos['sources'] = sources_infos
            variables_infos['betas'] = betas_infos
        return variables_infos
